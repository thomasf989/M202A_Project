"""Render posed synthetic reference images from a textured GLB.

Run with Blender, for example:

  blender -b -P tools/blender_render_references.py -- \
    --glb multiset_room.glb --output synthetic_refs \
    --spacing 0.6 --heights 0.45 1.50 --yaw-step 30 --fov-deg 70

The output folder contains images/ and references.json. Zip those contents and
upload the ZIP together with the SAME GLB using the web app's
"posed-images / MultiSet GLB + posed refs" format.

Blender cameras use +X right, +Y up, -Z forward, so references.json declares
camera_convention="opengl". PosedCapture converts them to the project's OpenCV
camera convention when the map is loaded.
"""
import argparse
import json
import math
from pathlib import Path
import sys

try:
    import bpy
    from mathutils import Matrix
except ImportError as e:
    raise SystemExit('Run this script through Blender, not normal Python.') from e


def args_parser():
    argv=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else []
    p=argparse.ArgumentParser()
    p.add_argument('--glb',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--spacing',type=float,default=.60)
    p.add_argument('--heights',type=float,nargs='+',default=[.45,1.50],help='Meters above mesh minimum Y')
    p.add_argument('--yaw-step',type=float,default=30.)
    p.add_argument('--fov-deg',type=float,default=70.)
    p.add_argument('--width',type=int,default=768)
    p.add_argument('--height',type=int,default=576)
    p.add_argument('--margin',type=float,default=.45,help='Keep camera grid this far inside X/Z mesh bounds')
    p.add_argument('--x-min',type=float);p.add_argument('--x-max',type=float)
    p.add_argument('--z-min',type=float);p.add_argument('--z-max',type=float)
    p.add_argument('--samples',type=int,default=32)
    return p.parse_args(argv)


def matrix_rows(m):
    return [[float(m[r][c]) for c in range(4)] for r in range(4)]


def main():
    a=args_parser();out=Path(a.output).resolve();images=out/'images';images.mkdir(parents=True,exist_ok=True)
    bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
    bpy.ops.import_scene.gltf(filepath=str(Path(a.glb).resolve()))
    meshes=[o for o in bpy.context.scene.objects if o.type=='MESH']
    if not meshes:raise RuntimeError('GLB contains no mesh objects')
    corners=[]
    for o in meshes:
        corners.extend([o.matrix_world @ c for c in o.bound_box])
    xs=[p.x for p in corners];ys=[p.y for p in corners];zs=[p.z for p in corners]
    xmin=a.x_min if a.x_min is not None else min(xs)+a.margin
    xmax=a.x_max if a.x_max is not None else max(xs)-a.margin
    zmin=a.z_min if a.z_min is not None else min(zs)+a.margin
    zmax=a.z_max if a.z_max is not None else max(zs)-a.margin
    floor=min(ys)
    if xmin>=xmax or zmin>=zmax:raise RuntimeError('Bounds/margin leave no camera area')

    scene=bpy.context.scene
    try: scene.render.engine='BLENDER_EEVEE_NEXT'
    except Exception: scene.render.engine='BLENDER_EEVEE'
    scene.render.resolution_x=a.width;scene.render.resolution_y=a.height;scene.render.resolution_percentage=100
    scene.render.image_settings.file_format='PNG';scene.render.film_transparent=False
    scene.render.image_settings.color_mode='RGB'
    scene.render.resolution_percentage=100
    scene.render.image_settings.color_depth='8'
    scene.world.color=(.5,.5,.5)

    cam_data=bpy.data.cameras.new('SyntheticCamera');cam=bpy.data.objects.new('SyntheticCamera',cam_data)
    scene.collection.objects.link(cam);scene.camera=cam
    cam_data.type='PERSP';cam_data.sensor_fit='HORIZONTAL';cam_data.angle=math.radians(a.fov_deg)
    fx=.5*a.width/math.tan(math.radians(a.fov_deg)/2)
    # Square pixels; fy follows horizontal focal in pixel units.
    fy=fx
    intr=dict(width=a.width,height=a.height,fx=fx,fy=fy,cx=(a.width-1)/2,cy=(a.height-1)/2,distortion=[0,0,0,0,0])

    def values(lo,hi,step):
        n=max(1,int(math.floor((hi-lo)/step))+1)
        return [lo+i*(hi-lo)/max(n-1,1) for i in range(n)]
    xvals=values(xmin,xmax,a.spacing);zvals=values(zmin,zmax,a.spacing)
    yaws=[math.radians(v) for v in range(0,360,max(1,int(round(a.yaw_step))))]
    frames=[];i=0
    for h in a.heights:
        y=floor+h
        for x in xvals:
            for z in zvals:
                for yaw in yaws:
                    i+=1;fid=f'synth-{i:06d}'
                    cam.location=(x,y,z);cam.rotation_euler=(0,yaw,0)
                    scene.render.filepath=str(images/f'{fid}.png')
                    bpy.ops.render.render(write_still=True)
                    frames.append(dict(id=fid,image=f'images/{fid}.png',intrinsics=intr,
                        camera_to_world=matrix_rows(cam.matrix_world),split='reference'))
    manifest=dict(format='posed-images-v1',camera_convention='opengl',intrinsics=intr,
        world_to_glb=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],frames=frames,
        generator=dict(source='blender synthetic views',glb=Path(a.glb).name,spacing_m=a.spacing,
            heights_m=a.heights,yaw_step_deg=a.yaw_step,fov_deg=a.fov_deg))
    (out/'references.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(f'Rendered {len(frames)} reference images to {out}')

if __name__=='__main__':main()
