"""Reference photos with explicit intrinsics and poses, lifted on the GLB."""
from pathlib import Path
import hashlib
import re
import numpy as np
from PIL import Image
from .polycam import Frame, rigid_matrix, camera_matrix, read_json, sha256
from .seven_scenes import SevenScenesCapture


class PosedCapture:
    dataset_format='posed-images'
    split_description='Only frames marked reference/train enter the map; explicit test frames remain held out'

    def __init__(self, root):
        self.root=Path(root).resolve();path=self.root/'references.json';self.meta=read_json(path)
        if self.meta.get('format')!='posed-images-v1':raise ValueError('references.json: format must be posed-images-v1')
        self.A=rigid_matrix(self.meta.get('world_to_glb',np.eye(4)),'world_to_glb');self.raw_alignment=self.A
        self.alignment='dataset';self.depth_scale=1.;self.glb=self.root/'room.glb'
        if not self.glb.is_file():raise ValueError('Missing room.glb')
        self.frames={};self.reference_ids=[];self.query_ids=[]
        convention=self.meta.get('camera_convention','opencv')
        if convention not in ['opencv','opengl']:raise ValueError('camera_convention must be opencv or opengl')
        C=np.diag([1.,-1.,-1.,1.]) if convention=='opengl' else np.eye(4)
        for i,r in enumerate(self.meta['frames']):
            id=str(r.get('id',f'photo-{i:06d}'))
            if not re.fullmatch(r'[a-zA-Z0-9_-]{1,100}',id) or id in self.frames:raise ValueError('Invalid or duplicate photo ID')
            image=(self.root/r['image']).resolve()
            if not image.is_relative_to(self.root) or not image.is_file():raise ValueError(f'Missing or unsafe reference image: {id}')
            c=r.get('intrinsics',self.meta.get('intrinsics'))
            if c is None:raise ValueError(f'Missing camera intrinsics: {id}')
            K=camera_matrix(c);w,h=int(c['width']),int(c['height'])
            if min(w,h)<2:raise ValueError('Invalid image size')
            if np.any(c.get('distortion',[0])):raise ValueError('Reference photos must be undistorted before import')
            with Image.open(image) as im:
                if im.size!=(w,h):raise ValueError(f'Photo size differs from intrinsics: {id}')
                if im.getexif().get(274,1)!=1:raise ValueError(f'Reference image must be stored upright, with pose/intrinsics for those pixels: {id}')
            T=rigid_matrix(r['camera_to_world'],id)
            self.frames[id]=Frame(id,image,path,path,path,K,self.A@T@C,w,h)
            split=r.get('split','reference')
            if split not in ['reference','train','test']:raise ValueError('Frame split must be reference, train or test')
            (self.query_ids if split=='test' else self.reference_ids).append(id)
        self.ids=list(self.frames)
        if len(self.reference_ids)<4:raise ValueError('Need at least four posed reference photos')
        self._mesh=None;self._depth_cache={};self._ray=None


    def world_points(self, frame, pixels, min_confidence=127, edge_threshold=.75):
        """Lift floating-point reference pixels through the exported GLB mesh.

        MeshLoc bilinearly samples the rendered depth map at the exact matched
        feature coordinate.  We follow that behavior and use an exact sparse
        mesh ray cast only when the bilinear sample is missing or lies on a
        strong depth discontinuity.  This avoids both the old integer rounding
        error and the old 3x3-flat-patch rejection that discarded many useful
        object/door/furniture-edge matches.
        """
        from .raster import mesh_depth, sample_depth_bilinear, raycast_pixels
        if self._mesh is None:
            from .visualization import load_mesh
            from trimesh.ray.ray_triangle import RayMeshIntersector
            self._mesh=load_mesh(self.glb)
            self._ray=RayMeshIntersector(self._mesh)
        if frame.id not in self._depth_cache:
            if len(self._depth_cache)>=6:self._depth_cache.pop(next(iter(self._depth_cache)))
            self._depth_cache[frame.id]=mesh_depth(
                self._mesh,frame.T_world_cv,frame.K,frame.width,frame.height,near=.05,far=20.)
        uv=np.asarray(pixels,dtype=float).reshape(-1,2)
        z,valid,spread=sample_depth_bilinear(self._depth_cache[frame.id],uv,min_depth=.12,max_depth=15.)
        safe_z=np.where(valid,z,0.0)
        pc=np.column_stack([(uv[:,0]-frame.K[0,2])/frame.K[0,0]*safe_z,
                            (uv[:,1]-frame.K[1,2])/frame.K[1,1]*safe_z,safe_z])
        xyz=pc@frame.T_world_cv[:3,:3].T+frame.T_world_cv[:3,3]

        # Near silhouettes/depth discontinuities bilinear depth can blend two
        # surfaces.  Resolve those pixels with the actual mesh intersection.
        need_ray=(~valid)|(spread>float(edge_threshold))
        if need_ray.any():
            try:
                ray_xyz,ray_valid=raycast_pixels(self._ray,frame.T_world_cv,frame.K,uv[need_ray],.12,15.)
                idx=np.flatnonzero(need_ray)
                good_idx=idx[ray_valid]
                xyz[good_idx]=ray_xyz[ray_valid]
                valid[idx]=ray_valid
            except Exception:
                # If ray casting is unavailable, retain only trustworthy
                # bilinear samples and reject ambiguous discontinuity samples.
                valid &= spread<=float(edge_threshold)
        return xyz,valid

    def fingerprint(self):
        h=hashlib.sha256((self.root/'references.json').read_bytes());h.update(sha256(self.glb).encode());h.update(b'posed-mesh-depth-v3-bilinear-raycast-15m')
        for id in self.ids:h.update(sha256(self.frames[id].image).encode())
        return h.hexdigest()
