"""Generate a high-coverage posed reference set from a textured MultiSet GLB.

Designed for random indoor phone viewpoints while keeping the MeshLoc-style
localizer unchanged.  The default ``random-phone`` profile deliberately covers:

* dense XY camera centers,
* standing / seated / high / Go2-like heights,
* full 360-degree yaw,
* upward/downward pitch variants,
* only camera centers that are not embedded in geometry,
* only views with useful geometric visibility,
* a bounded, spatially balanced reference count.

Blender 5.2+ PowerShell example::

  & "C:\\Program Files\\Blender Foundation\\Blender 5.2\\blender.exe" `
      -b -P .\\tools\\blender_render_references.py -- `
      --glb .\\data\\multiset\\apartment.glb `
      --output .\\synthetic_refs_random `
      --profile random-phone --max-views 2400

The output contains ``images/`` and ``references.json``.  Zip those two items
and upload the ZIP together with the SAME GLB in the local web app.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

try:
    import bpy
    from mathutils import Matrix, Vector
except ImportError as exc:  # pragma: no cover - Blender-only script
    raise SystemExit("Run this script through Blender, not normal Python.") from exc


# Blender glTF import maps the original GLB Y-up frame into Blender's Z-up frame.
BLENDER_WORLD_TO_GLB = Matrix((
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, -1.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
))


@dataclass(frozen=True)
class ViewSpec:
    x: float
    y: float
    z: float
    height: float
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    role: str
    ix: int
    iy: int

    @property
    def position_key(self):
        return (self.ix, self.iy, round(self.height, 3), self.role)


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--glb", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--profile", choices=["random-phone", "uniform"], default="random-phone")
    p.add_argument("--spacing", type=float, default=0.65,
                   help="Base XY spacing in metres. 0.55-0.75 is a strong indoor range.")
    p.add_argument("--heights", type=float, nargs="+", default=[0.45, 1.10, 1.45, 1.70],
                   help="Camera heights above the estimated floor.")
    p.add_argument("--primary-height", type=float, default=1.45,
                   help="Main phone-height layer used most densely.")
    p.add_argument("--yaw-step", type=float, default=30.0,
                   help="Core standing-phone yaw spacing.")
    p.add_argument("--pitches", type=float, nargs="+", default=[-25.0, -10.0, 0.0, 10.0, 25.0],
                   help="Pitch family. Negative looks downward, positive upward.")
    p.add_argument("--rolls", type=float, nargs="+", default=[0.0],
                   help="Optional camera roll angles. Keep 0 unless testing strongly rolled phones.")
    p.add_argument("--fov-deg", type=float, default=95.0,
                   help="Wide horizontal FOV gives overlap with normal phone lenses.")
    p.add_argument("--width", type=int, default=768)
    p.add_argument("--height", type=int, default=768)
    p.add_argument("--margin", type=float, default=0.28)
    p.add_argument("--clearance", type=float, default=0.10,
                   help="Reject camera centers too close to walls/furniture on multiple sides.")
    p.add_argument("--forward-clearance", type=float, default=0.22,
                   help="Reject a view when geometry is immediately in front of the lens.")
    p.add_argument("--min-hit-ratio", type=float, default=0.42,
                   help="Minimum fraction of coarse view rays that should hit mapped geometry.")
    p.add_argument("--ray-grid", type=int, default=3,
                   help="Odd ray fan size for geometric visibility scoring; 3 or 5 recommended.")
    p.add_argument("--max-views", type=int, default=2400,
                   help="Hard database-size budget. Candidate views are diversity-sampled to fit.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true",
                   help="Plan and score the reference set without rendering images.")
    p.add_argument("--x-min", type=float)
    p.add_argument("--x-max", type=float)
    p.add_argument("--y-min", "--z-min", dest="y_min", type=float)
    p.add_argument("--y-max", "--z-max", dest="y_max", type=float)
    return p.parse_args(argv)


def matrix_rows(m):
    return [[float(m[r][c]) for c in range(4)] for r in range(4)]


def values(lo, hi, step):
    n = max(1, int(math.floor((hi - lo) / step)) + 1)
    if n == 1:
        return [(lo + hi) * 0.5]
    return [lo + i * (hi - lo) / (n - 1) for i in range(n)]


def nearest(values_, target):
    return min(values_, key=lambda value: abs(float(value) - float(target)))


def stable_phase(spec):
    token = f"{spec.ix}:{spec.iy}:{spec.height:.3f}:{spec.role}:{spec.pitch_deg:.1f}:{spec.roll_deg:.1f}"
    return int(hashlib.sha256(token.encode()).hexdigest()[:8], 16)


def round_robin_take(items, count):
    """Take views while spreading selections across camera centers and yaw."""
    if count <= 0 or not items:
        return []
    groups = defaultdict(list)
    for item in items:
        groups[item.position_key].append(item)
    queues = []
    for key in sorted(groups):
        group = sorted(groups[key], key=lambda s: (s.pitch_deg, s.yaw_deg, s.roll_deg))
        if group:
            shift = stable_phase(group[0]) % len(group)
            group = group[shift:] + group[:shift]
        queues.append(deque(group))
    output = []
    while queues and len(output) < count:
        next_queues = []
        for queue in queues:
            if queue and len(output) < count:
                output.append(queue.popleft())
            if queue:
                next_queues.append(queue)
        queues = next_queues
    return output


def allocate_budget(buckets, max_views):
    """Stratified budget preserving the most useful random-phone pose families."""
    weights = {
        "core": 0.46,          # standing phone, horizontal, dense yaw
        "tilt": 0.27,          # standing phone, up/down pitch
        "alt_height": 0.18,    # seated/high/Go2 horizontal
        "alt_tilt": 0.09,      # sparse pitch variants at alternate heights
    }
    selected = []
    used = set()
    for role, weight in weights.items():
        quota = int(round(max_views * weight))
        for item in round_robin_take(buckets.get(role, []), quota):
            if item not in used:
                selected.append(item); used.add(item)
    # Redistribute unused budget across any remaining views in balanced role order.
    role_order = ["core", "tilt", "alt_height", "alt_tilt"]
    leftovers = {role: [item for item in buckets.get(role, []) if item not in used] for role in role_order}
    while len(selected) < max_views and any(leftovers.values()):
        progressed = False
        for role in role_order:
            if leftovers[role] and len(selected) < max_views:
                item = leftovers[role].pop(0)
                selected.append(item); used.add(item); progressed = True
        if not progressed:
            break
    return selected[:max_views]


def main():
    a = parse_args()
    if a.spacing <= 0 or a.yaw_step <= 0 or not (35 <= a.fov_deg < 150):
        raise ValueError("spacing/yaw-step must be positive and FOV should be 35..149 degrees")
    if a.ray_grid < 1 or a.ray_grid % 2 == 0:
        raise ValueError("--ray-grid must be an odd positive integer")
    if a.max_views < 24:
        raise ValueError("--max-views must be at least 24")

    glb = Path(a.glb).expanduser().resolve()
    if not glb.is_file():
        raise FileNotFoundError(f"GLB not found: {glb}")
    out = Path(a.output).expanduser().resolve()
    images = out / "images"
    images.mkdir(parents=True, exist_ok=True)
    for path in images.glob("synth-*.png"):
        path.unlink()

    print(f"[INFO] Importing GLB: {glb}")
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.import_scene.gltf(filepath=str(glb))
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not meshes:
        raise RuntimeError("GLB contains no mesh objects")

    corners, world_vertex_z = [], []
    for obj in meshes:
        corners.extend(obj.matrix_world @ Vector(corner) for corner in obj.bound_box)
        for vertex in obj.data.vertices:
            world_vertex_z.append((obj.matrix_world @ vertex.co).z)
    xs = [p.x for p in corners]; ys = [p.y for p in corners]; zs = [p.z for p in corners]
    floor_z = float(np.percentile(np.asarray(world_vertex_z, float), 1.0)) if world_vertex_z else min(zs)
    ceiling_z = float(np.percentile(np.asarray(world_vertex_z, float), 99.5)) if world_vertex_z else max(zs)
    xmin = a.x_min if a.x_min is not None else min(xs) + a.margin
    xmax = a.x_max if a.x_max is not None else max(xs) - a.margin
    ymin = a.y_min if a.y_min is not None else min(ys) + a.margin
    ymax = a.y_max if a.y_max is not None else max(ys) - a.margin
    if xmin >= xmax or ymin >= ymax:
        raise RuntimeError("Bounds/margin leave no camera area. Reduce --margin or specify explicit bounds.")

    print("[INFO] Imported Blender bounds:")
    print(f"       X: {min(xs):.3f} .. {max(xs):.3f} m")
    print(f"       Y: {min(ys):.3f} .. {max(ys):.3f} m")
    print(f"       Z: {min(zs):.3f} .. {max(zs):.3f} m (up)")
    print(f"[INFO] Robust floor estimate: {floor_z:.3f} m; upper geometry: {ceiling_z:.3f} m")

    scene = bpy.context.scene
    engine_set = False
    for engine in ("BLENDER_EEVEE", "BLENDER_EEVEE_NEXT", "BLENDER_WORKBENCH"):
        try:
            scene.render.engine = engine
            engine_set = True
            print(f"[INFO] Render engine: {engine}")
            break
        except Exception:
            pass
    if not engine_set:
        raise RuntimeError("Could not select a Blender render engine")

    scene.render.resolution_x = a.width
    scene.render.resolution_y = a.height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.render.film_transparent = False
    scene.world.use_nodes = True
    bg = scene.world.node_tree.nodes.get("Background") if scene.world.node_tree else None
    if bg is not None:
        bg.inputs["Color"].default_value = (0.65, 0.65, 0.65, 1.0)
        bg.inputs["Strength"].default_value = 0.85

    cam_data = bpy.data.cameras.new("SyntheticCamera")
    cam = bpy.data.objects.new("SyntheticCamera", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    cam_data.type = "PERSP"
    cam_data.sensor_fit = "HORIZONTAL"
    cam_data.angle = math.radians(a.fov_deg)
    cam_data.clip_start = 0.05
    cam_data.clip_end = 20.0
    fx = 0.5 * a.width / math.tan(math.radians(a.fov_deg) / 2.0)
    # Blender uses square pixels here, so fx == fy in pixel units.
    fy = fx
    intr = dict(width=a.width, height=a.height, fx=fx, fy=fy,
                cx=(a.width - 1) / 2.0, cy=(a.height - 1) / 2.0,
                distortion=[0, 0, 0, 0, 0])

    xvals = values(xmin, xmax, a.spacing)
    yvals = values(ymin, ymax, a.spacing)
    usable_heights = []
    for height in a.heights:
        z = floor_z + float(height)
        if z <= floor_z + 0.12 or z >= ceiling_z - 0.08:
            print(f"[WARN] Skipping height {height:.2f} m because it lies outside usable vertical extent")
        else:
            usable_heights.append(float(height))
    if not usable_heights:
        raise RuntimeError("No requested camera heights lie inside the scan")
    primary_height = nearest(usable_heights, a.primary_height)

    depsgraph = bpy.context.evaluated_depsgraph_get()

    def ray_distance(origin, direction, max_distance=20.0):
        direction = Vector(direction).normalized()
        try:
            hit, loc, _normal, _index, _obj, _matrix = scene.ray_cast(
                depsgraph, Vector(origin), direction, distance=float(max_distance))
        except TypeError:
            hit, loc, _normal, _index, _obj, _matrix = scene.ray_cast(
                depsgraph, Vector(origin), direction, float(max_distance))
        return float((loc - Vector(origin)).length) if hit else None

    # Remove centers that appear embedded in geometry.
    centers = []
    clearance_dirs = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(1,1,0),(1,-1,0),(-1,1,0),(-1,-1,0)]
    for ix, x in enumerate(xvals):
        for iy, y in enumerate(yvals):
            for height in usable_heights:
                z = floor_z + height
                near = 0
                for direction in clearance_dirs:
                    distance = ray_distance((x, y, z), direction, max(0.16, a.clearance * 1.7))
                    if distance is not None and distance < a.clearance:
                        near += 1
                if near < 2:
                    centers.append((float(x), float(y), float(z), height, ix, iy))
    print(f"[INFO] Usable camera centers: {len(centers)}")

    yaw_core = list(np.arange(0.0, 360.0, a.yaw_step, dtype=float))
    yaw_medium = list(np.arange(0.0, 360.0, max(60.0, a.yaw_step * 2.0), dtype=float))
    yaw_sparse = list(np.arange(0.0, 360.0, max(90.0, a.yaw_step * 3.0), dtype=float))
    zero_pitch = nearest(a.pitches, 0.0)
    nonzero_pitches = [float(p) for p in a.pitches if abs(float(p) - zero_pitch) > 1e-6]

    buckets = defaultdict(list)
    for x, y, z, height, ix, iy in centers:
        is_primary = abs(height - primary_height) < 1e-6
        checker = (ix + iy) % 2
        sparse_checker = (ix + 2 * iy) % 3
        if a.profile == "uniform":
            for yaw in yaw_core:
                for pitch in a.pitches:
                    for roll in a.rolls:
                        role = "core" if is_primary and abs(pitch - zero_pitch) < 1e-6 else "tilt"
                        buckets[role].append(ViewSpec(x,y,z,height,float(yaw),float(pitch),float(roll),role,ix,iy))
            continue

        if is_primary:
            # Core: every valid XY location, every core yaw, horizontal view.
            for yaw in yaw_core:
                for roll in a.rolls:
                    buckets["core"].append(ViewSpec(x,y,z,height,float(yaw),float(zero_pitch),float(roll),"core",ix,iy))
            # Tilt views use a coarser yaw schedule but still cover all locations.
            for pitch in nonzero_pitches:
                for yaw in yaw_medium:
                    buckets["tilt"].append(ViewSpec(x,y,z,height,float(yaw),pitch,0.0,"tilt",ix,iy))
        else:
            # Alternate heights are spatially staggered so we gain vertical coverage
            # without multiplying the database by every possible height.
            if checker == 0:
                for yaw in yaw_medium:
                    buckets["alt_height"].append(ViewSpec(x,y,z,height,float(yaw),float(zero_pitch),0.0,"alt_height",ix,iy))
            if sparse_checker == 0 and nonzero_pitches:
                # Only the largest up/down pitches add meaningful new vertical coverage.
                extremes = sorted(nonzero_pitches, key=lambda value: abs(value), reverse=True)[:2]
                for pitch in extremes:
                    for yaw in yaw_sparse:
                        buckets["alt_tilt"].append(ViewSpec(x,y,z,height,float(yaw),pitch,0.0,"alt_tilt",ix,iy))

    raw_count = sum(len(items) for items in buckets.values())
    planned = allocate_budget(buckets, a.max_views) if raw_count > a.max_views else [
        item for role in ("core","tilt","alt_height","alt_tilt") for item in buckets.get(role, [])]
    print(f"[INFO] Candidate views before budget: {raw_count}")
    print(f"[INFO] Pose-diverse planned views: {len(planned)} (budget {a.max_views})")
    print(f"[INFO] Planned roles: {dict(Counter(spec.role for spec in planned))}")

    def forward_vector(yaw_deg, pitch_deg):
        yaw = math.radians(float(yaw_deg)); pitch = math.radians(float(pitch_deg))
        return Vector((math.cos(pitch) * math.cos(yaw),
                       math.cos(pitch) * math.sin(yaw),
                       math.sin(pitch))).normalized()

    def view_geometry_score(spec):
        forward = forward_vector(spec.yaw_deg, spec.pitch_deg)
        close = ray_distance((spec.x, spec.y, spec.z), forward, max(0.5, a.forward_clearance * 1.8))
        if close is not None and close < a.forward_clearance:
            return None
        # Fan rays approximate whether the view actually overlaps mapped surfaces.
        offsets = np.linspace(-0.36 * a.fov_deg, 0.36 * a.fov_deg, a.ray_grid)
        depths = []
        for dyaw in offsets:
            for dpitch in offsets:
                ray = forward_vector(spec.yaw_deg + float(dyaw), spec.pitch_deg + float(dpitch))
                distance = ray_distance((spec.x, spec.y, spec.z), ray, 20.0)
                if distance is not None:
                    depths.append(distance)
        hit_ratio = len(depths) / float(a.ray_grid * a.ray_grid)
        if hit_ratio < a.min_hit_ratio:
            return None
        if depths:
            depth_span = float(np.percentile(depths, 90) - np.percentile(depths, 10))
            median_depth = float(np.median(depths))
        else:
            depth_span = 0.0; median_depth = float("nan")
        score = 0.70 * hit_ratio + 0.30 * min(1.0, depth_span / 3.0)
        return dict(hit_ratio=hit_ratio, depth_span_m=depth_span,
                    median_depth_m=median_depth, geometry_score=score)

    # Geometric filtering happens after diversity sampling, so expensive ray fans
    # are bounded by --max-views rather than every raw candidate.
    valid_specs = []
    skipped = Counter()
    for index, spec in enumerate(planned, 1):
        quality = view_geometry_score(spec)
        if quality is None:
            skipped["geometry"] += 1
            continue
        valid_specs.append((spec, quality))
        if index % 100 == 0 or index == len(planned):
            print(f"[INFO] View validation {index}/{len(planned)}; valid={len(valid_specs)}")

    if len(valid_specs) < 24:
        raise RuntimeError(f"Only {len(valid_specs)} valid views remain; reduce clearance/min-hit-ratio or inspect GLB scale.")
    print(f"[INFO] Final valid reference count: {len(valid_specs)}")
    print(f"[INFO] Final roles: {dict(Counter(spec.role for spec, _ in valid_specs))}")
    if a.dry_run:
        report = dict(profile=a.profile, raw_candidates=raw_count, planned=len(planned),
                      valid=len(valid_specs), roles=dict(Counter(spec.role for spec, _ in valid_specs)),
                      skipped=dict(skipped), settings=vars(a))
        out.mkdir(parents=True, exist_ok=True)
        (out / "generation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[DONE] Dry run only. Report: {out / 'generation_report.json'}")
        return

    frames = []
    for rendered, (spec, quality) in enumerate(valid_specs, 1):
        fid = f"synth-{rendered:06d}"
        forward = forward_vector(spec.yaw_deg, spec.pitch_deg)
        cam.location = (spec.x, spec.y, spec.z)
        cam.rotation_mode = "QUATERNION"
        quat = forward.to_track_quat("-Z", "Y")
        if abs(spec.roll_deg) > 1e-6:
            # Rotate around the world-space viewing direction after look-at orientation.
            roll_q = Matrix.Rotation(math.radians(spec.roll_deg), 4, forward).to_quaternion()
            quat = roll_q @ quat
        cam.rotation_quaternion = quat
        scene.render.filepath = str(images / f"{fid}.png")
        print(f"[RENDER] {rendered}/{len(valid_specs)} role={spec.role} "
              f"center=({spec.x:.2f},{spec.y:.2f},{spec.z:.2f}) "
              f"yaw={spec.yaw_deg:.0f} pitch={spec.pitch_deg:.0f}")
        bpy.ops.render.render(write_still=True)
        frames.append(dict(
            id=fid,
            image=f"images/{fid}.png",
            intrinsics=intr,
            camera_to_world=matrix_rows(cam.matrix_world),
            split="reference",
            synthetic=dict(
                role=spec.role,
                height_m=spec.height,
                yaw_deg=spec.yaw_deg,
                pitch_deg=spec.pitch_deg,
                roll_deg=spec.roll_deg,
                fov_deg=a.fov_deg,
                **quality,
            ),
        ))

    manifest = dict(
        format="posed-images-v1",
        camera_convention="opengl",
        intrinsics=intr,
        world_to_glb=matrix_rows(BLENDER_WORLD_TO_GLB),
        frames=frames,
        generator=dict(
            source="blender random-phone reference generator v5",
            glb=glb.name,
            profile=a.profile,
            spacing_m=a.spacing,
            heights_m=usable_heights,
            primary_height_m=primary_height,
            yaw_step_deg=a.yaw_step,
            pitches_deg=a.pitches,
            rolls_deg=a.rolls,
            fov_deg=a.fov_deg,
            raw_candidate_views=raw_count,
            planned_views=len(planned),
            rendered_views=len(frames),
            role_counts=dict(Counter(frame["synthetic"]["role"] for frame in frames)),
            blender_up_axis="Z",
            glb_up_axis="Y",
            note=("Reference set optimized for arbitrary indoor phone viewpoints. "
                  "Dense horizontal standing-phone views are combined with pitch, seated/high, "
                  "and Go2-height views under a bounded diversity budget."),
        ),
    )
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "references.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    report = dict(raw_candidates=raw_count, planned=len(planned), rendered=len(frames),
                  roles=manifest["generator"]["role_counts"], skipped=dict(skipped), settings=vars(a))
    (out / "generation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[DONE] Rendered {len(frames)} reference images to {out}")
    print(f"[DONE] Wrote pose manifest: {manifest_path}")
    print(f"[DONE] Wrote generation report: {out / 'generation_report.json'}")


if __name__ == "__main__":
    main()
