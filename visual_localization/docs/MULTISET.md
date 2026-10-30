# MultiSet integration: use it as a map and commercial baseline

This version intentionally does **not** call MultiSet inside our pose solver. That separation makes the project contribution defensible: MultiSet is the commercial VPS baseline; our algorithm runs on exported/open geometry and our own reference images.

## Recommended workflow

1. Build/test a MultiSet map and export the **textured GLB**. A MultiSet map export can also include point-cloud/splat assets, but this project uses GLB because it gives metric geometry plus texture.
2. Create your own reference image database in that same GLB coordinate frame. Two easy choices are:
   - real posed RGB images from your own camera/robot pipeline, packaged as `posed-images-v1`; or
   - synthetic views rendered from the exported GLB with `tools/blender_render_references.py`.
3. In the web app, add a room using:
   - **3D model:** the MultiSet-exported GLB;
   - **reference ZIP:** the ZIP containing `references.json` + `images/`.
4. Calibrate the iPhone camera and import its intrinsics JSON.
5. Localize a new phone RGB photo with our hierarchy:
   - DINOv2 global retrieval (coarse place recognition),
   - ALIKED + LightGlue local correspondence,
   - GLB/reference depth -> 2D-to-3D points,
   - PnP + RANSAC -> metric 6-DoF pose.
6. Optionally attach the MultiSet VPS pose JSON for the **same query image**. The app reports the position difference (and orientation difference when the camera convention is explicitly known). MultiSet is labelled as a baseline, not ground truth.

## Why a GLB alone is not enough

The exported mesh/point cloud is geometry, not MultiSet's proprietary VPS feature database. Our localizer still needs image appearance associated with known camera poses. Synthetic rendering is one way to create that appearance database without depending on the proprietary VPS internals.

## Pose JSON

A raw MultiSet result containing `poseFound`, `position`, `rotation`, `confidence`, and `mapIds` can be attached for position comparison. For orientation comparison, wrap it and declare the camera convention explicitly:

```json
{
  "camera_convention": "opengl",
  "multiset": {
    "poseFound": true,
    "position": {"x": 1.2, "y": 1.5, "z": -0.8},
    "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
    "confidence": 0.91,
    "mapIds": ["YOUR_MAP_ID"]
  }
}
```

Use a right-handed map-space response when comparing with the exported GLB. If your SDK/platform returns a different coordinate convention, convert it first rather than silently comparing incompatible frames.

## Aligning a separate reference map to the MultiSet map

If your reference poses come from Polycam/Go2 in a different world frame, collect at least three corresponding 3D control points in both frames, then run:

```bash
python -m wifi_localization align-multiset --points control_points.json --output multiset_alignment.json
```

`control_points.json`:

```json
{
  "source_points": [[0,0,0], [1,0,0], [0,0,1]],
  "target_points": [[2,0,3], [3,0,3], [2,0,4]]
}
```

The command estimates a rigid transform and reports RMSE. For research evaluation, use more than three well-spread points.

## Evaluation warning

MultiSet is a strong commercial reference, but it is not perfect ground truth. For the final report, a measured camera rig, AprilTags/control points, motion capture, or another independent ground-truth system is preferable for absolute accuracy.
