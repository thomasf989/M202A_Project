# Independent phone-image evaluation

Use `evaluate-phones` for Midterm Check 2 rather than evaluating only images held out from the same room scan.

Example `phone_queries.json`:

```json
{
  "queries": [
    {
      "id": "phone_high_01",
      "image": "queries/high_01.jpg",
      "intrinsics": "iphone_intrinsics.json",
      "ground_truth": {
        "T_world_camera_cv": [[1,0,0,1.2],[0,1,0,1.5],[0,0,1,2.0],[0,0,0,1]]
      },
      "multiset_pose": "multiset/high_01.json"
    }
  ]
}
```

`ground_truth` is optional. `multiset_pose` is optional and is treated as a commercial baseline, not ground truth.

Run:

```bash
python -m wifi_localization evaluate-phones \
  --data data/room \
  --db outputs/database \
  --manifest phone_queries.json \
  --output outputs/phone_evaluation
```

The output includes per-query JSON files, `results.csv`, and `summary.json`.

For the cross-view experiment, label your query IDs by condition (for example `low`, `phone_height`, `occluded`, `large_yaw`) and compute success/error by condition in your report.

## Camera-only temporal tracking

The module `wifi_localization.tracking.MappedFeatureTracker` can maintain pose between global relocalizations without an IMU. It starts from the 2D/3D inliers of a successful global pose, tracks the query pixels with pyramidal Lucas-Kanade optical flow, and runs metric PnP again in the next frame. When too few landmarks survive, it returns `needs_global_relocalization=true` so the application can run DINOv2 + LightGlue again.
