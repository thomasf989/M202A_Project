# Changelog

## v2.0

- Added optional DINOv2 global image retrieval for large viewpoint changes.
- Made ALIKED + LightGlue the preferred local correspondence stage after Top-K retrieval.
- Preserved RootSIFT TF-IDF + PnP as a CPU baseline/fallback for ablations.
- Added sparse mesh-consistency filtering for Polycam depth-lifted 3D points.
- Added geometric pose-confidence diagnostics.
- Added MultiSet exported-GLB workflow and optional MultiSet VPS pose comparison.
- Added rigid control-point alignment between a separate Go2/Polycam frame and a MultiSet map frame.
- Added Blender synthetic reference-view generation at configurable heights/yaws.
- Added camera-only mapped-landmark temporal tracking (optical flow + PnP, no IMU).
- Added independent phone-image evaluation manifests.
- Added Windows launch scripts.
- Added unit tests for MultiSet parsing/alignment, global-ranking logic, and camera-only tracking.
