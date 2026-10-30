# Visual localization v2.0 — MultiSet baseline + our own RGB localizer

Estimate a phone camera's metric 6-DoF pose inside a mapped room from a new RGB photo. This version is designed for the M202A project direction: **MultiSet is the commercial VPS baseline, while the project contribution is an independent visual-localization pipeline built on open/exported map geometry.**

## What changed from v1.4

- **Learned global retrieval:** optional DINOv2 ranks the most likely reference views before expensive matching.
- **Learned local matching is now the preferred path:** ALIKED + LightGlue runs on the Top-K candidate views instead of only being a late fallback.
- **Metric pose is still geometric:** learned features only find correspondences; PnP + RANSAC recovers the camera pose from 2D-to-3D matches.
- **Stronger 3D landmarks:** Polycam sensor depth is filtered against sparse ray intersections with the exported room mesh to reject large RGB-depth association errors.
- **Pose-quality diagnostics:** each successful solution reports a geometric `pose_confidence` score, inlier ratio, image-grid coverage, supporting references, and reprojection error.
- **MultiSet integration:** a MultiSet-exported textured GLB can be used as the room model, and a MultiSet VPS pose JSON can be attached to a query for baseline comparison. MultiSet is never fed into our solver.
- **Synthetic reference generator:** a Blender script can render posed reference images from a MultiSet/Polycam GLB at robot-height and phone-height viewpoints.
- **Control-point alignment utility:** fit a rigid transform when your Go2/Polycam reference poses live in a different frame from the MultiSet map.
- **Camera calibration remains first-class:** checkerboard calibration is supported and strongly recommended for meaningful metric evaluation.

## Recommended project architecture

```text
MultiSet exported textured GLB
          |
          +----> synthetic/real posed reference images
                        |
                        v
Live iPhone RGB -> DINOv2 retrieval -> Top-K views
                                     -> ALIKED + LightGlue
                                     -> 2D / 3D correspondences
                                     -> PnP + RANSAC
                                     -> OUR 6-DoF pose

Same query image -> MultiSet VPS -> MultiSet baseline pose
                                      |
OUR pose <-----------------------------+
         compare position / rotation / latency
```

The important research contribution is the **RGB-only cross-view localizer**, especially under large viewpoint differences between a low Unitree Go2 camera and a human-held iPhone.

## Start the local web app

Use Python 3.11 or 3.12.

macOS/Linux:

```bash
bash run_web.command
```

Windows:

```bat
run_web_windows.bat
```

The launcher creates `.venv`, installs the enhanced dependencies, and opens the local app. The first learned localization may download ALIKED/LightGlue and DINOv2 weights. Later runs use local caches.

1. Select **Add room**.
2. Upload a textured GLB and a reference-data ZIP.
3. Import calibrated iPhone intrinsics if available.
4. Upload a new phone image.
5. Optionally attach the MultiSet VPS pose JSON for that same image.
6. Localize, visualize the camera in 3D, and export the result.

## Using a MultiSet map export

MultiSet exports open map assets such as textured GLB/point-cloud formats, but the proprietary VPS feature map is not part of the export. Therefore a GLB by itself is **not** an image-localization database.

A practical workflow is:

1. Export the **textured GLB** from MultiSet.
2. Create known-pose reference images in the same map coordinate frame:
   - use your own registered Go2/iPhone reference frames, or
   - render synthetic views from the GLB with the included Blender tool.
3. Upload the MultiSet GLB + posed reference ZIP as a room.
4. Run our RGB-only localizer.
5. Attach MultiSet's pose result only for comparison.

See [`docs/MULTISET.md`](docs/MULTISET.md).

### Generate synthetic reference views

Example:

```bash
blender -b -P tools/blender_render_references.py -- \
  --glb multiset_room.glb \
  --output synthetic_refs \
  --spacing 0.60 \
  --heights 0.45 1.50 \
  --yaw-step 30 \
  --fov-deg 70
```

The output contains `images/` and `references.json`. Zip those contents and upload them with the same GLB. The two heights are useful for deliberately covering both Go2-like and phone-like viewpoints.

## Supported reference data

- Polycam raw export with corrected RGB, intrinsics, poses, depth and alignment metadata.
- Official/prepared 7-Scenes data.
- `posed-images-v1`: registered RGB reference images with known camera poses and intrinsics plus a metric GLB in the same coordinate frame.
- MultiSet-exported GLB + your own `posed-images-v1` references (real or synthetic).

A ZIP of ordinary photos with no camera pose information is not enough.

## Localization hierarchy

Preferred enhanced path:

```text
Query RGB
  -> DINOv2 global embedding
  -> Top-K reference retrieval
  -> ALIKED keypoints/descriptors
  -> LightGlue matching
  -> reference depth / mesh 3D lifting
  -> 2D-to-3D correspondences
  -> PnP + RANSAC + LM refinement
  -> geometric validity checks
  -> 6-DoF pose + confidence diagnostics
```

If DINOv2 is unavailable, the app keeps ALIKED/LightGlue but uses the existing RootSIFT TF-IDF database to shortlist candidate views. If learned matching is unavailable or fails, the CPU RootSIFT + PnP baseline is still attempted.

This separation is useful for an ablation table:

- RootSIFT retrieval + RootSIFT matching + PnP
- RootSIFT retrieval + LightGlue + PnP
- DINOv2 retrieval + LightGlue + PnP
- MultiSet commercial VPS baseline

## Camera calibration

For real metric accuracy, calibrate the exact iPhone lens/resolution used in the experiment. Do not mix 0.5x, 1x and telephoto images under one calibration file.

```bash
python -m wifi_localization calibrate \
  --images calibration_photos \
  --cols 9 --rows 6 --square-m 0.025 \
  --output iphone_intrinsics.json
```

Then import the JSON in the web app or use it with the CLI.

## CLI examples

```bash
python -m wifi_localization --help
```

Localize with the enhanced hierarchy and compare against a MultiSet baseline:

```bash
python -m wifi_localization localize \
  --data data/room \
  --db outputs/database \
  --query phone.jpg \
  --intrinsics iphone_intrinsics.json \
  --multiset-pose multiset_pose.json \
  --output outputs/query_01
```

Force the old CPU baseline for an ablation:

```bash
python -m wifi_localization localize ... --baseline-only
```

Align a separate Go2/Polycam map frame to MultiSet from corresponding control points:

```bash
python -m wifi_localization align-multiset \
  --points control_points.json \
  --output multiset_alignment.json
```

## Evaluation recommendations for Midterm Check 2

Collect 20–50 independent phone images and group them by viewpoint difficulty, for example:

- similar height/view to references;
- phone-height vs Go2-height reference images;
- large yaw/pitch changes;
- partial occlusion;
- low texture / repetitive areas.

Report:

- pose-return rate;
- position error and rotation error when independent ground truth exists;
- difference from MultiSet baseline;
- inlier count / inlier ratio;
- geometric pose confidence;
- processing latency;
- success rate by viewpoint condition.

**Do not call MultiSet ground truth.** It is a commercial comparison baseline. For final absolute accuracy, use an independent reference such as measured control points, AprilTags, UWB/motion capture, or a calibrated test rig.

## Dependencies

`requirements.txt` installs the CPU baseline. `requirements-enhanced.txt` adds PyTorch/torchvision/Kornia for ALIKED, LightGlue and DINOv2 inference.

Manual setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-enhanced.txt
python -m wifi_localization web --output outputs/web
```

For an offline DINOv2 first run, clone the DINOv2 repository separately and set `DINOV2_REPO` to its local path before starting the app. Model weights also need to be present in the PyTorch cache.

## Tests

```bash
python -m unittest discover -s tests -v
```

The suite covers geometric pose recovery, camera handling, dataset separation, room import/upload safety, MultiSet baseline parsing/alignment, retrieval ranking, and map-depth handling. Tests requiring optional learned weights or external real-image databases skip when those inputs are unavailable.

## Third-party components

Vendored LightGlue and ALIKED inference files retain their upstream notices under `wifi_localization/_vendor/lightglue/`. DINOv2 is loaded at runtime and is not vendored in this repository. Follow the applicable licenses for models, datasets and map providers.
