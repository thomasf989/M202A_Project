# M202A Project — Cross-View RGB Localization with MultiSet Baseline

This repository contains the current M202A visual-localization project for localizing a smartphone camera inside a mapped indoor environment and studying **cross-view localization between a human-held iPhone and a low-mounted Unitree Go2 camera**.

The project uses **MultiSet only as a commercial VPS baseline and an optional source of exported 3D geometry**. We do **not** train on, copy, or access MultiSet's proprietary localization database. Instead, our system builds and uses **its own reference database** of RGB views, known camera poses, and depth/3D map information, and then runs our own retrieval, feature matching, and geometric pose-estimation pipeline.

> **Main research question:** Can an RGB-only smartphone localize itself inside a 3D map despite a large viewpoint difference between a human-held phone and a quadruped robot, without relying on phone GPS or IMU in our own localization method?

The project website is in [`index.html`](index.html). The implementation is in [`visual_localization/`](visual_localization/).

---

## Current contribution

MultiSet is **not** treated as the project contribution. It provides:

- a commercial VPS baseline for comparison;
- an exported 3D environment such as a textured `.GLB`;
- an optional **baseline pose** for the same query image, used only for comparison rather than as ground truth.

Our own localization stack performs:

1. **Learned global image retrieval** using DINOv2 when available.
2. **Top-K reference selection** instead of matching against the entire database.
3. **ALIKED + LightGlue** local feature matching.
4. **2D-to-3D correspondence construction** using registered depth / map geometry.
5. **PnP + RANSAC** metric 6-DoF phone pose estimation.
6. **Pose validation and confidence diagnostics** using inlier ratio, reprojection error, image coverage, and reference support.
7. **Camera-only temporal tracking** using optical flow + mapped 3D landmarks + PnP, without requiring IMU measurements.
8. **Optional comparison against MultiSet** for position, orientation, and latency.

The older RootSIFT method remains available as a baseline so we can measure whether the learned pipeline actually improves localization.

---

## System architecture

```text
                    MAPPED ENVIRONMENT
             MultiSet / Polycam / Go2 map
                         |
                         v
               Exported textured GLB
                         |
              +----------+----------+
              |                     |
              v                     v
     Real registered views    Synthetic views
                           robot + phone heights
              |                     |
              +----------+----------+
                         |
                         v
                Reference database
                         |
                         v
Live iPhone RGB -> DINOv2 global retrieval
                         |
                         v
                Top-K reference views
                         |
                         v
                ALIKED + LightGlue
                         |
                         v
                 2D <-> 3D matches
                         |
                         v
                  PnP + RANSAC
                         |
                         v
              OUR RGB-only 6-DoF pose
                         |
                         +---------------------+
                                               |
Same phone image -> MultiSet VPS -> baseline pose
                                               |
                         +---------------------+
                         v
             Compare accuracy / latency
```

The key research challenge is the **large viewpoint difference** between the Go2 and the iPhone. A Go2 camera may observe the room from roughly robot height, while a phone is normally held around human chest/head height.

---

## Why this improves the original project

The original pipeline relied primarily on traditional RootSIFT retrieval and matching. That works as a useful baseline, but it can break when the phone and reference camera see the same room from very different heights and orientations.

The updated version improves the system in several places:

- **DINOv2 retrieval** improves coarse place recognition under larger appearance/viewpoint changes.
- **ALIKED + LightGlue** provides stronger local correspondence matching than a SIFT-only pipeline.
- **Top-K hierarchical localization** avoids running expensive matching against every reference image.
- **Mesh/depth consistency checks** reject unreliable 3D landmarks.
- **Calibrated camera intrinsics** improve metric PnP accuracy.
- **Synthetic reference rendering** deliberately covers robot-height and phone-height viewpoints.
- **Pose-confidence checks** allow the system to reject weak solutions rather than confidently returning a bad pose.
- **Temporal tracking** reduces repeated global localization and supports smoother camera-only operation.

---

## Using MultiSet with this project

### What MultiSet is used for

Recommended use:

```text
MultiSet mapping / VPS
      |
      +--> Export textured GLB -------------------+
      |                                           |
      +--> MultiSet VPS pose (optional baseline)  |
                                                  v
                                         OUR localization pipeline
```

A MultiSet-exported GLB provides useful **open 3D geometry**, but a GLB by itself is not a visual-localization database. After export, **we create our own reference database** from real registered images or synthetic views rendered from that geometry. Each reference contains an RGB image together with a known camera pose and usable depth/3D information. The phone query is matched against **our database**, not against MultiSet's proprietary internal VPS database.

Those references can come from:

- registered Go2/iPhone captures;
- Polycam reference images with poses/depth;
- supported benchmark datasets;
- synthetic views rendered from the exported MultiSet/Polycam GLB.

MultiSet's proprietary internal VPS feature map is **not used by our solver and is not part of our database**. MultiSet and our method therefore remain two separate localization systems: MultiSet produces a commercial baseline pose, while our pipeline independently estimates pose from our own reference data.

See [`visual_localization/docs/MULTISET.md`](visual_localization/docs/MULTISET.md) for detailed instructions.

---

## Synthetic cross-view references

One of the main additions is a Blender-based synthetic reference generator. This lets us deliberately render the same room from both **Go2-like** and **phone-like** camera heights.

Example:

```bash
cd visual_localization

blender -b -P tools/blender_render_references.py -- \
  --glb multiset_room.glb \
  --output synthetic_refs \
  --spacing 0.60 \
  --heights 0.45 1.50 \
  --yaw-step 30 \
  --fov-deg 70
```

Example interpretation:

```text
0.45 m -> approximate Go2-height viewpoint
1.50 m -> approximate human-held phone viewpoint
```

The generated images have known virtual camera poses, so they can be used as reference observations in our own localization database.

A useful M202A experiment is therefore:

> **Does adding reference views at both robot height and human-phone height improve RGB-only localization under severe viewpoint change?**

---

## Recommended Midterm Check 2 experiment

Collect approximately **20–50 independent iPhone query images** in one mapped room. Split them into viewpoint conditions such as:

- similar height/viewpoint to references;
- phone-height query vs. Go2-height references;
- large yaw or pitch difference;
- partial occlusion;
- low-texture or repetitive regions.

Compare at least these configurations:

| Method | Purpose |
|---|---|
| RootSIFT retrieval + RootSIFT matching + PnP | Original/traditional baseline |
| RootSIFT retrieval + LightGlue + PnP | Local-matcher ablation |
| DINOv2 retrieval + LightGlue + PnP | Updated proposed method |
| MultiSet VPS | Commercial baseline |

Report metrics such as:

- localization / pose-return rate;
- position error when independent ground truth is available;
- rotation error;
- difference from MultiSet baseline;
- PnP inlier count and inlier ratio;
- reprojection error;
- pose confidence;
- processing latency;
- success rate by viewpoint condition.

**Important:** MultiSet should be described as a **commercial baseline**, not absolute ground truth. For a final accuracy study, use an independent reference such as measured control points, AprilTags, UWB, motion capture, or another calibrated setup.

---

## Start the project

Use Python **3.11 or 3.12**.

### Windows

```bat
cd visual_localization
run_web_windows.bat
```

### macOS / Linux

```bash
cd visual_localization
bash run_web.command
```

The enhanced setup uses [`requirements-enhanced.txt`](visual_localization/requirements-enhanced.txt) for PyTorch, torchvision, Kornia, learned retrieval, and learned matching support.

The first learned run may need to download model weights. Later runs can use locally cached weights.

---

## Typical workflow

1. Create or export a mapped room.
2. Export a textured `.GLB` from MultiSet or use a supported Polycam/registered map.
3. Create real or synthetic reference images with known poses.
4. Calibrate the exact iPhone lens/resolution used in the experiment.
5. Add the room through the local web interface.
6. Upload a new iPhone RGB image.
7. Run our hierarchical localizer.
8. Optionally attach the MultiSet VPS result for the same image.
9. Inspect the estimated 6-DoF camera pose and confidence metrics.
10. Export results for evaluation.

More details are in [`visual_localization/README.md`](visual_localization/README.md).

---

## Camera calibration

For meaningful metric pose accuracy, calibrate the exact phone camera/lens and resolution used during testing.

Do not mix 0.5x, 1x, and telephoto images under one calibration.

Example:

```bash
cd visual_localization
python -m wifi_localization calibrate \
  --images calibration_photos \
  --cols 9 --rows 6 --square-m 0.025 \
  --output iphone_intrinsics.json
```

The resulting calibration can be imported into the web app or supplied to the CLI.

---

## CLI examples

From [`visual_localization/`](visual_localization/):

```bash
python -m wifi_localization --help
```

Enhanced localization with optional MultiSet comparison:

```bash
python -m wifi_localization localize \
  --data data/room \
  --db outputs/database \
  --query phone.jpg \
  --intrinsics iphone_intrinsics.json \
  --multiset-pose multiset_pose.json \
  --output outputs/query_01
```

Run only the original/traditional baseline:

```bash
python -m wifi_localization localize ... --baseline-only
```

Align a Go2/Polycam/reference coordinate system to the MultiSet frame using control points:

```bash
python -m wifi_localization align-multiset \
  --points control_points.json \
  --output multiset_alignment.json
```

---

## Camera-only temporal tracking

Once the first frame is globally localized, the system can reuse mapped 3D landmarks and track image points between nearby frames.

```text
Frame 1
  -> global retrieval + matching + PnP
  -> known 3D landmarks

Frame 2, 3, 4, ...
  -> optical-flow feature tracking
  -> tracked 2D pixels + existing 3D landmarks
  -> PnP

Tracking confidence becomes weak
  -> run global localization again
```

This reduces the need to perform global learned retrieval on every frame and does not require phone IMU input.

---

## Repository structure

```text
M202A_Project-updated/
|
|-- README.md                         <- project overview
|-- index.html                        <- project webpage
|
`-- visual_localization/
    |-- README.md                     <- detailed technical guide
    |-- CHANGELOG.md                  <- v2 changes
    |-- requirements.txt              <- CPU/baseline requirements
    |-- requirements-enhanced.txt     <- learned pipeline requirements
    |-- run_web_windows.bat
    |-- run_web.command
    |
    |-- docs/
    |   |-- MULTISET.md
    |   |-- PHONE_EVALUATION.md
    |   |-- multiset_pose.example.json
    |   |-- control_points.example.json
    |   `-- phone_queries.example.json
    |
    |-- tools/
    |   `-- blender_render_references.py
    |
    |-- wifi_localization/
    |   |-- retrieval.py              <- learned/global retrieval
    |   |-- learned.py                <- ALIKED + LightGlue
    |   |-- localization.py           <- PnP/RANSAC + validation
    |   |-- tracking.py               <- camera-only temporal tracking
    |   |-- multiset.py               <- MultiSet import/comparison/alignment
    |   |-- polycam.py                <- map/depth/reference handling
    |   |-- raster.py                 <- mesh/depth rendering
    |   |-- photo.py                  <- phone calibration/intrinsics
    |   |-- pipeline.py
    |   |-- app.py
    |   `-- ...
    |
    `-- tests/
```

---

## Current test status

The current regression suite completes successfully:

```text
Ran 26 tests
OK (skipped=5)
```

The skipped tests require optional learned-model dependencies, external room databases, or real-image benchmark data that are not bundled into the repository.

Covered functionality includes:

- PnP recovery with outliers;
- camera calibration/intrinsics handling;
- MultiSet pose parsing and coordinate alignment;
- retrieval ranking;
- room/map import safety;
- mesh/depth geometry;
- 7-Scenes dataset separation;
- camera-only landmark tracking.

---

## Important limitations

- A `.GLB` alone does **not** automatically provide image localization; known-pose reference views are still required.
- MultiSet's commercial VPS result is useful for comparison, but it should not be called ground truth.
- Synthetic views are reference observations with known virtual poses; they are not automatically ground truth for a real phone image.
- Cross-view localization can still fail in highly repetitive, textureless, strongly occluded, or visually changed areas.
- Metric accuracy depends strongly on correct phone intrinsics and correct alignment between the reference views and the 3D map.
- If the project requirement is strictly "no IMU," avoid relying on an ARKit/ARCore pose as part of the proposed localization method. The included temporal tracker is camera-only.

---

## M202A project statement

A concise description for the report/presentation is:

> **We use MultiSet only as a commercial VPS baseline and as a source of an exported 3D environment. Our own system performs RGB-only hierarchical visual localization using learned global retrieval, local feature matching, 2D-to-3D geometric correspondence, and PnP with RANSAC. We further investigate synthetic cross-view references at robot and human camera heights to improve localization under severe viewpoint differences.**

This framing makes the distinction between **existing commercial capability** and **our own technical contribution** explicit.

---

## Additional documentation

- [`visual_localization/README.md`](visual_localization/README.md) — full localization setup and CLI guide
- [`visual_localization/docs/MULTISET.md`](visual_localization/docs/MULTISET.md) — MultiSet workflow and coordinate alignment
- [`visual_localization/docs/PHONE_EVALUATION.md`](visual_localization/docs/PHONE_EVALUATION.md) — phone-query evaluation workflow
- [`visual_localization/CHANGELOG.md`](visual_localization/CHANGELOG.md) — detailed v2 changes
