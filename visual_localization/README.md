# Visual localization v6 — audited MeshLoc pipeline + high-coverage random-view references

Estimate a phone camera's metric 6-DoF pose inside a mapped room from a new RGB photo. MultiSet is used only as an optional **commercial VPS baseline** and as a source of exported open 3D geometry. The project's own solver builds and searches its **own posed reference database**.

## Primary localization pipeline

```text
new iPhone RGB photo
        ↓
NetVLAD + multi-crop DINOv2 retrieval (robust mode)
        ↓
reciprocal-rank fusion + pose-diverse top-50 references
        ↓
LoFTR indoor matching
        ↓
bilinear GLB mesh-depth lifting
+ exact ray fallback at depth discontinuities
        ↓
2D query pixels ↔ 3D map points
        ↓
P3P + LO-RANSAC (20 px, 10k–100k iterations)
        ↓
robust nonlinear refinement
        ↓
MSAC-weighted MeshLoc +REF position refinement
        ↓
6-DoF phone pose
```

This follows the main MeshLoc ECCV 2022 framework. The project-specific extension is the dense synthetic reference database generated from the exported textured GLB. For dense synthetic databases, v5 additionally suppresses near-duplicate retrieved camera poses so LoFTR's matching budget covers more distinct positions/orientations.

## v6: paper-audit fixes

The v6 audit fixed several issues that could reduce accuracy even when the high-level pipeline looked correct:

- bilinear depth sampling at the exact floating-point LoFTR match instead of integer rounding;
- exact sparse ray/mesh fallback near depth discontinuities;
- proper near-plane triangle clipping in the CPU depth renderer;
- MeshLoc's 20 px indoor RANSAC baseline and 10k–100k iteration range;
- MSAC-style residual weighting for the +REF camera-position refinement;
- hybrid NetVLAD + multi-crop DINOv2 retrieval for the synthetic-to-real domain gap;
- explicit solver modes so MeshLoc experiments cannot silently fall back to a different algorithm;
- optional full learned end-to-end known-pose validation.

The web app now offers **MeshLoc robust**, **MeshLoc paper baseline**, **Auto**, and **Legacy only**. Use one of the MeshLoc-only modes for experimental reporting.

See [`docs/MESHLOC_V6_FIXES.md`](docs/MESHLOC_V6_FIXES.md).

## v5: better reference coverage for random phone photos

The previous generator could still miss a random phone viewpoint because it sampled only a limited set of heights/pitches or generated too many near-duplicate views when made dense.

The new generator uses:

- standing-phone, seated-phone, high-phone, and Go2-like heights;
- 360° yaw coverage;
- upward/downward pitch variants;
- dense XY sampling;
- collision/forward-obstacle checks;
- coarse visibility scoring with scene ray casts;
- a fixed reference budget with stratified pose-diverse sampling;
- `quick`, `balanced`, and `dense` presets;
- `generation_report.json` so you can see how many views were planned/rendered by role.

For an apartment-sized scan, start with **balanced** rather than rendering every possible camera pose.

## Generate the recommended reference set on Windows

Put the MultiSet textured GLB at, for example:

```text
data\multiset\apartment.glb
```

First do a dry run:

```powershell
conda activate m202a
.\tools\generate_multiset_references_windows.ps1 `
  -Glb ".\data\multiset\apartment.glb" `
  -Preset balanced `
  -DryRun
```

Then render:

```powershell
.\tools\generate_multiset_references_windows.ps1 `
  -Glb ".\data\multiset\apartment.glb" `
  -Preset balanced
```

This creates:

```text
synthetic_refs_random/
  images/
  references.json
  generation_report.json
synthetic_refs_random.zip
```

Upload:

- **3D Model:** the same `apartment.glb`
- **Reference Photos and Coordinates:** `synthetic_refs_random.zip`

Your real iPhone test photo is uploaded later as the **query image**. It does not belong in the reference ZIP.

See [`docs/REFERENCE_SET_V5.md`](docs/REFERENCE_SET_V5.md) for preset details.

## Start the local web app

Recommended existing Conda environment:

```powershell
conda activate m202a
python -m pip install -r requirements-enhanced.txt
python -m wifi_localization web --output outputs\web
```

Then open:

```text
http://127.0.0.1:8765
```

The first enhanced run may download NetVLAD and LoFTR model weights. Later runs reuse local caches.

## MultiSet's role

MultiSet and our localizer are separate systems:

```text
MultiSet scan → exported textured GLB → OUR reference generator → OUR database

same phone image → MultiSet VPS → commercial baseline pose
same phone image → our MeshLoc-style pipeline → our pose
```

We do **not** train on, copy, or access MultiSet's proprietary VPS feature database. MultiSet's returned pose is a comparison baseline, not absolute ground truth.

## Reference format

The generated `references.json` is `posed-images-v1`. Each frame stores:

- RGB image path;
- known camera intrinsics;
- known `camera_to_world` pose;
- reference split;
- synthetic metadata such as role, camera height, yaw, pitch, and geometric visibility score.

The same GLB is used to lift matched reference pixels into metric 3D points.

## Evaluation

For Midterm Check 2, collect 20–50 independent phone photos and group them by difficulty:

- normal standing view;
- seated/low view;
- large yaw difference;
- upward/downward pitch;
- partial occlusion;
- repetitive or low-texture areas.

Report pose-return rate, position/rotation error when independent ground truth exists, difference from MultiSet, inlier count/ratio, reprojection error, and latency.

RGB-only localization cannot guarantee every possible photo. Blank walls, severe blur, changed environments, or unmapped surfaces may be ambiguous and should be rejected rather than assigned a fabricated pose.

## Useful docs

- [`docs/MESHLOC_METHOD.md`](docs/MESHLOC_METHOD.md) — MeshLoc-style method details
- [`docs/MESHLOC_V6_FIXES.md`](docs/MESHLOC_V6_FIXES.md) — paper-to-code audit and fixes
- [`docs/REFERENCE_SET_V5.md`](docs/REFERENCE_SET_V5.md) — v5 reference generation
- [`docs/MULTISET.md`](docs/MULTISET.md) — MultiSet export/baseline workflow
- [`docs/PHONE_EVALUATION.md`](docs/PHONE_EVALUATION.md) — phone-image evaluation

## Tests

```powershell
python -m pytest -q
```
