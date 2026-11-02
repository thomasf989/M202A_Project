# v6 MeshLoc audit fixes

This revision addresses the paper-to-code gaps found in the v5 audit.

## 1. Floating-point depth lifting

Matched reference pixels are no longer rounded to integer pixels and rejected
with a strict 3x3 flat-depth test. The primary path now bilinearly samples the
rendered mesh depth at the exact floating-point feature coordinate, matching the
behavior in the official MeshLoc 12-Scenes script more closely.

At strong depth discontinuities or missing raster samples, the implementation
uses a sparse exact ray/mesh intersection as a robustness fallback. This avoids
blending foreground/background depth at silhouettes.

## 2. Near-plane clipping

The CPU mesh depth rasterizer now clips triangles that cross the camera near
plane instead of dropping the entire triangle. This removes artificial holes in
depth maps near walls, furniture, and close geometry.

## 3. Paper RANSAC settings

`meshloc-paper` uses:

- Top-20 retrieved references;
- LoFTR indoor matching;
- PoseLib absolute pose estimation;
- 10,000 minimum RANSAC iterations;
- 100,000 maximum RANSAC iterations;
- 20 px reprojection threshold;
- +REF position refinement over a ±0.25 m volume with 0.05 m step.

The official MeshLoc 12-Scenes release uses DenseVLAD Top-20 retrieval. This
project does not ship the original DenseVLAD retrieval file/vocabulary because
our database consists of newly generated MultiSet/Blender views. Therefore the
paper baseline uses NetVLAD Top-20 while keeping the rest of the indoor pose
stack aligned with the release.

## 4. Robust position refinement

Nearby camera-center samples are now weighted by an MSAC-style truncated
reprojection residual instead of raw inlier count. A pose with fewer but much
cleaner correspondences can therefore beat a noisier pose with a couple more
borderline inliers.

## 5. Real-vs-synthetic retrieval gap

The recommended `meshloc` mode fuses NetVLAD and multi-crop DINOv2 using
reciprocal-rank fusion, then suppresses near-duplicate synthetic camera poses.
This is a project-specific robustness extension; `meshloc-paper` remains
NetVLAD-only.

## 6. No silent solver substitution

The web app exposes four explicit solver modes:

- `MeshLoc robust`: recommended for the synthetic MultiSet database;
- `MeshLoc paper baseline`: MeshLoc-only, no fallback;
- `Auto`: MeshLoc first, then legacy fallbacks;
- `Legacy only`: previous learned/RootSIFT system.

Use a MeshLoc-only mode when reporting MeshLoc success rate. `Auto` is for demos.
Every result JSON records both the requested solver mode and the method that
actually returned the pose.

## 7. LoFTR correspondence identity

MeshLoc's optional multi-3D-point merging assumes repeatable query feature
identity. Pairwise LoFTR matches do not naturally provide stable feature IDs
across reference pairs. Therefore merging and approximate covisibility remain
disabled by default. If manually enabled, the result JSON explicitly labels the
query-pixel clustering as experimental instead of presenting it as exact paper
behavior.

## 8. End-to-end validation

The normal regression suite now checks bilinear depth sampling, near-plane
clipping, MeshLoc-only no-fallback behavior, paper RANSAC settings, and pose
recovery.

An optional full learned integration test is included in
`tests/test_meshloc_e2e_optional.py`. Set:

```powershell
$env:M202A_E2E_DATA="C:\path\to\prepared_room"
$env:M202A_E2E_DB="C:\path\to\database"
python -m pytest -q tests\test_meshloc_e2e_optional.py
```

It localizes a known posed frame with the real NetVLAD/LoFTR/mesh/PoseLib stack,
excludes the exact query frame from retrieval, and checks translation and
rotation error.
