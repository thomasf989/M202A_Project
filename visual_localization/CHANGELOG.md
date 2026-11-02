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

## v3.0 — robust random-view localization

- Reworked Blender reference generation for wide square FOV coverage, multiple heights, optional pitch, robust floor estimation, and basic geometry-clearance rejection.
- Added multi-crop DINOv2 query retrieval for portrait/cropped/FOV-mismatched phone photos.
- Added pose-neighbour candidate expansion around high-scoring reference views.
- Increased ALIKED resolution/keypoint budget and added automatic CUDA use when available.
- Added fundamental-matrix RANSAC verification before 2D-to-3D lifting.
- Added EPNP/SQPNP/AP3P pose hypotheses, more tolerant but still conservative geometric acceptance, and stronger LM refinement.
- Added hybrid RootSIFT matching on the learned shortlist before the full CPU fallback.
- Extended posed-reference mesh-depth validity from 5 m to 15 m and mesh rasterization to 20 m.
- Broadened automatic phone focal hypotheses and increased processing resolution.
- Automatically rebuild stale feature databases after map/depth fingerprint upgrades.
- Improved web failure diagnostics and added a Windows helper for generating + zipping MultiSet synthetic references.

## v4 — MeshLoc-style primary localizer

- Replaced the primary DINOv2/ALIKED pipeline with a MeshLoc-style stack: NetVLAD top-50 retrieval, LoFTR indoor matching, mesh-depth 2D–3D lifting, P3P LO-RANSAC (PoseLib), nonlinear refinement, and MeshLoc position averaging.
- Added optional MeshLoc correspondence merging and on-the-fly covisibility components for ablation experiments.
- Kept the dense Blender synthetic camera-pose generator unchanged by design.
- Preserved the v3 learned and RootSIFT pipelines as fallbacks.

## v5 — High-coverage random-view reference design

- Replaced the simple synthetic grid with a bounded pose-diverse random-phone reference design.
- Added standing/seated/high/Go2 height families and multi-pitch coverage.
- Added geometric camera-center collision checks and ray-fan visibility validation.
- Added balanced reference budgeting so dense settings do not explode into tens of thousands of near-duplicates.
- Added `quick`, `balanced`, and `dense` Windows presets plus a dry-run mode.
- Added `generation_report.json` with role/count diagnostics.
- Added NetVLAD pose-diversity suppression before LoFTR to keep dense synthetic top-K retrieval from being dominated by near-identical camera poses.

## v6 — MeshLoc paper-audit and robustness fixes

- Replaced rounded/3x3-flat mesh-depth lifting with floating-point bilinear depth sampling.
- Added exact sparse ray/mesh fallback for missing or discontinuous depth samples.
- Fixed the CPU depth renderer to clip near-plane-crossing triangles rather than dropping whole triangles.
- Added an explicit `meshloc-paper` baseline using Top-20, 20 px reprojection threshold, and 10k–100k PoseLib RANSAC iterations.
- Added a recommended `meshloc` robust mode with NetVLAD + multi-crop DINOv2 reciprocal-rank fusion for real-vs-synthetic retrieval.
- Replaced raw-inlier-count position averaging weights with MSAC-style truncated residual weights.
- Added explicit solver selection in the browser and API; MeshLoc-only modes never silently run legacy fallbacks.
- Marked LoFTR cross-reference correspondence merging as experimental and kept it disabled by default.
- Added bilinear-depth, near-plane-clipping, no-fallback, paper-settings, and optional full learned end-to-end tests.
