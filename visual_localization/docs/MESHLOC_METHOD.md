# MeshLoc-style localization in v4

The v4 primary localizer follows the main framework of **Panek, Kukelova, and Sattler, “MeshLoc: Mesh-Based Visual Localization,” ECCV 2022** while deliberately retaining this project’s existing **dense Blender synthetic camera poses**.

## Default pipeline

1. **NetVLAD retrieval** (Pitts30K model), top 50 reference views.
2. **LoFTR indoor matching** (`indoor_new`) between the real query photo and each retrieved reference view. LoFTR is one of the local matchers evaluated in MeshLoc and was particularly strong in the paper’s reduced-resolution experiments.
3. **Mesh-based 2D→3D lifting**. Each matched reference pixel is intersected with / depth-rendered from the same textured GLB, creating query-pixel ↔ world-point correspondences.
4. **Individual 2D–3D matches by default**. This matches the simple MeshLoc variant the authors ultimately emphasize. Optional correspondence merging is implemented for ablations.
5. **P3P + LO-RANSAC** via PoseLib when available, with the paper-style high iteration range and nonlinear refinement. A portable OpenCV P3P + local robust Cauchy refinement fallback is included.
6. **Position averaging**. The estimated rotation is kept fixed while nearby camera centers are sampled; centers are weighted by their inlier support. Indoor defaults use a ±0.25 m volume and 0.05 m grid step, matching the 12 Scenes settings reported in the MeshLoc supplementary material.

## Optional paper ablations

`MeshLocSettings` also supports:

- `use_merge=True`: merge repeated 2D–3D matches for the same query feature using reprojection consensus and 3D point refinement.
- `use_covisibility=True`: construct the paper’s approximate on-the-fly covisibility graph from query features that match multiple reference views and solve each connected component independently.

The default disables covisibility because MeshLoc reports that its approximation can over-segment the correspondence set and reduce accuracy; their later experiments focus on the simpler individual-matches + position-averaging configuration.

## What is intentionally unchanged

The project still uses the **dense Blender synthetic camera-pose database** generated from the MultiSet-exported GLB. The user specifically requested retaining that part. MeshLoc’s retrieval/matching/pose-estimation ideas are applied *around* that database.

## First-run model downloads

The first enhanced run may download:

- NetVLAD Pitts30K weights (public HLoc checkpoint)
- LoFTR indoor weights (Kornia)

Afterwards the weights and room descriptors are cached locally.

## Citation

If this method is described in a report, cite MeshLoc and LoFTR/NetVLAD as appropriate rather than claiming these components were invented in this project.

## v5 adaptation for dense synthetic databases

MeshLoc's original experiments retrieve a fixed number of database images. In this project the Blender database can contain thousands of highly similar synthetic poses. A raw NetVLAD top-50 can therefore be dominated by several renders from almost the same camera center and orientation.

v5 keeps NetVLAD scoring but examines a larger retrieval pool and suppresses candidates that are both spatially and angularly near-duplicates. The requested top-50 is then back-filled if necessary, so this adaptation does not reduce the candidate count. This is a project-specific adaptation for dense synthetic references; the downstream LoFTR, mesh lifting, P3P/LO-RANSAC, nonlinear refinement, and position averaging remain MeshLoc-style.
