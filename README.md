# M202A RGB-only indoor visual localization

This repository localizes a new iPhone RGB image inside a previously scanned 3D environment. The current version combines a **MeshLoc-style visual-localization pipeline** with a **high-coverage synthetic reference generator** designed for random indoor phone viewpoints.

## Core idea

```text
MultiSet textured GLB
        ↓
our Blender reference generator
        ↓
our posed RGB reference database
        ↓
NetVLAD → LoFTR → mesh 2D/3D lifting → P3P/LO-RANSAC
        ↓
our 6-DoF phone pose
```

MultiSet is not our localization database. It is used only for exported 3D geometry and, optionally, as a commercial VPS baseline for comparison.

## Recommended next step

From `visual_localization` on Windows:

```powershell
conda activate m202a

# Inspect planned references without rendering.
.\tools\generate_multiset_references_windows.ps1 `
  -Glb ".\data\multiset\apartment.glb" `
  -Preset balanced `
  -DryRun

# Render the balanced random-view reference set.
.\tools\generate_multiset_references_windows.ps1 `
  -Glb ".\data\multiset\apartment.glb" `
  -Preset balanced

# Start the local app.
python -m wifi_localization web --output outputs\web
```

The balanced preset uses multiple camera heights, full yaw coverage, several pitch angles, geometry validity checks, and a bounded pose-diverse reference budget. See [`visual_localization/README.md`](visual_localization/README.md) for the full workflow.


## v6 MeshLoc audit

The visual localization module now includes audited MeshLoc-style depth lifting, near-plane clipping, 20 px paper-baseline RANSAC settings, robust position refinement, hybrid synthetic-domain retrieval, explicit no-fallback MeshLoc evaluation modes, and expanded tests. See `visual_localization/docs/MESHLOC_V6_FIXES.md`.
