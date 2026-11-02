# Robust random-view localization workflow (v3)

The goal of v3 is to localize a new phone RGB image from an arbitrary *informative* viewpoint inside the scanned 3D area. RGB-only localization cannot uniquely solve a blank wall, severe blur, an unseen room, or an environment that changed substantially after scanning; those cases are intentionally rejected rather than assigned a fake pose.

## 1. Regenerate the reference database

Do not reuse the older sparse synthetic reference ZIP. It was rendered with narrower FOV / less viewpoint coverage and remains a bottleneck even with the new matcher.

Windows helper:

```powershell
conda activate m202a
cd "C:\202 AS\M202A_Project-updated\visual_localization"
.\tools\generate_multiset_references_windows.ps1 -Glb ".\data\multiset\apartment.glb"
```

Default robust sampling:

- 0.90 m XY camera-center spacing
- camera heights 0.45, 1.25 and 1.60 m
- 60° yaw spacing
- 100° square synthetic FOV
- horizontal pitch by default; extra pitch can be enabled with `-Pitches @(-18,0,18)` if the database remains reasonably sized

The helper creates:

```text
synthetic_refs_robust/
  references.json
  images/
synthetic_refs_robust.zip
```

Upload the same MultiSet textured GLB as **3D Model** and `synthetic_refs_robust.zip` as **Reference Photos and Coordinates**.

## 2. Use a new room / allow the app to rebuild

v3 changes mesh-depth validity and feature indexing. The app now detects stale database fingerprints and rebuilds automatically, but creating a fresh room from the robust reference ZIP is the clearest experiment.

## 3. Localize a real phone image

Start:

```powershell
conda activate m202a
python -m wifi_localization web --output outputs\web
```

Upload an independent iPhone photo. The photo does not belong in the reference ZIP.

The v3 path is:

```text
phone RGB
  -> multi-crop DINOv2 retrieval
  -> pose-neighbour candidate expansion
  -> ALIKED + LightGlue
  -> fundamental-matrix RANSAC
  -> GLB 2D-to-3D lifting
  -> EPNP / SQPNP / AP3P RANSAC hypotheses
  -> LM refinement + geometry checks
```

If learned local matching fails, the app tries RootSIFT on the same global shortlist and then the original full RootSIFT baseline.

## 4. For best accuracy

- Use the same room state as the scan whenever possible.
- Avoid digital zoom and switching between 0.5x / 1x / telephoto while using one calibration.
- Calibrate the exact phone lens/resolution for final metric evaluation.
- Keep some textured structure in the image: furniture corners, posters, doors, shelves, appliances, etc.
- If a location repeatedly fails, inspect whether the textured GLB actually contains that surface and whether robust synthetic views cover that region.

## 5. If coverage is still insufficient

Increase spatial coverage first, not brute-force matching:

```powershell
.\tools\generate_multiset_references_windows.ps1 `
  -Glb ".\data\multiset\apartment.glb" `
  -Spacing 0.70 `
  -YawStep 45
```

For strongly tilted-phone experiments, add pitch variants:

```powershell
.\tools\generate_multiset_references_windows.ps1 `
  -Glb ".\data\multiset\apartment.glb" `
  -Spacing 0.85 `
  -YawStep 60 `
  -Pitches @(-18,0,18)
```

Watch the Blender printed reference count. More references improve coverage but increase first-time database/DINO indexing cost.
