# v5 high-coverage reference set

The v5 generator is designed for the project's hardest case: a new iPhone photo taken from an arbitrary informative viewpoint inside the same scanned 3D area.

## Why the old grid was not enough

A uniform grid with one height, horizontal pitch, and coarse yaw can leave a valid query with no nearby reference view. A much denser Cartesian product fixes coverage but creates tens of thousands of nearly duplicate images and makes NetVLAD/LoFTR unnecessarily expensive.

v5 uses a **bounded, pose-diverse reference design** instead:

- dense standing-phone views at the primary height;
- full 360-degree yaw coverage;
- up/down pitch variants;
- staggered seated, high-phone, and Go2-like heights;
- geometric collision filtering for camera centers;
- forward-obstacle rejection;
- coarse ray-fan visibility checks;
- a hard maximum reference budget;
- stratified selection so that the budget is spread across positions, yaws, heights, and pitch roles.

The localizer also suppresses near-duplicate NetVLAD results before LoFTR, which is important once the reference database becomes dense.

## Recommended Windows workflow

From `visual_localization`:

```powershell
conda activate m202a

# First inspect the planned set without rendering.
.\tools\generate_multiset_references_windows.ps1 `
  -Glb ".\data\multiset\apartment.glb" `
  -Preset balanced `
  -DryRun

# Then render it.
.\tools\generate_multiset_references_windows.ps1 `
  -Glb ".\data\multiset\apartment.glb" `
  -Preset balanced
```

The helper creates:

```text
synthetic_refs_random/
  images/
  references.json
  generation_report.json
synthetic_refs_random.zip
```

Upload `synthetic_refs_random.zip` as **Reference Photos and Coordinates** and the same textured GLB as **3D Model**.

## Presets

### quick

Useful for checking that the pipeline works before a long render.

- spacing: 0.85 m
- max views: 900
- yaw: 45°
- heights: 0.45, 1.20, 1.50 m
- pitches: -20°, 0°, +20°

### balanced — recommended

Best starting point for an apartment-sized scan.

- spacing: 0.65 m
- max views: 2400
- yaw: 30° core coverage
- heights: 0.45, 1.10, 1.45, 1.70 m
- pitches: -25°, -10°, 0°, +10°, +25°
- FOV: 95°

### dense

Use only after the balanced set works and you have identified genuine coverage failures.

- spacing: 0.50 m
- max views: 4000
- yaw: 30°
- heights: 0.45, 1.05, 1.35, 1.55, 1.75 m
- pitches: -30°, -15°, 0°, +15°, +30°

## Important limitation

No RGB-only method can guarantee a unique pose for every possible image. A blank wall, severe blur, large scene changes, or an area absent from the textured scan may be fundamentally ambiguous. The goal of v5 is high coverage for normal informative phone photos, plus correct rejection when visual evidence is insufficient.
