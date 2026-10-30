# Visual localization

## iPhone offline version (development)

The [iOS app and Xcode project](ios/README.md) run the RootSIFT/PnP pipeline directly on the phone. Prepare a room using the desktop app, then click **导出 iPhone 房间包** and transfer the resulting `.roomloc` file to the phone. The app imports it locally, captures or selects photos, displays camera poses in the room, and exports pose JSON. It does not require a running computer or server for localization.

The C++ solver and desktop-to-native export path have portable tests. The complete Apple SDK build and camera/device performance still require validation in Xcode on a Mac. This first mobile version uses the SIFT baseline; the desktop ALIKED/LightGlue fallback remains available on the desktop.

Estimate a camera's 6-DoF pose inside a mapped room from a new photo, view the result in a GLB model, and export the model with a camera marker.

## Start the local web app

Use Python 3.11 or 3.12. On macOS or Linux, open a terminal in this directory and run:

```bash
bash run_web.command
```

The launcher creates a virtual environment, installs dependencies, and opens the local web app. Keep the terminal open while using it. Stop with Control+C.

1. Select **Add room** and enter a name.
2. Upload a GLB model and a reference-data ZIP.
3. Wait for the reference database to finish building.
4. Upload a new photo to estimate its position and orientation.
5. Download the pose, marked GLB, or offline viewer. Saved rooms can be selected again later.

The repository contains source code, tests, and generic documentation. Supply models and reference data separately. Input photos, room scans, generated databases, caches, and experiment results are excluded.

## Reference data

Supported inputs:

- A Polycam raw export with camera intrinsics, poses, depth, and alignment metadata.
- Official or prepared 7-Scenes data with its sequence split.
- Registered reference photos with `references.json`, known camera intrinsics and poses, and a GLB in the same metric coordinate frame.

A ZIP of ordinary photos alone does not supply the geometry needed for localization. See [web upload format](docs/WEB_UPLOAD.md), [7-Scenes import](docs/SEVEN_SCENES.md), and the [reference JSON template](wifi_localization/web/reference-format.json).

## Method

The CPU baseline uses RootSIFT, a visual vocabulary with TF-IDF retrieval, reference-depth backprojection, and PnP with RANSAC. The web app can fall back to ALIKED and LightGlue matching. Enhanced matching downloads pretrained weights on first use; it does not train a new visual network.

Query images do not enter the reference database. Geometry checks can reject an unreliable pose. Automatic query-camera intrinsics are estimates; a calibrated profile can be imported through the app.

## Command line

```bash
python -m wifi_localization --help
```

For a manual installation:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-enhanced.txt
python -m wifi_localization web --output outputs/web
```

`requirements.txt` installs the baseline. `requirements-enhanced.txt` adds learned matching. `requirements-tested.txt` records dependency versions from the original validation environment.

## Tests

```bash
python -m unittest discover -s tests -v
```

Tests cover pose recovery, coordinate and depth handling, official train/test separation, resumable uploads, room preparation, switching, and export. Tests needing optional learned-matching dependencies or a local image database skip when those inputs are unavailable. No private test images or experimental records are included.

## Third-party components

Vendored LightGlue and ALIKED inference files retain their upstream notices. See [upstream components](wifi_localization/_vendor/lightglue/UPSTREAM.md) and [license](wifi_localization/_vendor/lightglue/LICENSE). Dataset users must follow the licenses supplied by their data providers.

