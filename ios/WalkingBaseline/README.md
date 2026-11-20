# Offline Room Wi-Fi Explorer

An iPhone app for photo localization, AR walking, and tapping scanned surfaces
for estimated Wi-Fi strength. Runtime is entirely on-device: no account, server,
network connection, photo upload, or Mac is needed after installation and room import.

## What this build does

- Captures an AR camera frame and calibrated intrinsics in sensor orientation.
- Runs a native OpenCV RootSIFT / TF-IDF / reciprocal feature matching / PnP-RANSAC
  localizer against an offline reference database.
- Registers the map to ARKit using the **capture-time** AR camera pose, then uses
  ARKit motion tracking as the user walks and turns. Relocate periodically for drift.
- Shows camera coordinates, viewing yaw, and the survey estimate at the phone.
- Finds the nearest scanned mesh surface along a screen tap and reports its estimate.
- Draws colored surface dots inside measured coverage (red −90 to green −40 dBm),
  using scanned geometry to hide dots behind nearer mapped surfaces.
- Imports multiple offline room packages, switches rooms, and keeps surveys separate.
- Imports/exports Wi-Fi surveys and records manually entered external readings at
  the localized phone position. Empty/unsupported regions show no measured coverage.

This is a native port of the repository's **CPU baseline**, not an on-device
conversion of its NetVLAD/DINOv2/LoFTR MeshLoc pipeline. It can reject difficult
photos; neural-model parity and robust independent-phone evaluation remain future work.
The simulator builds the UI/Swift tests with a deliberate native-localizer stub;
real image localization requires a physical iPhone in this build.

## Build and install

Requires Xcode, a complete XcodeGen installation, and an ARKit-compatible iPhone
running iOS 16+. From the repository root:

```bash
bash tools/install_opencv_ios.sh
# Prepare at least one room as below, or create Resources/Room as an empty folder
# and import a package after installing the app.
cd ios/WalkingBaseline
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
xcodegen generate
open WalkingBaseline.xcodeproj
```

In Xcode choose the WalkingBaseline scheme and your connected iPhone. Set your
own signing team in Signing & Capabilities. Enable Developer Mode and trust the
Mac if prompted, then Run. Grant camera permission. No paid account is required
for a personal development installation, subject to Apple's signing limits.

The framework installer pins official OpenCV 5.1.0 and verifies its SHA-256. It
contains a device arm64 library and an Intel simulator slice; the ARM simulator
therefore uses the explicit stub. Generated Xcode projects, private room packages,
reference databases, and vendor binaries are ignored by Git. A signing team or
Apple credentials are never stored in this repository.

## Prepare any future room (once, on a computer)

Install the preparation dependencies in a Python 3.11/3.12 environment:

```bash
python3.11 -m venv visual_localization/.venv
visual_localization/.venv/bin/python -m pip install -r visual_localization/requirements-iphone.txt
visual_localization/.venv/bin/python tools/prepare_iphone_room.py \
  --references /path/to/room_train.zip --model /path/to/same_scan.glb \
  --name "Living room" --work outputs/living-room-v1
```

Use a new `--work` folder for every different scan/version. Supported reference
formats are the existing importer's Polycam raw data, registered posed photos,
and supported 7-Scenes exports. A GLB plus unposed ordinary photos is insufficient.
The GLB, depth, intrinsics and reference poses must share the same metric frame.
Keep test photos outside the training ZIP.

AirDrop/copy `outputs/living-room-v1/room.wifiroom` to Files on the iPhone.
Open **Survey tools → Import room package**, select it, and select the room.
Repeat for other scans without rebuilding/reinstalling the app. The room library
persists across app restarts. Import validates format, lengths, and SHA-256 hashes.
Packages contain private scan/reference features: share only with intended users.

The supplied Jeffrey room was prepared from `11_5_2026_Train.zip` (88 frames)
and `11_5_2026.glb`. Its 22 Test frames are never in the reference package.
A local initial bundle is in `Resources/Room`; the matching `.wifiroom` file can
also be imported through Files. Future packages do not need to be bundled.

## Walk and inspect

1. Select the room; stand inside the corresponding physical scan.
2. Aim at recognizable textured furniture/corners, hold steady, and tap **Locate me**.
3. After a reliable pose is returned, walk/turn. Tap a mapped surface to inspect it.
4. Tap **Relocate** if the overlay drifts. Limited tracking hides readings and overlay.
5. Reset, backgrounding, interruptions and room switches clear AR alignment.
   Responses from a reset session are discarded, including delayed inference.

The app needs a real iPhone for camera tracking. An untextured wall, motion blur,
changed furniture, poor coverage, or repeated patterns can prevent localization.
Geometric acceptance does not guarantee a pose is correct; verify against physical
landmarks before relying on a survey location.

## Wi-Fi data (not yet supplied)

Apple exposes no general-purpose live RSSI API for ordinary iOS apps. This app
visualizes **external survey measurements**. It does not measure RSSI itself.
Import JSON, or enter a value measured by survey equipment at your current phone
position. Do not mix networks/BSSIDs/bands in one survey. Include those in `network`.
Export an empty survey from the app to get the selected room's exact map ID.

```json
{
  "mapID": "<exact mapID from room manifest or exported empty survey>",
  "network": "SSID / BSSID / 5 GHz",
  "radius": 1.5,
  "samples": [
    {"position": [0.2, 1.3, -0.8], "dbm": -62, "measuredAt": "2026-11-19T18:00:00Z"}
  ]
}
```

Positions are **GLB-frame XYZ meters**, not latitude/longitude or raw AR coordinates.
If your future survey uses a robot/map frame, register it to the GLB with a measured
rigid transform before importing. A transformed or rescanned room gets a new ID.

Values are bounded 3D inverse-distance interpolation over up to eight nearby samples,
within `radius` (0–5 m). They are approximate, not a propagation simulation, and do
not account for intervening walls. Surface dots are a sparse visualization; blank
areas mean no measured coverage. No fake Wi-Fi field is bundled.

## Validation

Run Swift tests with Xcode Product → Test on an iPhone simulator. They cover
matrix conventions, delayed-frame alignment, survey validation/room isolation,
coverage, nearest-surface picking, and truncated-package rejection.

`tools/evaluate_iphone_baseline.py` evaluates the Python equivalent on held-out
photos. It does not validate native parity, on-phone latency, or AR drift.
Record physical-device acceptance separately: photo pose vs surveyed landmarks,
five 2–3 m walking loops, turns, interruptions, offline/airplane mode, map switches,
and tapped-point estimates vs independent survey readings. No such physical-device
results or real Wi-Fi measurements have yet been collected for this integration.
