# iPhone walking baseline

This native SwiftUI/RealityKit app detects a printed image, fixes a blue cube in
AR world space, and lets you walk around it. It implements the marker-assisted
walking milestone. It does not yet display measured Wi-Fi or send camera frames
to the visual localizer.

## Build on a Mac

Requires Xcode with the iOS SDK, XcodeGen, and an ARKit-compatible iPhone with
iOS 16 or later. From this directory:

```bash
brew install xcodegen
xcodegen generate
open WalkingBaseline.xcodeproj
```

Choose your signing team in the application target, select the connected iPhone,
and run the WalkingBaseline scheme. Grant camera permission. The simulator cannot
validate physical camera tracking. Run the included unit tests from Xcode with
Product → Test on a supported destination.

## Print and run

1. Print `marker-print.pdf` at **100% / actual size**, without fit-to-page scaling.
   Measure the outer black square: it must be **16 cm wide**. The physical width
   configured in the app is 0.16 meters; a different print size corrupts scale.
2. Lay the marker flat on a stable, well-lit surface. Do not move it during a trial.
3. Point the phone at it. Wait for “Marker found,” then tap **Align to marker**.
4. A small orange sphere marks the origin and a blue cube appears 50 cm above it.
5. Walk away, turn, and return. The cube should remain fixed even when the printed
   image leaves the camera view. Re-align explicitly if drift accumulates.
6. Reset clears the previous alignment. Interrupted sessions also restart;
   limited tracking hides the content until tracking becomes normal again.

The marker defines a temporary local frame, not the Go2 map. Its image-plane
normal is local +Y. To connect to a surveyed map, supply the measured
`mapFromMarker` transform instead of identity at the alignment call.

## Acceptance test and evidence

On a real device, record model/iOS version, print width, lighting, and room:

- Check the cube is above the marker rather than behind/below it.
- Walk a measured 2–3 m route and turn away until the marker is out of view.
- Return and measure visible displacement against independently positioned targets.
- Repeat at least five times; log failures and drift before manual realignment.
- Cover the camera briefly: content must hide while tracking is limited.
- Background and foreground the app: previous alignment must be cleared.
- Move the print only between trials and verify Reset requires a new alignment.

No device results have been collected in this cloud environment. Swift sources
and XCTest cases are provided, but Xcode compilation and XCTest execution still
need to be performed on a Mac.

## Next integration

The existing visual localizer returns `T_world_camera_cv`. Convert its camera
basis with `Alignment.arCameraPose`, retain the ARFrame pose at capture time, and
compute `T_map_session` with `Alignment.mapFromSession`. Register the GLB world to
the Go2 map first. Accept only successful, geometrically verified localization
results for the same map and session; reject stale frames. Never use the pose at
response arrival as the capture pose.

The current desktop server intentionally rejects non-local hosts. It has not
been exposed or weakened for this app. Phone integration needs an explicitly
configured network service, frame IDs/timestamps, calibrated image intrinsics,
and session/map identity validation before automatic correction is enabled.
