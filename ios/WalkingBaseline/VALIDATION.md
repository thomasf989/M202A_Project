# Validation — November 19, 2026

## Scope

Based on GitHub `Localization_Testing` commit `030064e`, with the earlier iPhone
baseline preserved in its original checkout. Data source is the user's current
Desktop `杰佛瑞的Room` folder. No raw scan, photo, database, or binary framework is
committed to Git. No changes are made to main.

## Completed checks

- Unsigned physical-iPhone build with Xcode 27.2 and OpenCV 5.1.0: passed.
- Simulator build and UI launch: passed; the bundled room and no-survey state displayed.
- Swift core executable: 13 checks passed (interpolation, no coverage, map mismatch,
  nonfinite values, nearest-surface picking, no arbitrary depth, capture-time pose,
  invalid transform, real package import, repeat import, checksum and truncation).
- Python package/script compilation: passed.
- Reference export: 88 training frames only; held-out frame IDs are disjoint.
- Python CPU baseline on all 22 Test frames: 20 accepted, 2 rejected.
  Accepted median position error: 0.0193 m; maximum: 0.1053 m.
  Accepted median rotation error: 0.826°; maximum: 2.752°.

Those errors use camera poses from the same globally optimized Polycam scan.
They are not independent surveyed ground truth, native-code parity, a new-phone
photo evaluation, or an estimate of walking drift. Native matching uses exact
BF reciprocal matching instead of the Python FLANN approximation. On-phone timing
and accuracy must be measured separately. The full MeshLoc pipeline was not ported.

## Still required

- XCTest simulator execution: the runner stalled after compilation; do not count
  the XCTest suite as passed. The separate core executable checks above did run.
- Install with the user's Apple signing team on a physical iPhone.
- Verify offline camera localization on independently captured phone views.
- Verify map alignment, camera orientation, taps and repeated walking loops.
- Import a second independently scanned room and repeat the walking tests there.
- Collect/register real Wi-Fi measurements; check estimates against held-out readings.

There is no real Wi-Fi survey yet. The application deliberately shows no measured
coverage until one is imported or external readings are recorded.
