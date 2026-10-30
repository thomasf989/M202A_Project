#!/bin/bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
if ! command -v xcodebuild >/dev/null 2>&1; then
  echo "Install Xcode, open it once, then select it with xcode-select."
  exit 1
fi
swift test
xcodebuild -resolvePackageDependencies -project RoomLocate.xcodeproj -scheme RoomLocate
xcodebuild -project RoomLocate.xcodeproj -scheme RoomLocate -configuration Debug \
  -destination 'generic/platform=iOS Simulator' -derivedDataPath build/simulator \
  CODE_SIGNING_ALLOWED=NO build
xcodebuild -project RoomLocate.xcodeproj -scheme RoomLocate -configuration Release \
  -destination 'generic/platform=iOS' -derivedDataPath build/device \
  CODE_SIGNING_ALLOWED=NO build
echo "Archive checks, simulator build and unsigned device build passed. Sign with your own team to install or distribute."
