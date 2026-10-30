#!/bin/bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
if ! command -v xcodebuild >/dev/null 2>&1; then
  echo "Install Xcode, open it once, then select it with xcode-select."
  exit 1
fi
xcodebuild -resolvePackageDependencies -project RoomLocate.xcodeproj -scheme RoomLocate
xcodebuild -project RoomLocate.xcodeproj -scheme RoomLocate -configuration Debug \
  -destination 'generic/platform=iOS Simulator' -derivedDataPath build \
  CODE_SIGNING_ALLOWED=NO build
echo "Simulator build passed. Select your iPhone and Personal Team in Xcode for device installation."
