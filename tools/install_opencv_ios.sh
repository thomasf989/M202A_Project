#!/bin/bash
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
vendor="$repo/ios/WalkingBaseline/Vendor"
mkdir -p "$vendor" "$repo/ios/WalkingBaseline/Resources/Room"
archive="$vendor/opencv-5.1.0.zip"
curl -fL https://github.com/opencv/opencv/releases/download/5.1.0/opencv-5.1.0-ios-framework.zip -o "$archive"
expected=6fe1e1deda814e9f7eb6ca3e1b22289037c88738bdc81daf6f705813e8ae7e1c
actual=$(shasum -a 256 "$archive" | cut -d ' ' -f 1)
[[ "$actual" == "$expected" ]] || { echo 'OpenCV checksum mismatch'; exit 1; }
if [[ ! -d "$vendor/opencv2.framework" ]]; then unzip -q "$archive" -d "$vendor"; fi
echo 'OpenCV ready. Use a physical iPhone for localization; simulator supports UI/Swift tests.'
