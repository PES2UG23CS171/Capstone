#!/bin/zsh
# Build dist/VoiceISO-<version>-macOS-AppleSilicon.dmg.  Run from any directory
# with the build environment's python on PATH (or PYTHON=/path/to/python).
#
# The app is ad-hoc signed.  Removing the "Open Anyway" step on other people's
# Macs needs an Apple Developer ID and notarization.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PYTHON:-python}
VERSION=$(<packaging/VERSION)
APP=build/dist/VoiceISO.app
DMG=dist/VoiceISO-$VERSION-macOS-AppleSilicon.dmg

"$PY" -m PyInstaller --noconfirm --clean packaging/voiceiso.spec \
    --distpath build/dist --workpath build/work

echo "--- audit: nothing may link outside the bundle; stamp the real minimum macOS"
# The minimum depends on the machine and Python that built the libraries, so
# measure it.  A wrong value means a cryptic crash instead of a clear message.
MIN_MACOS=$("$PY" packaging/check_bundle.py "$APP")
echo "requires macOS $MIN_MACOS"
/usr/libexec/PlistBuddy -c "Set :LSMinimumSystemVersion $MIN_MACOS" "$APP/Contents/Info.plist"
codesign --force --sign - "$APP"            # editing Info.plist breaks the seal

echo "--- self-test of the built app"
"$APP/Contents/MacOS/VoiceISO" --self-test --self-test-out build/selftest-macos.json

echo "--- signature must still be valid after running (nothing may write into the app)"
codesign --verify --deep --strict "$APP"

echo "--- disk image"
STAGE=$(mktemp -d)
ditto "$APP" "$STAGE/VoiceISO.app"
ln -s /Applications "$STAGE/Applications"
cp packaging/READ_ME_FIRST.txt "$STAGE/Read Me First.txt"
mkdir -p dist
rm -f "$DMG"
hdiutil create -quiet -volname VoiceISO -srcfolder "$STAGE" -ov -format UDZO "$DMG"
rm -rf "$STAGE"
ls -lh "$DMG"
