"""Audit a built VoiceISO.app.  Prints the macOS version it really requires.

Fails if any binary links to something outside the bundle and the OS (a Homebrew
path, say), which works on the build machine and crashes on everyone else's.
"""
import os
import re
import subprocess
import sys

ALLOWED_PREFIXES = ("/System/", "/usr/lib/", "@rpath", "@loader_path", "@executable_path")
MACHO_MAGIC = (b"\xcf\xfa\xed\xfe", b"\xca\xfe\xba\xbe")


def macho_files(app: str):
    seen = set()
    for root, _, files in os.walk(app):
        for name in files:
            path = os.path.join(root, name)
            real = os.path.realpath(path)
            if os.path.islink(path) or real in seen:
                continue
            seen.add(real)
            with open(path, "rb") as f:
                if f.read(4) in MACHO_MAGIC:
                    yield path


def main(app: str) -> int:
    outside = set()
    required = (11, 0)
    for path in macho_files(app):
        listing = subprocess.run(["otool", "-L", path], capture_output=True, text=True).stdout
        for line in listing.splitlines()[1:]:
            dep = line.strip().split(" (")[0]
            if not dep.startswith(ALLOWED_PREFIXES):
                outside.add(f"{os.path.relpath(path, app)} -> {dep}")
        commands = subprocess.run(["otool", "-l", path], capture_output=True, text=True).stdout
        for version in re.findall(r"LC_BUILD_VERSION.*?minos (\d+)\.(\d+)", commands, re.S):
            required = max(required, (int(version[0]), int(version[1])))
    if outside:
        print("Binaries link outside the bundle:", *sorted(outside), sep="\n  ", file=sys.stderr)
        return 1
    print(f"{required[0]}.{required[1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
