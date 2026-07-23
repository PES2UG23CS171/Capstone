"""
Fetch the Microsoft DNSMOS P.835 model (sig_bak_ovr.onnx) into checkpoints/.

DNSMOS is a NON-INTRUSIVE perceptual metric (SIG / BAK / OVRL) and is the only
quality metric that works on real mic recordings.  The model is ~1 MB and is
distributed with the Microsoft DNS-Challenge repo (CC-licensed); we do not
redistribute it here, so this script downloads it on demand.

Once present, ``PipelineConfig`` auto-wires it (config.py __post_init__) and
``python -m voiceiso bench`` reports DNSMOS without any flag.

Usage
-----
    python -m scripts.fetch_dnsmos                 # default URL → checkpoints/
    python -m scripts.fetch_dnsmos --url <URL>     # override source
    python -m scripts.fetch_dnsmos --out path.onnx
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.request import urlopen

# Microsoft DNS-Challenge DNSMOS P.835 primary model (raw GitHub).
_DEFAULT_URL = (
    "https://raw.githubusercontent.com/microsoft/DNS-Challenge/master/"
    "DNSMOS/DNSMOS/sig_bak_ovr.onnx"
)


def main() -> int:
    ap = argparse.ArgumentParser(description="Download the DNSMOS sig_bak_ovr.onnx model")
    ap.add_argument("--url", default=_DEFAULT_URL)
    ap.add_argument("--out", default="checkpoints/sig_bak_ovr.onnx")
    args = ap.parse_args()

    out = Path(args.out)
    # A partial/failed prior run can leave a zero-byte corpse — treat anything
    # under 100 KB as absent (the real model is ~1.1 MB).
    if out.exists() and out.stat().st_size > 100_000:
        print(f"already present: {out} ({out.stat().st_size} bytes)")
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {args.url}\n        → {out}")
    data = b""
    try:
        with urlopen(args.url, timeout=60) as r:  # noqa: S310 - pinned GH raw URL
            data = r.read()
    except Exception as exc:  # noqa: BLE001
        print(f"urllib failed ({exc}); trying curl…", file=sys.stderr)
    if len(data) < 100_000:
        # Some environments return empty bodies through urllib for GitHub
        # redirects — fall back to curl -L (observed on the dev Mac).
        import subprocess
        gh = args.url.replace("raw.githubusercontent.com/", "github.com/").replace(
            "/master/", "/raw/master/") if "raw.githubusercontent" in args.url else args.url
        rc = subprocess.run(["curl", "-sL", "--fail", "--max-time", "240",
                             "-o", str(out), gh]).returncode
        if rc == 0 and out.exists():
            data = out.read_bytes()
    # Validate: size + protobuf-ish start (0x08 field-1 varint = ONNX ir_version).
    if len(data) < 100_000 or data[:1] != b"\x08":
        if out.exists():
            out.unlink()
        print(f"ERROR: download invalid ({len(data)} bytes).", file=sys.stderr)
        print("  Obtain sig_bak_ovr.onnx from the Microsoft DNS-Challenge repo "
              f"(DNSMOS/DNSMOS/) and place it at {out}.", file=sys.stderr)
        return 1
    out.write_bytes(data)
    print(f"wrote {len(data)} bytes (validated). PipelineConfig will auto-wire it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
