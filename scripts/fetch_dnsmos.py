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
    if out.exists():
        print(f"already present: {out} ({out.stat().st_size} bytes)")
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {args.url}\n        → {out}")
    try:
        with urlopen(args.url, timeout=30) as r:  # noqa: S310 - pinned GH raw URL
            data = r.read()
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: download failed ({exc}).", file=sys.stderr)
        print("  The repo does not redistribute the model. Obtain sig_bak_ovr.onnx "
              "from the Microsoft DNS-Challenge repo (DNSMOS/DNSMOS/) and place it "
              f"at {out}.", file=sys.stderr)
        return 1
    out.write_bytes(data)
    print(f"wrote {len(data)} bytes. PipelineConfig will auto-wire it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
