#!/usr/bin/env python3
"""Download the METLIN SMRT dataset into data/raw/ (roadmap step 1).

Data: figshare DOI 10.6084/m9.figshare.8038913 (Domingo-Almenara et al., 2019).
This is a placeholder: fill in the figshare file URLs / API call for your setup.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/raw/metlin_smrt", help="Output directory.")
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    raise SystemExit(
        "download_metlin: add the figshare download for METLIN SMRT "
        "(DOI 10.6084/m9.figshare.8038913) -> " + args.out
    )


if __name__ == "__main__":
    raise SystemExit(main())
