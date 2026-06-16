#!/usr/bin/env python3
"""Download the METLIN SMRT dataset from figshare (roadmap step 1).

Data: figshare DOI 10.6084/m9.figshare.8038913 (Domingo-Almenara et al.,
Nat. Commun. 2019). Files are resolved live via the figshare API, streamed to
``--out``, and verified against the supplied MD5. Already-present, valid files
are skipped.

Available files (name -> size):
  csv           SMRT_dataset.csv                      ~12 MB  (pubchem;rt;inchi)
  sdf           SMRT_dataset.sdf                       ~376 MB (3D molfiles + RT)
  fingerprints  SMRT_ECFP_1024_Fingerprints.txt       ~83 MB
  descriptors   SMRT_molecular_descriptors.zip        ~521 MB
  model         Deep learning model and results.zip   ~14 MB

The CSV is the smallest complete source and is enough for the whole elutediff
pipeline (elutediff.data.load_metlin parses it directly).

    python scripts/download_metlin.py                      # CSV only (default)
    python scripts/download_metlin.py --files csv sdf
    python scripts/download_metlin.py --all --out data/raw/metlin_smrt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

ARTICLE_ID = 8038913
API_URL = f"https://api.figshare.com/v2/articles/{ARTICLE_ID}"

# Friendly aliases -> exact figshare file names.
ALIASES = {
    "csv": "SMRT_dataset.csv",
    "sdf": "SMRT_dataset.sdf",
    "fingerprints": "SMRT_ECFP_1024_Fingerprints.txt",
    "descriptors": "SMRT_molecular_descriptors.zip",
    "model": "Deep learning model and results.zip",
}


def _list_files() -> dict[str, dict]:
    with urllib.request.urlopen(API_URL, timeout=30) as resp:
        article = json.load(resp)
    return {f["name"]: f for f in article["files"]}


def _md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _download(meta: dict, dest: Path) -> None:
    expected = meta.get("computed_md5") or meta.get("supplied_md5")
    if dest.exists() and expected and _md5(dest) == expected:
        print(f"  [skip] {dest.name} already present and verified")
        return

    total = int(meta.get("size", 0))
    done = 0
    print(f"  [get ] {dest.name} ({total / 1e6:.1f} MB)")
    with urllib.request.urlopen(meta["download_url"], timeout=60) as resp, open(dest, "wb") as out:
        while True:
            block = resp.read(1 << 20)
            if not block:
                break
            out.write(block)
            done += len(block)
            if total:
                pct = 100 * done / total
                sys.stdout.write(f"\r         {done / 1e6:7.1f}/{total / 1e6:.1f} MB ({pct:5.1f}%)")
                sys.stdout.flush()
    if total:
        sys.stdout.write("\n")

    if expected:
        actual = _md5(dest)
        if actual != expected:
            raise RuntimeError(f"MD5 mismatch for {dest.name}: {actual} != {expected}")
        print(f"  [ok  ] {dest.name} md5 verified")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/raw/metlin_smrt", help="Output directory.")
    ap.add_argument("--files", nargs="+", default=["csv"], choices=list(ALIASES),
                    help="Which files to fetch (default: csv).")
    ap.add_argument("--all", action="store_true", help="Fetch every file in the article.")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    available = _list_files()
    wanted = list(available) if args.all else [ALIASES[a] for a in args.files]

    print(f"Downloading {len(wanted)} file(s) to {out}/ from figshare article {ARTICLE_ID}")
    for name in wanted:
        if name not in available:
            print(f"  [warn] {name} not found in article; skipping")
            continue
        _download(available[name], out / name)
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
