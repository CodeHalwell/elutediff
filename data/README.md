# data/

Local data lives here and is **git-ignored** (only the directory structure is
tracked via `.gitkeep`).

- `raw/` — the downloaded METLIN SMRT dataset (SDF/molfiles + RT table).
  Fetch with `python scripts/download_metlin.py --out data/raw/metlin_smrt`.
  Source: Domingo-Almenara et al., *Nat. Commun.* 10, 5811 (2019);
  figshare DOI `10.6084/m9.figshare.8038913`.
- `processed/` — derived artifacts, e.g. the `(prompt, RT-density target)` JSONL
  produced by `scripts/build_targets.py`.

No raw chromatograms are provided by METLIN SMRT; RT-density targets here are
synthetic (Gaussian) by construction — see `docs/density-first-revision.md`.
