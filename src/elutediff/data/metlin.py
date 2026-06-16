"""Load and canonicalize the METLIN SMRT dataset (proposal Section 2).

METLIN SMRT provides, per molecule: RT (seconds), PubChem id, SDF/molfiles,
molecular descriptors, and ECFP fingerprints -- but *no raw chromatograms*.
This module turns the raw download into a tidy table the rest of the pipeline
consumes.

Reference: Domingo-Almenara et al., Nat. Commun. 10, 5811 (2019);
data at figshare DOI 10.6084/m9.figshare.8038913.

NOTE: scaffold. Wire up the real readers in roadmap step 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Molecule:
    """A single METLIN SMRT record."""

    pubchem_id: str
    smiles: str          # RDKit canonical SMILES
    rt_seconds: float
    mol_block: str | None = None  # raw SDF/molfile text, if retained


def load_metlin(path: str | Path) -> list[Molecule]:
    """Load METLIN SMRT records from ``path`` (SDF + RT table).

    Should: parse the SDF with RDKit, canonicalize SMILES, attach RT in seconds,
    drop unparseable / RT-missing records, and de-duplicate by canonical SMILES.
    """
    raise NotImplementedError(
        "load_metlin: implement METLIN SMRT reading (roadmap step 1). "
        "Download the dataset to data/raw/ via scripts/download_metlin.py."
    )
