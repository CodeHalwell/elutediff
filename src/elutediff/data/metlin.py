"""Load and canonicalize the METLIN SMRT dataset (proposal Section 2).

METLIN SMRT provides, per molecule: RT (seconds), PubChem id, SDF/molfiles,
molecular descriptors, and ECFP fingerprints -- but *no raw chromatograms*.
This module turns the raw download into a tidy list of :class:`Molecule`.

Supported inputs (``load_metlin`` auto-detects):
  * an SDF file (``.sdf``) -- RT read from a molecule property
    (``RETENTION_TIME`` / ``RT`` / ``rt``), PubChem id from the molecule title
    or a ``PUBCHEM``/``pubchem`` property;
  * a CSV file with columns for an identifier (``smiles`` or ``inchi``), an RT
    (``rt``/``retention_time``), and optionally ``pubchem``;
  * a directory containing one of the above (``.sdf`` preferred).

Reference: Domingo-Almenara et al., Nat. Commun. 10, 5811 (2019);
data at figshare DOI 10.6084/m9.figshare.8038913.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from rdkit import Chem

from elutediff.data.molecules import InvalidMoleculeError, canonical_smiles

_RT_KEYS = ("RETENTION_TIME", "retention_time", "RT", "rt", "Retention_Time")
_PUBCHEM_KEYS = ("PUBCHEM", "pubchem", "PUBCHEM_CID", "pubchem_cid", "CID", "cid")
_SMILES_KEYS = ("smiles", "SMILES", "canonical_smiles")
_INCHI_KEYS = ("inchi", "InChI", "INCHI")


@dataclass
class Molecule:
    """A single METLIN SMRT record."""

    pubchem_id: str
    smiles: str          # RDKit canonical SMILES
    rt_seconds: float
    mol_block: str | None = None  # raw molfile text, if retained


@dataclass
class LoadStats:
    """Summary of a load, so dropped records are visible rather than silent."""

    total: int = 0
    kept: int = 0
    unparseable: int = 0
    missing_rt: int = 0
    duplicates: int = 0

    def __str__(self) -> str:
        return (
            f"METLIN load: kept {self.kept}/{self.total} "
            f"(unparseable={self.unparseable}, missing_rt={self.missing_rt}, "
            f"duplicates={self.duplicates})"
        )


def _first(mapping, keys, getter):
    for k in keys:
        val = getter(mapping, k)
        if val not in (None, ""):
            return val
    return None


def _prop_getter(mol: Chem.Mol, key: str):
    return mol.GetProp(key) if mol.HasProp(key) else None


def _parse_rt(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_metlin(
    path: str | Path, keep_mol_block: bool = False, return_stats: bool = False
):
    """Load METLIN SMRT records from ``path``.

    Canonicalizes SMILES, attaches RT in seconds, drops unparseable / RT-missing
    records, and de-duplicates by canonical SMILES (first occurrence wins).

    Returns a ``list[Molecule]`` (or ``(molecules, LoadStats)`` if
    ``return_stats``).
    """
    file_path = _resolve_path(path)
    stats = LoadStats()

    if file_path.suffix.lower() == ".sdf":
        records = _iter_sdf(file_path, keep_mol_block, stats)
    else:
        records = _iter_csv(file_path, stats)

    seen: set[str] = set()
    molecules: list[Molecule] = []
    for mol in records:
        if mol.smiles in seen:
            stats.duplicates += 1
            continue
        seen.add(mol.smiles)
        molecules.append(mol)
    stats.kept = len(molecules)

    return (molecules, stats) if return_stats else molecules


def _resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_dir():
        for pattern in ("*.sdf", "*.csv"):
            hits = sorted(p.glob(pattern))
            if hits:
                return hits[0]
        raise FileNotFoundError(f"no .sdf or .csv found under {p}")
    if not p.exists():
        raise FileNotFoundError(f"no such file: {p}")
    return p


def _iter_sdf(path: Path, keep_mol_block: bool, stats: LoadStats):
    supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=True)
    for mol in supplier:
        stats.total += 1
        if mol is None:
            stats.unparseable += 1
            continue
        rt = _parse_rt(_first(mol, _RT_KEYS, _prop_getter))
        if rt is None:
            stats.missing_rt += 1
            continue
        smiles = Chem.MolToSmiles(mol)
        pubchem = _first(mol, _PUBCHEM_KEYS, _prop_getter)
        if not pubchem and mol.HasProp("_Name"):
            pubchem = mol.GetProp("_Name")
        yield Molecule(
            pubchem_id=str(pubchem or ""),
            smiles=smiles,
            rt_seconds=rt,
            mol_block=Chem.MolToMolBlock(mol) if keep_mol_block else None,
        )


def _iter_csv(path: Path, stats: LoadStats):
    with open(path, newline="") as fh:
        sample = fh.readline()
        fh.seek(0)
        reader = csv.DictReader(fh, delimiter=_sniff_delimiter(sample))
        get = dict.get
        for row in reader:
            stats.total += 1
            rt = _parse_rt(_first(row, _RT_KEYS, get))
            if rt is None:
                stats.missing_rt += 1
                continue
            smiles_raw = _first(row, _SMILES_KEYS, get)
            inchi = _first(row, _INCHI_KEYS, get)
            try:
                if smiles_raw:
                    smiles = canonical_smiles(smiles_raw)
                elif inchi:
                    mol = Chem.MolFromInchi(inchi)
                    if mol is None:
                        raise InvalidMoleculeError(inchi)
                    smiles = Chem.MolToSmiles(mol)
                else:
                    stats.unparseable += 1
                    continue
            except InvalidMoleculeError:
                stats.unparseable += 1
                continue
            yield Molecule(
                pubchem_id=str(_first(row, _PUBCHEM_KEYS, get) or ""),
                smiles=smiles,
                rt_seconds=rt,
            )


def _sniff_delimiter(header: str) -> str:
    """Pick ','/';'/tab from a header line (METLIN CSVs are sometimes ';')."""
    delimiter = ";" if header.count(";") > header.count(",") else ","
    if "\t" in header and header.count("\t") > header.count(delimiter):
        delimiter = "\t"
    return delimiter
