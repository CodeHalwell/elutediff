"""Molecule featurization: descriptors, ECFP fingerprints, atom/bond tables.

Backs the classical baselines (ECFP/descriptor models) and the higher
conditioning levels (atom/bond serialization). RDKit required (``chem`` extra).

NOTE: scaffold. Implement in roadmap step 1.
"""

from __future__ import annotations


def canonical_smiles(smiles: str) -> str:
    """Return RDKit canonical SMILES (raises on invalid input)."""
    raise NotImplementedError("canonical_smiles: RDKit canonicalization (roadmap step 1).")


def compute_descriptors(smiles: str, names: list[str]) -> dict[str, float]:
    """Compute the requested RDKit descriptors (MolWt, LogP, TPSA, HBD, HBA, ...)."""
    raise NotImplementedError("compute_descriptors: RDKit descriptors (roadmap step 1).")


def ecfp_fingerprint(smiles: str, radius: int = 2, n_bits: int = 2048):
    """Return a length-``n_bits`` ECFP (Morgan) fingerprint as a numpy array."""
    raise NotImplementedError("ecfp_fingerprint: Morgan fingerprint (roadmap step 1).")


def atom_bond_table(smiles: str) -> str:
    """Serialize atoms (type, aromaticity, charge) and bonds (type, stereo) to text.

    Used by conditioning level >= 3. Keep it compact and parser-friendly.
    """
    raise NotImplementedError("atom_bond_table: graph serialization (roadmap step 1).")
