"""Molecule featurization: descriptors, ECFP fingerprints, atom/bond tables.

Backs the classical baselines (ECFP/descriptor models) and the higher
conditioning levels (atom/bond serialization). RDKit required (``chem`` extra).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

# RDKit is chatty about parse failures; we surface those ourselves via return codes.
RDLogger.DisableLog("rdApp.*")


class InvalidMoleculeError(ValueError):
    """Raised when a SMILES string cannot be parsed by RDKit."""


@lru_cache(maxsize=8192)
def _mol(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise InvalidMoleculeError(f"unparseable SMILES: {smiles!r}")
    return mol


def canonical_smiles(smiles: str) -> str:
    """Return RDKit canonical SMILES (raises :class:`InvalidMoleculeError`)."""
    return Chem.MolToSmiles(_mol(smiles))


# Descriptor name -> callable. Names match ConditioningConfig.descriptors defaults.
_DESCRIPTORS = {
    "MolWt": Descriptors.MolWt,
    "LogP": Descriptors.MolLogP,
    "TPSA": Descriptors.TPSA,
    "HBD": rdMolDescriptors.CalcNumHBD,
    "HBA": rdMolDescriptors.CalcNumHBA,
    "RotatableBonds": rdMolDescriptors.CalcNumRotatableBonds,
    "NumRings": rdMolDescriptors.CalcNumRings,
    "AromaticRings": rdMolDescriptors.CalcNumAromaticRings,
    "FractionCSP3": rdMolDescriptors.CalcFractionCSP3,
    "HeavyAtomCount": lambda m: m.GetNumHeavyAtoms(),
}


def available_descriptors() -> list[str]:
    """Names recognized by :func:`compute_descriptors`."""
    return list(_DESCRIPTORS)


def compute_descriptors(smiles: str, names: list[str]) -> dict[str, float]:
    """Compute the requested RDKit descriptors for a SMILES string."""
    mol = _mol(smiles)
    out: dict[str, float] = {}
    for name in names:
        if name not in _DESCRIPTORS:
            raise KeyError(f"unknown descriptor {name!r}; see available_descriptors()")
        out[name] = float(_DESCRIPTORS[name](mol))
    return out


def ecfp_fingerprint(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """Return a length-``n_bits`` ECFP (Morgan) fingerprint as a uint8 array."""
    mol = _mol(smiles)
    gen = AllChem.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fp = gen.GetFingerprint(mol)
    arr = np.zeros(n_bits, dtype=np.uint8)
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def atom_bond_table(smiles: str) -> str:
    """Serialize atoms and bonds to a compact, parser-friendly text block.

    One line per atom (index, element, aromatic flag, formal charge, Hs) and one
    line per bond (atoms, order, aromatic/ring/stereo flags). Used by
    conditioning level >= 3.
    """
    mol = _mol(smiles)
    lines = ["atoms:"]
    for atom in mol.GetAtoms():
        lines.append(
            f"{atom.GetIdx()} {atom.GetSymbol()} "
            f"arom={int(atom.GetIsAromatic())} "
            f"q={atom.GetFormalCharge()} h={atom.GetTotalNumHs()}"
        )
    lines.append("bonds:")
    for bond in mol.GetBonds():
        lines.append(
            f"{bond.GetBeginAtomIdx()}-{bond.GetEndAtomIdx()} "
            f"order={bond.GetBondTypeAsDouble():g} "
            f"arom={int(bond.GetIsAromatic())} "
            f"ring={int(bond.IsInRing())} "
            f"stereo={str(bond.GetStereo()).replace('STEREO', '')}"
        )
    return "\n".join(lines)
