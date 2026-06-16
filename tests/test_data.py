import numpy as np
import pytest
from rdkit import Chem

from elutediff.config import SplitConfig
from elutediff.data.metlin import load_metlin
from elutediff.data.molecules import (
    InvalidMoleculeError,
    atom_bond_table,
    canonical_smiles,
    compute_descriptors,
    ecfp_fingerprint,
)
from elutediff.data.splits import make_split

# A small, chemically valid set spanning a couple of scaffolds.
SMILES = [
    "CCO", "CC(C)O", "CCCCO",                 # aliphatic alcohols
    "c1ccccc1", "c1ccccc1O", "Cc1ccccc1",      # benzene scaffold
    "c1ccncc1", "Cc1ccncc1",                    # pyridine scaffold
    "CC(=O)O", "CCC(=O)O",                      # carboxylic acids
]


def _write_sdf(path, smiles_rt):
    writer = Chem.SDWriter(str(path))
    for i, (smi, rt) in enumerate(smiles_rt):
        mol = Chem.MolFromSmiles(smi)
        mol.SetProp("_Name", f"CID{i}")
        mol.SetProp("RETENTION_TIME", str(rt))
        writer.write(mol)
    writer.close()


def test_canonical_smiles_and_invalid():
    assert canonical_smiles("OCC") == canonical_smiles("CCO")
    with pytest.raises(InvalidMoleculeError):
        canonical_smiles("not_a_molecule")


def test_descriptors():
    d = compute_descriptors("CCO", ["MolWt", "LogP", "HBD", "HBA"])
    assert 45 < d["MolWt"] < 47
    assert d["HBD"] == 1
    with pytest.raises(KeyError):
        compute_descriptors("CCO", ["NotADescriptor"])


def test_ecfp_shape_and_bits():
    fp = ecfp_fingerprint("c1ccccc1O", n_bits=1024)
    assert fp.shape == (1024,)
    assert fp.dtype == np.uint8
    assert fp.sum() > 0


def test_atom_bond_table_contents():
    table = atom_bond_table("CCO")
    assert table.startswith("atoms:")
    assert "bonds:" in table
    assert table.count("\n") >= 4  # header + 3 atoms + bonds header + 2 bonds


def test_load_sdf(tmp_path):
    sdf = tmp_path / "smrt.sdf"
    _write_sdf(sdf, [("CCO", 120.0), ("c1ccccc1", 480.0), ("CCO", 121.0)])  # dup CCO
    mols, stats = load_metlin(sdf, return_stats=True)
    assert stats.total == 3
    assert stats.duplicates == 1
    assert stats.kept == 2
    rts = {m.smiles: m.rt_seconds for m in mols}
    assert rts[canonical_smiles("CCO")] == 120.0  # first occurrence wins


def test_load_directory_autodetect(tmp_path):
    _write_sdf(tmp_path / "data.sdf", [("CCO", 100.0)])
    mols = load_metlin(tmp_path)
    assert len(mols) == 1 and mols[0].pubchem_id == "CID0"


def test_load_csv(tmp_path):
    csv = tmp_path / "smrt.csv"
    csv.write_text("smiles,rt,pubchem\nCCO,150.5,111\nbad!!,200,222\nc1ccccc1,300,333\n")
    mols, stats = load_metlin(csv, return_stats=True)
    assert stats.unparseable == 1
    assert {m.pubchem_id for m in mols} == {"111", "333"}


def test_random_split_partitions_all():
    cfg = SplitConfig(strategy="random", val_frac=0.2, test_frac=0.2, seed=0)
    s = make_split(SMILES, cfg)
    allidx = sorted(s.train_idx + s.val_idx + s.test_idx)
    assert allidx == list(range(len(SMILES)))
    assert sum(s.sizes()) == len(SMILES)


def test_scaffold_split_no_leakage():
    from rdkit.Chem.Scaffolds import MurckoScaffold

    cfg = SplitConfig(strategy="scaffold", val_frac=0.2, test_frac=0.2, seed=1)
    s = make_split(SMILES, cfg)
    # No (non-empty) scaffold should appear in more than one fold. Acyclic
    # molecules have an empty Murcko scaffold and are treated as singletons.
    def scaffolds(idxs):
        out = {
            MurckoScaffold.MurckoScaffoldSmiles(smiles=SMILES[i], includeChirality=False)
            for i in idxs
        }
        out.discard("")
        return out
    tr, va, te = scaffolds(s.train_idx), scaffolds(s.val_idx), scaffolds(s.test_idx)
    assert tr.isdisjoint(va) and tr.isdisjoint(te) and va.isdisjoint(te)
    assert sum(s.sizes()) == len(SMILES)


def test_cluster_split_partitions_all():
    cfg = SplitConfig(strategy="cluster", val_frac=0.2, test_frac=0.2,
                      cluster_cutoff=0.5, seed=0)
    s = make_split(SMILES, cfg)
    allidx = sorted(s.train_idx + s.val_idx + s.test_idx)
    assert allidx == list(range(len(SMILES)))
