"""GNN baseline tests (B3). Skipped unless the ``graph`` extra is installed."""

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

import torch  # noqa: E402

from elutediff.models.gnn import build_gnn, mol_to_graph, predict_rt  # noqa: E402


def test_mol_to_graph_shapes():
    g = mol_to_graph("CCO")  # 3 heavy atoms, 2 bonds -> 4 directed edges
    assert g.x.shape == (3, 5)
    assert g.edge_index.shape == (2, 4)
    assert g.edge_attr.shape == (4, 2)


def test_mol_to_graph_invalid():
    with pytest.raises(ValueError):
        mol_to_graph("not_a_molecule")


def test_gnn_forward_batch():
    from torch_geometric.loader import DataLoader

    smiles = ["CCO", "c1ccccc1O", "CC(=O)O", "c1ccncc1"]
    loader = DataLoader([mol_to_graph(s) for s in smiles], batch_size=4)
    model = build_gnn(hidden=16, layers=2)
    model.eval()
    with torch.no_grad():
        out = model(next(iter(loader)))
    assert out.shape == (4,)
    assert torch.isfinite(out).all()


def test_gnn_overfits_tiny_signal():
    # The model should be able to drive training loss down on a handful of graphs.
    smiles = ["CCO", "CCCCO", "c1ccccc1", "c1ccc2ccccc2c1", "CC(=O)O"]
    y = torch.tensor([50.0, 120.0, 300.0, 600.0, 90.0])
    graphs = [mol_to_graph(s) for s in smiles]

    from torch_geometric.loader import DataLoader

    for g, t in zip(graphs, y):
        g.y = t.view(1)
    loader = DataLoader(graphs, batch_size=5, shuffle=True)

    torch.manual_seed(0)
    model = build_gnn(hidden=32, layers=2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = torch.nn.MSELoss()

    first = None
    for _ in range(200):
        for batch in loader:
            opt.zero_grad()
            loss = loss_fn(model(batch), batch.y)
            loss.backward()
            opt.step()
        if first is None:
            first = loss.item()
    assert loss.item() < first  # learning happened


def test_predict_rt_runs():
    smiles = ["CCO", "c1ccccc1", "CC(=O)O"]
    model = build_gnn(hidden=16, layers=2)
    preds = predict_rt(model, smiles, batch_size=2)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (3,)
