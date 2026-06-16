"""B3: graph neural network / sparse graph transformer baseline (high bar).

A GNN/graph-transformer with proper inductive bias is expected to be the
strongest *point*-RT predictor; it is the bar DiffusionGemma's density output is
measured against (proposal Sections 8, 10). This module is wired against
PyTorch Geometric (the ``graph`` extra). Heavy imports are deferred so the rest
of the package imports without torch installed.

The reference recipe is GINE message passing with optional Laplacian positional
encodings, following GraphGPS (Rampasek et al., 2022) and LSPE (Dwivedi et al.,
2022). Swap in a full GPS layer once the GINE baseline is established.
"""

from __future__ import annotations

import numpy as np

_BOND_ORDER = {1.0: 0, 1.5: 1, 2.0: 2, 3.0: 3}


def mol_to_graph(smiles: str):
    """Convert a SMILES string to a PyG ``Data`` object (atom/bond features).

    Atom features: atomic number, degree, formal charge, aromaticity, total Hs.
    Bond features: bond-order class + aromatic flag. Edges are added in both
    directions (undirected graph).
    """
    import torch
    from rdkit import Chem
    from torch_geometric.data import Data

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"unparseable SMILES: {smiles!r}")

    x = torch.tensor(
        [
            [a.GetAtomicNum(), a.GetDegree(), a.GetFormalCharge(),
             int(a.GetIsAromatic()), a.GetTotalNumHs()]
            for a in mol.GetAtoms()
        ],
        dtype=torch.float,
    )

    edge_index, edge_attr = [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        feat = [_BOND_ORDER.get(b.GetBondTypeAsDouble(), 0), int(b.GetIsAromatic())]
        edge_index += [[i, j], [j, i]]
        edge_attr += [feat, feat]

    return Data(
        x=x,
        edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        if edge_index else torch.empty((2, 0), dtype=torch.long),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float)
        if edge_attr else torch.empty((0, 2), dtype=torch.float),
    )


def build_gnn(node_dim: int = 5, hidden: int = 128, layers: int = 4, dropout: float = 0.1):
    """Construct a GINE-based RT regressor (returns an ``nn.Module``).

    Output is a single scalar RT (seconds). Train with Huber/MSE loss and the
    same scaffold/cluster splits as every other baseline.
    """
    import torch
    from torch import nn
    from torch_geometric.nn import GINEConv, global_mean_pool

    class GNNRegressor(nn.Module):
        def __init__(self):
            super().__init__()
            self.in_proj = nn.Linear(node_dim, hidden)
            self.edge_proj = nn.Linear(2, hidden)
            self.convs = nn.ModuleList(
                GINEConv(nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                       nn.Linear(hidden, hidden)))
                for _ in range(layers)
            )
            self.norms = nn.ModuleList(nn.BatchNorm1d(hidden) for _ in range(layers))
            self.head = nn.Sequential(
                nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1)
            )

        def forward(self, data):
            x = self.in_proj(data.x)
            ea = self.edge_proj(data.edge_attr)
            for conv, norm in zip(self.convs, self.norms):
                x = torch.relu(norm(conv(x, data.edge_index, ea))) + x
            return self.head(global_mean_pool(x, data.batch)).squeeze(-1)

    return GNNRegressor()


def predict_rt(model, smiles: list[str], batch_size: int = 256) -> np.ndarray:
    """Run a trained GNN over SMILES and return predicted RT (seconds)."""
    import torch
    from torch_geometric.loader import DataLoader

    graphs = [mol_to_graph(s) for s in smiles]
    device = next(model.parameters()).device
    model.eval()
    out = []
    with torch.no_grad():
        for batch in DataLoader(graphs, batch_size=batch_size):
            out.append(model(batch.to(device)).cpu().numpy())
    return np.concatenate(out)
