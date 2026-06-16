#!/usr/bin/env python3
"""Train the B3 GNN / graph-transformer RT baseline -- the high bar (roadmap step 4).

Consumes the JSONL from scripts/build_targets.py (needs smiles, rt, split) and
trains the GINE regressor from elutediff.models.gnn. Requires the ``graph`` extra
(torch + torch-geometric); best on a GPU but runs on CPU for small sets.

    python scripts/train_gnn.py --data data/processed/targets.jsonl --epochs 100
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from elutediff.evaluation.point_rt import point_rt_metrics, tolerance_hit_rate


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="JSONL from scripts/build_targets.py")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--tolerances", type=float, nargs="+", default=[15.0, 30.0, 60.0])
    args = ap.parse_args()

    import torch
    from torch_geometric.loader import DataLoader

    from elutediff.models.gnn import build_gnn, mol_to_graph, predict_rt

    rows = [json.loads(line) for line in open(args.data, encoding="utf-8")]

    def fold(name):
        return [r for r in rows if r.get("split") == name]

    train_rows, val_rows, test_rows = fold("train"), fold("val"), fold("test")
    print(f"folds: train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def to_graphs(rs):
        graphs = []
        for r in rs:
            g = mol_to_graph(r["smiles"])
            g.y = torch.tensor([r["rt"]], dtype=torch.float)
            graphs.append(g)
        return graphs

    train_loader = DataLoader(to_graphs(train_rows), batch_size=args.batch_size, shuffle=True)

    model = build_gnn(hidden=args.hidden, layers=args.layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.HuberLoss(delta=30.0)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            loss = loss_fn(model(batch), batch.y)
            loss.backward()
            opt.step()
            total += loss.item() * batch.num_graphs
        if epoch % 10 == 0 or epoch == args.epochs:
            print(f"epoch {epoch:3d}/{args.epochs} | train loss {total / len(train_rows):.2f}")

    for name, rs in (("val", val_rows), ("test", test_rows)):
        if not rs:
            continue
        y = np.array([r["rt"] for r in rs])
        pred = predict_rt(model.to(device), [r["smiles"] for r in rs], batch_size=args.batch_size)
        m = point_rt_metrics(y, pred)
        hits = tolerance_hit_rate(y, pred, args.tolerances)
        print(
            f"[{name}] MAE {m['mae']:.1f}s medAE {m['median_ae']:.1f}s "
            f"RMSE {m['rmse']:.1f}s R2 {m['r2']:.3f} | "
            + " ".join(f"{k} {v*100:.0f}%" for k, v in hits.items())
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
