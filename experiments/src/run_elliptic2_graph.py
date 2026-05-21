import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from torch_geometric.nn import SAGEConv


class Elliptic2GraphSAGE(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.cls = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        return self.cls(h).squeeze(-1)


def split_components(component_labels: pd.DataFrame, seed: int):
    rng = random.Random(seed)
    train_components, val_components, test_components = set(), set(), set()
    for label, group in component_labels.groupby("ccLabel"):
        component_ids = group["ccId"].tolist()
        rng.shuffle(component_ids)
        n_total = len(component_ids)
        n_train = int(0.7 * n_total)
        n_val = int(0.15 * n_total)
        train_components.update(component_ids[:n_train])
        val_components.update(component_ids[n_train:n_train + n_val])
        test_components.update(component_ids[n_train + n_val:])
    return train_components, val_components, test_components


def build_graph(data_dir: Path):
    nodes = pd.read_csv(data_dir / "nodes.csv")
    component_labels = pd.read_csv(data_dir / "connected_components.csv")
    merged = nodes.merge(component_labels, on="ccId", how="left")
    merged = merged[merged["ccLabel"].isin(["licit", "suspicious"])].copy()
    merged["label"] = (merged["ccLabel"] == "suspicious").astype(np.float32)

    background_nodes_path = data_dir / "background_nodes.csv"
    background_feature_cols = []
    if background_nodes_path.exists():
        background_nodes = pd.read_csv(background_nodes_path)
        background_feature_cols = [col for col in background_nodes.columns if col.startswith("feat#")]
        merged = merged.merge(background_nodes, on="clId", how="left")
        if background_feature_cols:
            merged[background_feature_cols] = merged[background_feature_cols].fillna(0.0)

    edges = pd.read_csv(data_dir / "edges.csv")
    node_ids = set(merged["clId"].tolist())
    edges = edges[edges["clId1"].isin(node_ids) & edges["clId2"].isin(node_ids)].copy()

    node_to_idx = {node_id: idx for idx, node_id in enumerate(merged["clId"].tolist())}
    edges["src"] = edges["clId1"].map(node_to_idx)
    edges["dst"] = edges["clId2"].map(node_to_idx)

    degree_counts = pd.concat([edges["src"], edges["dst"]]).value_counts()
    component_sizes = merged["ccId"].map(merged.groupby("ccId").size()).astype(np.float32)
    degree = np.array([degree_counts.get(idx, 0) for idx in range(len(merged))], dtype=np.float32)
    component_size = component_sizes.to_numpy(dtype=np.float32)
    structural_features = np.stack(
        [
            np.log1p(degree),
            np.log1p(component_size),
            degree / np.maximum(component_size, 1.0),
            (degree <= 1).astype(np.float32),
            np.ones(len(merged), dtype=np.float32),
        ],
        axis=1,
    ).astype(np.float32)

    if background_feature_cols:
        node_background = merged[background_feature_cols].to_numpy(dtype=np.float32)
        node_background = np.log1p(np.clip(node_background, a_min=0.0, a_max=None))
        features = np.concatenate([structural_features, node_background], axis=1)
    else:
        features = structural_features

    src = edges["src"].to_numpy(dtype=np.int64)
    dst = edges["dst"].to_numpy(dtype=np.int64)
    edge_index = np.stack([np.concatenate([src, dst]), np.concatenate([dst, src])], axis=0)

    return merged, torch.tensor(features, dtype=torch.float32), torch.tensor(edge_index, dtype=torch.long)


def compute_metrics(logits, labels, mask):
    masked_logits = logits[mask]
    masked_labels = labels[mask]
    probs = torch.sigmoid(masked_logits).detach().cpu().numpy()
    gold = masked_labels.detach().cpu().numpy().astype(int)
    pred = (probs >= 0.5).astype(int)
    return {
        "roc_auc": round(float(roc_auc_score(gold, probs)), 4),
        "average_precision": round(float(average_precision_score(gold, probs)), 4),
        "precision": round(float(precision_score(gold, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(gold, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(gold, pred, zero_division=0)), 4),
        "positive_rate": round(float(gold.mean()), 4),
        "n": int(mask.sum().item()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.outdir.mkdir(parents=True, exist_ok=True)
    merged, x, edge_index = build_graph(args.data_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    component_labels = merged[["ccId", "ccLabel"]].drop_duplicates()
    train_components, val_components, test_components = split_components(component_labels, args.seed)

    train_mask = torch.tensor(merged["ccId"].isin(train_components).to_numpy(), dtype=torch.bool, device=device)
    val_mask = torch.tensor(merged["ccId"].isin(val_components).to_numpy(), dtype=torch.bool, device=device)
    test_mask = torch.tensor(merged["ccId"].isin(test_components).to_numpy(), dtype=torch.bool, device=device)
    labels = torch.tensor(merged["label"].to_numpy(), dtype=torch.float32, device=device)
    x = x.to(device)
    edge_index = edge_index.to(device)

    model = Elliptic2GraphSAGE(x.size(1), hidden_dim=args.hidden_dim).to(device)
    pos_train = labels[train_mask].sum().item()
    neg_train = train_mask.sum().item() - pos_train
    pos_weight = torch.tensor([max(neg_train / max(pos_train, 1.0), 1.0)], dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_state = None
    best_val_ap = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        logits = model(x, edge_index)
        loss = criterion(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(x, edge_index)
        train_metrics = compute_metrics(logits, labels, train_mask)
        val_metrics = compute_metrics(logits, labels, val_mask)
        epoch_record = {
            "epoch": epoch,
            "loss": round(float(loss.item()), 4),
            "train_average_precision": train_metrics["average_precision"],
            "val_average_precision": val_metrics["average_precision"],
        }
        history.append(epoch_record)
        if val_metrics["average_precision"] > best_val_ap:
            best_val_ap = val_metrics["average_precision"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        logits = model(x, edge_index)
    train_metrics = compute_metrics(logits, labels, train_mask)
    val_metrics = compute_metrics(logits, labels, val_mask)
    test_metrics = compute_metrics(logits, labels, test_mask)

    probs = torch.sigmoid(logits).detach().cpu().numpy()
    test_predictions = merged.loc[test_mask.cpu().numpy(), ["clId", "ccId", "ccLabel"]].copy()
    test_predictions["risk_score"] = probs[test_mask.cpu().numpy()]
    test_predictions.to_csv(args.outdir / "elliptic2_test_predictions.csv", index=False)

    summary = {
        "seed": args.seed,
        "epochs": args.epochs,
        "hidden_dim": args.hidden_dim,
        "lr": args.lr,
        "nodes": int(len(merged)),
        "edges": int(edge_index.size(1) // 2),
        "node_feature_dim": int(x.size(1)),
        "splits": {
            "train_components": len(train_components),
            "val_components": len(val_components),
            "test_components": len(test_components),
        },
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
    }
    (args.outdir / "elliptic2_summary.json").write_text(json.dumps(summary, indent=2))
    (args.outdir / "elliptic2_history.json").write_text(json.dumps(history, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
