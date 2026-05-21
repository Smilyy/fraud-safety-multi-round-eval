"""GNN risk-score probe.

Loads a cached static and temporal graph encoder and reports their risk-score
distributions on (a) the actual fraud test cases and (b) benign substitutes
generated from the same case role-backgrounds. Used to demonstrate the
fraud/benign separation produced by the encoder, supporting the paper's claim
that the safety-utility tradeoff is an LLM-conditioning issue rather than a
graph-quality issue.

Output: artifacts/gnn_probe.json
"""
import argparse
import json
from pathlib import Path
from statistics import mean, pstdev

import torch

try:
    from benign_control_utils import build_benign_multi_round
    from fraud_r1_joint_graph import (
        StaticGraphEncoder,
        TemporalGraphEncoder,
        build_global_stats,
        build_snapshot,
        get_multi_round_texts,
        load_paired_examples,
        peek_training_sample,
    )
except ModuleNotFoundError:
    from experiments.src.benign_control_utils import build_benign_multi_round
    from experiments.src.fraud_r1_joint_graph import (
        StaticGraphEncoder,
        TemporalGraphEncoder,
        build_global_stats,
        build_snapshot,
        get_multi_round_texts,
        load_paired_examples,
        peek_training_sample,
    )


def stats(values):
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": round(mean(values), 4),
        "std": round(pstdev(values), 4) if len(values) > 1 else 0.0,
        "min": round(min(values), 4),
        "max": round(max(values), 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-data", type=Path, required=True)
    parser.add_argument("--levelup-data", type=Path, required=True)
    parser.add_argument("--language", type=str, default="English")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--test-limit", type=int)
    parser.add_argument("--graph-cache-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    train_pairs, test_pairs = load_paired_examples(
        args.base_data,
        args.levelup_data,
        args.language,
        args.seed,
        args.test_fraction,
        split_manifest=args.split_manifest,
    )
    if args.train_limit:
        train_pairs = train_pairs[: args.train_limit]
    if args.test_limit:
        test_pairs = test_pairs[: args.test_limit]
    global_stats = build_global_stats(train_pairs)

    first_snap, _ = peek_training_sample(train_pairs, global_stats)
    in_dim = first_snap.x.shape[1]

    static = StaticGraphEncoder(in_dim).to(args.device)
    static.load_state_dict(torch.load(args.graph_cache_dir / "static_model.pt", map_location=args.device)["state_dict"])
    static.eval()
    temporal = TemporalGraphEncoder(in_dim).to(args.device)
    temporal.load_state_dict(torch.load(args.graph_cache_dir / "temporal_model.pt", map_location=args.device)["state_dict"])
    temporal.eval()

    fraud_static, fraud_temporal = [], []
    benign_static, benign_temporal = [], []
    for pair in test_pairs:
        ex = pair["levelup"]
        rounds = get_multi_round_texts(ex)
        if not rounds:
            continue
        snap = build_snapshot(ex, rounds, global_stats)
        with torch.no_grad():
            p_s, _ = static(snap)
            p_t, _ = temporal.forward_snapshot(snap, args.device)
        fraud_static.append(float(p_s.item()))
        fraud_temporal.append(float(p_t.item()))

        benign_pair_ex = pair["base"]
        benign_rounds = build_benign_multi_round(benign_pair_ex)
        if benign_rounds:
            b_snap = build_snapshot(benign_pair_ex, benign_rounds, global_stats)
            with torch.no_grad():
                p_s_b, _ = static(b_snap)
                p_t_b, _ = temporal.forward_snapshot(b_snap, args.device)
            benign_static.append(float(p_s_b.item()))
            benign_temporal.append(float(p_t_b.item()))

    snap_zero = build_snapshot(test_pairs[0]["levelup"], get_multi_round_texts(test_pairs[0]["levelup"]), global_stats)
    snap_zero.x.zero_()
    with torch.no_grad():
        p_s_zero, _ = static(snap_zero)
        p_t_zero, _ = temporal.forward_snapshot(snap_zero, args.device)

    output = {
        "graph_cache_dir": str(args.graph_cache_dir),
        "n_test_cases": len(test_pairs),
        "fraud": {
            "static": stats(fraud_static),
            "temporal": stats(fraud_temporal),
        },
        "benign": {
            "static": stats(benign_static),
            "temporal": stats(benign_temporal),
        },
        "fraud_vs_benign_gap": {
            "static": round(mean(fraud_static) - mean(benign_static), 4) if benign_static else None,
            "temporal": round(mean(fraud_temporal) - mean(benign_temporal), 4) if benign_temporal else None,
        },
        "zero_feature_baseline": {
            "static": round(float(p_s_zero.item()), 4),
            "temporal": round(float(p_t_zero.item()), 4),
        },
        "interpretation": (
            "Wide fraud-vs-benign gap supports the claim that the trained encoders discriminate "
            "fraud from benign cleanly. Zero-feature baseline that collapses toward the midpoint "
            "indicates the encoders use their inputs. Within-fraud std characterizes per-case "
            "discrimination: small std means the encoder is saturated on fraud and provides a "
            "binary-like signal to the LLM."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
