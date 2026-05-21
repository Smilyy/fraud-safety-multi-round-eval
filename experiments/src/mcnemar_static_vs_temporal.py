"""Paired McNemar test for static_graph vs temporal_graph on per-case safe_reject.

The frozen suite already runs paired permutation tests via significance_tests.py.
This script adds a McNemar exact test specifically on the binary safe_reject
outcome at each round (ESR@k) and on AUSR>=threshold, which is the standard
statistical test for paired binary outcomes and the one a reviewer expects to
see when comparing two treatments on the same subjects.

Output: artifacts/mcnemar_static_vs_temporal.json
"""
import argparse
import csv
import json
from math import factorial
from pathlib import Path


def mcnemar_exact_p(b: int, c: int):
    """Two-sided exact binomial p-value for McNemar's test.
    b = #cases where a is safe-reject and b is not
    c = #cases where b is safe-reject and a is not
    Concordant cases (both reject or both don't) are excluded."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    cumulative = sum(factorial(n) // (factorial(i) * factorial(n - i)) for i in range(k + 1))
    p_one_sided = cumulative / (2 ** n)
    return min(2 * p_one_sided, 1.0)


def load_predictions(paths):
    by_key = {}
    for path in paths:
        run_name = path.parent.name
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                key = (run_name, row["id"], row["attacker_mode"])
                by_key.setdefault(key, {})[row["context_mode"]] = row
    return by_key


def metric_for_row(row, metric):
    safe_reject = int(float(row["safe_reject"]))
    safe_round_raw = row.get("safe_reject_round")
    safe_round = int(float(safe_round_raw)) if safe_round_raw not in ("", None) else None
    if metric.startswith("ESR@"):
        k = int(metric.split("@")[1])
        return 1 if (safe_reject and safe_round is not None and safe_round <= k) else 0
    if metric == "any_reject":
        return safe_reject
    raise ValueError(metric)


def run_mcnemar(by_key, attacker_mode, metric):
    paired = []
    for key, ctx_rows in by_key.items():
        if key[2] != attacker_mode:
            continue
        if "static_graph" not in ctx_rows or "temporal_graph" not in ctx_rows:
            continue
        a = metric_for_row(ctx_rows["static_graph"], metric)
        b = metric_for_row(ctx_rows["temporal_graph"], metric)
        paired.append((a, b))

    if not paired:
        return None
    b_count = sum(1 for a, b in paired if a == 1 and b == 0)
    c_count = sum(1 for a, b in paired if a == 0 and b == 1)
    both_reject = sum(1 for a, b in paired if a == 1 and b == 1)
    neither = sum(1 for a, b in paired if a == 0 and b == 0)
    p_value = mcnemar_exact_p(b_count, c_count)
    return {
        "n_pairs": len(paired),
        "static_only_reject": b_count,
        "temporal_only_reject": c_count,
        "both_reject": both_reject,
        "neither_reject": neither,
        "discordant_pairs": b_count + c_count,
        "static_better_share": round(b_count / max(b_count + c_count, 1), 4),
        "p_value_two_sided": round(p_value, 6),
        "interpretation": (
            "p > 0.05 with non-trivial concordance => static_graph and temporal_graph are not "
            "statistically distinguishable on this binary outcome at this N; interpret directional "
            "differences as suggestive only."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--attacker-modes", nargs="+", default=["replay", "adaptive"])
    parser.add_argument("--metrics", nargs="+", default=["ESR@1", "ESR@2", "ESR@3", "ESR@4", "any_reject"])
    args = parser.parse_args()

    by_key = load_predictions(args.predictions)
    output = {"comparisons": {}}
    for am in args.attacker_modes:
        for metric in args.metrics:
            label = f"{am}__{metric}"
            result = run_mcnemar(by_key, am, metric)
            if result is not None:
                output["comparisons"][label] = result

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
