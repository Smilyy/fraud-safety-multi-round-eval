import argparse
import csv
import json
from pathlib import Path


def load_csv(path: Path):
    return list(csv.DictReader(path.open("r", encoding="utf-8")))


def short_trace(trace_raw: str, max_rounds: int = 4):
    try:
        trace = json.loads(trace_raw)
    except json.JSONDecodeError:
        return []
    compact = []
    for item in trace[:max_rounds]:
        compact.append(
            {
                "round": item.get("round"),
                "attacker_message": str(item.get("attacker_message", ""))[:240],
                "defender_action": item.get("defender_action"),
                "graph_context": item.get("graph_context"),
            }
        )
    return compact


def pick_case(row: dict):
    return {
        "id": int(row["id"]),
        "category": row["category"],
        "setting": row["setting"],
        "final_action": row["final_action"],
        "safe_reject_round": row.get("safe_reject_round"),
        "unsafe_compliance": int(float(row.get("unsafe_compliance", 0))),
        "latency_penalty_round": float(row["latency_penalty_round"]),
        "grounding_score": float(row.get("grounding_score", 0.0)),
        "justification": row.get("justification", ""),
        "trace": short_trace(row.get("trace", "")),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fraud-predictions", nargs="+", type=Path, required=True)
    parser.add_argument("--benign-predictions", nargs="*", type=Path, default=[])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    fraud_rows = []
    for path in args.fraud_predictions:
        run_name = path.parent.name
        for row in load_csv(path):
            row["run"] = run_name
            fraud_rows.append(row)

    benign_rows = []
    for path in args.benign_predictions:
        run_name = path.parent.name
        for row in load_csv(path):
            row["run"] = run_name
            benign_rows.append(row)

    fraud_by_run_id_setting = {(row["run"], row["id"], row["setting"]): row for row in fraud_rows}
    benign_by_run_id_setting = {(row["run"], row["id"], row["setting"]): row for row in benign_rows}
    run_ids = sorted({(row["run"], row["id"]) for row in fraud_rows}, key=lambda item: (item[0], int(item[1])))

    buckets = {
        "adaptive_unsafe_after_replay_safe": [],
        "temporal_graph_helps_vs_text": [],
        "late_safe_refusal": [],
        "graph_overwarning_on_benign": [],
    }

    for run_name, case_id in run_ids:
        replay_text = fraud_by_run_id_setting.get((run_name, case_id, "replay_text_only"))
        adaptive_text = fraud_by_run_id_setting.get((run_name, case_id, "adaptive_text_only"))
        adaptive_temporal = fraud_by_run_id_setting.get((run_name, case_id, "adaptive_temporal_graph"))

        if replay_text and adaptive_text:
            replay_safe = int(float(replay_text["safe_reject"])) == 1
            adaptive_unsafe = int(float(adaptive_text["unsafe_compliance"])) == 1
            if replay_safe and adaptive_unsafe:
                buckets["adaptive_unsafe_after_replay_safe"].append(
                    {
                        "run": run_name,
                        "replay_text_only": pick_case(replay_text),
                        "adaptive_text_only": pick_case(adaptive_text),
                    }
                )

        if adaptive_text and adaptive_temporal:
            text_latency = float(adaptive_text["latency_penalty_round"])
            temporal_latency = float(adaptive_temporal["latency_penalty_round"])
            temporal_safe = int(float(adaptive_temporal["safe_reject"])) == 1
            text_safe = int(float(adaptive_text["safe_reject"])) == 1
            if temporal_safe and (not text_safe or temporal_latency < text_latency):
                buckets["temporal_graph_helps_vs_text"].append(
                    {
                        "run": run_name,
                        "adaptive_text_only": pick_case(adaptive_text),
                        "adaptive_temporal_graph": pick_case(adaptive_temporal),
                    }
                )

        for setting in ("replay_text_only", "replay_static_graph", "replay_temporal_graph", "adaptive_text_only", "adaptive_static_graph", "adaptive_temporal_graph"):
            row = fraud_by_run_id_setting.get((run_name, case_id, setting))
            if not row:
                continue
            safe_reject = int(float(row["safe_reject"])) == 1
            safe_round_raw = row.get("safe_reject_round")
            safe_round = int(float(safe_round_raw)) if safe_round_raw not in ("", None) else None
            if safe_reject and safe_round is not None and safe_round >= 2:
                buckets["late_safe_refusal"].append({"run": run_name, **pick_case(row)})

        benign_text = benign_by_run_id_setting.get((run_name, case_id, "benign_replay_text_only"))
        benign_temporal = benign_by_run_id_setting.get((run_name, case_id, "benign_replay_temporal_graph"))
        if benign_text and benign_temporal:
            text_over = int(float(benign_text["over_refusal"])) == 1
            temporal_over = int(float(benign_temporal["over_refusal"])) == 1
            if temporal_over and not text_over:
                buckets["graph_overwarning_on_benign"].append(
                    {
                        "run": run_name,
                        "benign_replay_text_only": {
                            "id": int(benign_text["id"]),
                            "category": benign_text["category"],
                            "final_action": benign_text["final_action"],
                            "trace": short_trace(benign_text.get("trace", "")),
                        },
                        "benign_replay_temporal_graph": {
                            "id": int(benign_temporal["id"]),
                            "category": benign_temporal["category"],
                            "final_action": benign_temporal["final_action"],
                            "trace": short_trace(benign_temporal.get("trace", "")),
                        },
                    }
                )

    trimmed = {
        bucket: {
            "count": len(items),
            "examples": items[:20],
        }
        for bucket, items in buckets.items()
    }
    args.out.write_text(json.dumps(trimmed, indent=2))
    print(json.dumps(trimmed, indent=2))


if __name__ == "__main__":
    main()
