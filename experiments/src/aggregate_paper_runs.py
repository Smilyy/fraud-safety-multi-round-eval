import argparse
import json
from pathlib import Path
from statistics import mean, stdev


def load_json(path: Path):
    return json.loads(path.read_text())


def aggregate_metric(values):
    if not values:
        return None
    if len(values) == 1:
        return {"mean": round(float(values[0]), 4), "std": 0.0, "n": 1}
    return {"mean": round(float(mean(values)), 4), "std": round(float(stdev(values)), 4), "n": len(values)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--summary-file", type=str, default="fraud_r1_joint_summary.json")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    summaries = [load_json(run_dir / args.summary_file) for run_dir in args.run_dirs]
    settings = sorted({setting for summary in summaries for setting in summary.keys()})
    aggregated = {}
    for setting in settings:
        metric_names = sorted({metric for summary in summaries if setting in summary for metric in summary[setting].keys() if isinstance(summary[setting][metric], (int, float))})
        aggregated[setting] = {}
        for metric_name in metric_names:
            values = [summary[setting][metric_name] for summary in summaries if setting in summary and isinstance(summary[setting].get(metric_name), (int, float))]
            aggregated[setting][metric_name] = aggregate_metric(values)

    args.out.write_text(json.dumps(aggregated, indent=2))
    print(json.dumps(aggregated, indent=2))


if __name__ == "__main__":
    main()
