import argparse
import subprocess
from pathlib import Path


def run(cmd: list[str]):
    print("[paper-artifacts]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--seed-pattern", type=str, default="*_seed*")
    args = parser.parse_args()

    run_dirs = sorted(path for path in args.run_root.glob(args.seed_pattern) if path.is_dir())
    if not run_dirs:
        raise SystemExit(f"no run directories found under {args.run_root}")

    fraud_predictions = [run_dir / "fraud_r1_joint_predictions.csv" for run_dir in run_dirs if (run_dir / "fraud_r1_joint_predictions.csv").exists()]
    benign_predictions = [run_dir / "fraud_r1_benign_predictions.csv" for run_dir in run_dirs if (run_dir / "fraud_r1_benign_predictions.csv").exists()]
    summary_jsons = [run_dir / "fraud_r1_joint_summary.json" for run_dir in run_dirs if (run_dir / "fraud_r1_joint_summary.json").exists()]
    benign_summary_jsons = [run_dir / "fraud_r1_benign_summary.json" for run_dir in run_dirs if (run_dir / "fraud_r1_benign_summary.json").exists()]

    args.out_root.mkdir(parents=True, exist_ok=True)
    if len(summary_jsons) >= 2:
        run(
            [
                "python",
                "experiments/src/aggregate_paper_runs.py",
                "--run-dirs",
                *[str(run_dir) for run_dir in run_dirs if (run_dir / "fraud_r1_joint_summary.json").exists()],
                "--out",
                str(args.out_root / "fraud_r1_joint_aggregate.json"),
            ]
        )
    if len(benign_summary_jsons) >= 2:
        run(
            [
                "python",
                "experiments/src/aggregate_paper_runs.py",
                "--run-dirs",
                *[str(run_dir) for run_dir in run_dirs if (run_dir / "fraud_r1_benign_summary.json").exists()],
                "--summary-file",
                "fraud_r1_benign_summary.json",
                "--out",
                str(args.out_root / "fraud_r1_benign_aggregate.json"),
            ]
        )
    if fraud_predictions:
        run(
            [
                "python",
                "experiments/src/significance_tests.py",
                "--predictions",
                *[str(path) for path in fraud_predictions],
                "--out",
                str(args.out_root / "significance_tests.json"),
            ]
        )
        run(
            [
                "python",
                "experiments/src/analyze_attacker_gap.py",
                "--predictions",
                *[str(path) for path in fraud_predictions],
                "--out",
                str(args.out_root / "attacker_gap.json"),
            ]
        )
        failure_cmd = [
            "python",
            "experiments/src/extract_failure_cases.py",
            "--fraud-predictions",
            *[str(path) for path in fraud_predictions],
        ]
        if benign_predictions:
            failure_cmd.extend(["--benign-predictions", *[str(path) for path in benign_predictions]])
        failure_cmd.extend(["--out", str(args.out_root / "failure_cases.json")])
        run(failure_cmd)


if __name__ == "__main__":
    main()
