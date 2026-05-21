import argparse
import shlex
import subprocess
from pathlib import Path


def run_cmd(cmd: list[str]):
    print("[suite] running:", " ".join(shlex.quote(part) for part in cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-data", type=Path, required=True)
    parser.add_argument("--levelup-data", type=Path, required=True)
    parser.add_argument("--language", type=str, default="English")
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--test-limit", type=int)
    parser.add_argument("--graph-epochs", type=int, default=3)
    parser.add_argument("--temporal-backbone", type=str, default="gru", choices=["gru", "tgn"])
    parser.add_argument("--graph-device", type=str, default="cpu")
    parser.add_argument("--llm-device", type=str, default="cuda")
    parser.add_argument("--attacker-device", type=str, default="auto")
    parser.add_argument("--max-input-tokens", type=int, default=768)
    parser.add_argument("--max-threads", type=int, default=2)
    parser.add_argument("--attacker-model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--attacker-modes", nargs="+", default=["replay", "adaptive"])
    parser.add_argument("--with-benign-controls", action="store_true")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--runner", type=Path, default=Path("experiments/src/run_fraud_r1_joint_graph.py"))
    args = parser.parse_args()

    for seed in args.seeds:
        cache_dir = args.cache_root / f"seed{seed}_{args.temporal_backbone}_e{args.graph_epochs}"
        for model in args.models:
            model_slug = model.replace("/", "_").replace(":", "_")
            outdir = args.out_root / f"{model_slug}_seed{seed}"
            cmd = [
                "python",
                "-u",
                str(args.runner),
                "--base-data",
                str(args.base_data),
                "--levelup-data",
                str(args.levelup_data),
                "--language",
                args.language,
                "--model",
                model,
                "--attacker-model",
                args.attacker_model,
                "--graph-epochs",
                str(args.graph_epochs),
                "--graph-device",
                args.graph_device,
                "--graph-cache-dir",
                str(cache_dir),
                "--llm-device",
                args.llm_device,
                "--attacker-device",
                args.attacker_device,
                "--max-input-tokens",
                str(args.max_input_tokens),
                "--max-threads",
                str(args.max_threads),
                "--temporal-backbone",
                args.temporal_backbone,
                "--seed",
                str(seed),
                "--split-manifest",
                str(args.split_manifest),
                "--outdir",
                str(outdir),
                "--progress-every",
                "1",
                "--attacker-modes",
                *args.attacker_modes,
            ]
            if args.with_benign_controls:
                cmd.append("--with-benign-controls")
            if args.train_limit is not None:
                cmd.extend(["--train-limit", str(args.train_limit)])
            if args.test_limit is not None:
                cmd.extend(["--test-limit", str(args.test_limit)])
            run_cmd(cmd)


if __name__ == "__main__":
    main()
