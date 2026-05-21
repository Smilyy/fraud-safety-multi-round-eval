import argparse
import json
import os
import shlex
import subprocess
import time
from pathlib import Path


EXPECTED_RUN_FILES = (
    "fraud_r1_joint_predictions.csv",
    "fraud_r1_joint_summary.json",
    "fraud_r1_joint_category_summary.json",
    "split_meta.json",
)


def read_available_mem_gb() -> float | None:
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    kb = float(line.split()[1])
                    return kb / (1024.0 * 1024.0)
    except OSError:
        return None
    return None


def read_gpu_mem_used_mib() -> int | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    values = []
    for raw in result.stdout.strip().splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            values.append(int(raw))
        except ValueError:
            continue
    return max(values) if values else None


def wait_for_resources(min_free_mem_gb: float, max_gpu_mem_used_mib: int, poll_seconds: int):
    while True:
        available_mem = read_available_mem_gb()
        gpu_used = read_gpu_mem_used_mib()
        mem_ok = available_mem is None or available_mem >= min_free_mem_gb
        gpu_ok = gpu_used is None or gpu_used <= max_gpu_mem_used_mib
        if mem_ok and gpu_ok:
            return
        parts = []
        if available_mem is not None:
            parts.append(f"mem_available={available_mem:.2f}GiB")
        if gpu_used is not None:
            parts.append(f"gpu_used={gpu_used}MiB")
        joined = ", ".join(parts) if parts else "resource probe unavailable"
        print(f"[safe-suite] waiting for resources ({joined})")
        time.sleep(poll_seconds)


def run_cmd(cmd: list[str], env: dict[str, str], log_path: Path, cwd: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n[safe-suite] starting: {' '.join(shlex.quote(part) for part in cmd)}\n")
        handle.flush()
        process = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        handle.write(f"[safe-suite] exit_code={process.returncode}\n")
        handle.flush()
    return process.returncode


def run_complete(outdir: Path, with_benign_controls: bool, only_benign_controls: bool = False) -> bool:
    required = []
    if not only_benign_controls:
        required.extend(EXPECTED_RUN_FILES)
    if with_benign_controls:
        required.extend(["fraud_r1_benign_predictions.csv", "fraud_r1_benign_summary.json"])
    return all((outdir / name).exists() for name in required)


def build_env(thread_limit: int, allocator_max_split_size_mb: int) -> dict[str, str]:
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(thread_limit)
    env["MKL_NUM_THREADS"] = str(thread_limit)
    env["OPENBLAS_NUM_THREADS"] = str(thread_limit)
    env["NUMEXPR_NUM_THREADS"] = str(thread_limit)
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["PYTHONUNBUFFERED"] = "1"
    env["TRANSFORMERS_VERBOSITY"] = "error"
    env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    alloc_conf = env.get("PYTORCH_CUDA_ALLOC_CONF", "").strip()
    extra = f"max_split_size_mb:{allocator_max_split_size_mb}"
    env["PYTORCH_CUDA_ALLOC_CONF"] = f"{alloc_conf},{extra}".strip(",")
    return env


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
    parser.add_argument("--only-benign-controls", action="store_true")
    parser.add_argument("--defender-prompt-mode", type=str, default="default", choices=["default", "balanced_benign"])
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--runner", type=Path, default=Path("experiments/src/run_fraud_r1_joint_graph.py"))
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--thread-limit", type=int, default=2)
    parser.add_argument("--allocator-max-split-size-mb", type=int, default=128)
    parser.add_argument("--min-free-mem-gb", type=float, default=3.0)
    parser.add_argument("--max-gpu-mem-used-mib", type=int, default=1500)
    parser.add_argument("--poll-seconds", type=int, default=20)
    parser.add_argument("--sleep-seconds", type=int, default=5)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--job-manifest", type=Path)
    args = parser.parse_args()

    if args.only_benign_controls:
        args.with_benign_controls = True

    env = build_env(args.thread_limit, args.allocator_max_split_size_mb)
    jobs = []

    for seed in args.seeds:
        cache_dir = args.cache_root / f"seed{seed}_{args.temporal_backbone}_e{args.graph_epochs}"
        for model in args.models:
            model_slug = model.replace("/", "_").replace(":", "_")
            outdir = args.out_root / f"{model_slug}_seed{seed}"
            log_path = args.log_dir / f"{model_slug}_seed{seed}.log"
            cmd = [
                "python",
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
                str(args.progress_every),
                "--defender-prompt-mode",
                args.defender_prompt_mode,
                "--attacker-modes",
                *args.attacker_modes,
            ]
            if args.with_benign_controls:
                cmd.append("--with-benign-controls")
            if args.only_benign_controls:
                cmd.append("--only-benign-controls")
            if args.train_limit is not None:
                cmd.extend(["--train-limit", str(args.train_limit)])
            if args.test_limit is not None:
                cmd.extend(["--test-limit", str(args.test_limit)])
            jobs.append(
                {
                    "seed": seed,
                    "model": model,
                    "outdir": str(outdir),
                    "log_path": str(log_path),
                    "cmd": cmd,
                }
            )

    if args.job_manifest:
        args.job_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.job_manifest.write_text(json.dumps({"jobs": jobs}, indent=2))

    cwd = Path.cwd()
    failures = []
    for job in jobs:
        outdir = Path(job["outdir"])
        log_path = Path(job["log_path"])
        model = job["model"]
        seed = job["seed"]
        if args.skip_existing and run_complete(outdir, args.with_benign_controls, args.only_benign_controls):
            print(f"[safe-suite] skip completed model={model} seed={seed}")
            continue

        wait_for_resources(
            min_free_mem_gb=args.min_free_mem_gb,
            max_gpu_mem_used_mib=args.max_gpu_mem_used_mib,
            poll_seconds=args.poll_seconds,
        )
        print(f"[safe-suite] start model={model} seed={seed} log={log_path}")
        exit_code = run_cmd(job["cmd"], env=env, log_path=log_path, cwd=cwd)
        if exit_code != 0:
            failures.append({"model": model, "seed": seed, "exit_code": exit_code, "log_path": str(log_path)})
            print(f"[safe-suite] failed model={model} seed={seed} exit_code={exit_code}")
            if args.fail_fast:
                break
        else:
            print(f"[safe-suite] completed model={model} seed={seed}")
        time.sleep(args.sleep_seconds)

    if failures:
        raise SystemExit(json.dumps({"failures": failures}, indent=2))


if __name__ == "__main__":
    main()
