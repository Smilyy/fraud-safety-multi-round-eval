# Reproduction Guide

This guide maps each paper table / figure / claim to the exact command that produces it. All numbers in the paper come from artifacts in `experiments/results/paper_suite_frozen_final_256x20/artifacts/`.

---

## 0. Prerequisites

### Software

Conda is the recommended way to install — it handles the PyTorch + CUDA pairing automatically:

```bash
conda env create -f environment.yml
conda activate fraud-eval
```

No conda? Pip fallback (you will need to install a CUDA-compatible PyTorch wheel separately first):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Developed with Python 3.11, PyTorch 2.11 dev (CUDA 12.8), torch_geometric 2.7, transformers 5.3.
Any torch ≥ 2.2 with CUDA 12.x should reproduce the results.

### Hardware

| Component                         | Minimum    | Recommended            |
|-----------------------------------|------------|------------------------|
| RAM                               | 16 GB      | 32 GB                  |
| GPU                               | None (CPU) | 1× 12 GB VRAM (for LLM) |
| Disk (code + Fraud-R1 + artifacts)| 1 GB       | 1 GB                   |
| Disk (Elliptic2 raw + extracted)  | —          | 110 GB (optional)      |

The graph encoder always runs on CPU. The defender LLM (Qwen 2.5-1.5B-Instruct) runs comfortably on a 12 GB GPU; on CPU it works but is slow.

### Optional: OpenAI API

Two cells of the frozen suite use `openai:gpt-5.4-mini` as the defender. To rerun those, set:

```bash
export OPENAI_API_KEY=sk-...
```

The Qwen-only cells reproduce the central claims without an API key.

### Data

Fraud-R1 is vendored at `experiments/data/Fraud-R1/`.

Elliptic2 raw data (107 GB) is not in the repo. To fetch:

```bash
bash experiments/src/download_data.sh
```

This requires a Kaggle account and `kaggle` CLI configured.

---

## 1. Reproduce the frozen suite from scratch

This is the single command that produced every number in Section "Results" of the paper, modulo Elliptic2-side tables.

```bash
python experiments/src/run_safe_paper_suite.py \
  --base-data experiments/data/Fraud-R1/dataset/FP-base-full/FP-base-English.json \
  --levelup-data experiments/data/Fraud-R1/dataset/FP-levelup-full/FP-levelup-English.json \
  --language English \
  --split-manifest experiments/results/paper_shared_split_seed7/split_manifest.json \
  --train-limit 256 --test-limit 20 \
  --graph-epochs 3 --temporal-backbone gru \
  --graph-device cpu --llm-device cuda \
  --attacker-device auto \
  --max-input-tokens 768 --max-threads 2 --thread-limit 2 \
  --attacker-model Qwen/Qwen2.5-1.5B-Instruct \
  --attacker-modes replay adaptive \
  --models Qwen/Qwen2.5-1.5B-Instruct openai:gpt-5.4-mini \
  --seeds 7 11 \
  --cache-root experiments/results/paper_graph_cache_256 \
  --out-root experiments/results/paper_suite_frozen_final_256x20 \
  --log-dir experiments/logs/paper_suite_frozen_final_256x20 \
  --progress-every 10 \
  --with-benign-controls \
  --skip-existing
```

Walltime: ~6–10 hours on a single 12 GB GPU machine (depends on OpenAI rate limits for the gpt-5.4-mini cells).

Then regenerate the analysis artifacts:

```bash
python experiments/src/build_paper_artifacts.py \
  --run-root experiments/results/paper_suite_frozen_final_256x20 \
  --out-root experiments/results/paper_suite_frozen_final_256x20/artifacts
```

This produces `fraud_r1_joint_aggregate.json`, `fraud_r1_benign_aggregate.json`, `significance_tests.json`, `attacker_gap.json`, `failure_cases.json`, and `mcnemar_static_vs_temporal.json`.

---

## 2. Paper-table → command mapping

| Paper element | Numbers come from | How to regenerate |
|---|---|---|
| Table: main fraud-side results (`tab:fraud-main`)    | `paper_suite_frozen_final_256x20/artifacts/fraud_r1_joint_aggregate.json` | `build_paper_artifacts.py` |
| Table: benign over-refusal (`tab:benign-main`)       | `.../artifacts/fraud_r1_benign_aggregate.json`                            | `build_paper_artifacts.py` |
| Significance tests (replay & adaptive)               | `.../artifacts/significance_tests.json`                                   | `significance_tests.py` (invoked by `build_paper_artifacts.py`) |
| McNemar static vs temporal                           | `.../artifacts/mcnemar_static_vs_temporal.json`                           | `mcnemar_static_vs_temporal.py` |
| Attacker gap analysis (replay vs adaptive)           | `.../artifacts/attacker_gap.json`                                         | `analyze_attacker_gap.py` |
| GNN probe (encoder separates fraud vs benign)        | `.../artifacts/gnn_probe.json`                                            | `gnn_probe.py` (see below) |
| Failure-case catalogue                               | `.../artifacts/failure_cases.json`                                        | `extract_failure_cases.py` |
| Scale-consistency check                              | `paper_scale_consistency_512x40_seed7/...`                                | rerun `run_safe_paper_suite.py` with `--train-limit 512 --test-limit 40` |
| Prompt-calibration check (benign ORR mitigation)     | `paper_prompt_balance_256x20_seed7/...`                                   | rerun `run_safe_paper_suite.py` with the balanced prompt template |
| Figure: `tradeoff_scatter.pdf`                       | `paper/tradeoff_scatter.pdf`                                              | `paper/build_paper_assets.py` |
| Figure: `fraud_esr_curves.pdf`                       | `paper/fraud_esr_curves.pdf`                                              | `paper/build_paper_assets.py` |

### Stand-alone analyses

GNN encoder probe (no LLM needed):

```bash
python experiments/src/gnn_probe.py \
  --cache-dir experiments/results/paper_graph_cache_256/seed7_gru_e3 \
  --out experiments/results/paper_suite_frozen_final_256x20/artifacts/gnn_probe.json
```

Shuffle-risk ablation (Q2):

```bash
python experiments/src/run_fraud_r1_joint_graph.py \
  --base-data experiments/data/Fraud-R1/dataset/FP-base-full/FP-base-English.json \
  --levelup-data experiments/data/Fraud-R1/dataset/FP-levelup-full/FP-levelup-English.json \
  --shuffle-risk-scores \
  --split-manifest experiments/results/paper_shared_split_seed7/split_manifest.json \
  --graph-cache-dir experiments/results/paper_graph_cache_256/seed7_gru_e3 \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --attacker-model Qwen/Qwen2.5-1.5B-Instruct \
  --train-limit 256 --test-limit 20 \
  --attacker-modes replay adaptive \
  --outdir experiments/results/q2_shuffle_risk_seed7
```

---

## 3. Elliptic2 graph-backbone validation

Only needed for the Elliptic2 table in the paper. Skips cleanly if the data isn't downloaded.

```bash
python experiments/src/run_elliptic2_graph.py \
  --data-dir experiments/data/elliptic2/extracted \
  --outdir experiments/results/elliptic2_graph_seed7
```

---

## 4. Determinism

The frozen suite is deterministic given the split manifest and graph cache:

- `experiments/results/paper_shared_split_seed7/split_manifest.json` fixes train/test case ids.
- `experiments/results/paper_graph_cache_256/seed7_gru_e3/` and `seed11_gru_e3/` hold the trained graph-encoder weights used by the suite.

LLM sampling for the local Qwen defender is greedy (temperature 0). The `openai:gpt-5.4-mini` cells use the model's default decoding and so are not bit-exact reproducible; the directional claims and significance tests are robust across reruns.

---

## 5. Known issues

- The 84 historical result directories from exploratory pilots are excluded via `.gitignore`. None of them are referenced by paper claims.
- OpenAI rate limits can stall the `openai:gpt-5.4-mini` cells; rerun with `--skip-existing` to resume.
- Elliptic2 extraction can require ~100 GB of free disk; check disk space before running `download_data.sh`.
