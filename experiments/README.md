# Experiments

This folder contains the initial experiment scaffold and a runnable pilot aligned with the paper draft.

## What Is Here

- `data/Fraud-R1/`
  Public clone of the Fraud-R1 benchmark repository.
- `src/run_fraud_r1_pilot.py`
  Original pilot experiment on the English split of Fraud-R1.
- `src/run_fraud_r1_joint_graph.py`
  Current paper-aligned experiment with unified prompting, learned graph context, replay and adaptive attackers, and early-refusal metrics.
- `src/run_elliptic2_graph.py`
  Real graph-only validation experiment on Elliptic2.
- `results/fraud_r1_pilot_*/`
  Saved predictions and summaries from completed runs.

## Datasets

Primary dataset used now:

- `Fraud-R1`
  Source: `https://github.com/kaustpradalab/Fraud-R1`

Dataset status for the broader paper design:

- `Fraud-R1`: downloaded and used in the pilot.
- `Elliptic2`: Kaggle download started and is in progress.
- `Sting9`: official website and source repos are public, but the public dataset dump is not yet populated and the API currently requires authorization.

## Current Recommended Design

Use `run_fraud_r1_joint_graph.py` for the paper path.

It evaluates:

- `single_turn_text_only`
- `single_turn_static_graph`
- `single_turn_temporal_graph`
- `replay_text_only`
- `replay_static_graph`
- `replay_temporal_graph`
- `adaptive_text_only`
- `adaptive_static_graph`
- `adaptive_temporal_graph`

Key changes relative to the older pilot:

- one defender prompt template is shared across single-turn and multi-round settings;
- the graph branch uses learned graph encoders instead of a rule gate;
- the adaptive attacker conditions on the previous defender action;
- the main metrics are `ESR@k`, `AUSR`, `unsafe_compliance_rate`, and `avg_latency_penalized`.
- defender outputs now include structured evidence tags, enabling a prompt-grounding score;
- optional benign controls now measure over-refusal under `single_turn` and `replay` dialogue.

## Pilot Design

The current runnable study is a pilot, not the full final paper experiment.

The old pilot evaluates five settings on `Fraud-R1`:

- `single_turn_text`
- `single_turn_static`
- `multi_round_text`
- `multi_round_static`
- `multi_round_temporal`

Pilot notes:

- The text-only defender uses `google/flan-t5-small`.
- The graph-aware settings use lightweight graph summaries derived from `role_bg`, category labels, keyword signals, and multi-round accumulation.
- The old pilot uses a graph-risk gate before fallback to the LLM. Keep it only as an archival baseline, not as the main paper experiment.
- The current runner uses a paired design:
  - `FP-base` for single-turn evaluation
  - `FP-levelup` for multi-round evaluation
  - pairing is done by the original case `id`

## Joint Graph+LLM Design

The current experiment path is closer to the intended paper setup:

- train a static graph encoder on round-level escalation targets;
- train a temporal graph encoder on ordered graph snapshots over rounds;
- serialize learned graph outputs into interpretable prompt fields;
- evaluate the same defender prompt under `single_turn`, `replay`, and `adaptive` attack modes.

Current implementation details:

- `single_turn_*` now uses the paired `FP-base` example for the same case id;
- `replay_*` and `adaptive_*` use the paired `FP-levelup` conversation;
- optional `--split-manifest` support keeps train/test case ids fixed across runs;
- the graph target is an escalation-risk score derived from current and future fraud-pressure signals rather than raw round index.

Relevant files:

- `src/fraud_r1_joint_graph.py`
- `src/run_fraud_r1_joint_graph.py`
- `src/fraud_pipeline_utils.py`

## Elliptic2 Graph Validation

`Elliptic2` is the supporting real-graph dataset for this repo.

Use it to validate the graph backbone, not as a conversational benchmark.

Relevant file:

- `src/run_elliptic2_graph.py`

## How To Run

```bash
python experiments/src/run_fraud_r1_pilot.py \
  --base-data experiments/data/Fraud-R1/dataset/FP-base-full/FP-base-English.json \
  --levelup-data experiments/data/Fraud-R1/dataset/FP-levelup-full/FP-levelup-English.json \
  --language English \
  --limit 200 \
  --seed 7 \
  --outdir experiments/results/fraud_r1_pilot_200_seed7
```

Current paper path:

```bash
python experiments/src/run_fraud_r1_joint_graph.py \
  --base-data experiments/data/Fraud-R1/dataset/FP-base-full/FP-base-English.json \
  --levelup-data experiments/data/Fraud-R1/dataset/FP-levelup-full/FP-levelup-English.json \
  --language English \
  --model google/flan-t5-small \
  --attacker-model google/flan-t5-small \
  --graph-device cpu \
  --llm-device cuda \
  --max-input-tokens 768 \
  --max-threads 2 \
  --split-manifest experiments/results/fraud_r1_joint_stage2_seed7/split_manifest.json \
  --outdir experiments/results/fraud_r1_joint_stage2_seed7
```

For a quick smoke run, add `--train-limit 12 --test-limit 2`.

Recommended low-risk smoke command on this laptop:

```bash
python experiments/src/run_fraud_r1_joint_graph.py \
  --base-data experiments/data/Fraud-R1/dataset/FP-base-full/FP-base-English.json \
  --levelup-data experiments/data/Fraud-R1/dataset/FP-levelup-full/FP-levelup-English.json \
  --language English \
  --model google/flan-t5-small \
  --attacker-model google/flan-t5-small \
  --graph-device cpu \
  --llm-device cuda \
  --max-input-tokens 512 \
  --max-threads 2 \
  --temporal-backbone gru \
  --train-limit 4 \
  --test-limit 1 \
  --attacker-modes replay \
  --outdir experiments/results/fraud_r1_joint_safe_smoke
```

Model id options:

- Local Hugging Face model ids still work as before, for example `google/flan-t5-small`.
- API-backed OpenAI models are supported with the prefix `openai:`, for example `openai:gpt-5.4-mini`.
- When using an `openai:` model id, set `OPENAI_API_KEY` in the environment first.

To include benign controls in the same run, add:

```bash
  --with-benign-controls
```

This writes:

- `fraud_r1_benign_predictions.csv`
- `fraud_r1_benign_summary.json`

For larger comparative runs, reuse the same trained graph backbone:

```bash
python -u experiments/src/run_fraud_r1_joint_graph.py \
  --base-data experiments/data/Fraud-R1/dataset/FP-base-full/FP-base-English.json \
  --levelup-data experiments/data/Fraud-R1/dataset/FP-levelup-full/FP-levelup-English.json \
  --language English \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --attacker-model Qwen/Qwen2.5-1.5B-Instruct \
  --graph-epochs 3 \
  --graph-device cpu \
  --graph-cache-dir experiments/results/paper_graph_cache/seed7_gru_e3 \
  --llm-device cuda \
  --temporal-backbone gru \
  --train-limit 48 \
  --test-limit 20 \
  --attacker-modes replay adaptive \
  --split-manifest experiments/results/paper_shared_split_seed7/split_manifest.json \
  --outdir experiments/results/fraud_r1_joint_qwen15_gru_pilot20
```

Then switch only the defender backbone while keeping the same cache directory and split:

```bash
python -u experiments/src/run_fraud_r1_joint_graph.py \
  --base-data experiments/data/Fraud-R1/dataset/FP-base-full/FP-base-English.json \
  --levelup-data experiments/data/Fraud-R1/dataset/FP-levelup-full/FP-levelup-English.json \
  --language English \
  --model openai:gpt-5.4-mini \
  --attacker-model Qwen/Qwen2.5-1.5B-Instruct \
  --graph-epochs 3 \
  --graph-device cpu \
  --graph-cache-dir experiments/results/paper_graph_cache/seed7_gru_e3 \
  --llm-device cuda \
  --temporal-backbone gru \
  --train-limit 48 \
  --test-limit 20 \
  --attacker-modes replay adaptive \
  --split-manifest experiments/results/paper_shared_split_seed7/split_manifest.json \
  --outdir experiments/results/fraud_r1_joint_gpt54mini_gru_pilot20
```

Paper-suite driver for multiple models and seeds:

```bash
python experiments/src/run_joint_paper_suite.py \
  --base-data experiments/data/Fraud-R1/dataset/FP-base-full/FP-base-English.json \
  --levelup-data experiments/data/Fraud-R1/dataset/FP-levelup-full/FP-levelup-English.json \
  --language English \
  --split-manifest experiments/results/paper_shared_split_seed7/split_manifest.json \
  --train-limit 48 \
  --test-limit 20 \
  --graph-epochs 3 \
  --temporal-backbone gru \
  --graph-device cpu \
  --llm-device cuda \
  --attacker-model Qwen/Qwen2.5-1.5B-Instruct \
  --models Qwen/Qwen2.5-1.5B-Instruct openai:gpt-5.4-mini \
  --seeds 7 \
  --cache-root experiments/results/paper_graph_cache \
  --out-root experiments/results/paper_suite
```

Safer low-output suite runner for this machine:

```bash
python experiments/src/run_safe_paper_suite.py \
  --base-data experiments/data/Fraud-R1/dataset/FP-base-full/FP-base-English.json \
  --levelup-data experiments/data/Fraud-R1/dataset/FP-levelup-full/FP-levelup-English.json \
  --language English \
  --split-manifest experiments/results/paper_shared_split_seed7/split_manifest.json \
  --train-limit 256 \
  --test-limit 20 \
  --graph-epochs 3 \
  --temporal-backbone gru \
  --graph-device cpu \
  --llm-device cuda \
  --attacker-device auto \
  --max-input-tokens 768 \
  --max-threads 2 \
  --thread-limit 2 \
  --attacker-model Qwen/Qwen2.5-1.5B-Instruct \
  --attacker-modes replay adaptive \
  --models Qwen/Qwen2.5-1.5B-Instruct openai:gpt-5.4-mini \
  --seeds 7 11 \
  --cache-root experiments/results/paper_graph_cache_256 \
  --out-root experiments/results/paper_suite_safe_256x20 \
  --log-dir experiments/logs/paper_suite_safe_256x20 \
  --progress-every 10 \
  --skip-existing
```

This runner is safer than the older suite script because it:

- runs one job at a time;
- writes stdout/stderr to per-run log files instead of flooding the terminal;
- waits for free RAM / GPU headroom before each run;
- reuses cached graph models;
- can skip already-completed runs.

To regenerate paper artifacts from a run directory:

```bash
python experiments/src/build_paper_artifacts.py \
  --run-root experiments/results/paper_suite_repaired_256x20 \
  --out-root experiments/results/paper_suite_repaired_256x20/artifacts
```

Elliptic2 graph validation:

```bash
python experiments/src/run_elliptic2_graph.py \
  --data-dir experiments/data/elliptic2/extracted \
  --outdir experiments/results/elliptic2_graph_seed7
```

## Existing Saved Results

The main completed archived pilot runs are:

- `results/fraud_r1_pilot_200_seed7/`
- `results/fraud_r1_paired_200_seed7/`
- `results/fraud_r1_paired_full_english_seed7/`

Headline results on 200 randomly sampled English examples:

- `single_turn_text`: DSR `0.18`
- `multi_round_text`: DSR `0.75`
- `single_turn_static`: DSR `0.92`
- `multi_round_static`: DSR `0.995`
- `multi_round_temporal`: DSR `1.00`

Full English paired run on all `1071` examples:

- `single_turn_text`: DSR `0.169`
- `multi_round_text`: DSR `0.8086`
- `single_turn_static`: DSR `0.9384`
- `multi_round_static`: DSR `0.9981`
- `multi_round_temporal`: DSR `0.9991`

Interpretation of the archived pilot:

- the text-only defender is very weak in single-turn evaluation;
- later replay rounds become easier to reject than round 1;
- the graph-risk gate makes the old graph variants look stronger than they should;
- `network friendship` remains the hardest category across graph-aware variants.

These pilot artifacts are useful for bootstrapping, but they are not sufficient for the paper's final claims because:

- the prompt is not controlled across all settings;
- the main metric is final-turn success rather than early safe refusal;
- the graph module is heuristic in the pilot path;
- only the English Fraud-R1 split has been run so far;
- `Sting9` is planned but not yet runnable from public data as currently exposed.

## Repaired Paper Runs

The repaired protocol results live under:

- `results/paper_suite_repaired_256x20/`

These runs differ from the older saved suite in four important ways:

- adaptive attacker outputs are cleaned instead of carrying prompt-echo artifacts;
- long Fraud-R1 prompts are compacted before inference to reduce boilerplate drift;
- rationale grounding is scored from structured evidence tags;
- benign controls are reported alongside fraud-defense metrics.

## Sting9 Status

Official sources checked:

- Website: `https://sting9.org/dataset`
- GitHub org: `https://github.com/sting9-research`
- Dataset repo: `https://github.com/sting9-research/dataset`

Current status:

- the website advertises a GitHub dump and API access;
- the public dataset repository currently contains no data files;
- the live API endpoint is implemented as an authenticated service rather than a tokenless public feed.

Implication:

- `Sting9` is appropriate for the paper's dataset plan;
- it is not yet usable here as an experiment dataset unless the maintainers expose the dump or provide API access.
