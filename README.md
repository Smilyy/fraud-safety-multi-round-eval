# Rethinking Fraud Safety Evaluation
### Multi-Round Attacks Reveal Safety–Utility Tradeoffs in Graph-Context LLM Defenders

Companion code and data for the paper of the same title.

- **Paper PDF**: [`paper/main.pdf`](paper/main.pdf)
- **arXiv**: https://arxiv.org/abs/2605.20759
- **Authors**: Laura Jiang, Reza Ryan, Qian Li, Nasim Ferdosian (Curtin University)

---

## Abstract

Single-turn fraud safety evaluation under-measures real defender behavior. Under multi-round **replay** and **adaptive** attack regimes, graph-context defenders refuse fraud *earlier* than text-only defenders (replay temporal AUSR `0.978` vs text-only `0.847`, p = 0.0004 on the frozen suite), but **over-refuse benign traffic** at substantially higher rates (replay benign ORR rises from `0.36` text-only to `0.84`–`0.89` with graph context). Direct probing of the trained graph encoder plus paired shuffle-risk ablations localise this cost to **how the defender LLM consumes structured context**, not to graph-encoder quality.

---

## What's in this repo

```
paper/                            LaTeX source, final PDF, build assets, design notes
experiments/
  src/                            All training, evaluation, and analysis scripts
  data/Fraud-R1/                  Vendored copy of the Fraud-R1 conversation benchmark
  data/sting9/                    Planning notes for a third (unreleased) dataset
  results/
    paper_suite_frozen_final_256x20/    Frozen suite (3x3 design x 2 backbones x 2 seeds)
    paper_scale_consistency_512x40_seed7/   Larger-slice scale check
    paper_prompt_balance_256x20_seed7/      Minimal prompt-calibration check
    paper_shared_split_seed7/                Frozen test/train case-id manifest
    paper_graph_cache_256/                   Cached graph encoders (frozen)
REPRODUCE.md                      Per-table / per-figure reproduction commands
PR3_HANDOVER.md                   Curtin COMP6016 Progress Report 3 handover
environment.yml                   Conda environment (recommended)
requirements.txt                  Pip fallback (no conda)
.gitignore
```

The Elliptic2 raw data (107 GB) and exploratory / smoke / pilot result directories are excluded — see `experiments/src/download_data.sh` to fetch Elliptic2.

---

## Quickstart

**Step 1 — set up the environment (conda recommended):**

```bash
git clone https://github.com/Smilyy/fraud-safety-multi-round-eval.git
cd fraud-safety-multi-round-eval
conda env create -f environment.yml
conda activate fraud-eval
```

No conda? Use pip instead: `pip install -r requirements.txt`

**Step 2 — five-minute smoke run on CPU (no GPU required):**

```bash
python experiments/src/run_fraud_r1_joint_graph.py \
  --base-data experiments/data/Fraud-R1/dataset/FP-base-full/FP-base-English.json \
  --levelup-data experiments/data/Fraud-R1/dataset/FP-levelup-full/FP-levelup-English.json \
  --language English \
  --model google/flan-t5-small \
  --attacker-model google/flan-t5-small \
  --graph-device cpu --llm-device cpu \
  --max-input-tokens 512 --max-threads 2 \
  --temporal-backbone gru \
  --train-limit 4 --test-limit 1 \
  --attacker-modes replay \
  --outdir experiments/results/smoke_check
```

If this completes and writes `experiments/results/smoke_check/fraud_r1_joint_summary.json`, the pipeline is wired up correctly.

For the full reproduction of paper tables and figures, see [`REPRODUCE.md`](REPRODUCE.md).

---

## Video walkthrough script

> This section is a recording guide — open this page in a browser tab while you record.

**Target length: ~5 minutes.** No specific length is required; the goal is that someone who has never seen the project can follow along and reproduce a result.

---

### Section 1 — Intro (0:00 – 0:30)

Open `paper/main.pdf` on screen at the title page.

> "Hi, I'm Laura Jiang, student ID 22742957, Group 24. This is the project walkthrough for COMP6016 Progress Report 3. My project is a research paper titled *Rethinking Fraud Safety Evaluation: Multi-Round Attacks Reveal Safety–Utility Tradeoffs in Graph-Context LLM Defenders*, supervised by Reza Ryan at Curtin University. The paper is published on arXiv. In this video I'll walk through the repository, set up the environment, and run the pipeline."

---

### Section 2 — The GitHub repo (0:30 – 1:15)

Open **this page** in a browser. Scroll slowly through the README.

> "The repository is public on GitHub. The Abstract shows the main finding: under multi-round attack, graph-context defenders refuse fraud earlier — AUSR rises from 0.847 to 0.978 under replay — but they also over-refuse benign traffic, with ORR rising from 0.36 to 0.84–0.89. The repo contains all the experiment source code in `experiments/src/`, the Fraud-R1 dataset vendored under `experiments/data/`, the frozen result artifacts, and the paper PDF and LaTeX. `REPRODUCE.md` maps every paper table to the exact command that produced it."

---

### Section 3 — Clone and install (1:15 – 2:15)

Switch to a terminal. Type these commands:

```bash
git clone https://github.com/Smilyy/fraud-safety-multi-round-eval.git
cd fraud-safety-multi-round-eval
conda env create -f environment.yml
conda activate fraud-eval
```

> "Setup uses conda, which handles the PyTorch and CUDA pairing automatically. The `environment.yml` file creates a self-contained environment called `fraud-eval` with Python 3.11, PyTorch, torch-geometric, transformers, and all the data science dependencies. For users without conda there is also a `requirements.txt` pip fallback."

*(Edit out the long conda install output if it takes a while.)*

---

### Section 4 — Run the smoke check (2:15 – 4:15)

In the terminal, run:

```bash
python experiments/src/run_fraud_r1_joint_graph.py \
  --base-data experiments/data/Fraud-R1/dataset/FP-base-full/FP-base-English.json \
  --levelup-data experiments/data/Fraud-R1/dataset/FP-levelup-full/FP-levelup-English.json \
  --language English \
  --model google/flan-t5-small \
  --attacker-model google/flan-t5-small \
  --graph-device cpu --llm-device cpu \
  --temporal-backbone gru \
  --train-limit 4 --test-limit 1 \
  --attacker-modes replay \
  --outdir experiments/results/smoke_check
```

> "This runs the full pipeline end-to-end on a tiny four-case slice so it finishes in a couple of minutes on CPU. It trains both a static and a temporal graph encoder, then evaluates the defender under three context conditions — text-only, static graph, and temporal graph — against a replay attacker. The full frozen suite uses the same script with 256 training cases, 20 test cases, two backbones, and two seeds."

When it finishes:

```bash
cat experiments/results/smoke_check/fraud_r1_joint_summary.json | python -m json.tool | head -40
```

> "The output is a JSON summary and a per-case predictions CSV. The summary reports the main metrics for each cell: ESR at each round, AUSR, unsafe-compliance rate, and average latency."

---

### Section 5 — Show the frozen paper artifacts (4:15 – 5:00)

```bash
ls experiments/results/paper_suite_frozen_final_256x20/artifacts/
cat experiments/results/paper_suite_frozen_final_256x20/artifacts/significance_tests.json \
  | python -m json.tool | head -30
```

> "These are the actual artifacts the paper cites. `fraud_r1_joint_aggregate.json` and `fraud_r1_benign_aggregate.json` are the source of truth for the main tables. `significance_tests.json` contains the paired permutation tests — you can see the p-value of 0.0004 for the replay temporal vs text-only AUSR contrast. Every number in the paper traces back to one of these JSON files. Full reproduction instructions are in `REPRODUCE.md`. Thank you."

---

## Citation

```bibtex
@article{jiang2026rethinking,
  title  = {Rethinking Fraud Safety Evaluation: Multi-Round Attacks Reveal
            Safety--Utility Tradeoffs in Graph-Context LLM Defenders},
  author = {Jiang, Laura and Ryan, Reza and Li, Qian and Ferdosian, Nasim},
  year   = {2026},
  note   = {arXiv preprint}
}
```

---

## License

Code released under the MIT License (see `LICENSE`).
Fraud-R1 dataset retains its original license from
[`kaustpradalab/Fraud-R1`](https://github.com/kaustpradalab/Fraud-R1).

---

## Acknowledgments

This work was carried out at Curtin University. The author thanks Reza Ryan for
supervision and Qian Li and Nasim Ferdosian for advisory review of the paper draft.
