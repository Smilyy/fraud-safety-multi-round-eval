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
requirements.txt                  Python dependencies
.gitignore
```

The Elliptic2 raw data (107 GB) and exploratory / smoke / pilot result directories are excluded — see `experiments/src/download_data.sh` to fetch Elliptic2.

---

## Quickstart

Five-minute smoke run on CPU (no GPU required):

```bash
pip install -r requirements.txt

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
