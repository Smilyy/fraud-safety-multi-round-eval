# Elliptic2 Dataset

This directory is a placeholder. The raw Elliptic2 data is not included in the repository because it is ~107 GB.

## Role in the paper

Elliptic2 is used for **graph-backbone validation only** (Section: Elliptic2 graph validation). It is not a conversational benchmark and is not used for any of the main fraud-defence tables. The Fraud-R1 dataset drives every headline result.

## How to download

Elliptic2 is available on Kaggle:

1. Install the Kaggle CLI: `pip install kaggle`
2. Set up your Kaggle credentials: https://github.com/Kaggle/kaggle-api#api-credentials
3. Run:

```bash
kaggle datasets download -d elliptic/elliptic-bitcoin-dataset-2 -p experiments/data/elliptic2
unzip experiments/data/elliptic2/elliptic-bitcoin-dataset-2.zip -d experiments/data/elliptic2/extracted
```

The experiment script expects the extracted files at `experiments/data/elliptic2/extracted/`.

## Running the Elliptic2 validation

```bash
python experiments/src/run_elliptic2_graph.py \
  --data-dir experiments/data/elliptic2/extracted \
  --outdir experiments/results/elliptic2_graph_seed7
```
