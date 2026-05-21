#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$ROOT_DIR/data"

mkdir -p "$DATA_DIR"

# --- Fraud-R1 (vendored in repo, this is a fallback if not present) ---
if [ ! -d "$DATA_DIR/Fraud-R1/dataset" ]; then
  echo "Cloning Fraud-R1..."
  git clone https://github.com/kaustpradalab/Fraud-R1 "$DATA_DIR/Fraud-R1"
else
  echo "Fraud-R1 already present at $DATA_DIR/Fraud-R1"
fi

# --- Elliptic2 (~107 GB, requires Kaggle CLI) ---
ELLIPTIC_DIR="$DATA_DIR/elliptic2"
ELLIPTIC_EXTRACTED="$ELLIPTIC_DIR/extracted"

if [ ! -d "$ELLIPTIC_EXTRACTED" ]; then
  echo ""
  echo "Downloading Elliptic2 from Kaggle (~107 GB)..."
  echo "Make sure the Kaggle CLI is installed and credentials are configured."
  echo "See: https://github.com/Kaggle/kaggle-api#api-credentials"
  echo ""
  mkdir -p "$ELLIPTIC_DIR"
  kaggle datasets download -d elliptic/elliptic-bitcoin-dataset-2 -p "$ELLIPTIC_DIR"
  echo "Extracting..."
  unzip "$ELLIPTIC_DIR/elliptic-bitcoin-dataset-2.zip" -d "$ELLIPTIC_EXTRACTED"
  echo "Elliptic2 extracted to $ELLIPTIC_EXTRACTED"
else
  echo "Elliptic2 already extracted at $ELLIPTIC_EXTRACTED"
fi

# --- Sting9 (public repo, but dataset dump is currently empty) ---
if [ ! -d "$DATA_DIR/sting9" ]; then
  echo ""
  echo "Cloning Sting9 dataset repo..."
  git clone https://github.com/sting9-research/dataset "$DATA_DIR/sting9" || true
else
  echo "Sting9 already present at $DATA_DIR/sting9"
fi

echo ""
echo "Done. Dataset status:"
echo "  Fraud-R1:  $DATA_DIR/Fraud-R1/dataset"
echo "  Elliptic2: $ELLIPTIC_EXTRACTED"
echo "  Sting9:    $DATA_DIR/sting9 (NOTE: public dump may be empty — not used in paper)"
