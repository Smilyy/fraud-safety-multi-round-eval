#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$ROOT_DIR/data"

mkdir -p "$DATA_DIR"

if [ ! -d "$DATA_DIR/Fraud-R1" ]; then
  git clone https://github.com/kaustpradalab/Fraud-R1 "$DATA_DIR/Fraud-R1"
else
  echo "Fraud-R1 repo already exists at $DATA_DIR/Fraud-R1"
fi

if [ ! -d "$DATA_DIR/sting9" ]; then
  git clone https://github.com/sting9-research/dataset "$DATA_DIR/sting9"
else
  echo "Sting9 dataset repo already exists at $DATA_DIR/sting9"
fi

echo "Fraud-R1 files:"
find "$DATA_DIR/Fraud-R1/dataset" -maxdepth 2 -type f | sort

echo
echo "Sting9 repo contents:"
find "$DATA_DIR/sting9" -maxdepth 2 -type f | sort || true

cat <<'EOF'

Notes:
- Fraud-R1 is available through the public GitHub repository cloned above.
- The Hugging Face dataset endpoint is gated and was not used here.
- Sting9 has a public website and source repository, but the public dataset dump may not yet be populated.
- Elliptic2 is available through Kaggle and is large (~24GB).
EOF
