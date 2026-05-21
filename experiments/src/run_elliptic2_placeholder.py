import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()

    print("Elliptic2 experiment scaffold")
    print(f"data_dir={args.data_dir}")
    print("Next step once extraction is complete:")
    print("- inspect raw files")
    print("- identify node table, edge table, and labels")
    print("- build temporal graph snapshots")
    print("- run graph-only and temporal-graph baselines")


if __name__ == "__main__":
    main()
