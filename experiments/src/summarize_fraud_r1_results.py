import argparse
import json
from pathlib import Path


def load_json(path: Path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--category-summary", type=Path, required=True)
    args = parser.parse_args()

    summary = load_json(args.summary)
    category_summary = load_json(args.category_summary)

    print("Overall")
    print("setting,AUSR,ESR@1,ESR@2,unsafe_compliance_rate,avg_latency_penalized,n")
    for setting, stats in summary.items():
        print(
            f"{setting},{stats.get('AUSR')},{stats.get('ESR@1')},{stats.get('ESR@2')},"
            f"{stats.get('unsafe_compliance_rate')},{stats.get('avg_latency_penalized')},{stats['n']}"
        )

    print("\nByCategory")
    print("setting,category,AUSR,ESR@1,ESR@2,unsafe_compliance_rate,avg_latency_penalized,n")
    for setting, categories in category_summary.items():
        for category, stats in categories.items():
            print(
                f"{setting},{category},{stats.get('AUSR')},{stats.get('ESR@1')},{stats.get('ESR@2')},"
                f"{stats.get('unsafe_compliance_rate')},{stats.get('avg_latency_penalized')},{stats['n']}"
            )


if __name__ == "__main__":
    main()
