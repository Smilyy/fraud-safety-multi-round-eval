import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_ROOT = ROOT / "experiments" / "results" / "paper_suite_frozen_final_256x20" / "artifacts"
OUT_DIR = ROOT / "paper" / "generated"


def load_json(name: str):
    return json.loads((ARTIFACT_ROOT / name).read_text())


def metric_mean(block: dict, metric: str) -> float:
    return float(block[metric]["mean"])


def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_fraud_table(agg: dict):
    rows = [
        ("Replay text-only", "replay_text_only"),
        ("Replay static graph", "replay_static_graph"),
        ("Replay temporal graph", "replay_temporal_graph"),
        ("Adaptive text-only", "adaptive_text_only"),
        ("Adaptive static graph", "adaptive_static_graph"),
        ("Adaptive temporal graph", "adaptive_temporal_graph"),
    ]
    lines = [
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Setting & AUSR & ESR@1 & ESR@2 & ESR@4 & Unsafe & Latency \\",
        r"\midrule",
    ]
    for label, key in rows:
        block = agg[key]
        lines.append(
            f"{label} & "
            f"{metric_mean(block, 'AUSR'):.4f} & "
            f"{metric_mean(block, 'ESR@1'):.4f} & "
            f"{metric_mean(block, 'ESR@2'):.4f} & "
            f"{metric_mean(block, 'ESR@4'):.4f} & "
            f"{metric_mean(block, 'unsafe_compliance_rate'):.4f} & "
            f"{metric_mean(block, 'avg_latency_penalized'):.4f} \\\\"
        )
        if key == "replay_temporal_graph":
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def build_benign_table(agg: dict):
    rows = [
        ("Single-turn text-only", "benign_single_turn_text_only"),
        ("Single-turn static graph", "benign_single_turn_static_graph"),
        ("Single-turn temporal graph", "benign_single_turn_temporal_graph"),
        ("Replay text-only", "benign_replay_text_only"),
        ("Replay static graph", "benign_replay_static_graph"),
        ("Replay temporal graph", "benign_replay_temporal_graph"),
    ]
    lines = [
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Setting & ORR@1 & ORR@2 & ORR@4 & Final ORR & Latency \\",
        r"\midrule",
    ]
    for label, key in rows:
        block = agg[key]
        orr2 = metric_mean(block, "ORR@2") if "ORR@2" in block else metric_mean(block, "over_refusal_rate")
        orr4 = metric_mean(block, "ORR@4") if "ORR@4" in block else metric_mean(block, "over_refusal_rate")
        lines.append(
            f"{label} & "
            f"{metric_mean(block, 'ORR@1'):.4f} & "
            f"{orr2:.4f} & "
            f"{orr4:.4f} & "
            f"{metric_mean(block, 'over_refusal_rate'):.4f} & "
            f"{metric_mean(block, 'avg_latency_penalized'):.4f} \\\\"
        )
        if key == "benign_single_turn_temporal_graph":
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def build_sig_table(sig: dict):
    pairs = [
        ("Replay temporal vs replay text-only", "replay_temporal_graph__vs__replay_text_only"),
        ("Adaptive temporal vs adaptive text-only", "adaptive_temporal_graph__vs__adaptive_text_only"),
        ("Replay temporal vs replay static", "replay_temporal_graph__vs__replay_static_graph"),
        ("Adaptive temporal vs adaptive static", "adaptive_temporal_graph__vs__adaptive_static_graph"),
    ]
    lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Comparison & $\Delta$AUSR & $p$ & $\Delta$ latency \\",
        r"\midrule",
    ]
    for label, key in pairs:
        comp = sig["comparisons"][key]
        lines.append(
            f"{label} & "
            f"{comp['AUSR']['mean_diff_a_minus_b']:.4f} & "
            f"{comp['AUSR']['p_value']:.4f} & "
            f"{comp['avg_latency_penalized']['mean_diff_a_minus_b']:.4f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def plot_esr_curves(agg: dict, out_path: Path):
    rounds = [1, 2, 3, 4]
    replay_keys = [
        ("Text-only", "replay_text_only", "#1f77b4"),
        ("Static graph", "replay_static_graph", "#ff7f0e"),
        ("Temporal graph", "replay_temporal_graph", "#2ca02c"),
    ]
    adaptive_keys = [
        ("Text-only", "adaptive_text_only", "#1f77b4"),
        ("Static graph", "adaptive_static_graph", "#ff7f0e"),
        ("Temporal graph", "adaptive_temporal_graph", "#2ca02c"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), sharey=True)
    for ax, title, keys in [
        (axes[0], "Replay Fraud", replay_keys),
        (axes[1], "Adaptive Fraud", adaptive_keys),
    ]:
        for label, key, color in keys:
            ys = [metric_mean(agg[key], f"ESR@{r}") for r in rounds]
            ax.plot(rounds, ys, marker="o", linewidth=2, markersize=5, label=label, color=color)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(rounds)
        ax.set_xlabel("Round")
        ax.grid(alpha=0.25, linewidth=0.6)
    axes[0].set_ylabel("Early safe refusal")
    axes[0].set_ylim(0.7, 1.0)
    axes[1].legend(frameon=False, loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_tradeoff_scatter(fraud: dict, benign: dict, out_path: Path):
    contexts = [
        ("Text-only", "text_only", "#1f77b4"),
        ("Static graph", "static_graph", "#ff7f0e"),
        ("Temporal graph", "temporal_graph", "#2ca02c"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharex=True, sharey=True)
    for ax, title, fraud_prefix in [
        (axes[0], "Replay Tradeoff", "replay"),
        (axes[1], "Adaptive Tradeoff", "adaptive"),
    ]:
        for label, suffix, color in contexts:
            fraud_key = f"{fraud_prefix}_{suffix}"
            benign_key = f"benign_replay_{suffix}"
            x = metric_mean(benign[benign_key], "over_refusal_rate")
            y = metric_mean(fraud[fraud_key], "AUSR")
            ax.scatter([x], [y], s=85, color=color)
            ax.annotate(label, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.set_xlabel("Benign replay over-refusal")
    axes[0].set_ylabel("Fraud AUSR")
    axes[0].set_xlim(0.25, 1.0)
    axes[0].set_ylim(0.8, 1.0)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fraud = load_json("fraud_r1_joint_aggregate.json")
    benign = load_json("fraud_r1_benign_aggregate.json")
    sig = load_json("significance_tests.json")

    write_text(OUT_DIR / "fraud_main_table.tex", build_fraud_table(fraud))
    write_text(OUT_DIR / "benign_main_table.tex", build_benign_table(benign))
    write_text(OUT_DIR / "significance_table.tex", build_sig_table(sig))

    plot_esr_curves(fraud, OUT_DIR / "fraud_esr_curves.pdf")
    plot_tradeoff_scatter(fraud, benign, OUT_DIR / "tradeoff_scatter.pdf")


if __name__ == "__main__":
    main()
