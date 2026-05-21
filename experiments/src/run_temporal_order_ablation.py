import argparse
import json
import random
from pathlib import Path

import pandas as pd
import torch

try:
    from fraud_pipeline_utils import TextGenerationModel, build_defender_prompt, score_conversation, score_rationale_grounding, summarize_results
    from fraud_r1_joint_graph import build_global_stats, load_paired_examples
    from run_fraud_r1_joint_graph import load_or_train_graph_models, select_graph_context
except ModuleNotFoundError:
    from experiments.src.fraud_pipeline_utils import (
        TextGenerationModel,
        build_defender_prompt,
        score_conversation,
        score_rationale_grounding,
        summarize_results,
    )
    from experiments.src.fraud_r1_joint_graph import build_global_stats, load_paired_examples
    from experiments.src.run_fraud_r1_joint_graph import load_or_train_graph_models, select_graph_context


CONTEXT_MODES = ("text_only", "static_graph", "temporal_graph")
ORDER_VARIANTS = ("original", "reverse", "shuffle")


def reorder_rounds(reference_rounds: list[str], variant: str, case_id: int, seed: int):
    rounds = list(reference_rounds)
    if variant == "original":
        return rounds
    if variant == "reverse":
        return list(reversed(rounds))
    if variant == "shuffle":
        rng = random.Random(seed + int(case_id) * 1009)
        indices = list(range(len(rounds)))
        rng.shuffle(indices)
        return [rounds[idx] for idx in indices]
    raise ValueError(variant)


def evaluate_order_variants(
    test_pairs,
    defender,
    global_stats,
    static_model,
    temporal_model,
    graph_device: str,
    seed: int,
    progress_every: int = 10,
):
    rows = []
    for idx, pair in enumerate(test_pairs, start=1):
        example = pair["levelup"]
        reference_rounds = [item["generated_data"] for item in example["multi-rounds fraud"]]
        for order_variant in ORDER_VARIANTS:
            ordered_rounds = reorder_rounds(reference_rounds, variant=order_variant, case_id=example["id"], seed=seed)
            round_budget = len(ordered_rounds)
            for context_mode in CONTEXT_MODES:
                observed_messages = []
                defender_actions = []
                trace = []
                for round_idx, current_message in enumerate(ordered_rounds, start=1):
                    graph_context = select_graph_context(
                        context_mode,
                        example,
                        observed_messages + [current_message],
                        global_stats,
                        static_model,
                        temporal_model,
                        graph_device,
                    )
                    prompt = build_defender_prompt(
                        current_message=current_message,
                        history=observed_messages,
                        graph_context=graph_context,
                    )
                    action, raw = defender.act(prompt)
                    trace.append(
                        {
                            "round": round_idx,
                            "attacker_message": current_message,
                            "defender_action": action,
                            "defender_raw_output": raw,
                            "graph_context": graph_context,
                        }
                    )
                    observed_messages.append(current_message)
                    defender_actions.append(action)
                    if action in {"REJECT", "COMPLY"}:
                        break
                scored = score_conversation(defender_actions, round_budget=round_budget)
                final_step = trace[-1] if trace else None
                grounding = score_rationale_grounding(
                    final_step["defender_raw_output"] if final_step else "",
                    current_message=final_step["attacker_message"] if final_step else "",
                    graph_context=final_step["graph_context"] if final_step else None,
                )
                rows.append(
                    {
                        "id": example["id"],
                        "category": example["category"],
                        "setting": f"{order_variant}_{context_mode}",
                        "order_variant": order_variant,
                        "context_mode": context_mode,
                        "round_budget": round_budget,
                        "final_action": defender_actions[-1] if defender_actions else "ASK",
                        "raw_output": trace[-1]["defender_raw_output"] if trace else "",
                        "trace": json.dumps(trace, ensure_ascii=True),
                        **scored,
                        **grounding,
                    }
                )
        if progress_every and (idx % progress_every == 0 or idx == len(test_pairs)):
            print(f"[order-ablation] completed {idx}/{len(test_pairs)} cases")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-data", type=Path, required=True)
    parser.add_argument("--levelup-data", type=Path, required=True)
    parser.add_argument("--language", type=str, default="English")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--temporal-backbone", type=str, default="gru", choices=["gru", "tgn"])
    parser.add_argument("--graph-epochs", type=int, default=3)
    parser.add_argument("--graph-device", type=str, default="cpu", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--graph-cache-dir", type=Path)
    parser.add_argument("--llm-device", type=str, default="cuda", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--max-input-tokens", type=int, default=768)
    parser.add_argument("--max-threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--test-limit", type=int)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    train_pairs, test_pairs = load_paired_examples(
        args.base_data,
        args.levelup_data,
        args.language,
        args.seed,
        args.test_fraction,
        split_manifest=args.split_manifest,
    )
    if args.train_limit:
        train_pairs = train_pairs[:args.train_limit]
    if args.test_limit:
        test_pairs = test_pairs[:args.test_limit]

    global_stats = build_global_stats(train_pairs)
    graph_device = args.graph_device
    if graph_device == "auto":
        graph_device = "cuda" if torch.cuda.is_available() else "cpu"

    static_model, temporal_model = load_or_train_graph_models(
        train_pairs,
        global_stats,
        graph_device=graph_device,
        temporal_backbone=args.temporal_backbone,
        graph_epochs=args.graph_epochs,
        cache_dir=args.graph_cache_dir,
    )

    defender = TextGenerationModel(
        args.model,
        max_input_tokens=args.max_input_tokens,
        device_preference=args.llm_device,
        max_threads=args.max_threads,
    )
    rows = evaluate_order_variants(
        test_pairs,
        defender,
        global_stats,
        static_model,
        temporal_model,
        graph_device=graph_device,
        seed=args.seed,
        progress_every=args.progress_every,
    )
    df = pd.DataFrame(rows)
    df.to_csv(args.outdir / "temporal_order_ablation_predictions.csv", index=False)
    summary = summarize_results(df)
    (args.outdir / "temporal_order_ablation_summary.json").write_text(json.dumps(summary, indent=2))
    meta = {
        "train_cases": len(train_pairs),
        "test_cases": len(test_pairs),
        "seed": args.seed,
        "model": args.model,
        "temporal_backbone": args.temporal_backbone,
        "graph_epochs": args.graph_epochs,
        "graph_device": graph_device,
        "graph_cache_dir": str(args.graph_cache_dir) if args.graph_cache_dir else None,
        "llm_device": defender.device,
        "order_variants": list(ORDER_VARIANTS),
        "context_modes": list(CONTEXT_MODES),
        "split_manifest": str(args.split_manifest) if args.split_manifest else None,
    }
    (args.outdir / "temporal_order_ablation_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
