import argparse
import json
import random
from pathlib import Path

import pandas as pd
import torch

try:
    from benign_control_utils import build_benign_multi_round, build_benign_single_turn
    from fraud_pipeline_utils import (
        AdaptiveAttacker,
        TextGenerationModel,
        build_defender_prompt,
        score_benign_conversation,
        score_conversation,
        score_rationale_grounding,
        summarize_benign_results,
        summarize_by_category,
        summarize_results,
    )
    from fraud_r1_joint_graph import (
        build_global_stats,
        build_snapshot,
        build_static_context,
        build_temporal_context,
        create_static_model,
        create_temporal_model,
        get_multi_round_texts,
        load_paired_examples,
        train_static_model,
        train_temporal_model,
    )
except ModuleNotFoundError:
    from experiments.src.benign_control_utils import build_benign_multi_round, build_benign_single_turn
    from experiments.src.fraud_pipeline_utils import (
        AdaptiveAttacker,
        TextGenerationModel,
        build_defender_prompt,
        score_benign_conversation,
        score_conversation,
        score_rationale_grounding,
        summarize_benign_results,
        summarize_by_category,
        summarize_results,
    )
    from experiments.src.fraud_r1_joint_graph import (
        build_global_stats,
        build_snapshot,
        build_static_context,
        build_temporal_context,
        create_static_model,
        create_temporal_model,
        get_multi_round_texts,
        load_paired_examples,
        train_static_model,
        train_temporal_model,
    )


CONTEXT_MODES = ("text_only", "static_graph", "temporal_graph")


def load_or_train_graph_models(
    train_pairs,
    global_stats,
    graph_device: str,
    temporal_backbone: str,
    graph_epochs: int,
    cache_dir: Path | None = None,
):
    static_model_path = cache_dir / "static_model.pt" if cache_dir else None
    temporal_model_path = cache_dir / "temporal_model.pt" if cache_dir else None
    meta_path = cache_dir / "graph_cache_meta.json" if cache_dir else None

    if cache_dir and static_model_path.exists() and temporal_model_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if (
            meta.get("train_cases") == len(train_pairs)
            and meta.get("temporal_backbone") == temporal_backbone
            and meta.get("graph_epochs") == graph_epochs
        ):
            static_model = create_static_model(train_pairs, global_stats, device=graph_device)
            temporal_model = create_temporal_model(
                train_pairs,
                global_stats,
                device=graph_device,
                backbone=temporal_backbone,
            )
            static_payload = torch.load(static_model_path, map_location=graph_device)
            temporal_payload = torch.load(temporal_model_path, map_location=graph_device)
            static_model.load_state_dict(static_payload["state_dict"])
            temporal_model.load_state_dict(temporal_payload["state_dict"])
            static_model.eval()
            temporal_model.eval()
            print(f"[graph-cache] loaded cached graph models from {cache_dir}")
            return static_model, temporal_model

    static_model = train_static_model(
        train_pairs,
        global_stats,
        epochs=graph_epochs,
        device=graph_device,
    )
    temporal_model = train_temporal_model(
        train_pairs,
        global_stats,
        epochs=graph_epochs,
        device=graph_device,
        backbone=temporal_backbone,
    )

    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": static_model.state_dict()}, static_model_path)
        torch.save({"state_dict": temporal_model.state_dict()}, temporal_model_path)
        meta_path.write_text(
            json.dumps(
                {
                    "train_cases": len(train_pairs),
                    "temporal_backbone": temporal_backbone,
                    "graph_epochs": graph_epochs,
                },
                indent=2,
            )
        )
        print(f"[graph-cache] saved graph models to {cache_dir}")

    return static_model, temporal_model


class RiskScoreShuffler:
    """Precomputes per-case last-round risk_scores for static and temporal models on a fixed
    test set, then provides a deterministic case_id -> partner_case mapping. Used by the
    shuffle_risk graph-ablation mode to swap only the GNN-output scalar fields in each
    graph_context dict, leaving everything else (category, request_profile, entity_summary,
    etc.) intact. This isolates the LLM's use of the per-case GNN signal."""

    def __init__(self, test_pairs, global_stats, static_model, temporal_model, device, seed):
        self.static_scores = {}
        self.temporal_scores = {}
        self.temporal_trends = {}
        for pair in test_pairs:
            ex = pair["levelup"]
            case_id = ex["id"]
            rounds = get_multi_round_texts(ex)
            if not rounds:
                continue
            snap = build_snapshot(ex, rounds, global_stats)
            with torch.no_grad():
                p_s, _ = static_model(snap)
                p_t, _ = temporal_model.forward_snapshot(snap, device)
            cur = float(p_t.item())
            self.static_scores[case_id] = round(float(p_s.item()), 4)
            self.temporal_scores[case_id] = round(cur, 4)
            if len(rounds) > 1:
                prev_snap = build_snapshot(ex, rounds[:-1], global_stats)
                with torch.no_grad():
                    p_prev, _ = temporal_model.forward_snapshot(prev_snap, device)
                prev = float(p_prev.item())
                self.temporal_trends[case_id] = "increasing" if cur > prev + 1e-4 else "stable"
            else:
                self.temporal_trends[case_id] = "stable"

        ids = list(self.static_scores.keys())
        rng = random.Random(int(seed) ^ 0x5A17F1E)
        partners = ids[:]
        rng.shuffle(partners)
        for i, cid in enumerate(ids):
            if partners[i] == cid:
                j = (i + 1) % len(ids)
                partners[i], partners[j] = partners[j], partners[i]
        self.partner_of = dict(zip(ids, partners))

    def shuffled_static_score(self, case_id):
        return self.static_scores[self.partner_of[case_id]]

    def shuffled_temporal(self, case_id):
        pid = self.partner_of[case_id]
        return self.temporal_scores[pid], self.temporal_trends[pid]

    def export(self):
        return {
            "partner_of": {str(k): v for k, v in self.partner_of.items()},
            "static_scores": {str(k): v for k, v in self.static_scores.items()},
            "temporal_scores": {str(k): v for k, v in self.temporal_scores.items()},
            "temporal_trends": {str(k): v for k, v in self.temporal_trends.items()},
        }


def _apply_graph_ablation(graph_context, context_mode, case_id, shuffler, ablation_mode):
    if graph_context is None or ablation_mode == "none" or shuffler is None:
        return graph_context
    if ablation_mode != "shuffle_risk":
        raise ValueError(f"unknown graph ablation mode: {ablation_mode}")
    if case_id not in shuffler.partner_of:
        return graph_context
    new_ctx = dict(graph_context)
    if context_mode == "static_graph":
        new_ctx["risk_score"] = shuffler.shuffled_static_score(case_id)
    elif context_mode == "temporal_graph":
        partner_score, partner_trend = shuffler.shuffled_temporal(case_id)
        new_ctx["risk_score"] = partner_score
        new_ctx["risk_trend"] = partner_trend
        prev_hint = new_ctx.get("explanation_hint", "")
        if "risk moved from" in prev_hint:
            new_ctx["explanation_hint"] = f"shuffled-risk control: partner risk {partner_score:.3f}"
    return new_ctx


def select_graph_context(
    context_mode: str,
    example: dict,
    texts: list[str],
    global_stats,
    static_model,
    temporal_model,
    device: str,
    shuffler=None,
    ablation_mode: str = "none",
):
    if context_mode == "text_only":
        return None
    if context_mode == "static_graph":
        ctx = build_static_context(example, texts, global_stats, static_model, device)
    elif context_mode == "temporal_graph":
        ctx = build_temporal_context(example, texts, global_stats, temporal_model, device)
    else:
        raise ValueError(context_mode)
    return _apply_graph_ablation(ctx, context_mode, example.get("id"), shuffler, ablation_mode)


def evaluate_single_turn(
    test_pairs,
    defender,
    global_stats,
    static_model,
    temporal_model,
    graph_device: str,
    prompt_mode: str = "default",
    progress_every: int = 25,
    shuffler=None,
    ablation_mode: str = "none",
):
    rows = []
    for idx, pair in enumerate(test_pairs, start=1):
        example = pair["base"]
        current_message = example["generated text"]
        for context_mode in CONTEXT_MODES:
            graph_context = select_graph_context(
                context_mode,
                example,
                [current_message],
                global_stats,
                static_model,
                temporal_model,
                graph_device,
                shuffler=shuffler,
                ablation_mode=ablation_mode,
            )
            prompt = build_defender_prompt(
                current_message=current_message,
                history=[],
                graph_context=graph_context,
                prompt_mode=prompt_mode,
            )
            action, raw = defender.act(prompt)
            scored = score_conversation([action], round_budget=1)
            grounding = score_rationale_grounding(raw, current_message=current_message, graph_context=graph_context)
            rows.append(
                {
                    "id": example["id"],
                    "category": example["category"],
                    "setting": f"single_turn_{context_mode}",
                    "attacker_mode": "single_turn",
                    "context_mode": context_mode,
                    "round_budget": 1,
                    "final_action": action,
                    "raw_output": raw,
                    "trace": json.dumps(
                        [
                            {
                                "round": 1,
                                "attacker_message": current_message,
                                "defender_action": action,
                                "graph_context": graph_context,
                            }
                        ],
                        ensure_ascii=True,
                    ),
                    **scored,
                    **grounding,
                }
            )
        if progress_every and (idx % progress_every == 0 or idx == len(test_pairs)):
            print(f"[single_turn] completed {idx}/{len(test_pairs)} cases")
    return rows


def evaluate_multi_round(
    test_pairs,
    defender,
    global_stats,
    static_model,
    temporal_model,
    attacker_mode: str,
    attacker_model: str | None,
    graph_device: str,
    prompt_mode: str = "default",
    attacker_generator_kwargs: dict | None = None,
    progress_every: int = 25,
    shuffler=None,
    ablation_mode: str = "none",
):
    rows = []
    shared_generator = defender if attacker_mode == "adaptive" and attacker_model == getattr(defender, "model_id", None) else None
    attacker = AdaptiveAttacker(
        attacker_mode,
        attacker_model,
        generator=shared_generator,
        generator_kwargs=attacker_generator_kwargs,
    )
    for idx, pair in enumerate(test_pairs, start=1):
        example = pair["levelup"]
        reference_rounds = [item["generated_data"] for item in example["multi-rounds fraud"]]
        round_budget = len(reference_rounds)
        for context_mode in CONTEXT_MODES:
            observed_messages = []
            defender_actions = []
            trace = []
            for round_idx, reference_message in enumerate(reference_rounds, start=1):
                attacker_message = attacker.next_message(
                    category=example["category"],
                    reference_message=reference_message,
                    history=observed_messages,
                    defender_actions=defender_actions,
                    round_index=round_idx,
                )
                graph_context = select_graph_context(
                    context_mode,
                    example,
                    observed_messages + [attacker_message],
                    global_stats,
                    static_model,
                    temporal_model,
                    graph_device,
                    shuffler=shuffler,
                    ablation_mode=ablation_mode,
                )
                prompt = build_defender_prompt(
                    current_message=attacker_message,
                    history=observed_messages,
                    graph_context=graph_context,
                    prompt_mode=prompt_mode,
                )
                action, raw = defender.act(prompt)
                trace.append(
                    {
                        "round": round_idx,
                        "reference_message": reference_message,
                        "attacker_message": attacker_message,
                        "defender_action": action,
                        "defender_raw_output": raw,
                        "graph_context": graph_context,
                    }
                )
                observed_messages.append(attacker_message)
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
                    "setting": f"{attacker_mode}_{context_mode}",
                    "attacker_mode": attacker_mode,
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
            print(f"[{attacker_mode}] completed {idx}/{len(test_pairs)} cases")
    return rows


def evaluate_benign_single_turn(
    test_pairs,
    defender,
    global_stats,
    static_model,
    temporal_model,
    graph_device: str,
    prompt_mode: str = "default",
    progress_every: int = 25,
    shuffler=None,
    ablation_mode: str = "none",
):
    rows = []
    for idx, pair in enumerate(test_pairs, start=1):
        example = pair["base"]
        current_message = build_benign_single_turn(example)
        for context_mode in CONTEXT_MODES:
            graph_context = select_graph_context(
                context_mode,
                example,
                [current_message],
                global_stats,
                static_model,
                temporal_model,
                graph_device,
                shuffler=shuffler,
                ablation_mode=ablation_mode,
            )
            prompt = build_defender_prompt(
                current_message=current_message,
                history=[],
                graph_context=graph_context,
                prompt_mode=prompt_mode,
            )
            action, raw = defender.act(prompt)
            scored = score_benign_conversation([action], round_budget=1)
            grounding = score_rationale_grounding(raw, current_message=current_message, graph_context=graph_context)
            rows.append(
                {
                    "id": example["id"],
                    "category": example["category"],
                    "setting": f"benign_single_turn_{context_mode}",
                    "attacker_mode": "benign_single_turn",
                    "context_mode": context_mode,
                    "round_budget": 1,
                    "final_action": action,
                    "raw_output": raw,
                    "trace": json.dumps(
                        [
                            {
                                "round": 1,
                                "attacker_message": current_message,
                                "defender_action": action,
                                "graph_context": graph_context,
                            }
                        ],
                        ensure_ascii=True,
                    ),
                    **scored,
                    **grounding,
                }
            )
        if progress_every and (idx % progress_every == 0 or idx == len(test_pairs)):
            print(f"[benign_single_turn] completed {idx}/{len(test_pairs)} cases")
    return rows


def evaluate_benign_multi_round(
    test_pairs,
    defender,
    global_stats,
    static_model,
    temporal_model,
    graph_device: str,
    prompt_mode: str = "default",
    progress_every: int = 25,
    shuffler=None,
    ablation_mode: str = "none",
):
    rows = []
    for idx, pair in enumerate(test_pairs, start=1):
        example = pair["base"]
        benign_rounds = build_benign_multi_round(example)
        round_budget = len(benign_rounds)
        for context_mode in CONTEXT_MODES:
            observed_messages = []
            defender_actions = []
            trace = []
            for round_idx, benign_message in enumerate(benign_rounds, start=1):
                graph_context = select_graph_context(
                    context_mode,
                    example,
                    observed_messages + [benign_message],
                    global_stats,
                    static_model,
                    temporal_model,
                    graph_device,
                    shuffler=shuffler,
                    ablation_mode=ablation_mode,
                )
                prompt = build_defender_prompt(
                    current_message=benign_message,
                    history=observed_messages,
                    graph_context=graph_context,
                    prompt_mode=prompt_mode,
                )
                action, raw = defender.act(prompt)
                trace.append(
                    {
                        "round": round_idx,
                        "attacker_message": benign_message,
                        "defender_action": action,
                        "defender_raw_output": raw,
                        "graph_context": graph_context,
                    }
                )
                observed_messages.append(benign_message)
                defender_actions.append(action)
                if action in {"REJECT", "COMPLY"}:
                    break
            scored = score_benign_conversation(defender_actions, round_budget=round_budget)
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
                    "setting": f"benign_replay_{context_mode}",
                    "attacker_mode": "benign_replay",
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
            print(f"[benign_replay] completed {idx}/{len(test_pairs)} cases")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-data", type=Path, required=True)
    parser.add_argument("--levelup-data", type=Path, required=True)
    parser.add_argument("--language", type=str, default="English")
    parser.add_argument("--model", type=str, default="google/flan-t5-small")
    parser.add_argument("--attacker-model", type=str, default="google/flan-t5-small")
    parser.add_argument("--temporal-backbone", type=str, default="tgn", choices=["gru", "tgn"])
    parser.add_argument("--graph-epochs", type=int, default=8)
    parser.add_argument("--graph-device", type=str, default="cpu", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--graph-cache-dir", type=Path)
    parser.add_argument("--llm-device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--attacker-device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--max-input-tokens", type=int, default=768)
    parser.add_argument("--max-threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--test-limit", type=int)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--attacker-modes", nargs="+", default=["replay", "adaptive"])
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--with-benign-controls", action="store_true")
    parser.add_argument("--only-benign-controls", action="store_true")
    parser.add_argument("--defender-prompt-mode", type=str, default="default", choices=["default", "balanced_benign"])
    parser.add_argument("--graph-ablation-mode", type=str, default="none", choices=["none", "shuffle_risk"],
                        help="Post-process graph_context to ablate the GNN signal. 'shuffle_risk' substitutes risk_score (and risk_trend for temporal) with a deterministically-permuted partner case's value, preserving marginal distribution while breaking per-case correspondence.")
    args = parser.parse_args()

    if args.only_benign_controls:
        args.with_benign_controls = True

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

    shuffler = None
    if args.graph_ablation_mode == "shuffle_risk":
        shuffler = RiskScoreShuffler(
            test_pairs,
            global_stats,
            static_model,
            temporal_model,
            device=graph_device,
            seed=args.seed,
        )
        (args.outdir / "graph_ablation_shuffle.json").write_text(json.dumps(shuffler.export(), indent=2))
        print(f"[graph-ablation] shuffle_risk mode active: {len(shuffler.partner_of)} test cases permuted")

    defender = TextGenerationModel(
        args.model,
        max_input_tokens=args.max_input_tokens,
        device_preference=args.llm_device,
        max_threads=args.max_threads,
    )
    attacker_device = args.attacker_device
    if attacker_device == "auto":
        if args.attacker_model.startswith("openai:"):
            attacker_device = "auto"
        elif defender.backend == "local" and defender.device == "cuda" and args.attacker_model != args.model:
            attacker_device = "cpu"
        else:
            attacker_device = args.llm_device
    summary = None
    if not args.only_benign_controls:
        results = []
        results += evaluate_single_turn(
            test_pairs,
            defender,
            global_stats,
            static_model,
            temporal_model,
            graph_device=graph_device,
            prompt_mode=args.defender_prompt_mode,
            progress_every=args.progress_every,
            shuffler=shuffler,
            ablation_mode=args.graph_ablation_mode,
        )
        for attacker_mode in args.attacker_modes:
            results += evaluate_multi_round(
                test_pairs=test_pairs,
                defender=defender,
                global_stats=global_stats,
                static_model=static_model,
                temporal_model=temporal_model,
                attacker_mode=attacker_mode,
                attacker_model=args.attacker_model if attacker_mode == "adaptive" else None,
                graph_device=graph_device,
                prompt_mode=args.defender_prompt_mode,
                attacker_generator_kwargs={
                    "max_input_tokens": args.max_input_tokens,
                    "device_preference": attacker_device,
                    "max_threads": args.max_threads,
                },
                progress_every=args.progress_every,
                shuffler=shuffler,
                ablation_mode=args.graph_ablation_mode,
            )

        df = pd.DataFrame(results)
        df.to_csv(args.outdir / "fraud_r1_joint_predictions.csv", index=False)

        summary = summarize_results(df)
        category_summary = summarize_by_category(df)
        (args.outdir / "fraud_r1_joint_summary.json").write_text(json.dumps(summary, indent=2))
        (args.outdir / "fraud_r1_joint_category_summary.json").write_text(json.dumps(category_summary, indent=2))

    if args.with_benign_controls:
        benign_rows = []
        benign_rows += evaluate_benign_single_turn(
            test_pairs,
            defender,
            global_stats,
            static_model,
            temporal_model,
            graph_device=graph_device,
            prompt_mode=args.defender_prompt_mode,
            progress_every=args.progress_every,
            shuffler=shuffler,
            ablation_mode=args.graph_ablation_mode,
        )
        benign_rows += evaluate_benign_multi_round(
            test_pairs,
            defender,
            global_stats,
            static_model,
            temporal_model,
            graph_device=graph_device,
            prompt_mode=args.defender_prompt_mode,
            progress_every=args.progress_every,
            shuffler=shuffler,
            ablation_mode=args.graph_ablation_mode,
        )
        benign_df = pd.DataFrame(benign_rows)
        benign_df.to_csv(args.outdir / "fraud_r1_benign_predictions.csv", index=False)
        benign_summary = summarize_benign_results(benign_df)
        (args.outdir / "fraud_r1_benign_summary.json").write_text(json.dumps(benign_summary, indent=2))

    split_meta = {
        "train_cases": len(train_pairs),
        "test_cases": len(test_pairs),
        "seed": args.seed,
        "test_fraction": args.test_fraction,
        "split_manifest": str(args.split_manifest) if args.split_manifest else None,
        "model": args.model,
        "attacker_model": args.attacker_model,
        "temporal_backbone": args.temporal_backbone,
        "graph_epochs": args.graph_epochs,
        "graph_device": graph_device,
        "graph_cache_dir": str(args.graph_cache_dir) if args.graph_cache_dir else None,
        "llm_device": defender.device,
        "attacker_device": attacker_device,
        "max_input_tokens": args.max_input_tokens,
        "max_threads": args.max_threads,
        "attacker_modes": args.attacker_modes,
        "single_turn_source": "FP-base paired by id",
        "graph_target": "escalation_risk_v1",
        "benign_controls": bool(args.with_benign_controls),
        "only_benign_controls": bool(args.only_benign_controls),
        "defender_prompt_mode": args.defender_prompt_mode,
        "graph_ablation_mode": args.graph_ablation_mode,
    }
    (args.outdir / "split_meta.json").write_text(json.dumps(split_meta, indent=2))
    if summary is not None:
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
