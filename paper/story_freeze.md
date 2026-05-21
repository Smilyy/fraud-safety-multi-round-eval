# Frozen Core Story

Date frozen: 2026-04-22

## Main paper type

- Evaluation paper
- Measurement paper
- Insight / tradeoff paper

## Primary thesis

Single-turn fraud safety evaluation is too weak. Under multi-round replay and adaptive fraud escalation, defenders differ not only in whether they refuse, but when they refuse and at what benign-cost tradeoff.

## Strong claims allowed

- Multi-round fraud evaluation reveals behavior that single-turn evaluation misses.
- Early safe refusal is a more informative robustness quantity than final refusal alone.
- On the frozen suite, graph-context defenders improve fraud-side early safe refusal relative to text-only baselines.
- On the frozen suite, graph-context defenders substantially increase benign over-refusal.
- Temporal graph is directionally stronger than static graph and significantly better grounded, but not conclusively superior on the main refusal metrics.

## Claims not allowed as headlines

- Temporal graph is conclusively better than static graph.
- Graph context is broadly superior overall.
- The proposed defender solves fraud safety.
- Adaptive attack is universally harder than replay in every slice.

## Source of truth

- `experiments/results/paper_suite_frozen_final_256x20/artifacts/fraud_r1_joint_aggregate.json`
- `experiments/results/paper_suite_frozen_final_256x20/artifacts/fraud_r1_benign_aggregate.json`
- `experiments/results/paper_suite_frozen_final_256x20/artifacts/significance_tests.json`
- `experiments/results/paper_suite_frozen_final_256x20/artifacts/attacker_gap.json`
- `experiments/results/paper_suite_frozen_final_256x20/artifacts/failure_cases.json`

## Core numbers to anchor the draft

- Replay temporal vs text-only AUSR: `0.9781` vs `0.8469`
- Adaptive temporal vs text-only AUSR: `0.9625` vs `0.8375`
- Replay temporal vs text-only AUSR difference: `0.1313`, `p = 0.0004`
- Adaptive temporal vs text-only AUSR difference: `0.1250`, `p = 0.0027`
- Benign replay ORR: text-only `0.3625`, temporal `0.8375`, static `0.8875`

## Most dangerous reviewer objections

1. The frozen suite is still small and may not be stable beyond the repaired `256 x 20` slice.
2. Graph gains may reflect over-cautious prompting rather than genuinely better calibrated safety behavior.

## Targeted follow-up experiments only

1. Modest scale-consistency check on a larger repaired slice with the frozen attacker.
2. Minimal prompt-calibration check for benign over-refusal.

## Follow-up status

- Completed: scale-consistency check at `experiments/results/paper_scale_consistency_512x40_seed7/Qwen_Qwen2.5-1.5B-Instruct_seed7`
- Completed: benign-calibration check at `experiments/results/paper_prompt_balance_256x20_seed7/Qwen_Qwen2.5-1.5B-Instruct_seed7`

## Follow-up takeaways

- Larger-slice scale check keeps the main ordering:
  - replay temporal AUSR `0.9000` vs replay text-only `0.7688`
  - adaptive temporal AUSR `0.9125` vs adaptive text-only `0.7812`
- Larger-slice benign tradeoff remains severe:
  - benign replay temporal ORR `0.9500` vs benign replay text-only `0.3000`
- Minimal prompt calibration reduces benign over-refusal in several settings without collapsing the graph-side fraud advantage:
  - benign replay temporal ORR `0.85 -> 0.70`
  - benign single-turn temporal ORR `0.80 -> 0.60`
