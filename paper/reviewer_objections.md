# Reviewer Objections and Targeted Answers

Date: 2026-04-22

This note records the two most likely reviewer objections to the current paper framing and the smallest experiments run to address them.

## Objection 1: the frozen suite is too small

### Concern

The frozen `256 x 20` suite may not be large enough to trust the main ordering.

### Targeted experiment

- Run: `experiments/results/paper_scale_consistency_512x40_seed7/Qwen_Qwen2.5-1.5B-Instruct_seed7`
- Setting:
  - same frozen attacker family
  - Qwen backbone
  - seed 7
  - train limit 512
  - test limit 40
  - benign controls included

### Result

The main fraud-side ordering remains intact on the larger slice:

- replay text-only AUSR: `0.7688`
- replay temporal graph AUSR: `0.9000`
- adaptive text-only AUSR: `0.7812`
- adaptive temporal graph AUSR: `0.9125`

The benign tradeoff also remains:

- benign replay text-only ORR: `0.3000`
- benign replay temporal graph ORR: `0.9500`
- benign replay static graph ORR: `0.9750`

### Interpretation

This does not fully solve the scale question, but it is enough to argue that the core paper story is not an artifact of the smallest repaired slice.

## Objection 2: graph gains may just be over-cautious prompting

### Concern

Graph-context gains might simply reflect a defender that has become too eager to reject.

### Targeted experiment

- Run: `experiments/results/paper_prompt_balance_256x20_seed7/Qwen_Qwen2.5-1.5B-Instruct_seed7`
- Setting:
  - same frozen seed-7 repaired slice
  - same Qwen backbone
  - same attacker family
  - a minimal `balanced_benign` prompt variant

### Result

Benign over-refusal decreased in several important settings:

- benign replay text-only ORR: `0.40 -> 0.30`
- benign replay temporal graph ORR: `0.85 -> 0.70`
- benign single-turn temporal graph ORR: `0.80 -> 0.60`
- benign single-turn static graph ORR: `0.75 -> 0.55`

Fraud-side graph advantage did not collapse:

- replay text-only AUSR: `0.7875 -> 0.85`
- replay temporal graph AUSR: `0.9625 -> 0.9375`
- adaptive text-only AUSR: `0.75 -> 0.90`
- adaptive temporal graph AUSR: `0.875 -> 0.925`

### Interpretation

This is not a final mitigation method and should not replace the frozen baseline in the main paper. But it is useful evidence that at least part of the benign-cost problem is calibratable, which weakens the objection that graph gains are purely an artifact of extreme over-caution.
