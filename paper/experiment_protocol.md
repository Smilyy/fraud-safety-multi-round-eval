# Experiment Protocol

This document matches the current code path more closely than the original pilot notes.

For the current go/no-go run order and submission criteria, see `paper/submission_readiness.md`.

## Main Claim

Fixed single-turn prompts and final-turn-only success can overstate fraud-defense robustness. The main outcome should be early safe refusal under replay and adaptive multi-round attack, and learned graph context should improve that outcome.

## Dataset Roles

Primary end-to-end benchmark:

- `Fraud-R1`
  Use this for the conversational GNN+LLM study.

Supporting graph benchmark:

- `Elliptic2`
  Use this to validate the graph backbone on a real fraud graph with node/component labels.

Important constraint:

- `Elliptic2` is not a second conversational benchmark.
- `Fraud-R1` is the only dataset in this workspace that directly supports the joint reasoning conversation experiment.

## Core Experimental Axes

1. Evaluation mode:
   `single_turn`, `multi_round_replay`, `multi_round_adaptive`
2. Defender context:
   `text_only`, `static_graph`, `temporal_graph`
3. Backbone:
   start with one open-weight seq2seq/chat model, then add a second stronger model when compute allows

## Prompting Rule

Use one defender template everywhere.

- Same system role
- Same action space: `REJECT`, `ASK`, `COMPLY`
- Same safety policy
- The only difference between settings is the available history and graph context

## Fraud-R1 Pipeline

1. Split paired cases into train/test by case id.
2. Use `FP-base` as the paired single-turn source and `FP-levelup` as the paired multi-round source.
3. Train a static graph encoder on conversation-graph escalation-risk targets rather than raw round index.
4. Train a temporal graph encoder on ordered graph snapshots over rounds using the same escalation-risk target.
5. Store the split manifest so every rerun uses the same case ids.
6. Serialize learned graph outputs into interpretable prompt fields.
7. Evaluate the same defender prompt in:
   - `single_turn_*`
   - `replay_*`
   - `adaptive_*`
8. Report early safe refusal metrics, not only final rejection.

## Elliptic2 Pipeline

1. Build a node graph from `nodes.csv`, `edges.csv`, and `connected_components.csv`.
2. Split by connected component to avoid leakage.
3. Train a real GraphSAGE fraud-risk model.
4. Report node-level fraud-risk metrics on held-out components.
5. Use this experiment as graph-backbone validation, not as a replacement for Fraud-R1.

## Metrics

Primary:

- `ESR@k`
  Early Safe Refusal rate by round `k`
- `AUSR`
  Area under the safe-refusal curve across rounds
- `unsafe_compliance_rate`
- `avg_latency_penalized`
  mean refusal round with failures penalized as `T + 1`

Secondary:

- `non_compliance_rate`
- `avg_safe_rejection_round`
- category-wise `ESR@k`

Graph-benchmark metrics on `Elliptic2`:

- `roc_auc`
- `average_precision`
- `precision`
- `recall`
- `f1`

## Adaptive Attacker

The adaptive attacker must condition on defender behavior.

Minimum requirement:

- rewrite the next fraud message based on the previous defender action
- keep the same fraud goal
- change wording and pressure strategy after `ASK` or `REJECT`

Current code path:

- local seq2seq rewrite model with a defender-conditioned prompt

## Tables

Table 1:

- Fraud-R1 `single_turn` vs `replay` vs `adaptive` using `ESR@1`, `ESR@2`, `AUSR`

Table 2:

- `text_only` vs `static_graph` vs `temporal_graph` under replay attacker

Table 3:

- `text_only` vs `static_graph` vs `temporal_graph` under adaptive attacker

Table 4:

- Elliptic2 graph-only validation results

## Figures

- Safe-refusal-by-round curve
- Replay vs adaptive drop plot
- Example conversation with graph context and decision trajectory
- Elliptic2 precision-recall or risk-score histogram

## Failure Analysis

Collect at least 20 examples covering:

- late safe refusal after multiple persuasive rounds
- unsafe compliance after paraphrasing
- graph context that helps after text-only hesitation
- graph context that causes over-warning

## What Is Still Missing

- cross-seed results for the stronger API-backed backbone
- larger repaired runs beyond the current `256 x 20` comparative slice
- calibration of the adaptive attacker so it is consistently harder than replay for every backbone
- significance testing once the repaired multi-seed suite is complete

## Recommended Paper Framing

Claim the following:

- early refusal matters more than eventual rejection
- adaptive attackers are harder than replay attackers
- learned graph context helps the LLM refuse earlier

Do not claim the following unless the final results support it:

- that any graph variant is near-perfect
- that `Elliptic2` is a conversational transfer benchmark
- that replay-only results are enough to justify the word `adaptive`
