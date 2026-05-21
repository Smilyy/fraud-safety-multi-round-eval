# Submission Readiness Checklist

This is the minimum path from the current repo state to a defensible submission.

## Current Status

- Infrastructure: ready
- Safe execution path: ready
- Qwen repaired-slice attacker calibration: passed
- stronger API-backed backbone calibration: passed on the frozen two-seed slice
- Frozen benign controls: regenerated
- Frozen paper artifacts: regenerated
- Main empirical claim: provisionally supported on the frozen `256 x 20` suite

The attacker-calibration blocker is no longer the main issue. The current staged-template attacker in `experiments/src/fraud_pipeline_utils.py` passes the replay-vs-adaptive gate on the frozen two-seed Qwen slice and the frozen two-seed `openai:gpt-5.4-mini` slice. The remaining paper risk is not missing infrastructure but tradeoff quality: graph settings still show severe benign over-refusal, and the frozen evidence is still only the repaired `256 x 20` slice.

## Go / No-Go Criteria

The paper is not submission-ready unless all of the following are true.

1. Adaptive calibration passes on the repaired `256 x 20` slice.
   - Required:
     - `adaptive_text_only` must not be easier than `replay_text_only` on aggregate AUSR.
     - `adaptive_temporal_graph` must not be easier than `replay_temporal_graph` on aggregate AUSR.
   - Preferred:
     - `adaptive_*` is harder than `replay_*` for all three contexts.
  - Current status:
    - the structured staged-template attacker passed this gate on the repaired two-seed Qwen slice in `paper_suite_attacker_calibration_256x20_v10`.

2. The main graph-context result still holds after attacker calibration.
   - Required:
     - under adaptive attack, `temporal_graph` must beat `text_only` on AUSR.
     - under replay attack, `temporal_graph` must beat `text_only` on AUSR.

3. Results hold across at least two seeds for each backbone used in the main tables.
  - Minimum:
    - Qwen two-seed repaired suite
    - one stronger API-backed model two-seed repaired suite
  - Current status:
    - Qwen: passed on `paper_suite_attacker_calibration_256x20_v10`
    - `openai:gpt-5.4-mini`: passed on the frozen suite in `paper_suite_frozen_final_256x20`

4. Benign controls are reported and bounded.
   - Required:
     - include benign over-refusal results in the paper.
   - Preferred:
     - graph settings should not show a severe benign over-refusal regression relative to text-only.
   - Current status:
     - required: passed in `paper_suite_frozen_final_256x20/artifacts/fraud_r1_benign_aggregate.json`
     - preferred: failed; graph settings still over-refuse benign traffic far more often than text-only

5. Final paper artifacts exist for the exact version being written up.
   - Required:
     - aggregate summaries
     - significance tests
     - attacker-gap analysis
     - failure-case examples
   - Current status:
     - passed in `paper_suite_frozen_final_256x20/artifacts`

## Strict Run Order

Run these in order. Do not skip ahead.

1. Freeze the current adaptive attacker design.
   - The design is documented in `paper/adaptive_attacker_design.md`.
   - The current implementation lives in `experiments/src/fraud_pipeline_utils.py`.
   - Use one job at a time through `experiments/src/run_safe_paper_suite.py`.

2. Keep the Qwen repaired aggregate as the calibration reference.
   - Build:
     - `Qwen_Qwen2.5-1.5B-Instruct_agg_seeds7_11.json`
     - attacker-gap aggregate for both seeds

3. Run the stronger API-backed backbone on the same repaired slice and same split manifests.
   - Use the frozen attacker.
   - Run two seeds.

4. Build paper artifacts from the exact frozen suite.
   - Use:
     - `experiments/src/build_paper_artifacts.py`

5. Only then scale beyond `256 x 20`.
   - If scaling is needed for final confidence, do it after attacker freeze, not before.

## Safe Execution Rules

- Use only `experiments/src/run_safe_paper_suite.py` for comparative runs.
- Keep `--thread-limit 2`.
- Keep one active GPU job at a time.
- Reuse graph cache directories instead of retraining them unnecessarily.
- Prefer `graph-device cpu` and `llm-device cuda` on this machine.
- Do not launch larger runs while calibration is unresolved.

## Current Best Read

As of April 22, 2026, the repo is operationally submission-ready for the frozen `256 x 20` suite, but the paper is still empirically risky.

Why:

- The frozen suite now has attacker calibration, cross-backbone confirmation, benign controls, significance tests, attacker-gap analysis, and failure cases.
- The main graph-context result holds on the frozen suite:
  - replay `temporal_graph` AUSR = `0.9781` vs replay `text_only` = `0.8469`
  - adaptive `temporal_graph` AUSR = `0.9625` vs adaptive `text_only` = `0.8375`
- The strongest remaining weakness is benign over-refusal:
  - benign replay `text_only` over-refusal = `0.3625`
  - benign replay `temporal_graph` over-refusal = `0.8375`
  - benign replay `static_graph` over-refusal = `0.8875`
- So the paper can be written up now if it is framed honestly as a safety-performance tradeoff result, not as an unqualified win for graph context.

## Immediate Next Task

Write tables and figures from `paper_suite_frozen_final_256x20/artifacts`, and decide whether to submit this version as a calibrated small-scale study or spend more time improving benign over-refusal before scaling.
