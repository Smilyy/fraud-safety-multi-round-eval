# COMP6016 Progress Report 3 — Handover Document

| Field | Value |
|---|---|
| **Group No.** | 24 |
| **Project Name** | Rethinking Fraud Safety Evaluation: Multi-Round Attacks Reveal Safety–Utility Tradeoffs in Graph-Context LLM Defenders |
| **Supervisor** | Reza Ryan |
| **Student** | Laura Jiang |
| **Student ID** | 22742957 |
| **Date** | 2026-05-21 |
| **Source code repository** | https://github.com/Smilyy/fraud-safety-multi-round-eval |
| **arXiv preprint** | https://arxiv.org/abs/2605.20759 |
| **Video walkthrough** | [OneDrive link](https://curtin-my.sharepoint.com/:v:/g/personal/22742957_student_curtin_edu_au/IQDay-GdZmazSLJZe2sVpN-pAfLGCgXzAPCO-VGNH4Kqv7s?e=m0BZgx) |

> **Note on team structure.** This is nominally a group assignment; in practice it was completed solo because the project is research-oriented (a paper) rather than a full-stack software product. Reza Ryan is the formal supervisor. Qian Li and Nasim Ferdosian (Curtin staff) provided advisory review of the paper draft and are recognised as co-authors on the arXiv submission; on this Requirements Form, that contribution is recorded under *Acknowledgments* rather than the Student(s) column, which is reserved for student contributors.

---

## 1. Requirements Form

The functional and non-functional categories below have been adapted from the example template (which described a student–supervisor allocation system) to the actual deliverables of this research project. All ticks reflect the state of the repository as of the submission date.

### 1.1 Functional Requirements

#### Table 1 — Data ingestion and graph construction

| FR | Student | Non-functional | Occasionally | Somewhat | Mostly | **Fully** |
|---|---|:-:|:-:|:-:|:-:|:-:|
| FR 1.1 Load paired Fraud-R1 conversations (FP-base + FP-levelup, English split) | LJ |  |  |  |  | ✓ |
| FR 1.2 Build per-round heterogeneous conversation graph snapshot (12 node roles, 4 edge groups) | LJ |  |  |  |  | ✓ |
| FR 1.3 Compute escalation-risk target combining current and future-round signals | LJ |  |  |  |  | ✓ |
| FR 1.4 Build Elliptic2 node graph from `nodes.csv`, `edges.csv`, `connected_components.csv` | LJ |  |  |  | ✓ |  |
| FR 1.5 Persist fixed train/test split manifest by case id | LJ |  |  |  |  | ✓ |

#### Table 2 — Graph encoder training

| FR | Student | Non-functional | Occasionally | Somewhat | Mostly | **Fully** |
|---|---|:-:|:-:|:-:|:-:|:-:|
| FR 2.1 Train static SAGEConv encoder on escalation-risk targets (BCE loss) | LJ |  |  |  |  | ✓ |
| FR 2.2 Train temporal SAGEConv + GRU encoder on round-ordered snapshots | LJ |  |  |  |  | ✓ |
| FR 2.3 Cache trained encoder weights so defender-side reruns are deterministic | LJ |  |  |  |  | ✓ |
| FR 2.4 Serialize graph output into a structured JSON context block for the LLM prompt | LJ |  |  |  |  | ✓ |

#### Table 3 — Multi-round defender evaluation

| FR | Student | Non-functional | Occasionally | Somewhat | Mostly | **Fully** |
|---|---|:-:|:-:|:-:|:-:|:-:|
| FR 3.1 Run single-turn evaluation using one shared defender prompt template | LJ |  |  |  |  | ✓ |
| FR 3.2 Run multi-round **replay** evaluation with verbatim Fraud-R1 transcripts | LJ |  |  |  |  | ✓ |
| FR 3.3 Run multi-round **adaptive** evaluation with defender-conditioned attacker rewrites | LJ |  |  |  |  | ✓ |
| FR 3.4 Support `text_only` / `static_graph` / `temporal_graph` defender contexts under one prompt | LJ |  |  |  |  | ✓ |
| FR 3.5 Support multiple defender backbones (local Hugging Face + OpenAI API) | LJ |  |  |  | ✓ |  |
| FR 3.6 Run paired benign-control evaluation alongside fraud-side runs | LJ |  |  |  |  | ✓ |

#### Table 4 — Metrics, analysis, and reporting

| FR | Student | Non-functional | Occasionally | Somewhat | Mostly | **Fully** |
|---|---|:-:|:-:|:-:|:-:|:-:|
| FR 4.1 Report ESR@k, AUSR, unsafe-compliance rate, avg-latency-penalized, ORR | LJ |  |  |  |  | ✓ |
| FR 4.2 Score defender grounding from a fixed 14-tag evidence vocabulary | LJ |  |  |  |  | ✓ |
| FR 4.3 Run paired permutation significance tests with bootstrap CIs across seeds | LJ |  |  |  |  | ✓ |
| FR 4.4 Run McNemar test for static vs temporal first-round refusal | LJ |  |  |  |  | ✓ |
| FR 4.5 Run direct GNN probe and shuffle-risk ablation to localise observed costs | LJ |  |  |  |  | ✓ |
| FR 4.6 Aggregate per-cell results into paper tables and figures | LJ |  |  |  |  | ✓ |

### 1.2 Non-Functional Requirements

#### Table 5 — Reproducibility

| NFR | Student | Non-functional | Occasionally | Somewhat | Mostly | **Fully** |
|---|---|:-:|:-:|:-:|:-:|:-:|
| NFR 1.1 Fixed train/test split manifest is shared across every cell of the 3×3 design | LJ |  |  |  |  | ✓ |
| NFR 1.2 Cached graph-encoder weights are reused when only the defender backbone changes | LJ |  |  |  |  | ✓ |
| NFR 1.3 Local defender LLM uses greedy decoding (temperature = 0) | LJ |  |  |  |  | ✓ |

#### Table 6 — Experimental validity

| NFR | Student | Non-functional | Occasionally | Somewhat | Mostly | **Fully** |
|---|---|:-:|:-:|:-:|:-:|:-:|
| NFR 2.1 A single defender prompt template is used across all nine evaluation cells | LJ |  |  |  |  | ✓ |
| NFR 2.2 The adaptive attacker is constrained by a goal-preservation check | LJ |  |  |  |  | ✓ |
| NFR 2.3 Encoder behaviour is verified independently of the LLM via a probe + shuffle-risk ablation | LJ |  |  |  |  | ✓ |

#### Table 7 — Statistical reliability

| NFR | Student | Non-functional | Occasionally | Somewhat | Mostly | **Fully** |
|---|---|:-:|:-:|:-:|:-:|:-:|
| NFR 3.1 Headline contrasts are reported with p-values and bootstrap confidence intervals | LJ |  |  |  |  | ✓ |
| NFR 3.2 Multi-seed runs are reported for the local Qwen-1.5B backbone (seeds 7, 11) | LJ |  |  |  | ✓ |  |

#### Table 8 — Usability (for a future maintainer)

| NFR | Student | Non-functional | Occasionally | Somewhat | Mostly | **Fully** |
|---|---|:-:|:-:|:-:|:-:|:-:|
| NFR 4.1 The full frozen suite can be reproduced from a single driver script | LJ |  |  |  |  | ✓ |
| NFR 4.2 Each paper table / figure is mapped to its regenerating command in `REPRODUCE.md` | LJ |  |  |  |  | ✓ |
| NFR 4.3 Long runs are resumable (`--skip-existing`) and log per-run to disk | LJ |  |  |  |  | ✓ |
| NFR 4.4 A five-minute CPU-only smoke run is documented in `README.md` | LJ |  |  |  |  | ✓ |

#### Table 9 — Maintainability

| NFR | Student | Non-functional | Occasionally | Somewhat | Mostly | **Fully** |
|---|---|:-:|:-:|:-:|:-:|:-:|
| NFR 5.1 Code is modular by responsibility (graph builder, encoder, defender, attacker, scorer, aggregator) | LJ |  |  |  |  | ✓ |
| NFR 5.2 Configuration is via command-line flags; no hard-coded paths inside source files | LJ |  |  |  |  | ✓ |
| NFR 5.3 Cached intermediate state (split manifest, graph cache) decouples reruns from raw data | LJ |  |  |  |  | ✓ |

#### Table 10 — Compute footprint

| NFR | Student | Non-functional | Occasionally | Somewhat | Mostly | **Fully** |
|---|---|:-:|:-:|:-:|:-:|:-:|
| NFR 6.1 The graph encoder fits and trains on CPU | LJ |  |  |  |  | ✓ |
| NFR 6.2 The defender LLM runs on a single 12 GB consumer GPU (Qwen-1.5B) | LJ |  |  |  |  | ✓ |
| NFR 6.3 The suite can be re-scaled to a different case-count by swapping the split manifest | LJ |  |  |  |  | ✓ |

---

## 2. Comprehensive Documentation

The repository is documented in three layers, each with a different audience:

1. **`README.md`** at the repo root — the entry point. Summarises the paper's claim, lists what's in the repo, and provides a five-minute CPU-only smoke command. Audience: a developer who landed on the GitHub page from the paper.
2. **`REPRODUCE.md`** — per-table and per-figure reproduction commands, mapped to the exact frozen-suite artifacts they came from. Audience: a reviewer who needs to verify a specific number.
3. **`experiments/README.md`** — design rationale, dataset roles, full command examples for each script. Audience: a contributor extending the work (new dataset, new backbone, new attacker mode).

Design notes that informed the paper itself live in `paper/`:

- `paper/story_freeze.md` — the locked thesis, list of disallowed headline claims, source-of-truth artifact paths.
- `paper/experiment_protocol.md` — what each script is required to do and how it maps to the paper's tables.
- `paper/adaptive_attacker_design.md` — the staged-template generator and the goal-preservation filter used in the adaptive setting.
- `paper/reviewer_objections.md` — the two reviewer objections explicitly addressed in the final draft and where each is rebutted.
- `paper/submission_readiness.md` — the go/no-go checklist used to clear the arXiv submission.

The paper PDF itself, `paper/main.pdf`, is the authoritative description of methodology, metrics, and results.

---

## 3. Development Environment

### Reference environment

| Component | Version |
|---|---|
| OS | Linux (Ubuntu 22.04 / 24.04) |
| Python | 3.11 |
| PyTorch | 2.11 (CUDA 12.8 build) |
| torch_geometric | 2.7 |
| transformers | 5.3 |
| CUDA driver | ≥ 535 (for GPU runs) |

### Setup, from a clean checkout

Conda is the recommended approach — it resolves the PyTorch + CUDA pairing automatically and matches the environment the frozen suite was run in:

```bash
git clone https://github.com/Smilyy/fraud-safety-multi-round-eval.git
cd fraud-safety-multi-round-eval
conda env create -f environment.yml
conda activate fraud-eval
```

No conda? Pip fallback — install a CUDA-compatible PyTorch wheel first, then the rest:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

The frozen-suite numbers were produced with the versions listed in the table above. Any torch ≥ 2.2 with CUDA 12.x reproduces them.

### Optional: GPU-backed defender LLM

The local defender (Qwen 2.5-1.5B-Instruct) wants ~5 GB of VRAM. Any consumer GPU with 8 GB or more is sufficient. Without a GPU, every `--llm-device cuda` flag in `REPRODUCE.md` should be replaced with `--llm-device cpu`; runs slow down but still complete.

### Optional: OpenAI-backed defender LLM

The frozen suite includes two cells that use `openai:gpt-5.4-mini` as the defender. To rerun those cells:

```bash
export OPENAI_API_KEY=<your key>
```

The HTTP calls are stdlib `urllib`; no extra Python package is needed.

### A first sanity check

```bash
python experiments/src/run_fraud_r1_joint_graph.py \
  --base-data experiments/data/Fraud-R1/dataset/FP-base-full/FP-base-English.json \
  --levelup-data experiments/data/Fraud-R1/dataset/FP-levelup-full/FP-levelup-English.json \
  --language English \
  --model google/flan-t5-small \
  --attacker-model google/flan-t5-small \
  --graph-device cpu --llm-device cpu \
  --temporal-backbone gru \
  --train-limit 4 --test-limit 1 \
  --attacker-modes replay \
  --outdir experiments/results/smoke_check
```

Expected outcome: a `fraud_r1_joint_summary.json` file under `experiments/results/smoke_check/`. Walltime ~5 minutes on a recent laptop CPU.

---

## 4. Data Management

### Inputs

| Dataset | Role | Where it lives | Size |
|---|---|---|---|
| **Fraud-R1** | Primary conversational benchmark (drives every paper table) | Vendored at `experiments/data/Fraud-R1/` | 32 MB |
| **Elliptic2** | Real graph used to validate the graph backbone (Section: Elliptic2 graph validation) | *Not in repo.* Fetched by `experiments/src/download_data.sh` from Kaggle | 107 GB raw + extracted |
| **Sting9** | Originally planned third dataset | `experiments/data/sting9/` (planning notes only) | 200 KB |

Sting9's public dump is currently empty and its API requires authorisation; the paper documents this and does not draw claims from it. No code path requires Sting9.

### Intermediate and cached state

| Path | What it is | Why it's in the repo |
|---|---|---|
| `experiments/results/paper_shared_split_seed7/split_manifest.json` | Frozen train/test case-id split used by every cell | Without this, reruns would draw different test cases |
| `experiments/results/paper_graph_cache_256/` | Cached static and temporal graph-encoder weights | Lets a user swap defender backbones without retraining the encoder |

### Outputs

| Path | What it contains |
|---|---|
| `experiments/results/paper_suite_frozen_final_256x20/` | One sub-directory per (defender × seed); each holds `fraud_r1_joint_summary.json`, `fraud_r1_joint_predictions.csv`, `fraud_r1_benign_summary.json`, `fraud_r1_benign_predictions.csv`, and a per-category breakdown |
| `experiments/results/paper_suite_frozen_final_256x20/artifacts/` | Cross-cell aggregations: `fraud_r1_joint_aggregate.json`, `fraud_r1_benign_aggregate.json`, `significance_tests.json`, `attacker_gap.json`, `failure_cases.json`, `mcnemar_static_vs_temporal.json`, `gnn_probe.json`. These are the files the paper cites. |
| `experiments/results/paper_scale_consistency_512x40_seed7/` | Larger-slice scale check, replicating the main ordering at `512 × 40` |
| `experiments/results/paper_prompt_balance_256x20_seed7/` | Minimal prompt-calibration check, showing partial benign-ORR mitigation |

Exploratory result directories (smoke, pilot, validate, debug, repair, etc.) are excluded from the repo via `.gitignore`; they were used during development and are not referenced by any paper claim.

---

## 5. Testing and Quality

This is a research codebase rather than a production system, so testing is performed end-to-end (does the pipeline produce the right artifacts?) rather than via unit tests. Three layers of checking are in place:

1. **Smoke run** (Section 3 above). A four-case run that exercises the whole pipeline. Used as a regression check after any code change.
2. **Determinism check.** The frozen split manifest and the cached graph encoders make every Qwen-defender cell bit-exact reproducible across reruns; this is verified by checksumming `fraud_r1_joint_predictions.csv` after a clean rerun.
3. **Statistical robustness.** `significance_tests.py` runs paired permutation tests and bootstrap CIs on the same `n=80` pooled paired cases used everywhere in the paper. McNemar testing for static vs temporal first-round refusal is run by `mcnemar_static_vs_temporal.py`.

### Known issues and limitations

| Area | Status |
|---|---|
| OpenAI gpt-5.4-mini cells | Not bit-exact reproducible due to provider-side decoding; directional claims and p-values are robust across reruns. |
| Elliptic2 | Used as graph-backbone validation only, not as a conversational benchmark. The paper explicitly does not claim conversational transfer from Elliptic2. |
| Sting9 | Public dataset is currently empty; no claim depends on it. |
| Two-seed coverage | The local Qwen backbone has two seeds (7, 11); the OpenAI backbone has the same two seeds but is provider-non-deterministic, so the per-seed numbers should be treated as samples rather than independent replicates. |
| Adaptive-attacker uniformity | The adaptive attacker is not uniformly harder than replay in every cell; the paper documents this and does not claim universal adaptive-harder behaviour. |

---

## 6. Deployment and Infrastructure

This project is a research artifact, not a deployed system. The "deployment" path is reproduction: a user clones the repository, follows Section 3, and rerun the commands in `REPRODUCE.md`.

No CI/CD, no hosted service, no database; all intermediate state is file-based under `experiments/results/`.

A Docker image was not produced for this submission because the only host-specific complication is the PyTorch CUDA wheel (which a user should choose to match their CUDA driver). The `requirements.txt` + smoke-run workflow has been verified on a clean Python 3.11 virtual environment.

---

## 7. Future Work and Recommendations

For a follow-up student or contributor, the highest-value extensions, in order of leverage:

1. **Scale beyond `256 × 20`.** The current frozen suite is small. The `paper_scale_consistency_512x40_seed7` directory shows that the main ordering holds at `512 × 40`; the natural next step is `1024 × 80` and / or all `1071` English cases. The driver script supports this directly via `--train-limit` / `--test-limit` + a new split manifest.
2. **Stronger defender backbones.** The current suite uses Qwen-2.5-1.5B-Instruct and `openai:gpt-5.4-mini`. Adding Anthropic's Claude family and Meta's larger Llama variants would test whether the LLM-conditioning effect (graph context disproportionately raising benign ORR) generalises across model families.
3. **Calibrated graph-context consumption.** The probe + shuffle-risk ablation localises the benign-ORR cost to *how the LLM reads* the graph fields. A follow-up could replace the current JSON serialisation with a numerical-only or rank-only encoding, or with explicit instructions that distinguish "graph context is present" from "graph context is alarming".
4. **Adaptive-attacker hardening.** The adaptive attacker is not uniformly harder than replay across every cell. Reward-shaping the attacker against the *defender's* own refusal probability (rather than a static goal-preservation filter) would close that gap.
5. **Sting9 once data is available.** The paper documents Sting9's planning role; if the maintainers expose the dataset dump or grant API access, integrating it would give a third independent fraud benchmark.
6. **Real graph fusion.** Currently Fraud-R1 conversation graphs and Elliptic2 transaction graphs are evaluated separately. A future system could fuse them: use Elliptic2-pretrained node embeddings as initialisation for Fraud-R1 conversation entities that resolve to known addresses or organisations.

The frozen-suite design — fixed split manifest, cached graph encoders, single driver script — is intended to make all of these extensions one-flag changes rather than codebase rewrites.

---

## 8. Acknowledgments

This work was carried out at Curtin University. The author thanks Reza Ryan for supervision, and Qian Li and Nasim Ferdosian for advisory review of the paper draft.
