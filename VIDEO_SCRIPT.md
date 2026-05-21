# Video Walkthrough Script

Target length: **4 – 6 minutes**. Audience: a supervisor who has not seen the project before. The goal is to show enough that they can reproduce a result themselves.

Record with screen + voice. OBS, QuickTime, or Zoom local recording all work; the file just has to be uploadable to YouTube / OneDrive / Google Drive so the link can be embedded in the handover PDF.

---

## Suggested recording flow

### (00:00 – 00:30) Intro — who and what

**On screen**: the paper PDF open at the title page (`paper/main.pdf`).

**Say**:

> "Hi, I'm Laura Jiang, student ID 22742957, Group 24. This is the project walkthrough for COMP6016 Progress Report 3. My project is a research paper titled *Rethinking Fraud Safety Evaluation: Multi-Round Attacks Reveal Safety–Utility Tradeoffs in Graph-Context LLM Defenders*, supervised by Reza Ryan. The paper has been submitted to arXiv. In this video I'll show the code repository, install it from scratch, and run a working example."

### (00:30 – 01:15) The GitHub repository

**On screen**: browser, open https://github.com/Smilyy/fraud-safety-multi-round-eval.

Scroll slowly down the README. Pause briefly on:

- The TL;DR (highlight the two numbers: `0.978` vs `0.847` AUSR, and `0.36` → `0.84–0.89` ORR).
- The repo-structure tree.
- The "Quickstart" command block.

**Say**:

> "The repository is public. The README states the main finding: under multi-round attack, graph-context defenders refuse fraud earlier, but they over-refuse benign traffic. The directory layout has the paper LaTeX in `paper/`, the experiment code in `experiments/src/`, the primary dataset vendored in `experiments/data/Fraud-R1/`, and the frozen result artifacts in `experiments/results/paper_suite_frozen_final_256x20/`. There's a separate `REPRODUCE.md` that maps each paper table to the exact command that produces it."

### (01:15 – 02:00) Clone and install

**On screen**: terminal. Type these commands (the audience watches them execute):

```bash
git clone https://github.com/Smilyy/fraud-safety-multi-round-eval.git
cd fraud-safety-multi-round-eval
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

(You can `pip install -q` to keep the output tidy. If install is slow, edit the recording to cut the long pip output and resume once it's done.)

**Say** while it installs:

> "Setup is the standard Python workflow: clone the repo, make a virtual environment, install requirements. The only dependencies are PyTorch, torch-geometric, transformers, and the usual numpy / pandas / scikit-learn / matplotlib. There is no Docker image — the only host-specific piece is which PyTorch CUDA wheel to install, and the `requirements.txt` uses compatible-release pins so it works on most CUDA versions."

### (02:00 – 04:30) Run the smoke check

**On screen**: terminal, in the project root.

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

**Say** as it runs:

> "This command runs the full pipeline end-to-end on a tiny slice — four training cases and one test case — so you can see the whole thing in a couple of minutes on CPU. It trains the static and temporal graph encoders on a few conversation snapshots, then evaluates a small flan-t5 defender against the replay attacker under three context conditions: text-only, static graph, and temporal graph. The full frozen suite uses the same script with `--train-limit 256 --test-limit 20`, a stronger backbone, and both replay and adaptive attackers."

When it finishes:

```bash
ls experiments/results/smoke_check/
cat experiments/results/smoke_check/fraud_r1_joint_summary.json | python -m json.tool | head -40
```

**Say**:

> "The output is a JSON summary plus a per-case predictions CSV. The summary reports the main metrics for each cell: ESR-at-k, AUSR, unsafe-compliance rate, average latency. This same script populates the much larger frozen-suite directory that the paper cites."

### (04:30 – 05:30) Show the frozen-suite artifacts

**On screen**: terminal.

```bash
ls experiments/results/paper_suite_frozen_final_256x20/artifacts/
cat experiments/results/paper_suite_frozen_final_256x20/artifacts/significance_tests.json \
  | python -m json.tool | head -60
```

**Say**:

> "These are the actual paper artifacts. `fraud_r1_joint_aggregate.json` and `fraud_r1_benign_aggregate.json` are the source-of-truth for the main and benign tables. `significance_tests.json` has the paired permutation tests, including the `p = 0.0004` value for the replay temporal vs text-only AUSR contrast. `gnn_probe.json` is the direct encoder probe; `failure_cases.json` is the catalogue of representative failures. Every number quoted in the paper traces back to one of these JSON files."

### (05:30 – 06:00) Wrap-up

**On screen**: the README, scrolled to the Citation section.

**Say**:

> "To reproduce a specific paper table, see `REPRODUCE.md` for the per-table command. The full handover document, with the Requirements Form, dev environment, data management, testing, and future work, is on Blackboard. Thank you."

---

## After recording

1. Upload the file to YouTube as **Unlisted** (or to OneDrive / Google Drive with **anyone with the link can view**).
2. Copy the share URL.
3. Paste it into the **Video walkthrough** row in the table at the top of `PR3_HANDOVER.md`.
4. Re-export `PR3_HANDOVER.md` to PDF for Blackboard submission.

A clickable link in the final PDF is required. The link is "clickable" automatically if you export from Markdown via pandoc / Word / a modern browser print — the URL will be hyperlinked. Do not paste the URL as plain unformatted text in a way that breaks the hyperlink.

---

## Optional polish

- A quick draw-on-screen during the README scroll, circling the two AUSR numbers, makes the result instantly readable for a viewer who isn't a fraud-safety researcher.
- If your terminal has a tiny font, bump it up before recording. The marker is: a viewer should be able to read every command without zooming in.
- Cut the long pip-install output in post; nobody needs to watch 90 seconds of `Downloading torch...`.
