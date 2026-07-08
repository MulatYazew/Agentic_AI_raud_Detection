# Reply Mirror -- Multi-Agent Fraud Detection

A cooperative multi-agent system for the "Reply Mirror" AI agent challenge:
unsupervised/weakly-supervised fraud detection over MirrorPay transactions,
with no fraud-label column, five increasing-difficulty levels, and one
scored submission per level's evaluation set.

## Architecture

```
                     Ingestion / Identity Agent
              (loads tx/users/locations/sms/mails/audio,
               joins them via iban / biotag / phone / name)
                              |
        +---------+---------+---------+---------+
        |         |         |         |         |
   Behavior     Geo-Time  Network   Comm      Economic       <- cheap, run on
    Agent        Agent    (graph)   Agent      Agent            every transaction
        |         |         |         |         |
        +---------+---------+---------+---------+
                              |
                    Memory/Drift Agent
        (decayed cross-level fusion-weight prior, re-weighted
         by how much each signal actually varies this level;
         evolving lure-phrase lexicon)
                              |
                     weighted fusion score
                              |
                escalate only the most ambiguous
                transactions (near the decision boundary,
                bounded by --llm-budget)
                              |
                  Contextual Reasoning Agent (LLM)
              (+ Audio Agent transcribes only clips
                 tied to an escalated transaction)
                              |
                  Decision / Orchestrator Agent
        (re-fuse, cost-aware distribution-relative threshold,
         hard guardrails against 0%/100%/implausible flag rate)
                              |
                    outputs/{level}_{split}.txt
```

Design rationale, and everything discovered about the actual dataset
schema/joinability during development, is documented inline as module
docstrings in `reply_mirror/` -- e.g. `reply_mirror/identity.py` explains
the iban/biotag join, `reply_mirror/agents.py` explains why the amount
baseline is per-(sender, transaction_type), `reply_mirror/memory.py`
explains the cross-level weight adaptation. Read those before changing
scoring logic; the reasoning for each non-obvious choice lives there, not
here.

All agent classes (Behavior, Geo-Time, Network, Comm, Economic,
Contextual Reasoning, Audio) live in the single module
`reply_mirror/agents.py`, ordered cheap-to-expensive. The actual
*execution* -- loading a level, running the orchestrator, writing output,
validating it -- happens either via the notebook
(`codes/ai_agents_fraud_detection.ipynb`) or the CLI (`run.py`); both call
the exact same `reply_mirror.pipeline.run_one`, so there is one pipeline
implementation and two ways to drive it, not two implementations.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate   # or use the existing agentic_ai_env/
pip install -r requirements.txt
```

Copy/confirm `.env` has (already present in this repo, gitignored):

```
OPENAI_API_KEY=...      # OpenRouter key (base_url is openrouter.ai)
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_HOST=...
TEAM_NAME=...
```

**Known issue as of this writing**: the `OPENAI_API_KEY` currently in
`.env` is rejected by OpenRouter with `401 User not found` (verified with
a raw `curl`, not a bug in this code), and Langfuse 403s against
`LANGFUSE_HOST`. Both look like expired/rotated competition-issued
credentials. The pipeline **does not require a working key to run** --
`ContextualReasoningAgent` falls back to the mean of the cheap signal
scores on any LLM failure -- but the Contextual Reasoning Agent and Audio
Agent are only genuinely exercised once a valid key is in place. Refresh
the key before relying on `--use-llm` for a real submission.

## Running

Levels are auto-discovered from whatever folders exist under `dataset/`
(no hardcoded level list, so levels 4-5 work the moment they're added).
There are two interchangeable entrypoints -- pick whichever fits the
moment; both call the identical `reply_mirror.pipeline.run_one`.

### Option A: the notebook (primary way to run and inspect results)

Open `codes/ai_agents_fraud_detection.ipynb` (VS Code's Jupyter extension
or classic Jupyter both work against the `agentic_ai_env`/`.venv` kernel)
and run the cells top to bottom:

1. **Setup/imports** -- locates the repo root and imports `reply_mirror`.
2. **Discover levels** -- lists what's under `dataset/`.
3. **Configure this run** -- pick levels, split (`train` by default),
   whether to use the LLM/audio agents, and the LLM escalation budget.
4. **Run the pipeline** -- scores every configured level, writing
   `outputs/{level}_{split}.txt` and `outputs/debug_{level}_{split}.csv`.
5. **Inspect results** -- displays the top flagged transactions per level
   with their scores and reasons, for a visual face-validity check, plus
   what the Memory/Drift Agent has learned across levels so far.
6. **Validate output format** -- re-runs the hard-gate validator on every
   file just written.
7. **Submit against validation** -- separate, double-guarded section (see
   below); left set to not run by default.

For a headless/CI-style run of the same notebook:
`jupyter nbconvert --to notebook --execute codes/ai_agents_fraud_detection.ipynb`.

### Option B: the CLI

```bash
# Score one level's train pool (safe to run repeatedly -- iterate here)
python run.py --level The_Truman_Show --split train

# Score every discovered level's train pool
python run.py --all --split train

# Cheap-signal-only (no LLM calls, no audio transcription)
python run.py --level The_Truman_Show --split train --no-llm --no-audio

# Cap LLM escalations for a run
python run.py --level Deus_Ex --split train --llm-budget 40
```

### Validation split -- read before running, either option

**Only the first submission against each level's validation (eval) set
counts, and it cannot be undone.** `reply_mirror.pipeline.run_one` itself
refuses to score a validation split unless `confirm_validation=True` is
passed explicitly (raises `ValidationSplitGuardError` otherwise) -- this
is enforced at the pipeline level, not just in the CLI, so the notebook
gets the same protection. The CLI additionally requires `--i-am-sure`:

```bash
python run.py --level The_Truman_Show --split validation --i-am-sure
```

The notebook's validation section requires flipping
`CONFIRM_VALIDATION_SUBMIT = True` in that cell for the same reason. Do
this only when you are ready to submit that level for real. Iterate
against `train` first, in either option.

### Validating an output file

The output-format gate is a hard requirement (non-empty, not-all, IDs must
be a subset of that split's transaction IDs, ASCII, plausible flag rate).
Run this against every file before submitting:

```bash
python run.py --validate outputs/The_Truman_Show_validation.txt \
    --level The_Truman_Show --split validation
```

`run_one()` already runs this validator automatically at the end of every
scoring run (both entrypoints) and prints `Validator: OK/FAILED` plus any
warnings, but re-running it standalone against the exact file you're about
to submit is good practice.

## Memory across levels

`state/memory_store.json` (tracked in git) persists, per level+split run:
the fusion weights used, each signal's score standard deviation that run,
the chosen threshold quantile, the resulting flag rate, and an
accumulating lexicon of lure phrases the LLM has surfaced. Deleting this
file resets the pipeline to a cold-start prior (`memory.DEFAULT_WEIGHTS`)
-- normally you don't want to do that mid-competition, since it's exactly
the cross-level adaptivity the challenge scores.

`state/audio_transcript_cache.json` (gitignored -- regenerable, and
irrelevant to another machine's cache paths) caches Whisper transcripts by
file path + mtime so re-running a level doesn't re-transcribe audio.

## Repo layout

```
reply_mirror/
  config.py         settings, .env loading, level auto-discovery
  types.py          shared dataclasses (AgentResult, DatasetBundle, Identity)
  data_loading.py   Ingestion Agent
  identity.py       iban/biotag/phone/name join logic
  agents.py         every signal/reasoning agent (Behavior, Geo, Network,
                     Comm, Economic, Contextual Reasoning, Audio), cheap-to-expensive
  memory.py         Memory/Drift Agent
  llm_client.py     shared OpenRouter-hosted LLM client + Langfuse tracing
  orchestrator.py   Decision/Orchestrator Agent (fuse, escalate, threshold)
  pipeline.py       run_one(): the single per-level run function shared by
                     run.py and the notebook
  validator.py      output-format validator
run.py              CLI entrypoint (thin argparse wrapper over reply_mirror.pipeline.run_one)
codes/
  ai_agents_fraud_detection.ipynb   the interactive runner (see "Running" above) --
                                    imports reply_mirror, does not redefine any of its logic
```

### History: why this isn't the original prototype notebook's code

`codes/ai_agents_fraud_detection.ipynb` originally contained an early,
fully self-contained prototype (agent classes defined inline in notebook
cells). Several of its ideas (per-sender z-score baselines,
cheap-then-expensive escalation, quantile thresholding) carried over into
`reply_mirror/`, but it wasn't extended as-is because it had a real
correctness risk: it discovered and scored *every* folder under `dataset/`
in one run, including validation, and wrote output to
`./{level_name_with_train_or_validation_suffix_stripped}.txt` -- so a
train run and a validation run of the same level would silently overwrite
the same file. Given only the first validation submission counts, that
wasn't safe to build on. It also had no network/graph agent and no
cross-level memory. The notebook now imports the corrected, modular
`reply_mirror` package instead of containing its own copy of the logic.

## Design notes worth knowing before changing thresholds

- **Cost model**: the challenge PDF states the false-negative/false-positive
  asymmetry qualitatively but gives no numeric cost matrix. This pipeline
  assumes false negatives cost 5x false positives (`config.FN_COST` /
  `FP_COST`) -- an explicit, tunable, documented assumption, not a derived
  constant.
- **Flag-rate guardrails**: `config.MIN_FLAG_RATE` (2%) / `MAX_FLAG_RATE`
  (25%) hard-bound the orchestrator's threshold selection so the output can
  structurally never be empty or all-inclusive, and stays within a
  plausible range for the (unverifiable without labels) 15% recall floor.
- **Train vs. validation share no citizens**: verified empirically (zero
  IBAN overlap) that a level's train and validation pools are disjoint
  sets of people over the same in-world time period. Per-user history
  therefore cannot and does not carry across splits -- each file's
  behavioral baselines are built from that file alone. What *does* carry
  across levels is pattern-level knowledge (fusion weights, lure lexicon),
  via `state/memory_store.json`.
