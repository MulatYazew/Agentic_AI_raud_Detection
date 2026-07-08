"""Shared per-level pipeline run, used by both run.py (CLI) and the
notebook (codes/ai_agents_fraud_detection.ipynb). Kept in one place so the
two entrypoints can never drift apart -- the notebook is for running and
inspecting results interactively, not a second copy of the pipeline logic.
"""
from __future__ import annotations

from pathlib import Path

from .config import OUTPUT_DIR
from .data_loading import load_bundle
from .identity import build_identity_graph
from .memory import MemoryStore
from .orchestrator import FraudOrchestrator
from .validator import load_eval_transaction_ids, validate_submission


class ValidationSplitGuardError(RuntimeError):
    """Raised when a validation split is scored without an explicit confirmation.

    Only the first submission against each level's evaluation set counts and
    it cannot be undone -- this guard exists so that can't happen from a
    stray/looped notebook cell execution or an unattended script run.
    """


def run_one(
    level_name: str,
    split: str,
    folder: Path,
    memory: MemoryStore,
    *,
    use_llm: bool,
    use_audio: bool,
    llm_budget: int,
    output_dir: Path = OUTPUT_DIR,
    confirm_validation: bool = False,
) -> Path:
    if split == "validation" and not confirm_validation:
        raise ValidationSplitGuardError(
            f"refusing to score {level_name!r} validation split without confirm_validation=True: "
            "only the first submission per eval set counts and this cannot be undone."
        )

    print(f"\n=== {level_name} [{split}] ===")
    bundle = load_bundle(level_name, split, folder)
    audio_note = f", {len(bundle.audio_files)} audio clips" if bundle.audio_files else ""
    print(f"Loaded {len(bundle.transactions)} transactions, {len(bundle.users)} profiled citizens{audio_note}")

    graph = build_identity_graph(bundle)
    orchestrator = FraudOrchestrator(
        bundle, graph, memory, use_llm=use_llm, use_audio=use_audio, llm_budget=llm_budget
    )
    scored = orchestrator.run()
    flagged = orchestrator.select_flagged(scored)
    orchestrator.record_memory()
    orchestrator.flush_tracing()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{level_name}_{split}.txt"
    debug_path = output_dir / f"debug_{level_name}_{split}.csv"

    with open(out_path, "w", encoding="ascii") as f:
        for tid in flagged:
            f.write(tid + "\n")

    reason_free_cols = [c for c in scored.columns if not c.endswith("_reasons")]
    scored[reason_free_cols + ["reasons"]].to_csv(debug_path, index=False)

    n_llm = orchestrator.reasoning.used if orchestrator.reasoning else 0
    print(f"Scored {len(scored)} | Flagged {len(flagged)} ({len(flagged) / max(len(scored), 1):.1%}) | LLM calls used: {n_llm}")
    print(f"Fusion weights: { {k: round(v, 3) for k, v in orchestrator.weights_used.items()} }")
    print(f"Submission written: {out_path}")
    print(f"Debug scores written: {debug_path}")

    result = validate_submission(out_path, load_eval_transaction_ids(folder))
    status = "OK" if result.ok else "FAILED"
    print(f"Validator: {status} -- {result.stats}")
    for err in result.errors:
        print(f"  ERROR: {err}")
    for warn in result.warnings:
        print(f"  WARNING: {warn}")

    return out_path
