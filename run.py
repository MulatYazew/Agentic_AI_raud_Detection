#!/usr/bin/env python3
"""Reply Mirror multi-agent fraud detection -- CLI entrypoint.

The actual pipeline logic lives in reply_mirror/pipeline.py (run_one),
shared with the interactive notebook at
codes/ai_agents_fraud_detection.ipynb -- this file is just the argparse
layer on top of it.

Examples:
    python run.py --level The_Truman_Show --split train
    python run.py --all --split train
    python run.py --level The_Truman_Show --split validation --i-am-sure
    python run.py --validate outputs/The_Truman_Show_validation.txt --level The_Truman_Show --split validation

Safety note: only the FIRST submission against each level's validation
(eval) set counts, and it cannot be undone. Running with --split
validation requires the explicit --i-am-sure flag for that reason.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from reply_mirror.config import MAX_LLM_ESCALATIONS, OUTPUT_DIR, USE_AUDIO, USE_LLM, discover_levels
from reply_mirror.memory import MemoryStore
from reply_mirror.pipeline import run_one
from reply_mirror.validator import load_eval_transaction_ids, validate_submission


def main() -> None:
    parser = argparse.ArgumentParser(description="Reply Mirror multi-agent fraud detection")
    parser.add_argument("--level", help="Level name, e.g. The_Truman_Show")
    parser.add_argument("--all", action="store_true", help="Run every discovered level for the given split")
    parser.add_argument("--split", choices=["train", "validation"], default="train")
    parser.add_argument(
        "--i-am-sure",
        action="store_true",
        help="Required to run against a validation split -- only the first submission per eval set counts.",
    )
    parser.add_argument("--use-llm", dest="use_llm", action="store_true", default=USE_LLM)
    parser.add_argument("--no-llm", dest="use_llm", action="store_false")
    parser.add_argument("--use-audio", dest="use_audio", action="store_true", default=USE_AUDIO)
    parser.add_argument("--no-audio", dest="use_audio", action="store_false")
    parser.add_argument("--llm-budget", type=int, default=MAX_LLM_ESCALATIONS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--validate", type=Path, help="Validate an existing output file instead of running the pipeline")
    args = parser.parse_args()

    levels = discover_levels()
    if not levels:
        parser.error("no level folders discovered under the dataset directory")

    if args.validate:
        if not args.level:
            parser.error("--validate requires --level")
        lp = levels.get(args.level)
        if lp is None:
            parser.error(f"unknown level {args.level!r}; discovered: {sorted(levels)}")
        folder = lp.train if args.split == "train" else lp.validation
        if folder is None:
            parser.error(f"no {args.split} folder found for level {args.level!r}")

        result = validate_submission(args.validate, load_eval_transaction_ids(folder))
        print(f"Validating {args.validate} against {folder}")
        print(f"Stats: {result.stats}")
        for err in result.errors:
            print(f"ERROR: {err}")
        for warn in result.warnings:
            print(f"WARNING: {warn}")
        print("VALID" if result.ok else "INVALID")
        sys.exit(0 if result.ok else 1)

    if args.split == "validation" and not args.i_am_sure:
        parser.error(
            "refusing to run against a validation split without --i-am-sure: only the first "
            "submission per eval set counts and this cannot be undone."
        )

    if args.all:
        selected = sorted(levels.items())
    elif args.level:
        if args.level not in levels:
            parser.error(f"unknown level {args.level!r}; discovered: {sorted(levels)}")
        selected = [(args.level, levels[args.level])]
    else:
        parser.error("specify --level NAME or --all")

    memory = MemoryStore.load()

    for level_name, lp in selected:
        folder = lp.train if args.split == "train" else lp.validation
        if folder is None:
            print(f"Skipping {level_name}: no {args.split} folder found on disk.")
            continue
        run_one(
            level_name,
            args.split,
            folder,
            memory,
            use_llm=args.use_llm,
            use_audio=args.use_audio,
            llm_budget=args.llm_budget,
            output_dir=args.output_dir,
            confirm_validation=args.i_am_sure,
        )


if __name__ == "__main__":
    main()
