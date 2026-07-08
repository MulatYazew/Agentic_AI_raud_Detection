"""Output-format validator.

The brief is explicit that this is a hard gate: get this wrong and the
submission is void regardless of model quality. Checked here:

  * file is plain ASCII, one ID per line;
  * not empty;
  * does not flag every transaction;
  * every flagged ID is actually a transaction ID from the file being
    scored (nothing invented, nothing from a different split);
  * flag rate sits in a plausible band -- can't verify the real 15%
    recall floor without ground truth, so this is a best-effort proxy
    warning, not a hard failure, and is reported separately from the hard
    errors above.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .config import MAX_FLAG_RATE, MIN_FLAG_RATE


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)


def load_eval_transaction_ids(folder: Path) -> set[str]:
    tx = pd.read_csv(folder / "transactions.csv", usecols=["transaction_id"])
    return set(tx["transaction_id"].astype(str))


def validate_submission(
    output_path: Path,
    eval_transaction_ids: set[str],
    min_flag_rate: float = MIN_FLAG_RATE,
    max_flag_rate: float = MAX_FLAG_RATE,
) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []

    if not output_path.exists():
        return ValidationResult(False, [f"{output_path} does not exist"])

    raw = output_path.read_bytes()
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as exc:
        errors.append(f"file is not pure ASCII ({exc})")
        text = raw.decode("utf-8", errors="replace")

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    n_eval = len(eval_transaction_ids)

    if len(lines) == 0:
        errors.append("output is empty: no transactions flagged (invalid per competition rules)")

    unique_ids = set(lines)
    if n_eval and len(unique_ids) >= n_eval:
        errors.append("output flags all (or more than all) transactions (invalid per competition rules)")

    unknown = [i for i in lines if i not in eval_transaction_ids]
    if unknown:
        errors.append(
            f"{len(unknown)} flagged IDs are not present in this split's transactions.csv "
            f"(e.g. {unknown[:3]})"
        )

    dupes = len(lines) - len(unique_ids)
    if dupes:
        warnings.append(f"{dupes} duplicate line(s) in output")

    flag_rate = len(unique_ids) / n_eval if n_eval else 0.0
    if flag_rate < min_flag_rate:
        warnings.append(
            f"flag rate {flag_rate:.1%} is below the sanity floor {min_flag_rate:.0%} "
            "-- real recall could fall under the 15% valid-submission floor"
        )
    if flag_rate > max_flag_rate:
        warnings.append(
            f"flag rate {flag_rate:.1%} is above the sanity ceiling {max_flag_rate:.0%} "
            "-- risks reading as indiscriminate flagging"
        )

    stats = {
        "n_flagged_lines": len(lines),
        "n_unique_flagged": len(unique_ids),
        "n_eval_total": n_eval,
        "flag_rate": flag_rate,
    }
    return ValidationResult(ok=len(errors) == 0, errors=errors, warnings=warnings, stats=stats)
