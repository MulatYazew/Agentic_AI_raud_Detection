"""Shared dataclasses used across every agent in the pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class AgentResult:
    """Uniform output shape for every signal / reasoning agent."""

    score: float
    reasons: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetBundle:
    """Everything loaded from a single level/split folder (train or validation)."""

    name: str
    split: str
    path: Path
    transactions: pd.DataFrame
    users: list[dict[str, Any]]
    locations: list[dict[str, Any]]
    sms: list[dict[str, Any]]
    mails: list[dict[str, Any]]
    audio_files: list[Path]


@dataclass
class Identity:
    """Resolved identity for a citizen who appears as a transaction party."""

    citizen_id: str
    user: dict[str, Any]
    iban: str
    phone_numbers: set[str] = field(default_factory=set)
