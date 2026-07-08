"""Memory/Drift Agent: persists what worked across levels and re-weights.

This is the "adaptivity across levels" requirement made concrete. Two
things carry forward between levels via a small JSON file on disk
(state/memory_store.json, tracked in git so results are reproducible):

  1. Fusion weights: how much each signal agent's score should count
     toward the final risk score.
  2. A lure-phrase lexicon: suspicious phrases the LLM Contextual
     Reasoning Agent has surfaced on past levels, fed back into the cheap
     CommAgent's keyword list for future levels.

Crucially, per-level identity does NOT carry over -- verified empirically
that a level's train and validation pools share zero citizens, and
different levels obviously feature entirely different people. So this
store never keeps user-level facts, only level-level *pattern* statistics:
what weighting scheme was used, how spread out each signal's scores were,
what flag rate resulted. That is exactly the kind of "learned pattern"
that plausibly transfers even though the people don't.

Drift handling: a new level's starting fusion weights are a decayed
average of prior levels' weights (recent levels count more), then blended
with how much each signal actually varies *in the current level* --
a signal that stops discriminating anything this round (near-zero
variance) automatically loses influence rather than dragging the fused
score around on stale evidence. That's the "re-weight instead of
overfitting to stale patterns" behavior called for in the brief.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import MEMORY_STORE_PATH

DEFAULT_WEIGHTS: dict[str, float] = {
    "behavior": 0.28,
    "geo": 0.16,
    "network": 0.16,
    "comm": 0.18,
    "economic": 0.10,
    "reasoning": 0.12,
}
DECAY = 0.65
BLEND_ALPHA = 0.6  # weight given to prior/memory vs. current-level informativeness


def _normalize(weights: dict[str, float]) -> dict[str, float]:
    total = sum(max(w, 0.0) for w in weights.values())
    if total <= 1e-9:
        n = len(weights) or 1
        return {k: 1.0 / n for k in weights}
    return {k: max(w, 0.0) / total for k, w in weights.items()}


@dataclass
class MemoryStore:
    path: Path = MEMORY_STORE_PATH
    data: dict[str, Any] = field(default_factory=lambda: {"levels": {}, "lexicon": []})

    @classmethod
    def load(cls, path: Path = MEMORY_STORE_PATH) -> "MemoryStore":
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = {"levels": {}, "lexicon": []}
        data.setdefault("levels", {})
        data.setdefault("lexicon", [])
        return cls(path=path, data=data)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, sort_keys=True)

    def get_lexicon(self) -> list[str]:
        return list(dict.fromkeys(self.data.get("lexicon", [])))

    def add_lexicon_terms(self, terms: list[str]) -> None:
        existing = self.data.setdefault("lexicon", [])
        for term in terms:
            term = term.strip().lower()
            if term and term not in existing:
                existing.append(term)

    def prior_weights(self, exclude_level: str | None = None) -> dict[str, float]:
        """Decayed average of fusion weights from previously recorded levels.

        Records are keyed "{level}::{split}" so a level's train and
        validation runs don't clobber each other; exclude_level matches on
        the level-name prefix so scoring one split of a level never draws
        its prior from the other split of that same level.
        """
        records = [
            (name, rec) for name, rec in self.data["levels"].items()
            if not (exclude_level and name.startswith(f"{exclude_level}::")) and "fusion_weights" in rec
        ]
        if not records:
            return dict(DEFAULT_WEIGHTS)

        weighted_sum: dict[str, float] = {k: 0.0 for k in DEFAULT_WEIGHTS}
        total_weight = 0.0
        # Most-recently-recorded entries are later in dict insertion order;
        # give them more influence via exponential decay by recency rank.
        for rank, (_, rec) in enumerate(reversed(records)):
            w = DECAY ** rank
            for k, v in rec["fusion_weights"].items():
                weighted_sum[k] = weighted_sum.get(k, 0.0) + w * v
            total_weight += w

        return _normalize({k: v / total_weight for k, v in weighted_sum.items()})

    def adapt_weights(self, prior: dict[str, float], current_signal_std: dict[str, float]) -> dict[str, float]:
        total_std = sum(max(v, 0.0) for v in current_signal_std.values())
        if total_std <= 1e-9:
            return _normalize(prior)

        informativeness = {k: max(v, 0.0) / total_std for k, v in current_signal_std.items()}
        blended = {
            k: BLEND_ALPHA * prior.get(k, 0.0) + (1 - BLEND_ALPHA) * informativeness.get(k, 0.0)
            for k in prior
        }
        return _normalize(blended)

    def prior_threshold_quantile(self, default: float = 0.90) -> float:
        records = [rec for rec in self.data["levels"].values() if "threshold_quantile" in rec]
        if not records:
            return default
        return sum(r["threshold_quantile"] for r in records) / len(records)

    def record_level_run(
        self,
        level_name: str,
        split: str,
        fusion_weights: dict[str, float],
        signal_std: dict[str, float],
        threshold_quantile: float,
        flag_rate: float,
        n_transactions: int,
        lexicon_additions: list[str] | None = None,
    ) -> None:
        if lexicon_additions:
            self.add_lexicon_terms(lexicon_additions)

        self.data["levels"][f"{level_name}::{split}"] = {
            "level": level_name,
            "split": split,
            "run_at": datetime.now(timezone.utc).isoformat(),
            "n_transactions": n_transactions,
            "fusion_weights": fusion_weights,
            "signal_std": signal_std,
            "threshold_quantile": threshold_quantile,
            "flag_rate": flag_rate,
        }
        self.save()
