"""Configuration, environment loading, and level auto-discovery.

Levels are discovered from whatever folders actually exist under DATA_DIR
rather than a hardcoded name list, because (a) only 3 of the eventual 5
levels are present at the time this was written, and (b) the folder names
on disk are not perfectly regular (e.g. "Brave_New_World _validation" has a
stray space before the underscore) so a naive f"{level}_{split}" path never
finds it.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", REPO_ROOT / "dataset"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", REPO_ROOT / "outputs"))
STATE_DIR = Path(os.getenv("STATE_DIR", REPO_ROOT / "state"))
MEMORY_STORE_PATH = STATE_DIR / "memory_store.json"
AUDIO_CACHE_PATH = STATE_DIR / "audio_transcript_cache.json"

TEAM_NAME = os.getenv("TEAM_NAME", "MuNA")

USE_LLM = os.getenv("USE_LLM", "1") == "1"
USE_AUDIO = os.getenv("USE_AUDIO", "1") == "1"
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-4.1-mini")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
MAX_LLM_ESCALATIONS = int(os.getenv("MAX_LLM_ESCALATIONS", "40"))
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "tiny")

# Default asymmetric cost assumption: missing real fraud (false negative)
# is judged 5x costlier than blocking a legitimate transaction (false
# positive). Not given explicitly by the challenge PDF (it only states the
# relationship qualitatively) -- documented here so it is one auditable,
# tunable knob rather than a buried magic number.
FN_COST = float(os.getenv("FN_COST", "5.0"))
FP_COST = float(os.getenv("FP_COST", "1.0"))

# Guardrails so a threshold miscalibration can never produce an invalid
# submission (0 flags, all flags, or a rate implausible for this domain).
MIN_FLAG_RATE = float(os.getenv("MIN_FLAG_RATE", "0.02"))
MAX_FLAG_RATE = float(os.getenv("MAX_FLAG_RATE", "0.25"))

_SPLIT_SUFFIX_RE = re.compile(r"[\s_]+(train|validation)$", re.IGNORECASE)


def _canonicalize(name: str) -> str:
    stripped = _SPLIT_SUFFIX_RE.sub("", name)
    return re.sub(r"[\s_]+", "_", stripped).strip("_")


@dataclass(frozen=True)
class LevelPaths:
    name: str
    train: Path | None
    validation: Path | None


def discover_levels(data_dir: Path = DATA_DIR) -> dict[str, LevelPaths]:
    """Scan data_dir for {level}_train / {level}_validation folders.

    Returns a dict keyed by canonical level name (whitespace/underscore
    normalized, split suffix stripped) so callers can address a level
    without knowing the exact on-disk folder spelling.
    """
    found: dict[str, dict[str, Path]] = {}
    if not data_dir.exists():
        return {}

    for entry in sorted(data_dir.iterdir()):
        if not entry.is_dir():
            continue
        match = _SPLIT_SUFFIX_RE.search(entry.name)
        if not match:
            continue
        split = match.group(1).lower()
        canonical = _canonicalize(entry.name)
        found.setdefault(canonical, {})[split] = entry

    return {
        name: LevelPaths(name=name, train=paths.get("train"), validation=paths.get("validation"))
        for name, paths in found.items()
    }
