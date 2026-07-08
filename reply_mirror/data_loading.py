"""Ingestion Agent: loads one level/split folder into a DatasetBundle.

Column/vocabulary normalization is deliberately defensive: the PDF spec's
wording for transaction_type/payment_method ("bank transfer", "mobile
device") already does not match the actual CSV vocabulary ("transfer",
"mobile phone") in the levels we have. Since the whole premise of the
challenge is that levels 4-5 will drift further, nothing here assumes an
exact fixed vocabulary.
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd

from .types import DatasetBundle

REQUIRED_COLUMNS = [
    "transaction_id", "sender_id", "recipient_id", "transaction_type",
    "amount", "location", "payment_method", "sender_iban", "recipient_iban",
    "balance_after", "description", "timestamp",
]


def normalize_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", text.strip().lower())


def _load_json(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_bundle(level_name: str, split: str, folder: Path) -> DatasetBundle:
    tx = pd.read_csv(folder / "transactions.csv")
    tx.columns = [normalize_text(c).replace(" ", "_") for c in tx.columns]

    for col in REQUIRED_COLUMNS:
        if col not in tx.columns:
            tx[col] = ""

    for col in ["sender_id", "recipient_id", "transaction_type", "location",
                "payment_method", "sender_iban", "recipient_iban", "description"]:
        tx[col] = tx[col].fillna("").astype(str)

    tx["amount"] = pd.to_numeric(tx["amount"], errors="coerce").fillna(0.0)
    tx["balance_after"] = pd.to_numeric(tx["balance_after"], errors="coerce").fillna(0.0)
    tx["timestamp"] = pd.to_datetime(tx["timestamp"], errors="coerce")
    tx["transaction_type_norm"] = tx["transaction_type"].map(normalize_text)
    tx["payment_method_norm"] = tx["payment_method"].map(normalize_text)

    if tx["transaction_id"].duplicated().any():
        dup = int(tx["transaction_id"].duplicated().sum())
        raise ValueError(f"{folder}: {dup} duplicate transaction_id values")

    tx = tx.sort_values("timestamp", kind="stable").reset_index(drop=True)

    audio_dir = folder / "audio"
    audio_files = sorted(audio_dir.glob("*.mp3")) if audio_dir.exists() else []

    return DatasetBundle(
        name=level_name,
        split=split,
        path=folder,
        transactions=tx,
        users=_load_json(folder / "users.json"),
        locations=_load_json(folder / "locations.json"),
        sms=_load_json(folder / "sms.json"),
        mails=_load_json(folder / "mails.json"),
        audio_files=audio_files,
    )
