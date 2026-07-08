"""All signal / reasoning agents for the Reply Mirror fraud-detection pipeline.

Kept in one module (rather than one file per agent) by request. Ordered
cheap-to-expensive, matching the pipeline's own escalation order:

  BehaviorAgent, GeoAgent, NetworkAgent, CommAgent, EconomicAgent
      -- cheap, deterministic, no LLM call, run over every transaction.
  ContextualReasoningAgent
      -- expensive, LLM-gated, only run on the borderline slice of
         transactions the cheap agents couldn't confidently resolve
         (see orchestrator.py's escalation gate).
  AudioTranscriber
      -- on-demand helper for ContextualReasoningAgent: transcribes only
         the audio clip nearest an already-escalated transaction, never a
         whole level's audio folder upfront.

See each class's docstring for the reasoning behind its specific design
choices (several of which were corrected after being checked against the
actual dataset, not assumed).
"""
from __future__ import annotations

import json
import math
import re
import statistics
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path

import pandas as pd

from .config import AUDIO_CACHE_PATH, WHISPER_MODEL_SIZE
from .data_loading import normalize_text
from .identity import IdentityGraph
from .llm_client import invoke_json
from .types import AgentResult, DatasetBundle

# ---------------------------------------------------------------------------
# Behavior Agent: per-sender statistical anomaly scoring.
# ---------------------------------------------------------------------------
# Cheap, deterministic, runs over every transaction with no LLM calls. Each
# transaction is compared only against that *same sender's own prior
# transactions* (time-ordered, expanding window) -- never against other
# senders and never against future transactions, since a production fraud
# monitor only ever has the past to compare against.
#
# Amount anomaly uses a robust median/MAD z-score rather than mean/std:
# fraud sits in the tail, and a handful of large fraudulent transactions can
# drag a mean/std baseline toward themselves, masking the very anomaly it's
# supposed to catch. MAD is far less sensitive to that.
#
# The amount baseline is kept per (sender, transaction_type), not per sender
# alone: a sender's history is usually dominated by one or two recurring
# transaction types at very different scales (e.g. ~1000 rent transfers vs
# ~40 utility direct debits), and blending them into one baseline made every
# first payment in a smaller category look like an extreme anomaly purely
# from the scale mismatch -- confirmed against real data, where a routine
# $46 first-ever direct debit scored z=36 only because it was being compared
# against that sender's $950-1000 transfer history. Comparing within the
# same type avoids that false signal; a same-type baseline of 0 history
# simply skips the amount check (score 0) rather than borrowing a
# different-type baseline, and the separate "first-time transaction type"
# check still fires so the novelty itself is not lost.
#
# The "unusual hour" check is relative to each sender's own historical hour
# distribution rather than a fixed clock cutoff (e.g. "23:00-04:00 is bad"):
# the challenge explicitly calls out "shifting temporal habits" as a hacker
# tactic, so a fixed-hour rule would stop working the moment fraud timing
# shifts, while a per-sender baseline adapts automatically.

MIN_HISTORY_FOR_STATS = 3
Z_SCORE_ANOMALY_THRESHOLD = 3.0
VELOCITY_WINDOW_HOURS = 24
VELOCITY_BURST_COUNT = 5
MAD_SCALE = 1.4826  # scales MAD to be comparable to a normal std-dev


def _robust_z(value: float, history: list[float]) -> float:
    if len(history) < MIN_HISTORY_FOR_STATS:
        return 0.0
    median = statistics.median(history)
    mad = statistics.median([abs(x - median) for x in history]) * MAD_SCALE
    if mad < 1e-6:
        mad = max(median * 0.15, 1.0)
    return abs(value - median) / mad


def _hour_deviation(hour: int, prior_hours: list[int]) -> float:
    """Smallest circular distance (in hours) from `hour` to any prior hour."""
    if not prior_hours:
        return 0.0
    return min(min(abs(hour - h), 24 - abs(hour - h)) for h in prior_hours)


class BehaviorAgent:
    name = "behavior"

    def run_batch(self, tx: pd.DataFrame) -> pd.DataFrame:
        ordered = tx.sort_values(["sender_id", "timestamp"], kind="stable")

        amounts_by_type: dict[tuple[str, str], list[float]] = defaultdict(list)
        recipients: dict[str, set[str]] = defaultdict(set)
        methods: dict[str, set[str]] = defaultdict(set)
        types: dict[str, set[str]] = defaultdict(set)
        hours: dict[str, list[int]] = defaultdict(list)
        history_ts: dict[str, list[pd.Timestamp]] = defaultdict(list)

        results: dict[int, AgentResult] = {}

        for idx, row in ordered.iterrows():
            sender = str(row["sender_id"])
            amount = float(row["amount"])
            ttype = row["transaction_type"]
            ts = row["timestamp"]
            reasons: list[str] = []
            score = 0.0

            z = _robust_z(amount, amounts_by_type[(sender, ttype)])
            if z > Z_SCORE_ANOMALY_THRESHOLD:
                score += 0.35
                reasons.append(f"amount anomaly (robust z={z:.1f} vs sender's own '{ttype}' history)")

            hist_recipients = recipients[sender]
            if hist_recipients and row["recipient_id"] and row["recipient_id"] not in hist_recipients:
                score += 0.12
                reasons.append("first-time recipient for this sender")

            hist_methods = methods[sender]
            if hist_methods and row["payment_method"] and row["payment_method"] not in hist_methods:
                score += 0.08
                reasons.append("first-time payment method for this sender")

            hist_types = types[sender]
            if hist_types and row["transaction_type"] not in hist_types:
                score += 0.10
                reasons.append("first-time transaction type for this sender")

            if pd.notna(ts):
                trailing = [t for t in history_ts[sender] if ts - t <= pd.Timedelta(hours=VELOCITY_WINDOW_HOURS)]
                if len(trailing) >= VELOCITY_BURST_COUNT:
                    score += 0.20
                    reasons.append(f"velocity burst: {len(trailing)} transactions in trailing {VELOCITY_WINDOW_HOURS}h")

                dev = _hour_deviation(ts.hour, hours[sender])
                if len(hours[sender]) >= MIN_HISTORY_FOR_STATS and dev >= 6:
                    score += 0.12
                    reasons.append(f"transacting {dev:.0f}h outside sender's usual hour pattern")

            if float(row["balance_after"]) < 0:
                score += 0.20
                reasons.append("negative balance after transaction")

            results[idx] = AgentResult(min(score, 1.0), reasons, {"amount_z": round(z, 3)})

            amounts_by_type[(sender, ttype)].append(amount)
            if row["recipient_id"]:
                recipients[sender].add(row["recipient_id"])
            if row["payment_method"]:
                methods[sender].add(row["payment_method"])
            types[sender].add(row["transaction_type"])
            if pd.notna(ts):
                hours[sender].append(ts.hour)
                history_ts[sender].append(ts)

        return pd.DataFrame(
            {
                "behavior_score": {i: r.score for i, r in results.items()},
                "behavior_reasons": {i: r.reasons for i, r in results.items()},
            }
        ).reindex(tx.index)


# ---------------------------------------------------------------------------
# Geo-Time Agent: impossible-travel / location-mismatch detection.
# ---------------------------------------------------------------------------
# Cross-references a citizen's independent GPS pings (locations.json) against
# the location a transaction claims to happen at, irrespective of what the
# transaction itself says. Only meaningful for transaction types that assert
# a physical location -- empirically that's "in-person payment" and
# "withdrawal" (location = "City - Venue"); e-commerce's location field is a
# merchant name with no city component, and transfer/direct debit have no
# location at all, so those are left at neutral score rather than forcing a
# geo claim that isn't there.
#
# Distance/speed is computed from a city-name -> centroid lookup built from
# every GPS ping in the bundle (not just this citizen's), since we have no
# offline geocoder. This is an approximation but is enough to catch the
# clearly-impossible cases (thousands of km covered in minutes) that matter
# for fraud signal.

MAX_PLAUSIBLE_SPEED_KMH = 900.0  # generous: commercial flight cruise speed
RECENT_WINDOW_DAYS = 7


def _haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _extract_city(location: str) -> str | None:
    if not location or " - " not in location:
        return None
    return normalize_text(location.split(" - ")[0])


class GeoAgent:
    name = "geo"

    def __init__(self, graph: IdentityGraph):
        self.graph = graph
        self.city_centroid = self._build_city_centroids(graph.locations_df)

    @staticmethod
    def _build_city_centroids(loc_df: pd.DataFrame) -> dict[str, dict[str, float]]:
        if loc_df.empty:
            return {}
        keyed = loc_df.assign(_city_norm=loc_df["city"].map(normalize_text))
        grouped = keyed.groupby("_city_norm")[["lat", "lng"]].mean()
        return grouped.to_dict("index")

    def run_batch(self, tx: pd.DataFrame) -> pd.DataFrame:
        results: dict[int, AgentResult] = {}

        for idx, row in tx.iterrows():
            reasons: list[str] = []
            score = 0.0
            sender = str(row["sender_id"])
            city = _extract_city(row["location"])
            ts = row["timestamp"]

            if city and self.graph.is_profiled(sender) and pd.notna(ts):
                pings = self.graph.get_locations(sender)
                prior = pings[pings["timestamp"] < ts]

                if not prior.empty:
                    last = prior.iloc[-1]
                    last_city = normalize_text(last["city"])
                    if last_city != city:
                        tx_coords = self.city_centroid.get(city)
                        elapsed_h = max((ts - last["timestamp"]).total_seconds() / 3600, 1e-6)
                        if tx_coords is not None:
                            dist_km = _haversine_km(last["lat"], last["lng"], tx_coords["lat"], tx_coords["lng"])
                            speed_kmh = dist_km / elapsed_h
                            if speed_kmh > MAX_PLAUSIBLE_SPEED_KMH:
                                score += 0.45
                                reasons.append(
                                    f"impossible travel: ~{dist_km:.0f}km in {elapsed_h:.1f}h "
                                    f"(~{speed_kmh:.0f}km/h) from last known location {last['city']}"
                                )
                            else:
                                score += 0.15
                                reasons.append(f"new city vs last known location ({last['city']})")
                        else:
                            score += 0.15
                            reasons.append(f"new city vs last known location ({last['city']}, distance unknown)")

                recent = pings[
                    (pings["timestamp"] >= ts - pd.Timedelta(days=RECENT_WINDOW_DAYS)) & (pings["timestamp"] <= ts)
                ]
                if not recent.empty:
                    recent_cities = set(recent["city"].map(normalize_text))
                    if city not in recent_cities:
                        score += 0.10
                        reasons.append("location outside citizen's recent 7-day geo footprint")

            results[idx] = AgentResult(min(score, 1.0), reasons, {})

        return pd.DataFrame(
            {
                "geo_score": {i: r.score for i, r in results.items()},
                "geo_reasons": {i: r.reasons for i, r in results.items()},
            }
        ).reindex(tx.index)


# ---------------------------------------------------------------------------
# Network/Graph Agent: sender-recipient graph structure signals.
# ---------------------------------------------------------------------------
# Not present in the earlier prototype at all. Looks at the transaction graph
# shape rather than any single transaction in isolation:
#
#   * fan-out bursts -- one sender suddenly paying many distinct recipients
#     in a short window (account takeover distributing funds out);
#   * fan-in bursts -- one recipient suddenly collecting from many distinct
#     senders in a short window (mule account aggregating funds);
#   * pass-through / mule chains -- a recipient forwards a similar amount
#     onward again soon after receiving it, i.e. money doesn't stay;
#   * brand-new sender->recipient edges that carry a top-decile amount for
#     that file, which the pure per-sender behavior agent can miss when a
#     sender has too little history to establish its own baseline.
#
# Everything here is computed from the same transactions.csv file being
# scored (train and validation share no citizens, see reply_mirror.identity),
# using sliding-window / sorted-list scans rather than nested per-row
# dataframe filters, so it stays roughly linear in the number of transactions
# per sender/recipient rather than quadratic in the whole file.

FANOUT_WINDOW = pd.Timedelta(hours=48)
FANOUT_MIN_DISTINCT = 4
PASSTHROUGH_WINDOW = pd.Timedelta(hours=48)
PASSTHROUGH_RATIO = 0.20
PASSTHROUGH_MIN_AMOUNT = 50.0


class NetworkAgent:
    name = "network"

    def run_batch(self, tx: pd.DataFrame) -> pd.DataFrame:
        results: dict[int, AgentResult] = {idx: AgentResult(0.0, [], {}) for idx in tx.index}
        valid_ts = tx["timestamp"].notna()

        self._fan_burst(tx[valid_ts & (tx["sender_id"] != "")], "sender_id", "recipient_id", results,
                         "fan-out burst: {n} distinct recipients from this sender in trailing 48h")
        self._fan_burst(tx[valid_ts & (tx["recipient_id"] != "")], "recipient_id", "sender_id", results,
                         "fan-in burst: {n} distinct senders into this recipient in trailing 48h")
        self._pass_through(tx[valid_ts], results)
        self._new_high_value_edge(tx, results)

        return pd.DataFrame(
            {
                "network_score": {i: r.score for i, r in results.items()},
                "network_reasons": {i: r.reasons for i, r in results.items()},
            }
        ).reindex(tx.index)

    @staticmethod
    def _fan_burst(
        df: pd.DataFrame, group_col: str, counterpart_col: str,
        results: dict[int, AgentResult], message: str,
    ) -> None:
        for _, grp in df.groupby(group_col):
            grp = grp.sort_values("timestamp")
            times = grp["timestamp"].to_numpy()
            counterparts = grp[counterpart_col].to_numpy()
            idxs = grp.index.to_numpy()

            lo = 0
            window_counts: dict[str, int] = defaultdict(int)
            distinct = 0
            for hi in range(len(grp)):
                c = counterparts[hi]
                if window_counts[c] == 0 and c:
                    distinct += 1
                window_counts[c] += 1

                while times[hi] - times[lo] > FANOUT_WINDOW:
                    c_lo = counterparts[lo]
                    window_counts[c_lo] -= 1
                    if window_counts[c_lo] == 0 and c_lo:
                        distinct -= 1
                    lo += 1

                if distinct >= FANOUT_MIN_DISTINCT and c:
                    r = results[idxs[hi]]
                    r.score = min(r.score + 0.25, 1.0)
                    r.reasons.append(message.format(n=distinct))

    @staticmethod
    def _pass_through(tx: pd.DataFrame, results: dict[int, AgentResult]) -> None:
        outgoing_by_sender: dict[str, list[tuple[pd.Timestamp, float, int]]] = defaultdict(list)
        for idx, row in tx[tx["sender_id"] != ""].iterrows():
            outgoing_by_sender[row["sender_id"]].append((row["timestamp"], float(row["amount"]), idx))
        for sender in outgoing_by_sender:
            outgoing_by_sender[sender].sort(key=lambda t: t[0])

        for idx, row in tx[tx["recipient_id"] != ""].iterrows():
            amount = float(row["amount"])
            if amount < PASSTHROUGH_MIN_AMOUNT:
                continue
            outs = outgoing_by_sender.get(row["recipient_id"])
            if not outs:
                continue
            out_times = [o[0] for o in outs]
            ts = row["timestamp"]
            start = bisect_right(out_times, ts)
            end = bisect_right(out_times, ts + PASSTHROUGH_WINDOW)
            match = next(
                (o for o in outs[start:end] if abs(o[1] - amount) <= PASSTHROUGH_RATIO * amount), None
            )
            if match is None:
                continue

            r_in = results[idx]
            r_in.score = min(r_in.score + 0.30, 1.0)
            r_in.reasons.append("pass-through pattern: recipient forwards a similar amount onward within 48h (mule-like)")

            r_out = results[match[2]]
            r_out.score = min(r_out.score + 0.30, 1.0)
            r_out.reasons.append("pass-through pattern: this outgoing transfer mirrors a recent incoming amount (mule-like)")

    @staticmethod
    def _new_high_value_edge(tx: pd.DataFrame, results: dict[int, AgentResult]) -> None:
        if tx.empty:
            return
        p90 = tx["amount"].quantile(0.90)
        if p90 <= 0:
            return
        seen_edges: set[tuple[str, str]] = set()
        for idx, row in tx.sort_values("timestamp").iterrows():
            if not row["sender_id"] or not row["recipient_id"]:
                continue
            edge = (row["sender_id"], row["recipient_id"])
            is_new = edge not in seen_edges
            seen_edges.add(edge)
            if is_new and float(row["amount"]) >= p90:
                r = results[idx]
                r.score = min(r.score + 0.15, 1.0)
                r.reasons.append(f"new sender-recipient edge carrying a top-decile amount ({row['amount']:.0f})")


# ---------------------------------------------------------------------------
# Communication Agent (cheap tier): keyword scan of nearby SMS/mail.
# ---------------------------------------------------------------------------
# Runs over every transaction with no LLM call -- just a substring scan of
# the sender's messages in a lookback window before the transaction. This is
# the same idea as the prototype's CommunicationAgent but tied to a specific
# resolved citizen (via IdentityGraph) instead of a loose text-similarity
# guess, and its keyword list can be extended at construction time with
# lexicon the Memory/Drift Agent has learned from LLM escalations on earlier
# levels.
#
# Kept intentionally dumb and fast: this is the first-pass filter. Anything
# it flags as merely "elevated but ambiguous" is exactly what should be
# escalated to the LLM-based Contextual Reasoning Agent, not resolved here.

BASE_SUSPICIOUS_KEYWORDS = [
    # "bit.ly" and a bare "verify" were tried and dropped: in this dataset
    # legitimate city-service SMS (water maintenance, recycling reminders)
    # also use link shorteners and the word "verify", so those alone fire
    # on the majority of transactions and add no discriminative signal --
    # see identity/comm exploration notes. Kept to compound, more specific
    # phrasing instead.
    "security alert", "account lock", "suspicious login", "customs",
    "pay now", "otp", "one-time code", "one time passcode",
    "reset your password", "gift card", "act now immediately",
    "suspend your account", "confirm your identity", "confirm your account",
    "unusual activity", "verify your identity", "verify your account",
    "your account has been", "avoid suspension", "wire the funds",
    "amaz0n", "paypa1", "microsofl", "app1e",
]
COMM_LOOKBACK_DAYS = 5  # tight window: a monthly rent/salary transfer routinely falls within
# *some* 14-day message window by coincidence (confirmed against real data -- a routine
# "Rent payment" kept getting flagged purely because an unrelated phishing email had
# landed two weeks earlier). 5 days keeps this closer to "plausibly drove this
# transaction" while the LLM Contextual Reasoning Agent, which does get the fuller
# 14-day window, is what actually adjudicates borderline cases like this.
COMM_MAX_SCORE = 0.75
COMM_SCORE_PER_HIT = 0.08


class CommAgent:
    name = "comm"

    def __init__(self, graph: IdentityGraph, extra_keywords: list[str] | None = None):
        self.graph = graph
        seen = set()
        self.keywords: list[str] = []
        for kw in BASE_SUSPICIOUS_KEYWORDS + list(extra_keywords or []):
            norm = normalize_text(kw)
            if norm and norm not in seen:
                seen.add(norm)
                self.keywords.append(norm)

    def _nearby_text(self, citizen_id: str, ts: pd.Timestamp) -> str:
        if pd.isna(ts):
            return ""
        lo = ts - pd.Timedelta(days=COMM_LOOKBACK_DAYS)
        parts: list[str] = []

        sms = self.graph.get_sms(citizen_id)
        if not sms.empty:
            window = sms[(sms["timestamp"] >= lo) & (sms["timestamp"] <= ts)]
            parts.extend(window["raw"].tolist())

        mails = self.graph.get_mails(citizen_id)
        if not mails.empty:
            window = mails[(mails["timestamp"] >= lo) & (mails["timestamp"] <= ts)]
            parts.extend(window["raw"].tolist())

        return normalize_text("\n".join(parts))

    def run_batch(self, tx: pd.DataFrame) -> pd.DataFrame:
        results: dict[int, AgentResult] = {}
        text_cache: dict[tuple[str, object], str] = {}

        for idx, row in tx.iterrows():
            sender = str(row["sender_id"])
            if not self.graph.is_profiled(sender):
                results[idx] = AgentResult(0.0, [], {})
                continue

            cache_key = (sender, row["timestamp"])
            text = text_cache.get(cache_key)
            if text is None:
                text = self._nearby_text(sender, row["timestamp"])
                text_cache[cache_key] = text

            hits = [kw for kw in self.keywords if kw in text]
            score = min(len(hits) * COMM_SCORE_PER_HIT, COMM_MAX_SCORE)
            results[idx] = AgentResult(score, hits[:6], {"hit_count": len(hits)})

        return pd.DataFrame(
            {
                "comm_score": {i: r.score for i, r in results.items()},
                "comm_reasons": {i: r.reasons for i, r in results.items()},
            }
        ).reindex(tx.index)


# ---------------------------------------------------------------------------
# Economic Agent: transaction-magnitude signal.
# ---------------------------------------------------------------------------
# Deliberately simple and file-relative rather than fixed absolute
# thresholds (e.g. "$10,000 is high") because amount scale can legitimately
# differ a lot between levels (salaries, typical rents, typical purchase
# sizes all vary). Percentile-within-file bands adapt automatically.
#
# No flat per-transaction-type multiplier: an earlier version added a bump
# for "transfer" specifically, but transfers are the routine salary/rent
# backbone of this dataset (up to 90% of a level's volume), not an inherently
# riskier category, so that bump only diluted the signal. Transaction-type
# novelty for a given sender is already captured more precisely by the
# Behavior Agent's "first-time transaction type for this sender" check.
#
# The top-percentile amount bump is damped for recurring same-pair payments:
# confirmed against real data that a level's highest earner gets their
# routine monthly salary transfer flagged as "top 1% amount" every single
# month purely because their salary is large relative to the file, even
# though it's the same sender, same recipient, and near-identical amount
# every time -- i.e. exactly the opposite of anomalous for that pair. A
# (sender, recipient) pair with prior history at a similar amount has that
# bump reduced sharply; a first-time or amount-shifted payment to a pair
# keeps the full bump.

RECURRING_MIN_PRIOR = 2
RECURRING_AMOUNT_TOLERANCE = 0.15  # within 15% of the pair's own historical median
RECURRING_DAMPING = 0.2


class EconomicAgent:
    name = "economic"

    def run_batch(self, tx: pd.DataFrame) -> pd.DataFrame:
        results: dict[int, AgentResult] = {}
        amounts = tx["amount"]
        p75 = amounts.quantile(0.75) if not amounts.empty else 0.0
        p95 = amounts.quantile(0.95) if not amounts.empty else 0.0
        p99 = amounts.quantile(0.99) if not amounts.empty else 0.0

        pair_history: dict[tuple[str, str], list[float]] = defaultdict(list)
        ordered = tx.sort_values("timestamp", kind="stable")

        for idx, row in ordered.iterrows():
            amount = float(row["amount"])
            reasons: list[str] = []
            score = 0.0

            if p99 > 0 and amount >= p99:
                score, tier = 0.40, "top 1% amount for this level"
            elif p95 > 0 and amount >= p95:
                score, tier = 0.25, "top 5% amount for this level"
            elif p75 > 0 and amount >= p75:
                score, tier = 0.08, "above-median amount for this level"
            else:
                score, tier = 0.0, None

            pair = (str(row["sender_id"]), str(row["recipient_id"]))
            hist = pair_history[pair]
            if tier and len(hist) >= RECURRING_MIN_PRIOR:
                median_hist = sorted(hist)[len(hist) // 2]
                if median_hist > 0 and abs(amount - median_hist) / median_hist <= RECURRING_AMOUNT_TOLERANCE:
                    score *= RECURRING_DAMPING
                    tier = f"{tier} (damped: matches this pair's recurring amount)"

            if tier:
                reasons.append(tier)
            if row["recipient_id"]:
                hist.append(amount)

            results[idx] = AgentResult(min(score, 1.0), reasons, {"amount": amount})

        return pd.DataFrame(
            {
                "economic_score": {i: r.score for i, r in results.items()},
                "economic_reasons": {i: r.reasons for i, r in results.items()},
            }
        ).reindex(tx.index)


# ---------------------------------------------------------------------------
# Contextual Reasoning Agent (expensive tier, LLM-gated).
# ---------------------------------------------------------------------------
# This is the only agent in the whole pipeline that calls an LLM, and even
# then only for the borderline slice of transactions the cheap agents
# couldn't confidently resolve (see orchestrator.py's escalation gate) --
# that's the "reserve expensive reasoning for ambiguous cases" requirement.
#
# Given one transaction plus everything the cheap agents already found, it
# pulls the sender's free-text profile description and nearby SMS/mail (and
# an audio transcript, if one is available), and asks the model to make the
# final judgment call a human fraud analyst would: is there a plausible
# legitimate explanation, or does this look like a lure-driven fraud? It's
# also asked to surface any new lure phrasing it noticed, which feeds back
# into the cheap CommAgent's keyword list for future levels via the
# Memory/Drift Agent -- this is how the "expensive" tier makes the "cheap"
# tier smarter over time instead of being a one-off cost sink.

REASONING_CONTEXT_WINDOW_DAYS = 14
REASONING_MAX_MESSAGES = 6


class ContextualReasoningAgent:
    name = "reasoning"

    def __init__(self, graph: IdentityGraph, session_id: str, budget: int):
        self.graph = graph
        self.session_id = session_id
        self.budget = budget
        self.used = 0
        self._cache: dict[str, AgentResult] = {}

    def available(self) -> bool:
        return self.used < self.budget

    def _nearby_messages(self, citizen_id: str, ts: pd.Timestamp) -> tuple[list[str], list[str]]:
        if pd.isna(ts):
            return [], []
        lo = ts - pd.Timedelta(days=REASONING_CONTEXT_WINDOW_DAYS)
        hi = ts + pd.Timedelta(hours=6)

        sms = self.graph.get_sms(citizen_id)
        sms_texts = []
        if not sms.empty:
            window = sms[(sms["timestamp"] >= lo) & (sms["timestamp"] <= hi)]
            sms_texts = window["raw"].tolist()[:REASONING_MAX_MESSAGES]

        mails = self.graph.get_mails(citizen_id)
        mail_texts = []
        if not mails.empty:
            window = mails[(mails["timestamp"] >= lo) & (mails["timestamp"] <= hi)]
            mail_texts = [m[:800] for m in window["raw"].tolist()[:REASONING_MAX_MESSAGES]]

        return sms_texts, mail_texts

    def run(
        self,
        row: pd.Series,
        evidence: dict[str, AgentResult],
        audio_transcript: str | None = None,
    ) -> AgentResult:
        transaction_id = str(row["transaction_id"])
        if transaction_id in self._cache:
            return self._cache[transaction_id]

        fallback_score = sum(r.score for r in evidence.values()) / max(len(evidence), 1)
        default = {
            "risk": fallback_score,
            "label": "uncertain",
            "confidence": 0.2,
            "reasons": ["llm unavailable or budget exhausted; used mean of cheap signal scores"],
            "new_lure_phrases": [],
        }

        if not self.available():
            result = AgentResult(fallback_score, default["reasons"], {"label": "uncertain", "skipped": True})
            self._cache[transaction_id] = result
            return result

        sender = str(row["sender_id"])
        user = self.graph.get_user(sender)
        profile = user.get("description", "") if user else "(no profile on file: unregistered counterparty)"
        sms_texts, mail_texts = self._nearby_messages(sender, row["timestamp"])

        evidence_packed = {
            name: {"score": round(res.score, 3), "reasons": res.reasons[:4]}
            for name, res in evidence.items()
        }

        audio_block = f"\nNearby voice message transcript:\n{audio_transcript}\n" if audio_transcript else ""

        prompt = f"""
You are the final fraud adjudication analyst for a payments platform in the
year 2087. Decide whether this ONE transaction is most likely FRAUDULENT or
LEGITIMATE.

Transaction:
  id: {row['transaction_id']}
  type: {row['transaction_type']}
  amount: {row['amount']}
  timestamp: {row['timestamp']}
  sender -> recipient: {row['sender_id']} -> {row['recipient_id']}
  payment_method: {row['payment_method']}
  location: {row['location']}
  description: {row['description']}

Sender's self-reported profile (habits, travel, scam susceptibility; may be
in a language other than English):
{profile}

Cheap automated signal scores already computed for this transaction (0-1,
treat as hints, not verdicts):
{json.dumps(evidence_packed, ensure_ascii=False)}

Recent SMS to the sender (within {REASONING_CONTEXT_WINDOW_DAYS} days before this transaction):
{json.dumps(sms_texts, ensure_ascii=False)}

Recent emails to the sender:
{json.dumps(mail_texts, ensure_ascii=False)}
{audio_block}
Weigh whether there is a plausible legitimate explanation (matches a
recurring bill, a stated travel/spending habit from the profile, a routine
payment) against corroborating red flags (a phishing/social-engineering
message shortly before this transaction that the profile says this person
is susceptible to, behavior far outside their norm, a mule-like pattern).

Return strict JSON only, no markdown fences, no commentary outside the JSON:
{{"risk": 0.0, "label": "fraud" or "legitimate", "confidence": 0.0,
  "reasons": ["short reason", "short reason"],
  "new_lure_phrases": ["short phrase from the messages above that reads as a scam/phishing lure, if any"]}}
""".strip()

        self.used += 1
        data = invoke_json(prompt, self.session_id, "ContextualReasoningAgent", default)

        risk = float(data.get("risk", default["risk"]) or default["risk"])
        reasons = [str(r) for r in (data.get("reasons") or [])][:6]
        lure_phrases = [str(p) for p in (data.get("new_lure_phrases") or [])][:6]

        result = AgentResult(
            min(max(risk, 0.0), 1.0),
            reasons,
            {"label": data.get("label"), "confidence": data.get("confidence"), "lure_phrases": lure_phrases},
        )
        self._cache[transaction_id] = result
        return result


# ---------------------------------------------------------------------------
# Audio Agent: transcribes voice messages, but only on demand.
# ---------------------------------------------------------------------------
# Deus Ex is the first level with an `audio/` folder (filenames encode
# `YYYYMMDD_HHMMSS-first_last.mp3`, matched to a citizen by name via
# IdentityGraph). Transcribing all of them upfront for every level would be
# exactly the kind of blanket expensive pass the brief warns against. Instead
# this only transcribes the single nearest clip for a citizen who already has
# a transaction escalated to the Contextual Reasoning Agent, and the result
# is folded into that same LLM call rather than issued as an independent
# score.
#
# Uses faster-whisper locally (already vendored in this repo's env) instead
# of a paid transcription API: CPU-only, no per-call network cost, and the
# clip volume here is small enough that a "tiny" model is fast. Results are
# cached on disk by (path, mtime) so re-running a level doesn't re-transcribe
# audio it already processed.

_AUDIO_NAME_RE = re.compile(r"(\d{8})_(\d{6})-(.+)")
AUDIO_NEARBY_WINDOW_DAYS = 14


class AudioTranscriber:
    def __init__(self, bundle: DatasetBundle, graph: IdentityGraph, model_size: str = WHISPER_MODEL_SIZE):
        self.bundle = bundle
        self.graph = graph
        self.model_size = model_size
        self._model = None
        self._cache = self._load_cache()
        self._index = self._build_index()

    def has_audio(self) -> bool:
        return bool(self._index)

    def _build_index(self) -> dict[str, list[tuple[pd.Timestamp, Path]]]:
        person_key_to_citizen = {}
        for citizen_id, user in self.graph.citizen_to_user.items():
            key = normalize_text(f"{user.get('first_name', '')}_{user.get('last_name', '')}").replace(" ", "_")
            person_key_to_citizen[key] = citizen_id

        index: dict[str, list[tuple[pd.Timestamp, Path]]] = defaultdict(list)
        for path in self.bundle.audio_files:
            m = _AUDIO_NAME_RE.match(path.stem)
            if not m:
                continue
            date_part, time_part, person = m.groups()
            ts = pd.to_datetime(date_part + time_part, format="%Y%m%d%H%M%S", errors="coerce")
            person_key = normalize_text(person).replace(" ", "_")
            citizen_id = person_key_to_citizen.get(person_key)
            if citizen_id and pd.notna(ts):
                index[citizen_id].append((ts, path))

        for citizen_id in index:
            index[citizen_id].sort(key=lambda t: t[0])
        return index

    def nearby_clip(self, citizen_id: str, ts: pd.Timestamp) -> Path | None:
        if pd.isna(ts):
            return None
        best_path, best_delta = None, None
        for clip_ts, path in self._index.get(citizen_id, []):
            delta = abs((clip_ts - ts).total_seconds())
            if delta <= AUDIO_NEARBY_WINDOW_DAYS * 86400 and (best_delta is None or delta < best_delta):
                best_path, best_delta = path, delta
        return best_path

    def _load_cache(self) -> dict[str, dict]:
        if AUDIO_CACHE_PATH.exists():
            with open(AUDIO_CACHE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_cache(self) -> None:
        AUDIO_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(AUDIO_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(self._cache, f, indent=2)

    def _model_instance(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
        return self._model

    def transcribe(self, path: Path) -> str | None:
        key = str(path)
        mtime = path.stat().st_mtime
        cached = self._cache.get(key)
        if cached and cached.get("mtime") == mtime:
            return cached["text"]

        try:
            segments, _ = self._model_instance().transcribe(str(path), beam_size=1)
            text = " ".join(seg.text.strip() for seg in segments).strip()
        except Exception:
            return None

        self._cache[key] = {"mtime": mtime, "text": text}
        self._save_cache()
        return text

    def transcript_for(self, citizen_id: str, ts: pd.Timestamp) -> str | None:
        path = self.nearby_clip(citizen_id, ts)
        if path is None:
            return None
        return self.transcribe(path)
