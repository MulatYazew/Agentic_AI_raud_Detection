"""Decision/Orchestrator Agent: fuses signals, escalates, thresholds, decides.

Pipeline per level/split:

  1. Run all five cheap signal agents over every transaction (no LLM).
  2. Fuse them into a preliminary score using weights sourced from the
     Memory/Drift Agent (a decayed average of what worked on prior levels,
     blended with how much each signal actually varies in *this* level).
  3. Escalate only the transactions closest to the likely decision boundary
     to the Contextual Reasoning Agent (LLM), up to a fixed budget --
     these are the cases where cheap signals disagree or sit right at the
     margin, i.e. exactly where expensive judgment earns its keep.
  4. Re-fuse with the reasoning score folded in for escalated rows.
  5. Pick a flag count via a cost-aware, distribution-relative rule with
     hard guardrails so the output can never be empty, complete, or an
     implausible flag rate -- see config.MIN_FLAG_RATE / MAX_FLAG_RATE.
  6. Record this run's weights/thresholds/signal spread back into memory
     so the next level starts from an informed prior.
"""
from __future__ import annotations

import math

import pandas as pd

from .agents import (
    AudioTranscriber,
    BehaviorAgent,
    CommAgent,
    ContextualReasoningAgent,
    EconomicAgent,
    GeoAgent,
    NetworkAgent,
)
from .config import MAX_FLAG_RATE, MIN_FLAG_RATE
from .identity import IdentityGraph
from .llm_client import generate_session_id, get_langfuse_client
from .memory import MemoryStore
from .types import AgentResult, DatasetBundle

CHEAP_SIGNALS = ["behavior", "geo", "network", "comm", "economic"]
MAD_K = 2.5  # outlier cutoff: median + MAD_K * robust-MAD


def _renormalized_fuse(row_scores: dict[str, float], weights: dict[str, float]) -> float:
    total_w = sum(weights.get(name, 0.0) for name in row_scores)
    if total_w <= 1e-9:
        return 0.0
    return sum(weights.get(name, 0.0) * score for name, score in row_scores.items()) / total_w


def _choose_flag_count(scores: pd.Series, prior_quantile: float) -> tuple[int, float]:
    n = len(scores)
    if n == 0:
        return 0, prior_quantile

    median = scores.median()
    mad = (scores - median).abs().median() * 1.4826
    if mad < 1e-9:
        mad = max(float(scores.std(ddof=0)), 1e-6)
    mad_cut = median + MAD_K * mad
    mad_rate = float((scores >= mad_cut).mean())

    if MIN_FLAG_RATE <= mad_rate <= MAX_FLAG_RATE:
        rate = mad_rate
    else:
        rate = min(max(1 - prior_quantile, MIN_FLAG_RATE), MAX_FLAG_RATE)

    target_n = int(round(rate * n))
    target_n = max(target_n, math.ceil(MIN_FLAG_RATE * n))
    target_n = min(target_n, math.floor(MAX_FLAG_RATE * n))
    target_n = max(1, min(target_n, n - 1))
    return target_n, 1 - (target_n / n)


class FraudOrchestrator:
    def __init__(
        self,
        bundle: DatasetBundle,
        graph: IdentityGraph,
        memory: MemoryStore,
        use_llm: bool,
        use_audio: bool,
        llm_budget: int,
    ):
        self.bundle = bundle
        self.graph = graph
        self.memory = memory
        self.use_llm = use_llm
        self.session_id = generate_session_id()

        self.cheap_agents = {
            "behavior": BehaviorAgent(),
            "geo": GeoAgent(graph),
            "network": NetworkAgent(),
            "comm": CommAgent(graph, extra_keywords=memory.get_lexicon()),
            "economic": EconomicAgent(),
        }
        self.reasoning = ContextualReasoningAgent(graph, self.session_id, budget=llm_budget) if use_llm else None
        self.audio = AudioTranscriber(bundle, graph) if (use_audio and use_llm and bundle.audio_files) else None

    def run(self) -> pd.DataFrame:
        tx = self.bundle.transactions
        scored = pd.DataFrame(index=tx.index)
        scored["transaction_id"] = tx["transaction_id"]
        scored["amount"] = tx["amount"]

        for name, agent in self.cheap_agents.items():
            out = agent.run_batch(tx)
            scored[f"{name}_score"] = out[f"{name}_score"]
            scored[f"{name}_reasons"] = out[f"{name}_reasons"]

        prior_weights = self.memory.prior_weights(exclude_level=self.bundle.name)
        current_std = {name: float(scored[f"{name}_score"].std(ddof=0) or 0.0) for name in self.cheap_agents}
        weights = self.memory.adapt_weights(prior_weights, current_std)

        scored["cheap_fused_score"] = [
            _renormalized_fuse({name: scored.at[i, f"{name}_score"] for name in self.cheap_agents}, weights)
            for i in scored.index
        ]

        scored["reasoning_score"] = float("nan")
        scored["reasoning_reasons"] = [[] for _ in range(len(scored))]
        lure_phrases: list[str] = []

        if self.reasoning is not None and self.reasoning.budget > 0 and len(scored) > 0:
            prior_quantile = self.memory.prior_threshold_quantile()
            prelim_n, _ = _choose_flag_count(scored["cheap_fused_score"], prior_quantile)
            ranked = scored.sort_values("cheap_fused_score", ascending=False).reset_index(drop=True)
            cut_pos = min(prelim_n, len(ranked) - 1)
            cut_score = ranked.loc[cut_pos, "cheap_fused_score"]

            distance = (scored["cheap_fused_score"] - cut_score).abs()
            escalate_idx = distance.sort_values().head(self.reasoning.budget).index

            for idx in escalate_idx:
                row = tx.loc[idx]
                evidence = {
                    name: AgentResult(
                        scored.at[idx, f"{name}_score"], scored.at[idx, f"{name}_reasons"], {}
                    )
                    for name in self.cheap_agents
                }
                audio_transcript = None
                sender = str(row["sender_id"])
                if self.audio is not None and self.graph.is_profiled(sender):
                    audio_transcript = self.audio.transcript_for(sender, row["timestamp"])

                result = self.reasoning.run(row, evidence, audio_transcript)
                scored.at[idx, "reasoning_score"] = result.score
                scored.at[idx, "reasoning_reasons"] = result.reasons
                lure_phrases.extend(result.metadata.get("lure_phrases", []) or [])

        def _final_row_score(i: int) -> float:
            row_scores = {name: scored.at[i, f"{name}_score"] for name in self.cheap_agents}
            r_score = scored.at[i, "reasoning_score"]
            if pd.notna(r_score):
                row_scores["reasoning"] = float(r_score)
            return _renormalized_fuse(row_scores, weights)

        scored["final_score"] = [_final_row_score(i) for i in scored.index]

        def _combine_reasons(i: int) -> str:
            parts = []
            for name in list(self.cheap_agents) + ["reasoning"]:
                for reason in scored.at[i, f"{name}_reasons"]:
                    parts.append(f"{name}:{reason}")
            return " | ".join(parts)[:1500]

        scored["reasons"] = [_combine_reasons(i) for i in scored.index]

        self.weights_used = weights
        self.signal_std = current_std
        self.lure_phrases = list(dict.fromkeys(lure_phrases))
        return scored

    def select_flagged(self, scored: pd.DataFrame) -> list[str]:
        prior_quantile = self.memory.prior_threshold_quantile()
        target_n, quantile_used = _choose_flag_count(scored["final_score"], prior_quantile)

        ranked = scored.sort_values(["final_score", "amount"], ascending=[False, False])
        flagged = ranked.head(target_n)["transaction_id"].astype(str).tolist()

        self.threshold_quantile = quantile_used
        self.flag_rate = target_n / max(len(scored), 1)
        return flagged

    def record_memory(self) -> None:
        self.memory.record_level_run(
            level_name=self.bundle.name,
            split=self.bundle.split,
            fusion_weights=self.weights_used,
            signal_std=self.signal_std,
            threshold_quantile=self.threshold_quantile,
            flag_rate=self.flag_rate,
            n_transactions=len(self.bundle.transactions),
            lexicon_additions=self.lure_phrases,
        )

    def flush_tracing(self) -> None:
        client = get_langfuse_client()
        if client is not None:
            try:
                client.flush()
            except Exception:
                pass
