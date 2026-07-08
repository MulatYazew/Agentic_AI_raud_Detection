"""Shared LLM client. Only the Contextual Reasoning Agent calls this, and
only for the small borderline slice of transactions the cheap agents can't
confidently resolve -- see orchestrator.py for the escalation gate.

Reuses the same OpenRouter-hosted model + Langfuse tracing setup already
configured in this repo's .env (OPENAI_API_KEY is actually an OpenRouter
key here, matching the prior prototype notebook). Langfuse is
best-effort instrumentation only: any failure to reach it must never break
scoring, so it's wrapped defensively throughout.
"""
from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from typing import Any

from .config import LLM_BASE_URL, LLM_MODEL, TEAM_NAME

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


@lru_cache(maxsize=1)
def get_llm():
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=0.2,
        max_retries=1,
        timeout=25,
        base_url=LLM_BASE_URL,
        api_key=os.getenv("OPENAI_API_KEY"),
        default_headers={
            "HTTP-Referer": "http://localhost",
            "X-Title": "Reply Mirror Fraud Detection",
        },
    )


@lru_cache(maxsize=1)
def get_langfuse_client():
    try:
        from langfuse import Langfuse

        if not os.getenv("LANGFUSE_PUBLIC_KEY"):
            return None
        return Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )
    except Exception:
        return None


def generate_session_id() -> str:
    import ulid

    return f"{TEAM_NAME.replace(' ', '-')}-{ulid.new().str}"


def _extract_json(text: str) -> dict[str, Any]:
    fence = _JSON_FENCE_RE.search(text)
    payload = fence.group(1) if fence else text
    return json.loads(payload)


def invoke_json(prompt: str, session_id: str, agent_name: str, default: dict[str, Any]) -> dict[str, Any]:
    """Call the LLM once, parse strict JSON, fall back to `default` on any failure."""
    from langchain_core.messages import HumanMessage

    try:
        config: dict[str, Any] = {
            "metadata": {"langfuse_session_id": session_id, "agent_name": agent_name}
        }
        try:
            from langfuse.langchain import CallbackHandler

            if get_langfuse_client() is not None:
                config["callbacks"] = [CallbackHandler()]
        except Exception:
            pass

        resp = get_llm().invoke([HumanMessage(content=prompt)], config=config)
        text = resp.content if hasattr(resp, "content") else str(resp)
        return _extract_json(text)
    except Exception:
        return default
