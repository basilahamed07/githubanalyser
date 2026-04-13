from __future__ import annotations

from copy import deepcopy
from typing import Any


DEFAULT_PRICING_PER_1M_TOKENS = {
    "gpt-4o": {
        "input": 2.50,
        "cached_input": 1.25,
        "output": 10.00,
    }
}


def resolve_model_pricing(model_name: str) -> dict[str, float]:
    normalized = (model_name or "").strip().lower()

    if normalized in DEFAULT_PRICING_PER_1M_TOKENS:
        return deepcopy(DEFAULT_PRICING_PER_1M_TOKENS[normalized])

    if normalized.startswith("gpt-4o"):
        return deepcopy(DEFAULT_PRICING_PER_1M_TOKENS["gpt-4o"])

    return {}


def summarize_usage_metadata(usage_metadata: dict[str, dict[str, Any]]) -> dict[str, Any]:
    models: list[dict[str, Any]] = []
    totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cached_input_tokens": 0,
        "non_cached_input_tokens": 0,
    }

    for model_name, model_usage in usage_metadata.items():
        input_tokens = int(model_usage.get("input_tokens", 0) or 0)
        output_tokens = int(model_usage.get("output_tokens", 0) or 0)
        total_tokens = int(model_usage.get("total_tokens", input_tokens + output_tokens) or 0)

        input_details = model_usage.get("input_token_details") or {}
        cached_input_tokens = int(input_details.get("cache_read", 0) or 0)
        non_cached_input_tokens = max(input_tokens - cached_input_tokens, 0)

        model_summary = {
            "model_name": model_name,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "cached_input_tokens": cached_input_tokens,
            "non_cached_input_tokens": non_cached_input_tokens,
            "input_token_details": input_details,
            "output_token_details": model_usage.get("output_token_details") or {},
        }
        models.append(model_summary)

        totals["input_tokens"] += input_tokens
        totals["output_tokens"] += output_tokens
        totals["total_tokens"] += total_tokens
        totals["cached_input_tokens"] += cached_input_tokens
        totals["non_cached_input_tokens"] += non_cached_input_tokens

    return {
        "models": models,
        "totals": totals,
    }


def estimate_usage_cost(usage_summary: dict[str, Any], pricing_model_name: str) -> dict[str, Any]:
    pricing = resolve_model_pricing(pricing_model_name)
    totals = usage_summary.get("totals", {})

    non_cached_input_tokens = int(totals.get("non_cached_input_tokens", 0) or 0)
    cached_input_tokens = int(totals.get("cached_input_tokens", 0) or 0)
    output_tokens = int(totals.get("output_tokens", 0) or 0)

    if not pricing:
        return {
            "pricing_model": pricing_model_name,
            "pricing_found": False,
            "rates_per_1m_tokens": {},
            "input_cost_usd": None,
            "cached_input_cost_usd": None,
            "output_cost_usd": None,
            "estimated_cost_usd": None,
        }

    input_cost = (non_cached_input_tokens / 1_000_000) * pricing["input"]
    cached_input_cost = (cached_input_tokens / 1_000_000) * pricing.get(
        "cached_input",
        pricing["input"],
    )
    output_cost = (output_tokens / 1_000_000) * pricing["output"]

    return {
        "pricing_model": pricing_model_name,
        "pricing_found": True,
        "rates_per_1m_tokens": pricing,
        "input_cost_usd": input_cost,
        "cached_input_cost_usd": cached_input_cost,
        "output_cost_usd": output_cost,
        "estimated_cost_usd": input_cost + cached_input_cost + output_cost,
    }
