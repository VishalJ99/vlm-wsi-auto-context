# ABOUTME: Model pricing lookup and review cost estimation utilities.
# ABOUTME: Used by reviewer scripts to attach token-cost estimates to metadata.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class ModelPricing:
    input_per_million_usd: float
    output_per_million_usd: float


# Keep this dictionary easy to extend with exact model IDs or aliases.
# Keys should be lowercase.
MODEL_PRICING_USD_PER_MILLION: Dict[str, ModelPricing] = {
    # Gemini 3 Pro
    "gemini-3-pro-preview": ModelPricing(input_per_million_usd=2.0, output_per_million_usd=12.0),
    "gemini-3-pro": ModelPricing(input_per_million_usd=2.0, output_per_million_usd=12.0),
    "google/gemini-3-pro-preview": ModelPricing(input_per_million_usd=2.0, output_per_million_usd=12.0),
    "google/gemini-3-pro": ModelPricing(input_per_million_usd=2.0, output_per_million_usd=12.0),
    # Gemini 3.1 Pro
    "gemini-3.1-pro-preview": ModelPricing(input_per_million_usd=2.0, output_per_million_usd=12.0),
    "gemini-3.1-pro": ModelPricing(input_per_million_usd=2.0, output_per_million_usd=12.0),
    "google/gemini-3.1-pro-preview": ModelPricing(input_per_million_usd=2.0, output_per_million_usd=12.0),
    "google/gemini-3.1-pro": ModelPricing(input_per_million_usd=2.0, output_per_million_usd=12.0),
    # Gemini 3 Flash
    "gemini-3-flash-preview": ModelPricing(input_per_million_usd=0.5, output_per_million_usd=3.0),
    "gemini-3-flash": ModelPricing(input_per_million_usd=0.5, output_per_million_usd=3.0),
    "google/gemini-3-flash-preview": ModelPricing(input_per_million_usd=0.5, output_per_million_usd=3.0),
    "google/gemini-3-flash": ModelPricing(input_per_million_usd=0.5, output_per_million_usd=3.0),
}


def _as_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def lookup_model_pricing(model_name: str) -> Optional[dict]:
    """Resolve model pricing by exact key or by stripping provider prefix."""
    key = (model_name or "").strip().lower()
    if not key:
        return None

    # Cost estimation is intentionally limited to Gemini models.
    if "gemini" not in key:
        return None

    pricing = MODEL_PRICING_USD_PER_MILLION.get(key)
    matched_key = key
    if pricing is None and "/" in key:
        # Allow provider-prefixed model IDs, e.g. anthropic/..., google/...
        leaf = key.split("/")[-1]
        pricing = MODEL_PRICING_USD_PER_MILLION.get(leaf)
        matched_key = leaf

    if pricing is None:
        return None

    return {
        "matched_key": matched_key,
        "input_per_million_usd": pricing.input_per_million_usd,
        "output_per_million_usd": pricing.output_per_million_usd,
    }


def estimate_review_cost_usd(model_name: str, usage: Optional[dict]) -> Optional[dict]:
    """Estimate review cost; bills thoughts as output tokens."""
    if not usage:
        return None
    pricing = lookup_model_pricing(model_name)
    if pricing is None:
        return None

    prompt_tokens = _as_int(usage.get("prompt_tokens"))
    output_tokens = _as_int(usage.get("output_tokens"))
    thoughts_tokens = _as_int(usage.get("thoughts_tokens")) or 0

    if prompt_tokens is None:
        return None
    if output_tokens is None:
        output_tokens = 0

    billable_output_tokens = output_tokens + thoughts_tokens
    input_cost = (prompt_tokens / 1_000_000.0) * pricing["input_per_million_usd"]
    output_cost = (billable_output_tokens / 1_000_000.0) * pricing["output_per_million_usd"]
    total_cost = input_cost + output_cost

    return {
        "pricing_model_key": pricing["matched_key"],
        "pricing_input_per_million_usd": pricing["input_per_million_usd"],
        "pricing_output_per_million_usd": pricing["output_per_million_usd"],
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "thoughts_tokens": thoughts_tokens,
        "billable_output_tokens": billable_output_tokens,
        "estimated_input_cost_usd": input_cost,
        "estimated_output_cost_usd": output_cost,
        "estimated_total_cost_usd": total_cost,
    }
