# ABOUTME: Patch ranking strategies for ICL example curation.
# ABOUTME: Factory function for creating rankers by name.
"""
ICL Patch Ranking Strategies

This module provides pluggable ranking strategies for selecting the best
ICL examples from candidate patches. Unlike samplers (which operate on masks),
rankers perform perceptual/semantic evaluation of actual images.

Available rankers:
- vlm: VLM-based perceptual ranking (best for foreground)
- mi-diversity: Mutual information diversity selection (best for background)

Usage:
    from ablation.rankers import get_ranker

    # VLM ranker for foreground (OpenRouter)
    fg_ranker = get_ranker(
        "vlm",
        backend="openrouter",
        model="google/gemini-2.0-flash-001"
    )

    # VLM ranker for foreground (local vLLM)
    fg_ranker = get_ranker(
        "vlm",
        backend="vllm",
        port=8000,
        model="Qwen/Qwen3-VL-8B-Instruct-FP8"
    )

    # MI diversity ranker for background
    bg_ranker = get_ranker("mi-diversity", n_bins=32, seed=42)

    # Rank and save patches
    selected_indices, reasoning = fg_ranker.rank(
        patches=candidate_patches,
        patch_names=patch_filenames,
        label='foreground',
        k=3,
        icl_dir=output_path
    )
"""

from typing import Optional

from .base import PatchRanker
from .vlm_ranker import VLMRanker
from .mi_diversity_selector import MIDiversityRanker

__all__ = [
    "PatchRanker",
    "VLMRanker",
    "MIDiversityRanker",
    "get_ranker",
    "AVAILABLE_RANKERS",
]

AVAILABLE_RANKERS = ["vlm", "mi-diversity"]


def get_ranker(
    name: str,
    **kwargs
) -> PatchRanker:
    """
    Create a ranker by name.

    Args:
        name: Ranker type. One of:
            - "vlm": VLM-based perceptual ranking
            - "mi-diversity": Mutual information diversity selection
        **kwargs: Ranker-specific parameters:
            For "vlm":
            - backend (str): "vllm" or "openrouter" (default: "openrouter")
            - model (str): Model name (uses backend default if not specified)
            - port (int): vLLM server port (default: 8000)
            - timeout (int): Request timeout in seconds (default: 120)
            - fg_prompt (str): Custom foreground ranking prompt
            - bg_prompt (str): Custom background ranking prompt
            For "mi-diversity":
            - n_bins (int): Histogram bins per channel (default: 32)
            - seed (int): Random seed for tie-breaking

    Returns:
        Configured PatchRanker instance

    Raises:
        ValueError: If ranker name is not recognized

    Examples:
        # VLM ranker with OpenRouter
        ranker = get_ranker("vlm", backend="openrouter", model="google/gemini-2.0-flash-001")

        # VLM ranker with local vLLM
        ranker = get_ranker("vlm", backend="vllm", port=8000)

        # MI diversity ranker
        ranker = get_ranker("mi-diversity", seed=42)
    """
    rankers = {
        "vlm": VLMRanker,
        "mi-diversity": MIDiversityRanker,
    }

    if name not in rankers:
        raise ValueError(
            f"Unknown ranker: '{name}'. "
            f"Available: {list(rankers.keys())}"
        )

    return rankers[name](**kwargs)
