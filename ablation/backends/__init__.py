# ABOUTME: VLM Backend abstraction for patch classification.
# ABOUTME: Supports vLLM (local) and OpenRouter (cloud) providers.
"""
VLM Backend abstraction for patch classification.

Usage:
    from ablation.backends import get_backend

    # Local vLLM (Qwen)
    backend = get_backend("vllm", model="Qwen/Qwen3-VL-8B-Instruct-FP8")

    # OpenRouter (Gemini)
    backend = get_backend("openrouter", model="google/gemini-2.0-flash-001")

    # Using model alias
    backend = get_backend("openrouter", model="gemini-flash")

    # Classify a patch
    result = backend.classify_patch(image_pil, prompt, icl_examples, icl_mode)
"""

from typing import Optional

from .base import VLMBackend
from .vllm_backend import VLLMBackend
from .openrouter_backend import OpenRouterBackend

__all__ = [
    "VLMBackend",
    "VLLMBackend",
    "OpenRouterBackend",
    "get_backend",
    "AVAILABLE_BACKENDS",
]

AVAILABLE_BACKENDS = ["vllm", "openrouter"]


def get_backend(
    name: str,
    model: Optional[str] = None,
    **kwargs
) -> VLMBackend:
    """
    Create a VLM backend by name.

    Args:
        name: Backend type ("vllm" or "openrouter")
        model: Model identifier. For vllm, this is the model name as registered
               in vLLM. For openrouter, this can be a full model ID or an alias
               (e.g., "gemini-flash", "gpt-4o").
        **kwargs: Provider-specific parameters:
            - url (str): For vllm, server URL (default: http://localhost:8000/v1)
            - api_key (str): For openrouter, API key (or use env var)
            - timeout (int): Request timeout in seconds
            - max_retries (int): Retry count on failure

    Returns:
        Configured VLMBackend instance

    Raises:
        ValueError: If backend name is not recognized

    Examples:
        # Local Qwen via vLLM
        backend = get_backend("vllm", model="Qwen/Qwen3-VL-8B-Instruct-FP8")

        # Gemini via OpenRouter
        backend = get_backend("openrouter", model="gemini-flash")

        # Custom vLLM URL
        backend = get_backend("vllm", url="http://gpu-server:8000/v1")
    """
    backends = {
        "vllm": VLLMBackend,
        "openrouter": OpenRouterBackend,
    }

    if name not in backends:
        raise ValueError(
            f"Unknown backend: '{name}'. Available: {list(backends.keys())}"
        )

    return backends[name](model=model, **kwargs)
