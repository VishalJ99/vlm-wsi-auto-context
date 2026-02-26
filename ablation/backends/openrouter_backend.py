# ABOUTME: VLM backend for OpenRouter API (Gemini, GPT-4V, Claude, etc.).
# ABOUTME: Uses OPENROUTER_API_KEY or OPENAI_API_KEY env var for authentication.

import asyncio
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import requests
from PIL import Image

from .base import VLMBackend

# Add parent directories to path for imports
sys.path.insert(0, str(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from utils.vlm_utils import build_icl_messages, encode_image_base64, parse_vlm_output


class OpenRouterBackend(VLMBackend):
    """Backend for OpenRouter API supporting multiple cloud VLM providers."""

    BASE_URL = "https://openrouter.ai/api/v1"
    DEFAULT_MODEL = "google/gemini-2.0-flash-001"

    def __init__(
        self,
        model: str = None,
        api_key: str = None,
        timeout: int = 120,
        max_retries: int = 3,
        reasoning_effort: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize OpenRouter backend.

        Args:
            model: OpenRouter model ID (e.g., google/gemini-3-flash-preview)
            api_key: OpenRouter API key (or set OPENROUTER_API_KEY env var)
            timeout: Request timeout in seconds
            max_retries: Number of retry attempts
            reasoning_effort: Optional OpenRouter reasoning effort (low/medium/high)
        """
        super().__init__(
            model=model or self.DEFAULT_MODEL,
            timeout=timeout,
            max_retries=max_retries
        )

        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.reasoning_effort = reasoning_effort
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key required. Set OPENROUTER_API_KEY or OPENAI_API_KEY env var, "
                "or pass api_key parameter."
            )

    def _get_headers(self) -> dict:
        """Get HTTP headers for OpenRouter API."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/wsi-agents",  # Optional: for OpenRouter analytics
        }

    def classify_patch(
        self,
        image_pil: Image.Image,
        prompt: str,
        icl_examples: Optional[List[dict]] = None,
        icl_mode: str = "single",
        allowed_labels: Optional[List[str]] = None
    ) -> Tuple[str, str]:
        """Synchronous patch classification via OpenRouter."""
        # Encode image
        b64_image = encode_image_base64(image_pil, resize=True)

        # Build messages with ICL
        messages = build_icl_messages(b64_image, prompt, icl_examples, icl_mode)

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 500,
            "temperature": 0.0
        }
        if self.reasoning_effort:
            payload["reasoning"] = {"effort": self.reasoning_effort}

        try:
            response = requests.post(
                f"{self.BASE_URL}/chat/completions",
                json=payload,
                headers=self._get_headers(),
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            answer = result["choices"][0]["message"]["content"]
            return parse_vlm_output(answer, allowed_labels)
        except requests.exceptions.Timeout:
            print(f"Warning: OpenRouter request timed out", file=sys.stderr)
            return ("unknown", "NA")
        except requests.exceptions.RequestException as e:
            print(f"Warning: OpenRouter API error: {e}", file=sys.stderr)
            return ("unknown", "NA")
        except (KeyError, IndexError) as e:
            print(f"Warning: Failed to parse OpenRouter response: {e}", file=sys.stderr)
            return ("unknown", "NA")

    async def classify_patch_async(
        self,
        session: aiohttp.ClientSession,
        semaphore: asyncio.Semaphore,
        patch_info: dict,
        prompt: str,
        icl_examples: Optional[List[dict]] = None,
        icl_mode: str = "single",
        allowed_labels: Optional[List[str]] = None,
        debug: bool = False,
        output_dir: Optional[str] = None
    ) -> Tuple[Tuple[int, int], str, str]:
        """Async patch classification with rate limiting."""
        grid_pos = patch_info['grid_pos']
        image_pil = patch_info['image']

        # Debug: save patch to disk
        if debug and output_dir:
            row, col = grid_pos
            wsi_x = patch_info.get('wsi_x', 0)
            wsi_y = patch_info.get('wsi_y', 0)
            debug_dir = os.path.join(output_dir, 'debug_patches')
            os.makedirs(debug_dir, exist_ok=True)
            patch_path = os.path.join(debug_dir, f'{row}_{col}_{wsi_x}_{wsi_y}.png')
            image_pil.save(patch_path)

        # Encode image
        b64_image = encode_image_base64(image_pil, resize=True)

        # Build messages
        messages = build_icl_messages(b64_image, prompt, icl_examples, icl_mode)

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 500,
            "temperature": 0.0
        }
        if self.reasoning_effort:
            payload["reasoning"] = {"effort": self.reasoning_effort}

        async with semaphore:
            for attempt in range(self.max_retries):
                try:
                    async with session.post(
                        f"{self.BASE_URL}/chat/completions",
                        json=payload,
                        headers=self._get_headers(),
                        timeout=aiohttp.ClientTimeout(total=self.timeout)
                    ) as resp:
                        if resp.status != 200:
                            error_text = await resp.text()
                            raise aiohttp.ClientError(f"HTTP {resp.status}: {error_text[:200]}")
                        result = await resp.json()
                        answer = result["choices"][0]["message"]["content"]
                        class_label, quality = parse_vlm_output(answer, allowed_labels)
                        return (grid_pos, class_label, quality)
                except asyncio.TimeoutError:
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
                    else:
                        print(f"Warning: OpenRouter timeout at {grid_pos}", file=sys.stderr)
                        return (grid_pos, "unknown", "NA")
                except Exception as e:
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
                    else:
                        print(f"Warning: OpenRouter error at {grid_pos}: {e}", file=sys.stderr)
                        return (grid_pos, "unknown", "NA")

        return (grid_pos, "unknown", "NA")

    def get_config(self) -> Dict[str, Any]:
        """Return backend configuration (excludes API key for security)."""
        config = super().get_config()
        config.update({
            "backend": "openrouter",
            "base_url": self.BASE_URL,
            "reasoning_effort": self.reasoning_effort,
        })
        return config
