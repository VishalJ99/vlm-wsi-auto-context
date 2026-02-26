# ABOUTME: Abstract base class for VLM backend implementations.
# ABOUTME: Defines interface for sync/async patch classification.

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import asyncio
from PIL import Image


class VLMBackend(ABC):
    """
    Abstract base class for VLM backend providers.

    Backends handle API transport (URL, auth, retries) while message
    construction is shared via utils/vlm_utils.py.
    """

    def __init__(self, model: str, timeout: int = 120, max_retries: int = 3, **kwargs):
        """
        Initialize backend.

        Args:
            model: Model identifier (provider-specific)
            timeout: Request timeout in seconds
            max_retries: Number of retry attempts on failure
            **kwargs: Provider-specific configuration
        """
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries

    @abstractmethod
    def classify_patch(
        self,
        image_pil: Image.Image,
        prompt: str,
        icl_examples: Optional[List[dict]] = None,
        icl_mode: str = "single",
        allowed_labels: Optional[List[str]] = None
    ) -> Tuple[str, str]:
        """
        Synchronous patch classification.

        Args:
            image_pil: PIL Image to classify
            prompt: Classification prompt
            icl_examples: Optional ICL examples [{"image_base64": str, "label": str}]
            icl_mode: ICL structure mode (single, multi-turn, etc.)

        Returns:
            (class_label, quality_label) tuple
        """
        pass

    @abstractmethod
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
        """
        Async patch classification with rate limiting.

        Args:
            session: aiohttp ClientSession for connection pooling
            semaphore: Semaphore for rate limiting
            patch_info: Dict with 'image' (PIL), 'grid_pos', 'wsi_x', 'wsi_y'
            prompt: Classification prompt
            icl_examples: Optional ICL examples
            icl_mode: ICL structure mode
            debug: Save debug patches
            output_dir: Output directory for debug

        Returns:
            (grid_pos, class_label, quality_label) tuple
        """
        pass

    def get_config(self) -> Dict[str, Any]:
        """Return backend configuration for metadata logging."""
        return {
            "type": self.__class__.__name__,
            "model": self.model,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }

    @staticmethod
    def parse_response(answer: str, allowed_labels: Optional[List[str]] = None) -> Tuple[str, str]:
        """Parse VLM response to (class_label, quality_label)."""
        try:
            from utils.vlm_utils import parse_vlm_output
            return parse_vlm_output(answer, allowed_labels)
        except Exception:
            return ("unknown", "NA")
