# ABOUTME: VLM-based perceptual ranking for ICL patch selection.
# ABOUTME: Uses comparative judgment to select top-k most representative patches.
"""
VLM Ranker - Perceptual Patch Ranking using Vision Language Models

Uses closed-ended prompting to select top-k patches from candidates.
Shows all patches to the VLM in a single message with indices, asks it
to pick the best representatives.

Key insight from LOGBOOK: Context matters - seeing all patches together
enables comparative judgment vs scoring individually.
"""

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from PIL import Image

from .base import PatchRanker

# Add parent directories to path for imports
sys.path.insert(0, str(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from ablation.backends import get_backend
from utils.vlm_utils import encode_image_base64

DEFAULT_MAX_TOKENS = 512
DEFAULT_GEMINI_MODEL = "gemini-3-flash-preview"

# Default prompts for ranking
DEFAULT_FG_PROMPT = """You are selecting the best foreground (tissue) patches for in-context learning.

Below are {n_candidates} candidate patches extracted from histopathology whole slide images.
Each patch is labeled with an index (0 to {max_idx}).

Select exactly {k} patches that:
1. Show the clearest, most representative tissue structure
2. Have good visual quality (not blurry, not too dark/light)
3. Are diverse (different tissue patterns if possible)

Respond with ONLY a JSON object (no markdown, no explanation):
{{"selected_indices": [list of {k} integers], "reasoning": "Explanation for selection"}}"""

DEFAULT_BG_PROMPT = """You are selecting background (non-tissue) patches for in-context learning.

Below are {n_candidates} candidate patches from histopathology whole slide images.
Each patch is labeled with an index (0 to {max_idx}).

Select exactly {k} patches that:
1. Clearly show background (glass/empty space, no tissue)
2. Are unambiguous examples of non-tissue areas
3. Are diverse if possible

Respond with ONLY a JSON object (no markdown, no explanation):
{{"selected_indices": [list of {k} integers], "reasoning": "Brief explanation"}}"""

# Merged prompt for selecting both FG and BG in a single call
# Provides full context of both distributions for better selection
DEFAULT_MERGED_PROMPT = """You are selecting the best foreground (tissue) patches and background (non-tissue) patches for in-context learning.

Below are {n} candidate patches ({n_fg} foreground candidates, {n_bg} background candidates) extracted from histopathology whole slide images.
Each patch is labeled with an index (0 to {max_idx}).

Select exactly {k_fg} FOREGROUND patches that:
1. Show the clearest, most representative tissue structure
2. Have good visual quality (not blurry, not too dark/light)
3. Are diverse (different tissue patterns if possible)

Select exactly {k_bg} BACKGROUND patches that:
1. Clearly show background (glass/empty space, no tissue)
2. Are unambiguous examples of non-tissue areas
3. Are diverse if possible

Respond with ONLY a JSON object (no markdown, no explanation):
{{"foreground_indices": [list of {k_fg} integers], "background_indices": [list of {k_bg} integers], "reasoning": "Brief explanation"}}"""

# Class descriptions for dynamic domain background
CLASS_DESCRIPTIONS = {
    "tissue": """**Tissue Characteristics**:
- Cellular material with visible nuclei
- Fibrous texture with biological structure
- May show part of biological structures like glomeruli, tubules, arteries, vessels or epithelial cells""",

    "background": """**Background Characteristics**:
- Uniform white, off-white, or grey color
- No cellular structure or nuclei
- Smooth texture with minimal variation""",

    "paraffin_mounting_medium": """**Paraffin Artifacts**:
- Opaque material, distinct color from true positive background
- Varies from faint/translucent to solid purple/pink
- May contain streaks or scratches
- May contain dark specks - distinctly different from cells/nuclear features in tissue""",

    "pen_ink_marks": """**Pen/Ink Marks**:
- Opaque colored marks
- Distinct lack of features present in tissue (no nuclei, no cellular structure)
- Non-biological texture"""
}

# Multi-class prompt template with noise-aware design
MULTICLASS_BASE_PROMPT = """# Context: Histopathology Image Analysis

You are analyzing high-magnification patches from a digitized histopathology slide.

## Domain Background

{domain_background}

## Your Task

You are given {n_candidates} patches with noisy labels.
Visually verify and select the best representatives.

## Available Classes

{counts_block}

## Step-by-Step Instructions

**Step 1: Visual Pattern Recognition**
Scan all patches and identify visual features that distinguish each class.

**Step 2: Cross-Verify Labels**
Compare suggested label against visual appearance. Trust what you see if they conflict.

**Step 3: Select Representatives**
For each class, select patches that are visually unambiguous and typical examples.

**Step 4: Document Observations**
Note distinctive features you observed for each class.

## Patches

Below are {n_candidates} candidate patches indexed [0] to [{max_idx}].
Each patch shows its SUGGESTED class label - verify visually before selecting.

## Output Format

Respond with ONLY valid JSON (no markdown). Begin with `{{`:

{response_format}"""


class VLMRanker(PatchRanker):
    """
    VLM-based perceptual ranking for ICL patch selection.

    Uses closed-ended prompting: "Given N patches, pick the top-k
    that best represent tissue." Returns indices + reasoning.

    Supports both local vLLM and OpenRouter backends via the
    existing ablation.backends abstraction.
    """

    def __init__(
        self,
        backend: str = "openrouter",
        model: Optional[str] = None,
        port: int = 8000,
        timeout: int = 120,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        max_retries: int = 3,
        vlm_image_size: Optional[int] = None,
        fg_prompt: Optional[str] = None,
        bg_prompt: Optional[str] = None,
        openrouter_reasoning_effort: Optional[str] = None,
        gemini_use_vertex: bool = True,
        gemini_credentials: Optional[str] = None,
        gemini_location: str = "global",
        gemini_thinking_level: Optional[str] = None,
        gemini_include_thoughts: bool = False,
        **kwargs
    ):
        """
        Initialize VLM ranker.

        Args:
            backend: VLM backend - "vllm", "openrouter", or "gemini"
            model: Model name (uses backend default if not specified)
            port: vLLM server port (only used if backend="vllm")
            timeout: Request timeout in seconds
            max_tokens: Max completion tokens per ranking call
            max_retries: Number of retry attempts on transient failures
            vlm_image_size: Optional square image size for VLM query payloads
            fg_prompt: Custom prompt template for foreground ranking
            bg_prompt: Custom prompt template for background ranking
            openrouter_reasoning_effort: Optional OpenRouter reasoning effort (low/medium/high)
            gemini_use_vertex: Use Google Vertex AI mode for Gemini SDK backend
            gemini_credentials: Path to GCP service account JSON (if using Vertex)
            gemini_location: Google Cloud location for Vertex calls
            gemini_thinking_level: Optional Gemini thinking level (e.g., Low/High)
            gemini_include_thoughts: Include thought summaries in Gemini response
            **kwargs: Additional backend parameters
        """
        super().__init__(**kwargs)

        self.backend_name = backend
        self.port = port
        self.timeout = timeout
        self.max_tokens = max(1, int(max_tokens))
        self.max_retries = max(1, int(max_retries))
        if vlm_image_size is not None:
            parsed_size = int(vlm_image_size)
            if parsed_size < 1:
                raise ValueError("vlm_image_size must be >= 1")
            self.vlm_image_size = parsed_size
        else:
            self.vlm_image_size = None
        self.vlm_backend = None
        self.model = model
        self.openrouter_reasoning_effort = openrouter_reasoning_effort

        # Gemini SDK settings
        self.gemini_use_vertex = bool(gemini_use_vertex)
        self.gemini_credentials = gemini_credentials
        self.gemini_location = gemini_location
        self.gemini_thinking_level = gemini_thinking_level
        self.gemini_include_thoughts = bool(gemini_include_thoughts)
        self._gemini_client = None
        self._gemini_config = None

        if backend == "gemini":
            normalized_model = (model or DEFAULT_GEMINI_MODEL).strip()
            if normalized_model.startswith("google/"):
                normalized_model = normalized_model.split("/", 1)[1]
            self.model = normalized_model
        else:
            # Configure existing HTTP backends
            backend_kwargs = {"timeout": timeout}
            if backend == "vllm":
                backend_kwargs["url"] = f"http://localhost:{port}/v1"
            elif backend == "openrouter" and self.openrouter_reasoning_effort:
                backend_kwargs["reasoning_effort"] = self.openrouter_reasoning_effort
            self.vlm_backend = get_backend(backend, model=model, **backend_kwargs)
            self.model = self.vlm_backend.model

        # Prompts
        self.fg_prompt = fg_prompt or DEFAULT_FG_PROMPT
        self.bg_prompt = bg_prompt or DEFAULT_BG_PROMPT

    def _ensure_gemini_client(self) -> None:
        """Lazy-init Gemini SDK client/config to avoid hard dependency for non-gemini backends."""
        if self._gemini_client is not None and self._gemini_config is not None:
            return

        from google import genai
        from google.genai import types

        if self.gemini_use_vertex:
            creds_path = Path(self.gemini_credentials) if self.gemini_credentials else None
            if creds_path and creds_path.exists():
                with creds_path.open("r", encoding="utf-8") as f:
                    creds = json.load(f)
                project_id = creds.get("project_id")
                if project_id:
                    os.environ["GOOGLE_CLOUD_PROJECT"] = str(project_id)
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(creds_path.absolute())
            elif creds_path and not creds_path.exists():
                if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
                    raise FileNotFoundError(f"Gemini credentials file not found: {self.gemini_credentials}")
            elif not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
                raise ValueError(
                    "Gemini Vertex mode requires credentials. Provide gemini_credentials "
                    "or set GOOGLE_APPLICATION_CREDENTIALS."
                )
            os.environ["GOOGLE_CLOUD_LOCATION"] = self.gemini_location
            os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"

        config_kwargs: Dict[str, Any] = {
            "temperature": 0.0,
            "max_output_tokens": self.max_tokens if self.max_tokens else None,
        }
        if self.gemini_thinking_level:
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_level=self.gemini_thinking_level,
                include_thoughts=self.gemini_include_thoughts,
            )

        self._gemini_client = genai.Client()
        self._gemini_config = types.GenerateContentConfig(**config_kwargs)

    def rank(
        self,
        patches: List[Image.Image],
        patch_names: List[str],
        label: str,
        k: int,
        icl_dir: Path,
    ) -> Tuple[List[int], str]:
        """
        Rank patches using VLM comparative judgment and save selected to ICL dir.

        Args:
            patches: List of candidate patch images
            patch_names: List of filenames for each patch
            label: 'foreground' or 'background'
            k: Number of patches to select
            icl_dir: Output ICL directory

        Returns:
            (selected_indices, reasoning)
        """
        n = len(patches)

        # Handle trivial cases
        if n == 0:
            return [], "No candidates provided"

        if n <= k:
            # All candidates selected
            selected_indices = list(range(n))
            self._save_selected_patches(patches, patch_names, selected_indices, label, icl_dir)
            return selected_indices, f"All {n} candidates selected (n <= k)"

        # Build prompt
        prompt_template = self.fg_prompt if label == "foreground" else self.bg_prompt
        prompt = prompt_template.format(
            n_candidates=n,
            max_idx=n - 1,
            k=k
        )

        # Query VLM with all patches
        response_text = self._query_vlm_batch(patches, prompt)

        # Parse response
        selected_indices, reasoning = self._parse_selection_response(response_text, k, n)

        # Handle parse failure - fall back to first k
        if selected_indices is None:
            selected_indices = list(range(k))
            reasoning = f"Parse failed, using first {k}. Raw response: {response_text[:200]}"

        # Save selected patches
        self._save_selected_patches(patches, patch_names, selected_indices, label, icl_dir)

        return selected_indices, reasoning

    def rank_merged(
        self,
        fg_patches: List[Image.Image],
        fg_names: List[str],
        bg_patches: List[Image.Image],
        bg_names: List[str],
        k_fg: int,
        k_bg: int,
        icl_dir: Path,
    ) -> Tuple[List[int], List[int], str]:
        """
        Rank FG and BG patches together using merged prompt.

        Provides full context of both distributions to VLM for better selection.
        FG patches are indexed 0 to n_fg-1, BG patches are indexed n_fg to n_total-1.

        Args:
            fg_patches: List of foreground candidate images
            fg_names: List of foreground filenames
            bg_patches: List of background candidate images
            bg_names: List of background filenames
            k_fg: Number of foreground patches to select
            k_bg: Number of background patches to select
            icl_dir: Output ICL directory

        Returns:
            (fg_selected_indices, bg_selected_indices, reasoning)
            Indices are in the combined space (FG: 0 to n_fg-1, BG: n_fg to n_total-1)
        """
        n_fg = len(fg_patches)
        n_bg = len(bg_patches)
        n_total = n_fg + n_bg

        # Handle trivial cases
        if n_fg == 0 and n_bg == 0:
            return [], [], "No candidates provided"

        # Combine patches: FG first, then BG
        all_patches = fg_patches + bg_patches
        all_names = fg_names + bg_names

        # Handle case where we have fewer candidates than requested
        actual_k_fg = min(k_fg, n_fg)
        actual_k_bg = min(k_bg, n_bg)

        if n_fg <= k_fg and n_bg <= k_bg:
            # All candidates selected
            fg_indices = list(range(n_fg))
            bg_indices = list(range(n_fg, n_total))
            self._save_selected_patches(fg_patches, fg_names, list(range(n_fg)), 'foreground', icl_dir)
            self._save_selected_patches(bg_patches, bg_names, list(range(n_bg)), 'background', icl_dir)
            return fg_indices, bg_indices, f"All candidates selected (n_fg={n_fg} <= k_fg={k_fg}, n_bg={n_bg} <= k_bg={k_bg})"

        # Build merged prompt
        prompt = DEFAULT_MERGED_PROMPT.format(
            n=n_total,
            n_fg=n_fg,
            n_bg=n_bg,
            max_idx=n_total - 1,
            k_fg=actual_k_fg,
            k_bg=actual_k_bg
        )

        # Query VLM with all patches
        response_text = self._query_vlm_batch(all_patches, prompt)
        self._last_response = response_text

        # Parse response
        fg_indices, bg_indices, reasoning = self._parse_merged_response(
            response_text, actual_k_fg, actual_k_bg, n_total
        )

        # Handle parse failure - fall back to first k from each pool
        if fg_indices is None or bg_indices is None:
            fg_indices = list(range(actual_k_fg))
            bg_indices = list(range(n_fg, n_fg + actual_k_bg))
            reasoning = f"Parse failed, using first k. Raw response: {response_text[:200]}"

        # Convert indices to local space for saving
        fg_local_indices = [i for i in fg_indices if i < n_fg]
        bg_local_indices = [i - n_fg for i in bg_indices if i >= n_fg]

        # Save selected patches
        self._save_selected_patches(fg_patches, fg_names, fg_local_indices, 'foreground', icl_dir)
        self._save_selected_patches(bg_patches, bg_names, bg_local_indices, 'background', icl_dir)

        return fg_indices, bg_indices, reasoning

    def rank_multiclass(
        self,
        patches_by_class: Dict[str, Tuple[List[Image.Image], List[str]]],
        k_per_class: Dict[str, int],
        class_descriptions: Optional[Dict[str, str]],
        icl_dir: Path,
    ) -> Dict[str, Tuple[List[int], str]]:
        """
        Rank patches for multiple classes in a single VLM call.

        Returns global indices over the flat list per class.
        """
        classes = [c for c in k_per_class.keys() if c in patches_by_class]
        if not classes:
            return {}

        # Build flattened patch list with class labels
        all_patches: List[Image.Image] = []
        all_names: List[str] = []
        patch_labels: List[str] = []
        class_to_global: Dict[str, List[int]] = {}

        for class_name in classes:
            patches, names = patches_by_class[class_name]
            class_to_global[class_name] = []
            for local_idx, patch in enumerate(patches):
                global_idx = len(all_patches)
                all_patches.append(patch)
                all_names.append(names[local_idx])
                patch_labels.append(class_name)
                class_to_global[class_name].append(global_idx)

        n_total = len(all_patches)
        if n_total == 0:
            return {c: ([], "No candidates provided") for c in classes}

        # Handle trivial cases: all classes have n <= k
        all_trivial = True
        trivial_results: Dict[str, Tuple[List[int], str]] = {}
        for class_name in classes:
            n_class = len(patches_by_class[class_name][0])
            k_req = k_per_class.get(class_name, 0)
            if n_class == 0:
                trivial_results[class_name] = ([], "No candidates provided")
                continue
            if n_class <= k_req:
                trivial_results[class_name] = (
                    class_to_global[class_name],
                    f"All {n_class} candidates selected (n <= k)"
                )
            else:
                all_trivial = False

        if all_trivial:
            for class_name, (indices, _) in trivial_results.items():
                self._save_selected_patches(all_patches, all_names, indices, class_name, icl_dir)
            return trivial_results

        # Build prompt with actual k per class (cap at available)
        actual_k = {
            class_name: min(k_per_class.get(class_name, 0), len(patches_by_class[class_name][0]))
            for class_name in classes
        }
        prompt = self._build_multiclass_prompt(
            classes,
            actual_k,
            class_descriptions,
            n_candidates=n_total,
            max_idx=n_total - 1,
        )

        # Query VLM with all patches
        response_text = self._query_vlm_batch(all_patches, prompt, patch_labels=patch_labels)
        self._last_response = response_text

        # Parse response (global indices)
        selected_global, reasoning = self._parse_multiclass_response(
            response_text, classes, n_total
        )

        if selected_global is None:
            selected_global = {class_name: [] for class_name in classes}
            reasoning = f"Parse failed, no selections. Raw response: {response_text[:200]}"

        # Save using global indices as-is
        results: Dict[str, Tuple[List[int], str]] = {}
        for class_name in classes:
            global_indices = selected_global.get(class_name, [])
            self._save_selected_patches(all_patches, all_names, global_indices, class_name, icl_dir)
            results[class_name] = (global_indices, reasoning)

        return results

    def _parse_merged_response(
        self,
        response: str,
        k_fg: int,
        k_bg: int,
        n: int
    ) -> Tuple[Optional[List[int]], Optional[List[int]], str]:
        """
        Parse VLM response for merged FG+BG selection.

        Args:
            response: Raw VLM response text
            k_fg: Expected number of FG selections
            k_bg: Expected number of BG selections
            n: Total number of candidates

        Returns:
            (fg_indices or None, bg_indices or None, reasoning)
        """
        try:
            response_clean = response.strip()
            if response_clean.startswith("```"):
                match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_clean, re.DOTALL)
                if match:
                    response_clean = match.group(1)
                else:
                    response_clean = re.sub(r'```(?:json)?', '', response_clean).strip()

            # Find JSON object with both keys
            json_match = re.search(r'\{.*"foreground_indices".*"background_indices".*\}', response_clean, re.DOTALL)
            if not json_match:
                json_match = re.search(r'\{.*"background_indices".*"foreground_indices".*\}', response_clean, re.DOTALL)

            if json_match:
                data = json.loads(json_match.group())
                fg_indices = data.get("foreground_indices", [])
                bg_indices = data.get("background_indices", [])
                reasoning = data.get("reasoning", "")

                # Validate indices
                valid_fg = [i for i in fg_indices if isinstance(i, int) and 0 <= i < n]
                valid_bg = [i for i in bg_indices if isinstance(i, int) and 0 <= i < n]

                return valid_fg[:k_fg], valid_bg[:k_bg], reasoning

        except (json.JSONDecodeError, KeyError, TypeError):
            pass

        return None, None, response

    def _build_multiclass_prompt(
        self,
        classes: List[str],
        k_per_class: Dict[str, int],
        class_descriptions: Optional[Dict[str, str]],
        n_candidates: int,
        max_idx: int,
    ) -> str:
        # Build domain background from CLASS_DESCRIPTIONS for classes present
        domain_parts = []
        for class_name in classes:
            if class_name in CLASS_DESCRIPTIONS:
                domain_parts.append(CLASS_DESCRIPTIONS[class_name])
        domain_background = "\n\n".join(domain_parts) if domain_parts else "No domain descriptions available."

        # Build counts block
        counts_block = "\n".join(
            f"- {class_name}: select {k_per_class.get(class_name, 0)} patches"
            for class_name in classes
        )

        # Build response format with feature_observations
        feature_obs = ", ".join(f'"{c}": "observed features"' for c in classes)
        class_indices = ", ".join(f'"{c}": [indices]' for c in classes)
        response_format = '{\n  "feature_observations": {' + feature_obs + '},\n  ' + class_indices + ',\n  "reasoning": "Brief explanation of selections and any label corrections"\n}'

        return MULTICLASS_BASE_PROMPT.format(
            domain_background=domain_background,
            counts_block=counts_block,
            n_candidates=n_candidates,
            max_idx=max_idx,
            response_format=response_format,
        )

    def _query_vlm_batch(
        self,
        patches: List[Image.Image],
        prompt: str,
        patch_labels: Optional[List[str]] = None,
    ) -> str:
        """
        Send batch of images with indices to VLM.

        Args:
            patches: List of patch images
            prompt: Ranking prompt

        Returns:
            Raw VLM response text
        """
        if self.backend_name == "gemini":
            return self._query_gemini_batch(patches, prompt, patch_labels=patch_labels)

        # Build multi-image message content
        content = []

        for idx, patch in enumerate(patches):
            # Add index label (with optional class label)
            label = f"[Patch {idx}]"
            if patch_labels and idx < len(patch_labels):
                label = f"[Patch {idx} | class: {patch_labels[idx]}]"
            content.append({
                "type": "text",
                "text": label
            })
            # Add image
            query_patch = self._prepare_patch_for_query(patch)
            # Preserve historical smart_resize behavior when no explicit VLM image size is requested.
            b64 = encode_image_base64(query_patch, resize=self.vlm_image_size is None)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"}
            })

        # Add prompt at end
        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]

        # Build payload
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": 0.0
        }
        if self.backend_name == "openrouter" and self.openrouter_reasoning_effort:
            payload["reasoning"] = {"effort": self.openrouter_reasoning_effort}

        # Get backend URL and headers
        if self.backend_name == "vllm":
            url = f"http://localhost:{self.port}/v1/chat/completions"
            headers = {"Content-Type": "application/json"}
        else:
            url = f"{self.vlm_backend.BASE_URL}/chat/completions"
            headers = self.vlm_backend._get_headers()

        # Make request
        try:
            response = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            response_json = response.json()
            return response_json["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            return f"API Error: {e}"
        except (KeyError, IndexError) as e:
            return f"Response parse error: {e}. Full response: {response_json}"

    def _query_gemini_batch(
        self,
        patches: List[Image.Image],
        prompt: str,
        patch_labels: Optional[List[str]] = None,
    ) -> str:
        """Send batch to Gemini SDK (Vertex or AI Studio depending on config)."""
        try:
            self._ensure_gemini_client()
        except Exception as e:
            return f"Gemini init error: {e}"

        contents: List[Any] = []
        for idx, patch in enumerate(patches):
            label = f"[Patch {idx}]"
            if patch_labels and idx < len(patch_labels):
                label = f"[Patch {idx} | class: {patch_labels[idx]}]"
            contents.append(label)
            query_patch = self._prepare_patch_for_query(patch)
            contents.append(query_patch)
        contents.append(prompt)

        for attempt in range(self.max_retries):
            try:
                response = self._gemini_client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=self._gemini_config,
                )
                return (response.text or "").strip()
            except Exception as e:
                if attempt == self.max_retries - 1:
                    return f"Gemini API Error: {e}"
                time.sleep(2 ** attempt)

        return "Gemini API Error: unknown"

    def _prepare_patch_for_query(self, patch: Image.Image) -> Image.Image:
        """Prepare patch image for ranking query transport without mutating original patch."""
        query_patch = patch.convert("RGB")
        if self.vlm_image_size:
            query_patch = query_patch.resize(
                (self.vlm_image_size, self.vlm_image_size),
                resample=Image.BICUBIC,
            )
        return query_patch

    def _parse_multiclass_response(
        self,
        response: str,
        classes: List[str],
        n_total: int,
    ) -> Tuple[Optional[Dict[str, List[int]]], str]:
        """
        Parse VLM response for multiclass selection (global flat-list indices).

        Returns:
            (selected_indices_by_class or None, reasoning)
        """
        try:
            response_clean = response.strip()
            if response_clean.startswith("```"):
                match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_clean, re.DOTALL)
                if match:
                    response_clean = match.group(1)
                else:
                    response_clean = re.sub(r'```(?:json)?', '', response_clean).strip()

            # Try to parse the first JSON object
            json_match = re.search(r'\{.*\}', response_clean, re.DOTALL)
            if not json_match:
                return None, response

            data = json.loads(json_match.group())

            # Extract feature observations if present
            feature_obs = data.get("feature_observations", {})
            if feature_obs:
                self._last_feature_observations = feature_obs

            reasoning = data.get("reasoning", "")

            selections: Dict[str, List[int]] = {}
            for class_name in classes:
                raw_indices = data.get(class_name, [])
                if not isinstance(raw_indices, list):
                    raw_indices = []

                # Validate indices only (global flat list)
                valid = [i for i in raw_indices if isinstance(i, int) and 0 <= i < n_total]
                selections[class_name] = valid

            return selections, reasoning

        except (json.JSONDecodeError, TypeError):
            return None, response

    def _parse_selection_response(
        self,
        response: str,
        k: int,
        n: int
    ) -> Tuple[Optional[List[int]], str]:
        """
        Parse VLM response to extract selected indices and reasoning.

        Args:
            response: Raw VLM response text
            k: Expected number of selections
            n: Total number of candidates

        Returns:
            (selected_indices or None, reasoning)
        """
        # Try to extract JSON
        try:
            # Remove markdown fencing if present
            response_clean = response.strip()
            if response_clean.startswith("```"):
                # Find JSON between fences
                match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_clean, re.DOTALL)
                if match:
                    response_clean = match.group(1)
                else:
                    # Try just removing fences
                    response_clean = re.sub(r'```(?:json)?', '', response_clean).strip()

            # Find JSON object
            json_match = re.search(r'\{[^{}]*"selected_indices"[^{}]*\}', response_clean, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                indices = data.get("selected_indices", [])
                reasoning = data.get("reasoning", "")

                # Validate indices
                valid_indices = [i for i in indices if isinstance(i, int) and 0 <= i < n]

                # Ensure we have exactly k indices
                if len(valid_indices) >= k:
                    return valid_indices[:k], reasoning
                elif len(valid_indices) > 0:
                    # Pad with remaining indices if needed
                    remaining = [i for i in range(n) if i not in valid_indices]
                    while len(valid_indices) < k and remaining:
                        valid_indices.append(remaining.pop(0))
                    return valid_indices, reasoning + " (padded due to insufficient selections)"

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            pass

        return None, response

    def get_config(self) -> Dict[str, Any]:
        """Return ranker configuration for metadata logging."""
        config = super().get_config()
        config.update({
            "backend": self.backend_name,
            "model": self.model,
            "port": self.port if self.backend_name == "vllm" else None,
            "timeout": self.timeout,
            "max_tokens": self.max_tokens,
            "max_retries": self.max_retries,
            "vlm_image_size": self.vlm_image_size,
            "openrouter_reasoning_effort": (
                self.openrouter_reasoning_effort if self.backend_name == "openrouter" else None
            ),
            "gemini_use_vertex": self.gemini_use_vertex if self.backend_name == "gemini" else None,
            "gemini_location": self.gemini_location if self.backend_name == "gemini" else None,
            "gemini_thinking_level": self.gemini_thinking_level if self.backend_name == "gemini" else None,
            "gemini_include_thoughts": self.gemini_include_thoughts if self.backend_name == "gemini" else None,
        })
        return config

    def get_raw_response(self) -> Optional[str]:
        """Get the raw VLM response from the last ranking call."""
        return getattr(self, '_last_response', None)
