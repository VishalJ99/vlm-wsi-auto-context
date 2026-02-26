#!/usr/bin/env python3
# ABOUTME: Unified artifact QC pipeline - runs perception, claim extraction, strength classification, and verdict aggregation
# ABOUTME: Takes stage1 output dir as input, outputs to stage2_output with DVC-trackable structure
"""
Unified Artifact QC Pipeline

Runs the full 4-stage artifact detection pipeline:
1. Stage 1: Perception - VLM describes artifacts at 4 orientations
2. Stage 2: Claim-Evidence - Extract structured claims, aggregate across orientations
3. Stage 3: Strength - Classify evidence strength (SD/WA/SA)
4. Stage 4: Verdicts - Aggregate votes to INCLUDE/EXCLUDE/CONFLICT

Usage:
    python run_artifact_qc_pipeline.py --stage1-dir stage1_output/anon_xxx/model/timestamp/
    python run_artifact_qc_pipeline.py --stage1-dir stage1_output/anon_xxx/model/timestamp/ --model google/gemini-2.5-flash
    python run_artifact_qc_pipeline.py --stage1-dir stage1_output/anon_xxx/model/timestamp/ --skip-qc
"""

import argparse
import base64
import io
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from openai import OpenAI
from PIL import Image

# Import reproducibility utilities
from utils.reproducibility import require_clean_state, create_reproduce_command
from utils.wsi_backend import close_wsi, get_pyramid_info, load_wsi, read_region_rgb


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_MODEL = "google/gemini-3-flash-preview"
DEFAULT_MAX_DIM = 1024
ROTATIONS = [0, 90, 180, 270]
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_VLLM_BASE_URL = "http://localhost:8000/v1"
DEFAULT_BACKEND = "openrouter"
DEFAULT_VERTEX_LOCATION = "global"
DEFAULT_VERTEX_CREDENTIALS: Optional[str] = None

# Retry configuration
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2  # seconds
SKIP_QC_PERCEPTION_MESSAGE = (
    "SKIPPED artifact QC model calls; synthetic EXCLUDE-all verdicts used."
)

_VERTEX_CLIENT = None
_VERTEX_CLIENT_CACHE_KEY: Optional[Tuple[Optional[str], str]] = None
_VERTEX_CLIENT_LOCK = threading.Lock()
VERTEX_CREDENTIALS: Optional[str] = DEFAULT_VERTEX_CREDENTIALS
VERTEX_LOCATION: str = DEFAULT_VERTEX_LOCATION

# Standard 7 artifacts
STANDARD_ARTIFACTS = [
    "ink_or_pen_marks",
    "debris",
    "labels",
    "air_bubbles",
    "cracks",
    "tissue_folds",
    "paraffin_mounting_medium",
]

# =============================================================================
# Prompts
# =============================================================================

TASK1_PROMPT = """You are looking at a whole slide image tissue core under medium magnification. Describe what you see. Check for the following:
- Ink or pen marks
- Large, prominent pieces of debris
- Labels with printed text
- Large, prominent air bubbles
- Cracks (in the glass slide, not the tissue)
- Significant tissue folds
- Significant regions of Paraffin/mounting medium (only report if large regions white background also visible)
- Other notable artifacts"""

STAGE2_EXTRACT_PROMPT = """Extract artifact claims from this WSI region description.

For each artifact type, extract:
- present: true if mentioned, false if explicitly absent or not mentioned
- evidence: copy the relevant text verbatim from the input where possible

OUTPUT JSON FORMAT:
Always include these 7 standard keys:
{{
  "ink_or_pen_marks": {{"present": bool, "evidence": "..."}},
  "debris": {{"present": bool, "evidence": "..."}},
  "labels": {{"present": bool, "evidence": "..."}},
  "air_bubbles": {{"present": bool, "evidence": "..."}},
  "cracks": {{"present": bool, "evidence": "..."}},
  "tissue_folds": {{"present": bool, "evidence": "..."}},
  "paraffin_mounting_medium": {{"present": bool, "evidence": "..."}}
}}

If the input has additional artifact sections (e.g., "Streaks/Scratches", "Other Artifacts"), add them as separate keys.

---
INPUT:
{stage1_text}

OUTPUT JSON:"""

STRENGTH_CLASSIFICATION_PROMPT = """You are classifying artifact severity for whole slide image quality control.

Task: For each artifact claim, classify as WA (Weak Agree) or SA (Strong Agree).
- WA = Minor issue, unlikely to significantly affect tissue classifier
- SA = Significant issue, likely to confuse tissue classifier

<artifact_criteria>
ink_or_pen_marks:
  SA: Clearly identified as ink/pen mark (even if unfocused, out-of-focus, or partially visible)
  WA: Hedging/uncertainty ("could be ink but difficult to distinguish from X", "appears to be ink or possibly Y")

labels:
  SA: Clearly identified as label or printed text
  WA: Uncertainty about what is seen ("label or could be X")

debris:
  SA: Large, prominent, or varying sizes
  WA: If **only** references "small scattered specks"

air_bubbles:
  SA: "Large air bubbles", "prominent" or "varying sizes"
  WA: If **only** references "small air bubbles"

cracks:
  SA: Large, significant, major, across slide, prominent
  WA: Small, minor, hairline, tiny

tissue_folds:
  SA: Significant, major, prominent, throughout, severe, multiple
  WA: Small, minor, slight, knick, edge only, single small

paraffin_mounting_medium:
  SA: TWO background classes visible - colored medium AND white background both present
      - Colored region alongside white slide background
      - Distinct bounded area of medium visible against white background
  WA: Single uniform background OR inference only:
      - Only white background visible (normal)
      - Only medium visible, no white background contrast
      - "Grainy/textured" without color contrast
      - Inferred from air bubbles ("air bubbles indicate medium")
      - Streaks/swirls without distinct colored regions
</artifact_criteria>

<claims_to_classify>
{claims_text}
</claims_to_classify>

<output_format>
Return ONLY valid JSON matching this structure:
{output_template}
</output_format>

JSON:"""


# =============================================================================
# WSI Utilities
# =============================================================================

def find_optimal_level(resolutions: dict, bbox_w: int, bbox_h: int, target_max_dim: int) -> Tuple[int, float]:
    """Find optimal pyramid level where bbox max dimension is closest to target."""
    best_level = 0
    best_diff = float('inf')

    for level in range(resolutions['level_count']):
        downsample = resolutions['level_downsamples'][level]
        region_max_dim = max(bbox_w / downsample, bbox_h / downsample)
        diff = abs(region_max_dim - target_max_dim)
        if diff < best_diff:
            best_diff = diff
            best_level = level

    return best_level, resolutions['level_downsamples'][best_level]


def extract_bbox_region(
    wsi_path: str,
    bbox_l0: List[int],
    max_dim: int,
    wsi_reader: str = "cucim",
    force_read_l0: bool = False,
    verbose: bool = True,
) -> Tuple[Image.Image, int, float, str]:
    """Extract bbox region from WSI.

    By default, choose the pyramid level whose projected bbox max dimension is
    closest to ``max_dim``. If ``force_read_l0`` is True, always read from level
    0 and downsample to ``max_dim`` afterward.
    """
    wsi, resolved_wsi_reader = load_wsi(wsi_path, wsi_reader)
    try:
        resolutions = get_pyramid_info(wsi, resolved_wsi_reader)

        x1, y1, x2, y2 = bbox_l0
        bbox_w = x2 - x1
        bbox_h = y2 - y1

        if force_read_l0:
            level = 0
            downsample = 1.0
            read_w = bbox_w
            read_h = bbox_h
            if verbose:
                print(
                    f"      L0 bbox: {bbox_w}x{bbox_h} -> Level 0 (forced); "
                    f"downsampling to max_dim={max_dim}"
                )
        else:
            level, downsample = find_optimal_level(resolutions, bbox_w, bbox_h, max_dim)

            if verbose:
                print(
                    f"      L0 bbox: {bbox_w}x{bbox_h} -> Level "
                    f"{level}/{resolutions['level_count']-1} (downsample {downsample}x)"
                )

            read_w = int(bbox_w / downsample)
            read_h = int(bbox_h / downsample)

        region_np = read_region_rgb(
            wsi,
            resolved_wsi_reader,
            x=x1,
            y=y1,
            width=read_w,
            height=read_h,
            level=level,
        )
        region_pil = Image.fromarray(region_np)
    finally:
        close_wsi(wsi, resolved_wsi_reader)

    current_max = max(region_pil.size)
    if current_max > max_dim:
        scale = max_dim / current_max
        new_w = int(region_pil.size[0] * scale)
        new_h = int(region_pil.size[1] * scale)
        region_pil = region_pil.resize((new_w, new_h), Image.LANCZOS)

    return region_pil, level, downsample, resolved_wsi_reader


def rotate_image(img: Image.Image, degrees: int) -> Image.Image:
    """Rotate PIL Image by specified degrees (0, 90, 180, 270) clockwise."""
    if degrees == 0:
        return img
    return img.rotate(-degrees, expand=True)


# =============================================================================
# VLM Utilities
# =============================================================================

def encode_pil_to_base64(img: Image.Image) -> str:
    """Encode PIL Image to base64 string."""
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def get_client(
    backend: str,
    openrouter_url: str,
    vllm_url: str,
    api_key: Optional[str] = None
) -> OpenAI:
    """Get OpenAI-compatible client for OpenRouter or local vLLM."""
    backend = (backend or DEFAULT_BACKEND).lower()
    if backend == "vllm":
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY"
        return OpenAI(api_key=resolved_key, base_url=vllm_url)

    if backend == "openrouter":
        resolved_key = api_key or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError("Missing OpenRouter API key. Set OPENROUTER_API_KEY/OPENAI_API_KEY or pass --api-key.")
        return OpenAI(api_key=resolved_key, base_url=openrouter_url)

    raise ValueError(f"Unsupported OpenAI-compatible backend: {backend}")


def normalize_model_name_for_backend(model: str, backend: str) -> str:
    """Normalize model IDs when backend naming conventions differ."""
    backend = (backend or DEFAULT_BACKEND).lower()
    if backend == "vertex" and model.startswith("google/"):
        return model.split("/", 1)[1]
    return model


def get_vertex_client(vertex_credentials: Optional[str], vertex_location: str):
    """Lazily initialize a Vertex-enabled google-genai client."""
    global _VERTEX_CLIENT, _VERTEX_CLIENT_CACHE_KEY

    try:
        from google import genai
    except ImportError as exc:
        raise ImportError(
            "google-genai is required for --backend vertex. Install it with `pip install google-genai`."
        ) from exc

    resolved_location = vertex_location or DEFAULT_VERTEX_LOCATION
    resolved_creds_path = str(Path(vertex_credentials).expanduser().resolve()) if vertex_credentials else None
    cache_key = (resolved_creds_path, resolved_location)

    with _VERTEX_CLIENT_LOCK:
        if _VERTEX_CLIENT is not None and _VERTEX_CLIENT_CACHE_KEY == cache_key:
            return _VERTEX_CLIENT

        if vertex_credentials:
            creds_path = Path(vertex_credentials).expanduser()
            if creds_path.exists():
                with open(creds_path, "r", encoding="utf-8") as f:
                    creds = json.load(f)
                project_id = creds.get("project_id")
                if project_id:
                    os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(creds_path.resolve())
            else:
                env_creds = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
                if env_creds:
                    print(
                        f"Warning: Vertex credentials not found at {vertex_credentials}; "
                        "using GOOGLE_APPLICATION_CREDENTIALS from environment",
                    )
                else:
                    raise FileNotFoundError(
                        f"Vertex credentials file not found: {vertex_credentials}. "
                        "Provide a valid --vertex-credentials path or set "
                        "GOOGLE_APPLICATION_CREDENTIALS."
                    )
        elif not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
            raise ValueError(
                "Vertex backend requires credentials. Provide --vertex-credentials "
                "or set GOOGLE_APPLICATION_CREDENTIALS."
            )

        os.environ["GOOGLE_CLOUD_LOCATION"] = resolved_location
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"

        _VERTEX_CLIENT = genai.Client()
        _VERTEX_CLIENT_CACHE_KEY = cache_key
        return _VERTEX_CLIENT


def query_vertex_image(
    img: Image.Image,
    prompt: str,
    model: str,
    vertex_credentials: Optional[str],
    vertex_location: str,
) -> str:
    """Query Vertex Gemini with image + prompt."""
    from google.genai import types

    client = get_vertex_client(vertex_credentials, vertex_location)
    vertex_model = normalize_model_name_for_backend(model, backend="vertex")
    config = types.GenerateContentConfig(temperature=0, max_output_tokens=2000)

    response = client.models.generate_content(
        model=vertex_model,
        contents=[img.convert("RGB"), prompt],
        config=config,
    )
    return (response.text or "").strip()


def query_vertex_text(
    prompt: str,
    model: str,
    vertex_credentials: Optional[str],
    vertex_location: str,
) -> str:
    """Query Vertex Gemini with text-only prompt."""
    from google.genai import types

    client = get_vertex_client(vertex_credentials, vertex_location)
    vertex_model = normalize_model_name_for_backend(model, backend="vertex")
    config = types.GenerateContentConfig(temperature=0, max_output_tokens=2000)

    response = client.models.generate_content(
        model=vertex_model,
        contents=[prompt],
        config=config,
    )
    return (response.text or "").strip()


def query_vlm_image(
    img: Image.Image,
    prompt: str,
    model: str,
    backend: str,
    openrouter_url: str,
    vllm_url: str,
    api_key: Optional[str] = None
) -> str:
    """Query VLM with image and prompt via OpenAI-compatible endpoint (with retry)."""
    backend = (backend or DEFAULT_BACKEND).lower()
    if backend != "vertex":
        base64_image = encode_pil_to_base64(img)
        client = get_client(backend, openrouter_url, vllm_url, api_key)

        messages = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                {"type": "text", "text": prompt}
            ]
        }]

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            if backend == "vertex":
                return query_vertex_image(
                    img,
                    prompt,
                    model,
                    vertex_credentials=VERTEX_CREDENTIALS,
                    vertex_location=VERTEX_LOCATION,
                )

            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=2000,
                temperature=0
            )
            return completion.choices[0].message.content
        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                print(f"        VLM retry {attempt+1}/{MAX_RETRIES} after {delay}s: {e}")
                time.sleep(delay)

    return f"ERROR: {last_error}"


def query_vlm_text(
    prompt: str,
    model: str,
    backend: str,
    openrouter_url: str,
    vllm_url: str,
    api_key: Optional[str] = None
) -> str:
    """Query VLM with text-only prompt via OpenAI-compatible endpoint (with retry)."""
    backend = (backend or DEFAULT_BACKEND).lower()
    if backend != "vertex":
        client = get_client(backend, openrouter_url, vllm_url, api_key)
        messages = [{"role": "user", "content": prompt}]

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            if backend == "vertex":
                return query_vertex_text(
                    prompt,
                    model,
                    vertex_credentials=VERTEX_CREDENTIALS,
                    vertex_location=VERTEX_LOCATION,
                )

            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=2000,
                temperature=0
            )
            return completion.choices[0].message.content
        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                print(f"        VLM retry {attempt+1}/{MAX_RETRIES} after {delay}s: {e}")
                time.sleep(delay)

    return f"ERROR: {last_error}"


# =============================================================================
# Stage 1: Perception
# =============================================================================

def run_stage1_perception(
    img: Image.Image,
    model: str,
    backend: str,
    openrouter_url: str,
    vllm_url: str,
    api_key: Optional[str] = None
) -> Dict[str, str]:
    """
    Run Stage 1 perception on image at all 4 orientations.

    Returns:
        {"0": "response...", "90": "response...", "180": "...", "270": "..."}
    """
    results = {}

    def query_single(rotation: int) -> Tuple[int, str]:
        rotated = rotate_image(img, rotation)
        response = query_vlm_image(
            rotated,
            TASK1_PROMPT,
            model,
            backend,
            openrouter_url,
            vllm_url,
            api_key,
        )
        return rotation, response

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(query_single, r): r for r in ROTATIONS}

        for future in as_completed(futures):
            try:
                rotation, response = future.result()
                results[str(rotation)] = response
                print(f"        {rotation}deg: {len(response)} chars")
            except Exception as e:
                rotation = futures[future]
                results[str(rotation)] = f"ERROR: {e}"

    return results


# =============================================================================
# Stage 2: Claim-Evidence Extraction
# =============================================================================

def extract_claims(
    stage1_text: str,
    model: str,
    backend: str,
    openrouter_url: str,
    vllm_url: str,
    api_key: Optional[str] = None
) -> Dict:
    """Extract claim-evidence JSON from Stage 1 text."""
    prompt = STAGE2_EXTRACT_PROMPT.format(stage1_text=stage1_text)

    try:
        response_text = query_vlm_text(prompt, model, backend, openrouter_url, vllm_url, api_key)

        # Extract JSON from response
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                json_str = json_match.group(0)
            else:
                return {"error": "No JSON found in response", "raw": response_text}

        return json.loads(json_str)

    except json.JSONDecodeError as e:
        return {"error": f"JSON parse error: {e}", "raw": response_text}
    except Exception as e:
        return {"error": str(e)}


def aggregate_orientations(stage2_by_orientation: Dict[str, Dict]) -> Dict[str, Dict[str, str]]:
    """
    Aggregate 4 orientation Stage 2 outputs into single structure.

    Returns:
        {claim: {orientation: evidence, ...}, ...}
        Only includes orientations where present=True
    """
    aggregated = {}

    for orientation, claims in stage2_by_orientation.items():
        if not isinstance(claims, dict):
            continue
        for claim, data in claims.items():
            if isinstance(data, dict) and data.get('present'):
                if claim not in aggregated:
                    aggregated[claim] = {}
                aggregated[claim][orientation] = data.get('evidence', '')

    return aggregated


def run_stage2_extraction(
    perception_results: Dict[str, str],
    model: str,
    backend: str,
    openrouter_url: str,
    vllm_url: str,
    api_key: Optional[str] = None
) -> Tuple[Dict, Dict]:
    """
    Run Stage 2 claim extraction on all orientations (parallel).

    Returns:
        (stage2_by_orientation, aggregated_claims)
    """
    stage2_by_orientation = {}

    def extract_single(orientation: str) -> Tuple[str, Dict]:
        stage1_text = perception_results.get(orientation, '')

        if not stage1_text or stage1_text.startswith('ERROR'):
            return orientation, {"error": "No Stage 1 output"}

        stage2_json = extract_claims(stage1_text, model, backend, openrouter_url, vllm_url, api_key)
        return orientation, stage2_json

    # Parallel extraction
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(extract_single, o): o for o in ['0', '90', '180', '270']}

        for future in as_completed(futures):
            try:
                orientation, stage2_json = future.result()
                stage2_by_orientation[orientation] = stage2_json

                if "error" in stage2_json:
                    print(f"        {orientation}deg: SKIPPED (no Stage 1 output)")
                else:
                    present = [k for k, v in stage2_json.items() if isinstance(v, dict) and v.get("present")]
                    print(f"        {orientation}deg: {', '.join(present) if present else 'none'}")
            except Exception as e:
                orientation = futures[future]
                stage2_by_orientation[orientation] = {"error": str(e)}
                print(f"        {orientation}deg: ERROR - {e}")

    # Aggregate
    aggregated = aggregate_orientations(stage2_by_orientation)

    return stage2_by_orientation, aggregated


# =============================================================================
# Stage 3a: Strength Classification
# =============================================================================

def classify_strength(
    claims: List[Dict],
    model: str,
    backend: str,
    openrouter_url: str,
    vllm_url: str,
    api_key: Optional[str] = None
) -> Dict[str, str]:
    """
    Classify evidence strength for present=true claims via LLM.

    Args:
        claims: List of {"artifact": str, "orientation": str, "evidence": str}

    Returns:
        Dict mapping "artifact_orientation" to "WA" or "SA"
    """
    if not claims:
        return {}

    # Build claims text
    claims_text_lines = []
    for i, c in enumerate(claims, 1):
        claims_text_lines.append(
            f'{i}. {c["artifact"]} @ {c["orientation"]}deg: "{c["evidence"]}"'
        )
    claims_text = "\n".join(claims_text_lines)

    # Build output template
    output_lines = []
    for c in claims:
        output_lines.append(f'  "{c["artifact"]}_{c["orientation"]}": "WA or SA"')
    output_template = "{\n" + ",\n".join(output_lines) + "\n}"

    prompt = STRENGTH_CLASSIFICATION_PROMPT.format(
        claims_text=claims_text,
        output_template=output_template
    )

    try:
        response_text = query_vlm_text(prompt, model, backend, openrouter_url, vllm_url, api_key)

        # Extract JSON
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                json_str = json_match.group(0)
            else:
                print(f"        WARNING: No JSON found in strength response")
                return {}

        result = json.loads(json_str)

        # Validate values
        validated = {}
        for key, value in result.items():
            if value in ("WA", "SA"):
                validated[key] = value
            else:
                validated[key] = "WA"

        return validated

    except Exception as e:
        print(f"        ERROR in strength classification: {e}")
        return {}


def run_stage3_strength(
    stage2_by_orientation: Dict[str, Dict],
    model: str,
    backend: str,
    openrouter_url: str,
    vllm_url: str,
    api_key: Optional[str] = None
) -> Dict[str, Dict[str, str]]:
    """
    Run Stage 3a strength classification.

    Returns:
        {orientation: {artifact: "SD"|"WA"|"SA", ...}, ...}
    """
    strength_results = {str(r): {} for r in ROTATIONS}

    # Collect all present=true claims for batch LLM call
    claims_to_classify = []

    for orientation in ['0', '90', '180', '270']:
        stage2 = stage2_by_orientation.get(orientation, {})

        for artifact in STANDARD_ARTIFACTS:
            claim_data = stage2.get(artifact, {})

            if isinstance(claim_data, dict):
                present = claim_data.get('present', False)
                evidence = claim_data.get('evidence', '')

                if not present:
                    strength_results[orientation][artifact] = "SD"
                elif evidence:
                    claims_to_classify.append({
                        "artifact": artifact,
                        "orientation": orientation,
                        "evidence": evidence
                    })
                else:
                    strength_results[orientation][artifact] = "WA"
            else:
                strength_results[orientation][artifact] = "SD"

    # Classify present=true claims
    if claims_to_classify:
        print(f"        Classifying {len(claims_to_classify)} claims...")
        classifications = classify_strength(
            claims_to_classify,
            model,
            backend,
            openrouter_url,
            vllm_url,
            api_key,
        )

        for claim in claims_to_classify:
            key = f"{claim['artifact']}_{claim['orientation']}"
            strength = classifications.get(key, "WA")
            strength_results[claim['orientation']][claim['artifact']] = strength

    return strength_results


# =============================================================================
# Stage 4: Verdict Aggregation
# =============================================================================

def aggregate_votes(votes: Dict[str, str]) -> Tuple[str, Dict[str, int]]:
    """
    Aggregate votes across orientations to produce a verdict.

    Logic:
        - SA >= 3: INCLUDE (strong evidence of artifact)
        - SA == 2: CONFLICT (tie - needs reasoning agent)
        - SA <= 1: EXCLUDE (insufficient evidence)

    Returns:
        (verdict, counts) where verdict is INCLUDE/EXCLUDE/CONFLICT
    """
    sd = sum(1 for v in votes.values() if v == "SD")
    wa = sum(1 for v in votes.values() if v == "WA")
    sa = sum(1 for v in votes.values() if v == "SA")

    counts = {"SD": sd, "WA": wa, "SA": sa}

    if sa >= 3:
        return "INCLUDE", counts
    elif sa == 2:
        return "CONFLICT", counts  # Needs reasoning agent to decide
    else:
        return "EXCLUDE", counts  # sa == 0 or sa == 1


def run_stage4_verdicts(strength_results: Dict[str, Dict[str, str]]) -> Dict[str, Dict]:
    """
    Run Stage 4 verdict aggregation.

    Returns:
        {artifact: {votes: {...}, counts: {...}, verdict: "..."}, ...}
    """
    verdicts = {}

    for artifact in STANDARD_ARTIFACTS:
        votes = {}
        for orientation in ['0', '90', '180', '270']:
            votes[orientation] = strength_results.get(orientation, {}).get(artifact, "SD")

        verdict, counts = aggregate_votes(votes)
        verdicts[artifact] = {
            "votes": votes,
            "counts": counts,
            "verdict": verdict
        }

    return verdicts


def build_exclude_all_strength() -> Dict[str, Dict[str, str]]:
    """Build synthetic Stage 3 strength output with SD votes for all artifacts."""
    return {
        str(rotation): {artifact: "SD" for artifact in STANDARD_ARTIFACTS}
        for rotation in ROTATIONS
    }


# =============================================================================
# Main Pipeline
# =============================================================================

def process_bbox(
    wsi_path: str,
    bbox_l0: List[int],
    output_dir: Path,
    model: str,
    max_dim: int,
    wsi_reader: str,
    force_read_l0: bool,
    skip_qc: bool,
    backend: str,
    openrouter_url: str,
    vllm_url: str,
    api_key: Optional[str] = None
) -> Dict:
    """
    Process a single bbox through the full pipeline.

    Returns:
        Summary dict with verdicts
    """
    # Extract bbox region
    print(f"      Extracting bbox region...")
    img, level, downsample, resolved_wsi_reader = extract_bbox_region(
        wsi_path,
        bbox_l0,
        max_dim,
        wsi_reader=wsi_reader,
        force_read_l0=force_read_l0,
    )
    print(f"      Extracted: {img.size}")

    # Save bbox image
    img.save(output_dir / "bbox_region.png")
    print(f"      Saved: bbox_region.png")

    if skip_qc:
        print("      Skip QC enabled: writing synthetic EXCLUDE-all artifact outputs...")
        perception_results = {str(rotation): SKIP_QC_PERCEPTION_MESSAGE for rotation in ROTATIONS}
        with open(output_dir / "stage1_artifact_perception.json", 'w') as f:
            json.dump(perception_results, f, indent=2)

        aggregated_claims = {}
        with open(output_dir / "stage2_claim_evidence.json", 'w') as f:
            json.dump(aggregated_claims, f, indent=2)

        strength_results = build_exclude_all_strength()
        with open(output_dir / "stage3_strength.json", 'w') as f:
            json.dump(strength_results, f, indent=2)

        verdicts = run_stage4_verdicts(strength_results)
        with open(output_dir / "stage4_verdicts.json", 'w') as f:
            json.dump(verdicts, f, indent=2)

        include_count = sum(1 for v in verdicts.values() if v["verdict"] == "INCLUDE")
        exclude_count = sum(1 for v in verdicts.values() if v["verdict"] == "EXCLUDE")
        conflict_count = sum(1 for v in verdicts.values() if v["verdict"] == "CONFLICT")

        print(f"      Verdicts: INCLUDE={include_count}, EXCLUDE={exclude_count}, CONFLICT={conflict_count}")
        return {
            "include": include_count,
            "exclude": exclude_count,
            "conflict": conflict_count,
            "wsi_reader": resolved_wsi_reader,
            "skip_qc": True,
        }

    # Stage 1: Perception
    print(f"      Stage 1: Perception...")
    perception_results = run_stage1_perception(
        img, model, backend, openrouter_url, vllm_url, api_key
    )
    with open(output_dir / "stage1_artifact_perception.json", 'w') as f:
        json.dump(perception_results, f, indent=2)

    # Stage 2: Claim-Evidence Extraction
    print(f"      Stage 2: Claim-Evidence Extraction...")
    stage2_by_orientation, aggregated_claims = run_stage2_extraction(
        perception_results, model, backend, openrouter_url, vllm_url, api_key
    )
    with open(output_dir / "stage2_claim_evidence.json", 'w') as f:
        json.dump(aggregated_claims, f, indent=2)

    # Stage 3: Strength Classification
    print(f"      Stage 3: Strength Classification...")
    strength_results = run_stage3_strength(
        stage2_by_orientation, model, backend, openrouter_url, vllm_url, api_key
    )
    with open(output_dir / "stage3_strength.json", 'w') as f:
        json.dump(strength_results, f, indent=2)

    # Stage 4: Verdict Aggregation
    print(f"      Stage 4: Verdict Aggregation...")
    verdicts = run_stage4_verdicts(strength_results)
    with open(output_dir / "stage4_verdicts.json", 'w') as f:
        json.dump(verdicts, f, indent=2)

    # Summary
    include_count = sum(1 for v in verdicts.values() if v["verdict"] == "INCLUDE")
    exclude_count = sum(1 for v in verdicts.values() if v["verdict"] == "EXCLUDE")
    conflict_count = sum(1 for v in verdicts.values() if v["verdict"] == "CONFLICT")

    print(f"      Verdicts: INCLUDE={include_count}, EXCLUDE={exclude_count}, CONFLICT={conflict_count}")

    return {
        "include": include_count,
        "exclude": exclude_count,
        "conflict": conflict_count,
        "wsi_reader": resolved_wsi_reader,
        "skip_qc": False,
    }


def run_pipeline(
    stage1_dir: str,
    output_base: str = "stage2_output",
    model: str = DEFAULT_MODEL,
    max_dim: int = DEFAULT_MAX_DIM,
    wsi_reader: str = "cucim",
    force_read_l0: bool = False,
    skip_qc: bool = False,
    backend: str = DEFAULT_BACKEND,
    openrouter_url: str = OPENROUTER_BASE_URL,
    vllm_url: str = DEFAULT_VLLM_BASE_URL,
    api_key: Optional[str] = None,
    parser: Optional[argparse.ArgumentParser] = None,
    skip_repro_check: bool = False,
    skip_dvc_check: bool = False,
) -> Path:
    """
    Run the full artifact QC pipeline.

    Args:
        stage1_dir: Path to stage1 output directory
        output_base: Base directory for outputs (DVC-tracked)
        model: OpenRouter model to use
        max_dim: Max dimension for bbox extraction
        parser: ArgumentParser for reproduce.txt generation
        skip_repro_check: If True, skip git/DVC state check (for batch mode)

    Returns:
        Path to output directory
    """
    stage1_path = Path(stage1_dir)

    # === REPRODUCIBILITY CHECK ===
    if skip_repro_check:
        state_info = {"bypassed": True, "reason": "batch mode", "git_hash": "batch_deferred"}
    else:
        state_info = require_clean_state([stage1_dir], skip_dvc_check=skip_dvc_check)
        if state_info.get("bypassed"):
            print(f"Warning: Reproducibility check bypassed: {state_info.get('reason')}")

    # Load metadata
    metadata_path = stage1_path / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {stage1_path}")

    with open(metadata_path) as f:
        metadata = json.load(f)

    wsi_path = metadata.get("wsi_path")
    if not wsi_path or not Path(wsi_path).exists():
        raise FileNotFoundError(f"WSI not found: {wsi_path}")

    wsi_id = Path(wsi_path).stem

    # Load bboxes
    bboxes_path = stage1_path / "bboxes.json"
    if not bboxes_path.exists():
        raise FileNotFoundError(f"bboxes.json not found in {stage1_path}")

    with open(bboxes_path) as f:
        bboxes_data = json.load(f)

    # Handle different formats
    if isinstance(bboxes_data, dict) and "detected_regions" in bboxes_data:
        bboxes = bboxes_data["detected_regions"]
    elif isinstance(bboxes_data, list):
        bboxes = bboxes_data
    else:
        raise ValueError(f"Unknown bboxes.json format")

    print("=" * 60)
    print("ARTIFACT QC PIPELINE")
    print("=" * 60)
    print(f"Stage1 dir: {stage1_dir}")
    print(f"WSI: {wsi_path}")
    print(f"Backend: {backend}")
    print(f"Model: {model}")
    print(f"WSI reader (requested): {wsi_reader}")
    print(f"Force read L0: {force_read_l0}")
    print(f"Skip QC: {skip_qc}")
    print(f"Bboxes: {len(bboxes)}")
    print()

    # Create output directory
    model_slug = model.replace("/", "_").replace("-", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_base) / wsi_id / model_slug / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output dir: {output_dir}")
    print()

    # Process each bbox
    summaries = []
    for i, bbox in enumerate(bboxes):
        # Get bbox coordinates
        if isinstance(bbox, dict):
            bbox_l0 = bbox.get("bbox_level0") or bbox.get("bbox") or bbox.get("box_2d")
            label = bbox.get("label", f"bbox_{i}")
        else:
            bbox_l0 = bbox
            label = f"bbox_{i}"

        if not bbox_l0 or len(bbox_l0) != 4:
            print(f"  [{i+1}/{len(bboxes)}] Invalid bbox, skipping")
            continue

        # Create bbox subdir with coordinate format
        x1, y1, x2, y2 = bbox_l0
        bbox_dir_name = f"{x1}_{y1}_{x2}_{y2}"
        bbox_output_dir = output_dir / bbox_dir_name
        bbox_output_dir.mkdir(parents=True, exist_ok=True)

        print(f"  [{i+1}/{len(bboxes)}] {bbox_dir_name}")

        try:
            summary = process_bbox(
                wsi_path,
                bbox_l0,
                bbox_output_dir,
                model,
                max_dim,
                wsi_reader,
                force_read_l0,
                skip_qc,
                backend,
                openrouter_url,
                vllm_url,
                api_key,
            )
            summaries.append({
                "bbox": bbox_dir_name,
                "summary": summary
            })
        except Exception as e:
            print(f"      ERROR: {e}")
            summaries.append({
                "bbox": bbox_dir_name,
                "error": str(e)
            })

        print()

    # Save run metadata with reproducibility info
    resolved_wsi_readers = sorted(
        {
            entry["summary"].get("wsi_reader")
            for entry in summaries
            if isinstance(entry, dict)
            and isinstance(entry.get("summary"), dict)
            and entry["summary"].get("wsi_reader")
        }
    )

    run_metadata = {
        "stage1_dir": str(stage1_dir),
        "wsi_path": wsi_path,
        "wsi_id": wsi_id,
        "backend": backend,
        "model": model,
        "max_dim": max_dim,
        "force_read_l0": bool(force_read_l0),
        "skip_qc": bool(skip_qc),
        "wsi_reader_requested": wsi_reader,
        "wsi_reader_resolved": resolved_wsi_readers[0] if len(resolved_wsi_readers) == 1 else resolved_wsi_readers,
        "timestamp": timestamp,
        "bbox_count": len(bboxes),
        "summaries": summaries,
        "git_hash": state_info.get("git_hash", "unknown"),
        "reproducibility_bypassed": state_info.get("bypassed", False),
        "created_at": datetime.now().isoformat()
    }

    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(run_metadata, f, indent=2)

    # === GENERATE REPRODUCE.TXT ===
    if parser is not None:
        reproduce_path = output_dir / "reproduce.txt"
        create_reproduce_command(parser, str(reproduce_path), git_hash=state_info.get("git_hash"))
        print(f"Saved reproduce.txt: {reproduce_path}")

    print("=" * 60)
    print(f"DONE - Output: {output_dir}")
    print("=" * 60)

    return output_dir


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Unified Artifact QC Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single dir mode
    python run_artifact_qc_pipeline.py --stage1-dir stage1_output/anon_xxx/model/timestamp/

    # Batch mode (parallel processing)
    python run_artifact_qc_pipeline.py --batch dir1/ dir2/ dir3/
"""
    )

    parser.add_argument(
        '--stage1-dir',
        type=str,
        help='Path to stage1 output directory (contains bboxes.json, metadata.json)'
    )
    parser.add_argument(
        '--batch',
        nargs='+',
        help='Process multiple stage1 dirs in parallel (space-delimited). Skips git/DVC checks until end.'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=DEFAULT_MODEL,
        help=f'Model name served by selected backend (default: {DEFAULT_MODEL})'
    )
    parser.add_argument(
        '--backend',
        choices=['openrouter', 'vllm', 'vertex'],
        default=DEFAULT_BACKEND,
        help=f'VLM backend (default: {DEFAULT_BACKEND})'
    )
    parser.add_argument(
        '--openrouter-url',
        type=str,
        default=OPENROUTER_BASE_URL,
        help=f'OpenRouter-compatible base URL (default: {OPENROUTER_BASE_URL})'
    )
    parser.add_argument(
        '--vllm-url',
        type=str,
        default=DEFAULT_VLLM_BASE_URL,
        help=f'Local vLLM OpenAI-compatible base URL (default: {DEFAULT_VLLM_BASE_URL})'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        default=None,
        help='Optional API key override (OpenRouter requires a key; vLLM typically does not).'
    )
    parser.add_argument(
        '--vertex-credentials',
        type=str,
        default=DEFAULT_VERTEX_CREDENTIALS,
        help='Optional Vertex service account JSON path. If unset, use GOOGLE_APPLICATION_CREDENTIALS.',
    )
    parser.add_argument(
        '--vertex-location',
        type=str,
        default=DEFAULT_VERTEX_LOCATION,
        help=f'Vertex location (default: {DEFAULT_VERTEX_LOCATION}).',
    )
    parser.add_argument(
        '--output-base',
        type=str,
        default='stage2_output',
        help='Base output directory (default: stage2_output)'
    )
    parser.add_argument(
        '--max-dim',
        type=int,
        default=DEFAULT_MAX_DIM,
        help=f'Max dimension for bbox extraction (default: {DEFAULT_MAX_DIM})'
    )
    parser.add_argument(
        '--force-read-l0',
        action='store_true',
        help='Force reading bbox crops at level 0, then downsample to --max-dim.'
    )
    parser.add_argument(
        '--skip-qc',
        action='store_true',
        help='Skip artifact QC model calls and synthesize EXCLUDE-all verdict outputs for all bboxes.'
    )
    parser.add_argument(
        '--skip-stage2',
        dest='skip_qc',
        action='store_true',
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--wsi-reader",
        choices=["auto", "openslide", "cucim"],
        default="cucim",
        help="WSI reader backend for bbox extraction (default: cucim).",
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='Max parallel workers for batch mode (default: 4)'
    )
    parser.add_argument(
        '--skip-dvc-check',
        action='store_true',
        help='Bypass DVC clean-state check (still checks git)'
    )

    return parser


def main():
    global VERTEX_CREDENTIALS, VERTEX_LOCATION

    parser = create_parser()
    args = parser.parse_args()
    VERTEX_CREDENTIALS = args.vertex_credentials
    VERTEX_LOCATION = args.vertex_location

    # Validate args
    if not args.batch and not args.stage1_dir:
        parser.error("Either --stage1-dir or --batch is required")

    if args.batch:
        # === BATCH MODE ===
        print("=" * 60)
        print("BATCH MODE - Processing multiple dirs in parallel")
        print("=" * 60)

        # Pre-batch: git/DVC check BEFORE any processing
        print("Checking git/DVC state...")
        state_info = require_clean_state([args.batch[0]], skip_dvc_check=args.skip_dvc_check)
        if state_info.get("bypassed"):
            print(f"  Bypassed: {state_info.get('reason')}")
        else:
            print(f"  Git hash: {state_info.get('git_hash', 'unknown')}")
        print()

        print(f"Dirs to process: {len(args.batch)}")
        print(f"Backend: {args.backend}")
        print(f"Model: {args.model}")
        print(f"Skip QC: {args.skip_qc}")
        print(f"Output base: {args.output_base}")

        results = []

        # Process all dirs in parallel
        n_workers = min(len(args.batch), args.workers)
        if n_workers > 1:
            has_isyntax = False
            for stage1_dir in args.batch:
                metadata_path = Path(stage1_dir) / "metadata.json"
                try:
                    with open(metadata_path) as f:
                        meta = json.load(f)
                    wsi_path = str(meta.get("wsi_path", ""))
                    if Path(wsi_path).suffix.lower() == ".isyntax":
                        has_isyntax = True
                        break
                except Exception:
                    # Keep default workers when metadata probing fails.
                    continue
            if has_isyntax:
                print("Detected .isyntax batch input; forcing workers=1 for pyisyntax stability.")
                n_workers = 1
        print(f"Workers: {n_workers}")
        print()

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(
                    run_pipeline,
                    stage1_dir=d,
                    output_base=args.output_base,
                    model=args.model,
                    max_dim=args.max_dim,
                    wsi_reader=args.wsi_reader,
                    force_read_l0=args.force_read_l0,
                    skip_qc=args.skip_qc,
                    backend=args.backend,
                    openrouter_url=args.openrouter_url,
                    vllm_url=args.vllm_url,
                    api_key=args.api_key,
                    parser=None,  # Skip reproduce.txt in batch mode
                    skip_repro_check=True
                ): d for d in args.batch
            }

            for future in as_completed(futures):
                dir_name = futures[future]
                try:
                    output_dir = future.result()
                    results.append((dir_name, str(output_dir), None))
                    print(f"  DONE: {Path(dir_name).parts[-3]} -> {output_dir}")
                except Exception as e:
                    results.append((dir_name, None, str(e)))
                    print(f"  FAILED: {Path(dir_name).parts[-3]} - {e}")

        # Print summary
        print()
        print("=" * 60)
        print("BATCH COMPLETE")
        print("=" * 60)
        success_count = sum(1 for _, _, err in results if err is None)
        fail_count = len(results) - success_count
        print(f"Success: {success_count}, Failed: {fail_count}")
        print()

        for d, out, err in results:
            dir_short = Path(d).parts[-3] if len(Path(d).parts) >= 3 else d
            if err:
                print(f"  FAILED: {dir_short}")
                print(f"          {err}")
            else:
                print(f"  OK: {dir_short}")
                print(f"      -> {out}")
    else:
        # === SINGLE DIR MODE ===
        run_pipeline(
            stage1_dir=args.stage1_dir,
            output_base=args.output_base,
            model=args.model,
            max_dim=args.max_dim,
            wsi_reader=args.wsi_reader,
            force_read_l0=args.force_read_l0,
            skip_qc=args.skip_qc,
            backend=args.backend,
            openrouter_url=args.openrouter_url,
            vllm_url=args.vllm_url,
            api_key=args.api_key,
            parser=parser,
            skip_dvc_check=args.skip_dvc_check,
        )


if __name__ == "__main__":
    main()
