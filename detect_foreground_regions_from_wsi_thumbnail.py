#!/usr/bin/env python3
# ABOUTME: Stage 1 - Detect tissue bounding boxes from WSI thumbnail using VLM backends.
# ABOUTME: Outputs to stage1_output/{wsi_id}/{model}/{timestamp}/ with full reproducibility.
"""
Stage 1: Detect Foreground Regions from WSI Thumbnail

Uses a VLM backend (OpenRouter, local vLLM, or Vertex AI Gemini) to detect tissue
bounding boxes in a WSI thumbnail,
converts normalized coordinates to level 0 WSI pixel coordinates.

Output structure:
    stage1_output/{wsi_id}/{model}/{timestamp}/
        thumbnail.png          - What VLM sees
        vlm_responses/         - Raw VLM responses per orientation
        bboxes.json            - Detected bboxes with L0 coords
        bbox_overlay.png       - Thumbnail with bboxes drawn
        metadata.json          - Full reproducibility metadata
        reproduce.txt          - Command to replicate
        [bbox_regions/]        - Optional L0 bbox crops (with --save-bbox-region)

Usage:
    python detect_foreground_regions_from_wsi_thumbnail.py --wsi anon_6b692277.svs
    python detect_foreground_regions_from_wsi_thumbnail.py --wsi slide.svs --max-dim 2048
    python detect_foreground_regions_from_wsi_thumbnail.py --wsi slide.svs --model google/gemini-3-pro-preview
"""

import argparse
import ast
import base64
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from openai import OpenAI
from PIL import Image

from utils.wsi_backend import (
    close_wsi,
    get_pyramid_info,
    is_isyntax_path,
    load_wsi,
    read_region_rgb,
)


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_PROMPT_GEMINI = """You are looking at a whole slide image containing tissue core biopsies at low magnification.

First, count how many separate tissue cores you see in the image.
Then, draw a bounding box around each tissue core.

Output a JSON array of bounding boxes in normalized 0-1000 coordinates: [{"box_2d": [y_min, x_min, y_max, x_max]}]

Each tissue core **must** have its own separate bounding box - do **not** merge multiple cores into a single box."""

DEFAULT_PROMPT_QWEN = """You are looking at a whole slide image containing tissue core biopsies at low magnification.

Locate each separate tissue core and output a JSON array in normalized 0-1000 coordinates:
[{"bbox_2d": [x_min, y_min, x_max, y_max], "label": "tissue"}]

Rules:
- Use [x_min, y_min, x_max, y_max]
- One box per tissue core (do not merge multiple cores)
- Output JSON only"""

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_VLLM_BASE_URL = "http://localhost:8000/v1"
DEFAULT_BACKEND = "openrouter"
DEFAULT_MODEL = "google/gemini-3-flash-preview"
DEFAULT_VERTEX_MODEL = "gemini-3-flash-preview"
DEFAULT_VERTEX_LOCATION = "global"
DEFAULT_VERTEX_CREDENTIALS: Optional[str] = None
DEFAULT_MAX_DIM = 1024
DEFAULT_MERGE_OVERLAP_THRESHOLD = 0.2
OUTPUT_BASE_DIR = "stage1_output"

_VERTEX_CLIENT = None
_VERTEX_CLIENT_CACHE_KEY: Optional[Tuple[Optional[str], str]] = None
_VERTEX_CLIENT_LOCK = threading.Lock()


# =============================================================================
# Helper Functions
# =============================================================================

def sanitize_model_name(model: str) -> str:
    """Sanitize model name for use in file paths."""
    return model.replace("/", "_").replace(":", "_").replace("-", "_")


def rotate_image(img: Image.Image, degrees: int) -> Image.Image:
    """Rotate PIL Image by specified degrees (0, 90, 180, 270) clockwise."""
    if degrees == 0:
        return img
    # PIL rotates counter-clockwise, so negate for clockwise rotation
    return img.rotate(-degrees, expand=True)


def transform_bbox_to_rot0(bbox: List[float], rotation: int) -> Tuple[float, float, float, float]:
    """
    Transform bbox from rotated space back to 0° space.

    BBox format: [y_min, x_min, y_max, x_max] (Gemini format) in 0-1000 normalized coords.

    For PIL rotate(-degrees, expand=True) which rotates clockwise:
    - 90° CW: original (x,y) -> rotated (y, W-x), inverse: (x', y') -> (W-y', x')
    - 180°: original (x,y) -> rotated (W-x, H-y), inverse: same
    - 270° CW: original (x,y) -> rotated (H-y, x), inverse: (y', H-x')

    In normalized 0-1000 coords, W=H=1000.
    """
    y1, x1, y2, x2 = bbox

    if rotation == 0:
        return (y1, x1, y2, x2)
    elif rotation == 90:
        # Inverse of 90° CW rotation
        new_y1 = 1000 - x2
        new_x1 = y1
        new_y2 = 1000 - x1
        new_x2 = y2
        return (new_y1, new_x1, new_y2, new_x2)
    elif rotation == 180:
        # Inverse of 180° rotation
        new_y1 = 1000 - y2
        new_x1 = 1000 - x2
        new_y2 = 1000 - y1
        new_x2 = 1000 - x1
        return (new_y1, new_x1, new_y2, new_x2)
    elif rotation == 270:
        # Inverse of 270° CW rotation (= 90° CCW)
        new_y1 = x1
        new_x1 = 1000 - y2
        new_y2 = x2
        new_x2 = 1000 - y1
        return (new_y1, new_x1, new_y2, new_x2)
    else:
        raise ValueError(f"Unsupported rotation: {rotation}")


def bbox_overlaps(a: Tuple, b: Tuple) -> bool:
    """Check if two bboxes overlap. Format: (y1, x1, y2, x2)."""
    return not (a[3] <= b[1] or b[3] <= a[1] or a[2] <= b[0] or b[2] <= a[0])


def bbox_overlap_min_area(a: Tuple, b: Tuple) -> float:
    """
    Compute overlap as intersection area divided by min(box area).

    Returns:
        float in [0, 1]. 0 if no overlap or if any bbox has zero area.
    """
    inter_y1 = max(a[0], b[0])
    inter_x1 = max(a[1], b[1])
    inter_y2 = min(a[2], b[2])
    inter_x2 = min(a[3], b[3])

    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_area = inter_h * inter_w

    if inter_area <= 0:
        return 0.0

    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    min_area = min(area_a, area_b)

    if min_area <= 0:
        return 0.0

    return inter_area / min_area


def is_giant_bbox(bboxes: List, threshold: float = 0.8) -> bool:
    """
    Check if single bbox covers >threshold of normalized 0-1000 space.

    Used to detect when VLM returns a single bbox covering most/all of the thumbnail,
    which indicates it failed to properly segment individual tissue regions.
    """
    if not bboxes or len(bboxes) != 1:
        return False

    bbox = bboxes[0]
    # Extract coords (handle dict or list format)
    if isinstance(bbox, dict):
        coords = bbox.get("box_2d") or bbox.get("bbox_2d") or bbox.get("bbox")
    else:
        coords = bbox

    if not coords or len(coords) != 4:
        return False

    y1, x1, y2, x2 = coords
    area = (x2 - x1) * (y2 - y1)
    total_area = 1000 * 1000
    coverage = area / total_area

    return coverage > threshold


def merge_overlapping_bboxes(
    bboxes: List[Tuple],
    overlap_threshold: float = DEFAULT_MERGE_OVERLAP_THRESHOLD
) -> List[Tuple]:
    """
    Merge bboxes into unified hulls using iterative greedy merge.

    Uses overlap defined as intersection area / min(box area).
    """
    if not bboxes:
        return []

    merged = [tuple(b) for b in bboxes]
    changed = True

    while changed:
        changed = False
        new_merged = []
        used = set()

        for i, box_a in enumerate(merged):
            if i in used:
                continue

            hull = list(box_a)
            for j, box_b in enumerate(merged):
                if j <= i or j in used:
                    continue

                if bbox_overlap_min_area(tuple(hull), box_b) >= overlap_threshold:
                    # Merge: expand hull to encompass both
                    hull = [
                        min(hull[0], box_b[0]),  # y1
                        min(hull[1], box_b[1]),  # x1
                        max(hull[2], box_b[2]),  # y2
                        max(hull[3], box_b[3]),  # x2
                    ]
                    used.add(j)
                    changed = True

            new_merged.append(tuple(hull))

        merged = new_merged

    return merged


def parse_json(json_output: str) -> str:
    """Remove markdown fencing from JSON output."""
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith("```"):
            json_output = "\n".join(lines[i+1:])
            json_output = json_output.split("```")[0]
            break
    return json_output.strip()


def resolve_api_settings(
    backend: str,
    openrouter_url: str,
    vllm_url: str,
    api_key: Optional[str] = None
) -> Tuple[str, str]:
    """Resolve base URL and API key for OpenAI-compatible backends."""
    backend = (backend or DEFAULT_BACKEND).lower()
    if backend == "vllm":
        key = api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY"
        return vllm_url, key

    if backend == "openrouter":
        key = api_key or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("Missing OpenRouter API key. Set OPENROUTER_API_KEY/OPENAI_API_KEY or pass --api-key.")
        return openrouter_url, key

    raise ValueError(f"Unsupported OpenAI-compatible backend: {backend}")


def normalize_model_name_for_backend(model: str, backend: str) -> str:
    """
    Normalize model IDs when switching between backend naming conventions.

    OpenRouter often uses provider-prefixed IDs (e.g. google/gemini-3-flash-preview),
    while Vertex expects the bare Gemini model name.
    """
    backend = (backend or DEFAULT_BACKEND).lower()
    if backend == "vertex" and model.startswith("google/"):
        return model.split("/", 1)[1]
    return model


def get_vertex_client(vertex_credentials: Optional[str], vertex_location: str):
    """Lazily initialize and cache a Vertex-enabled google-genai client."""
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
                        file=sys.stderr,
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


def query_vertex_contents(
    model: str,
    contents: List,
    vertex_credentials: Optional[str],
    vertex_location: str,
    temperature: Optional[float] = None,
) -> str:
    """Call Gemini through Vertex AI via google-genai SDK."""
    from google.genai import types

    client = get_vertex_client(vertex_credentials, vertex_location)
    vertex_model = normalize_model_name_for_backend(model, backend="vertex")
    config = None
    if temperature is not None:
        config = types.GenerateContentConfig(temperature=temperature)

    response = client.models.generate_content(
        model=vertex_model,
        contents=contents,
        config=config,
    )
    return (response.text or "").strip()


def infer_bbox_coord_order(coord_order_arg: str, model: str, prompt: str) -> str:
    """Infer bbox coordinate order. Returns 'xyxy' or 'yxxy'."""
    if coord_order_arg in {"xyxy", "yxxy"}:
        return coord_order_arg

    prompt_l = (prompt or "").lower()
    if re.search(r"\[\s*x_min\s*,\s*y_min\s*,\s*x_max\s*,\s*y_max\s*\]", prompt_l):
        return "xyxy"
    if re.search(r"\[\s*y_min\s*,\s*x_min\s*,\s*y_max\s*,\s*x_max\s*\]", prompt_l):
        return "yxxy"
    if re.search(r"\[\s*x\s*,\s*y\s*,\s*x\s*,\s*y\s*\]", prompt_l):
        return "xyxy"
    if re.search(r"\[\s*y\s*,\s*x\s*,\s*y\s*,\s*x\s*\]", prompt_l):
        return "yxxy"

    # Fallback: infer by first mention order of x_min vs y_min
    ix = prompt_l.find("x_min")
    iy = prompt_l.find("y_min")
    if ix != -1 and iy != -1:
        return "xyxy" if ix < iy else "yxxy"

    if "qwen" in (model or "").lower():
        return "xyxy"
    return "yxxy"


def normalize_bbox_to_yxxy(bbox: List[float], coord_order: str) -> List[float]:
    """
    Normalize bbox to internal [y1, x1, y2, x2] format.

    Input:
    - coord_order='yxxy': [y1, x1, y2, x2]
    - coord_order='xyxy': [x1, y1, x2, y2]
    """
    if coord_order == "xyxy":
        x1, y1, x2, y2 = bbox
        return [y1, x1, y2, x2]
    return list(bbox)


def pad_bbox_normalized(bbox_norm: List[float], padding_fraction: float) -> List[float]:
    """Expand a normalized bbox by a percentage in all dimensions."""
    x1, y1, x2, y2 = bbox_norm
    width = x2 - x1
    height = y2 - y1

    pad_x = width * padding_fraction
    pad_y = height * padding_fraction

    new_x1 = max(0, x1 - pad_x)
    new_y1 = max(0, y1 - pad_y)
    new_x2 = min(1000, x2 + pad_x)
    new_y2 = min(1000, y2 + pad_y)

    return [new_x1, new_y1, new_x2, new_y2]


def normalized_to_level0(bbox_norm: List[float], wsi_w: int, wsi_h: int) -> Tuple[int, int, int, int]:
    """Convert normalized (0-1000) bbox to level 0 WSI pixel coords."""
    x1 = int(bbox_norm[0] / 1000 * wsi_w)
    y1 = int(bbox_norm[1] / 1000 * wsi_h)
    x2 = int(bbox_norm[2] / 1000 * wsi_w)
    y2 = int(bbox_norm[3] / 1000 * wsi_h)

    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1

    return (x1, y1, x2, y2)


def normalized_to_thumbnail(bbox_norm: List[float], thumb_w: int, thumb_h: int) -> Tuple[int, int, int, int]:
    """Convert normalized (0-1000) bbox to thumbnail pixel coords."""
    x1 = int(bbox_norm[0] / 1000 * thumb_w)
    y1 = int(bbox_norm[1] / 1000 * thumb_h)
    x2 = int(bbox_norm[2] / 1000 * thumb_w)
    y2 = int(bbox_norm[3] / 1000 * thumb_h)

    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1

    return (x1, y1, x2, y2)


# =============================================================================
# Core Functions
# =============================================================================

def _run_with_progress(func, message: str):
    """Run a function with live elapsed time indicator."""
    import time
    import threading
    import sys

    stop_event = threading.Event()
    result = [None]
    exception = [None]

    def worker():
        try:
            result[0] = func()
        except Exception as e:
            exception[0] = e
        finally:
            stop_event.set()

    def progress():
        start = time.time()
        while not stop_event.is_set():
            elapsed = time.time() - start
            sys.stdout.write(f"\r{message} ({elapsed:.0f}s)...")
            sys.stdout.flush()
            stop_event.wait(timeout=1.0)
        elapsed = time.time() - start
        sys.stdout.write(f"\r{message} ({elapsed:.1f}s) done\n")
        sys.stdout.flush()

    worker_thread = threading.Thread(target=worker)
    progress_thread = threading.Thread(target=progress)

    worker_thread.start()
    progress_thread.start()

    worker_thread.join()
    progress_thread.join()

    if exception[0]:
        raise exception[0]
    return result[0]


def create_thumbnail(
    wsi_path: str,
    max_dim: int = DEFAULT_MAX_DIM,
    wsi_reader: str = "cucim",
) -> Tuple[Image.Image, int, int, str]:
    """
    Create thumbnail from WSI using highest pyramid level (fastest).

    Returns:
        (thumbnail_pil, wsi_width, wsi_height, resolved_wsi_reader)
    """
    # pyisyntax can segfault when used through the threaded progress wrapper.
    if is_isyntax_path(wsi_path):
        print("Opening WSI...")
        wsi, resolved_wsi_reader = load_wsi(wsi_path, wsi_reader)
        print("Opening WSI done")
    else:
        wsi, resolved_wsi_reader = _run_with_progress(
            lambda: load_wsi(wsi_path, wsi_reader),
            "Opening WSI",
        )

    try:
        pyramid = get_pyramid_info(wsi, resolved_wsi_reader)
        wsi_width, wsi_height = pyramid["level_dimensions"][0]
        print(f"WSI Level 0: {wsi_width} x {wsi_height} px")

        # Use highest pyramid level (smallest/fastest) then resize.
        highest_level = pyramid["level_count"] - 1
        level_width, level_height = pyramid["level_dimensions"][highest_level]
        print(f"Using pyramid level {highest_level}: {level_width} x {level_height} px")

        if resolved_wsi_reader == "isyntax":
            print("Reading thumbnail...")
            thumbnail_np = read_region_rgb(
                wsi,
                resolved_wsi_reader,
                x=0,
                y=0,
                width=level_width,
                height=level_height,
                level=highest_level,
            )
            print("Reading thumbnail done")
        else:
            thumbnail_np = _run_with_progress(
                lambda: read_region_rgb(
                    wsi,
                    resolved_wsi_reader,
                    x=0,
                    y=0,
                    width=level_width,
                    height=level_height,
                    level=highest_level,
                ),
                "Reading thumbnail",
            )
    finally:
        close_wsi(wsi, resolved_wsi_reader)

    thumbnail_pil = Image.fromarray(thumbnail_np)

    # Resize to target max_dim (maintaining aspect ratio)
    current_max = max(level_width, level_height)
    if current_max != max_dim:
        scale = max_dim / current_max
        target_w = int(level_width * scale)
        target_h = int(level_height * scale)
        thumbnail_pil = thumbnail_pil.resize((target_w, target_h), Image.LANCZOS)

    print(f"Thumbnail: {thumbnail_pil.size[0]} x {thumbnail_pil.size[1]} px")

    return thumbnail_pil, wsi_width, wsi_height, resolved_wsi_reader


def extract_bbox_region_l0(
    wsi,
    resolved_wsi_reader: str,
    bbox_l0: List[int],
    max_dim: int,
) -> Image.Image:
    """Extract one bbox from level 0 and downsample to max_dim if needed."""
    x1, y1, x2, y2 = [int(v) for v in bbox_l0]
    bbox_w = max(1, x2 - x1)
    bbox_h = max(1, y2 - y1)

    region_np = read_region_rgb(
        wsi,
        resolved_wsi_reader,
        x=x1,
        y=y1,
        width=bbox_w,
        height=bbox_h,
        level=0,
    )
    region_pil = Image.fromarray(region_np)

    current_max = max(region_pil.size)
    if current_max > max_dim:
        scale = max_dim / current_max
        new_w = max(1, int(region_pil.size[0] * scale))
        new_h = max(1, int(region_pil.size[1] * scale))
        region_pil = region_pil.resize((new_w, new_h), Image.LANCZOS)

    return region_pil


def save_bbox_regions_from_level0(
    wsi_path: str,
    processed_bboxes: List[Dict],
    output_dir: Path,
    max_dim: int,
    wsi_reader: str,
) -> None:
    """
    Save optional per-bbox region PNGs under output_dir/bbox_regions/.

    Each bbox is read at level 0 and then downsampled to max_dim (long edge).
    """
    if not processed_bboxes:
        return

    bbox_regions_dir = output_dir / "bbox_regions"
    bbox_regions_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving bbox regions from L0 (max dim: {max_dim})...")
    wsi, resolved_wsi_reader = load_wsi(wsi_path, wsi_reader)
    try:
        total = len(processed_bboxes)
        for i, bbox in enumerate(processed_bboxes, start=1):
            x1, y1, x2, y2 = bbox["bbox_level0"]
            bbox_dir_name = f"{x1}_{y1}_{x2}_{y2}"
            bbox_dir = bbox_regions_dir / bbox_dir_name
            bbox_dir.mkdir(parents=True, exist_ok=True)

            region_pil = extract_bbox_region_l0(
                wsi,
                resolved_wsi_reader,
                [x1, y1, x2, y2],
                max_dim,
            )
            output_path = bbox_dir / "bbox_region.png"
            region_pil.save(output_path)
            print(f"  [{i}/{total}] Saved: {output_path}")
    finally:
        close_wsi(wsi, resolved_wsi_reader)


def encode_pil_to_base64(img: Image.Image) -> str:
    """Encode PIL Image to base64 string."""
    import io
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def query_vlm_pil(
    img: Image.Image,
    prompt: str,
    model: str = DEFAULT_MODEL,
    backend: str = DEFAULT_BACKEND,
    openrouter_url: str = OPENROUTER_BASE_URL,
    vllm_url: str = DEFAULT_VLLM_BASE_URL,
    api_key: Optional[str] = None,
    vertex_credentials: Optional[str] = None,
    vertex_location: str = DEFAULT_VERTEX_LOCATION,
) -> str:
    """Call selected VLM backend with PIL image."""
    backend = (backend or DEFAULT_BACKEND).lower()
    if backend == "vertex":
        return query_vertex_contents(
            model=model,
            contents=[img, prompt],
            vertex_credentials=vertex_credentials,
            vertex_location=vertex_location,
        )

    base64_image = encode_pil_to_base64(img)
    base_url, resolved_api_key = resolve_api_settings(backend, openrouter_url, vllm_url, api_key)

    client = OpenAI(
        api_key=resolved_api_key,
        base_url=base_url
    )

    messages = [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
            {"type": "text", "text": prompt}
        ]
    }]

    completion = client.chat.completions.create(
        model=normalize_model_name_for_backend(model, backend),
        messages=messages,
    )
    return completion.choices[0].message.content


def repair_json_with_llm(
    malformed_json: str,
    repair_model: str,
    coord_order: str = "yxxy",
    backend: str = DEFAULT_BACKEND,
    openrouter_url: str = OPENROUTER_BASE_URL,
    vllm_url: str = DEFAULT_VLLM_BASE_URL,
    api_key: Optional[str] = None,
    vertex_credentials: Optional[str] = None,
    vertex_location: str = DEFAULT_VERTEX_LOCATION,
    max_repair_attempts: int = 3
) -> Optional[str]:
    """
    Use an OpenAI-compatible model to repair malformed bbox JSON.

    Args:
        malformed_json: The malformed JSON string to repair
        repair_model: Model to use for repair
        coord_order: Expected output coordinate order ("yxxy" or "xyxy")
        max_repair_attempts: Max repair attempts before giving up

    Returns:
        Repaired JSON string, or None if repair fails
    """
    current_json = malformed_json

    for attempt in range(1, max_repair_attempts + 1):
        print(f"  JSON repair attempt {attempt}/{max_repair_attempts}...")

        # Truncate if too long
        max_len = 2000
        if len(current_json) > max_len:
            half = max_len // 2
            truncated = current_json[:half] + "\n... [TRUNCATED] ...\n" + current_json[-half:]
        else:
            truncated = current_json

        if coord_order == "xyxy":
            expected_schema = '{"bbox_2d": [x_min, y_min, x_max, y_max], "label": "tissue"}'
        else:
            expected_schema = '{"box_2d": [y_min, x_min, y_max, x_max], "label": "tissue"}'

        repair_prompt = f"""Fix ONLY the STRUCTURE of this malformed JSON. DO NOT change any numerical values.

Expected format is an array of bounding box objects:
[
  {expected_schema},
  ...
]

IMPORTANT:
- Preserve ALL original coordinate numbers exactly as they appear
- Fix missing/extra brackets, quotes, commas
- Convert bare arrays like [[0,670,50,720]] to proper objects with box_2d key
- Remove any text/explanation, keep only JSON

Malformed JSON:
{truncated}

Output ONLY the repaired JSON array, no explanation."""

        try:
            backend_l = (backend or DEFAULT_BACKEND).lower()
            if backend_l == "vertex":
                repaired = query_vertex_contents(
                    model=repair_model,
                    contents=[repair_prompt],
                    vertex_credentials=vertex_credentials,
                    vertex_location=vertex_location,
                    temperature=0.0,
                )
            else:
                base_url, resolved_api_key = resolve_api_settings(backend_l, openrouter_url, vllm_url, api_key)
                client = OpenAI(
                    api_key=resolved_api_key,
                    base_url=base_url
                )

                completion = client.chat.completions.create(
                    model=normalize_model_name_for_backend(repair_model, backend_l),
                    messages=[{"role": "user", "content": repair_prompt}],
                    temperature=0
                )
                repaired = completion.choices[0].message.content

            repaired = parse_json(repaired)

            # Validate it's parseable
            try:
                ast.literal_eval(repaired)
                print(f"  Repair attempt {attempt} produced valid JSON")
                return repaired
            except Exception as parse_err:
                print(f"  Repair attempt {attempt} still invalid: {parse_err}")
                current_json = repaired

        except Exception as e:
            print(f"  JSON repair attempt {attempt} failed: {e}")

    print(f"  All {max_repair_attempts} repair attempts failed")
    return None


def parse_bboxes_response(response_text: str, coord_order: str = "yxxy") -> Optional[List[Dict]]:
    """
    Parse bounding boxes from VLM response.

    Handles:
    - Markdown fencing (```json ... ```)
    - Both [{"bbox_2d": [...]}] and [{"bbox": [...]}] formats
    - Single object vs array
    - Coordinate validation (0-1000 range)
    - Conversion to internal [y, x, y, x] via coord_order

    Returns list of bbox dicts or None if parsing fails.
    """
    try:
        json_str = parse_json(response_text)
        data = ast.literal_eval(json_str)

        # Normalize to list
        if isinstance(data, dict):
            data = [data]

        if not isinstance(data, list):
            return None

        result = []
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                continue

            # Extract coordinates (support multiple key variants)
            bbox = item.get("bbox_2d") or item.get("box_2d") or item.get("bbox")
            if not bbox or not isinstance(bbox, list) or len(bbox) != 4:
                continue

            bbox = normalize_bbox_to_yxxy(bbox, coord_order)

            # Validate coordinate range after normalization
            if not all(0 <= c <= 1000 for c in bbox):
                print(f"Warning: bbox {i} has out-of-range coordinates: {bbox}")
                bbox = [max(0, min(1000, c)) for c in bbox]

            result.append({
                "bbox_2d": bbox,
                "label": item.get("label", f"tissue_{i+1}")
            })

        return result if result else None

    except Exception as e:
        print(f"Warning: Could not parse JSON response: {e}")
        return None


def draw_bboxes_overlay(
    thumbnail: Image.Image,
    bboxes: List[dict],
    save_path: str
) -> None:
    """Plot bounding boxes on thumbnail and save."""
    colors = ['red', 'green', 'blue', 'yellow', 'orange', 'pink', 'purple', 'cyan']

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(thumbnail)

    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        x1, y1, x2, y2 = bbox['bbox_thumbnail']

        rect = mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2,
            edgecolor=color,
            facecolor='none'
        )
        ax.add_patch(rect)

        label = bbox.get('label', f'bbox_{i}')
        ax.text(
            x1 + 5, y1 + 15,
            label,
            color=color,
            fontsize=10,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
        )

    ax.set_title('Stage 1: Detected Foreground Tissue Regions', fontsize=14, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved overlay: {save_path}")


def generate_output_dir(wsi_path: str, model: str) -> Path:
    """
    Generate output directory path for Stage 1.

    Structure: stage1_output/{case_name}/{model_sanitized}/{YYYYMMDD_HHMMSS}/
    """
    case_name = Path(wsi_path).stem
    model_dir = sanitize_model_name(model)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(OUTPUT_BASE_DIR) / case_name / model_dir / timestamp


def load_prompt(prompt_arg: Optional[str], model: str) -> str:
    """Load prompt from string or file path, or choose a model-aware default."""
    if prompt_arg and os.path.isfile(prompt_arg):
        return Path(prompt_arg).read_text().strip()
    if prompt_arg:
        return prompt_arg
    if "qwen" in (model or "").lower():
        return DEFAULT_PROMPT_QWEN
    return DEFAULT_PROMPT_GEMINI


def save_metadata(
    output_dir: Path,
    args: argparse.Namespace,
    wsi_path: str,
    wsi_w: int,
    wsi_h: int,
    thumbnail: Image.Image,
    wsi_reader: str,
    state_info: Dict,
    processed_bboxes: List[Dict],
    prompt: str,
    vlm_response: str
) -> None:
    """Save full metadata for reproducibility."""
    metadata = {
        "wsi_path": os.path.abspath(wsi_path),
        "wsi_dimensions": {"width": wsi_w, "height": wsi_h},
        "thumbnail_dimensions": {"width": thumbnail.size[0], "height": thumbnail.size[1]},
        "model": args.model,
        "backend": args.backend,
        "vertex_location": args.vertex_location if args.backend == "vertex" else None,
        "vertex_credentials": args.vertex_credentials if args.backend == "vertex" else None,
        "wsi_reader": wsi_reader,
        "bbox_coord_order": args.coord_order,
        "resolved_bbox_coord_order": args._resolved_coord_order if hasattr(args, "_resolved_coord_order") else args.coord_order,
        "prompt": prompt,
        "max_dim": args.max_dim,
        "padding": args.padding,
        "merge_overlap_threshold": args.merge_overlap_threshold,
        "detected_regions": processed_bboxes,
        "regions_count": len(processed_bboxes),
        "vlm_response_raw": vlm_response,
        "created_at": datetime.now().isoformat(),
        "git_hash": state_info.get("git_hash", "unknown"),
        "reproducibility_bypassed": state_info.get("bypassed", False)
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata: {metadata_path}")


def save_metadata_tta(
    output_dir: Path,
    args: argparse.Namespace,
    wsi_path: str,
    wsi_w: int,
    wsi_h: int,
    thumbnail: Image.Image,
    wsi_reader: str,
    state_info: Dict,
    processed_bboxes: List[Dict],
    prompt: str,
    per_orientation_responses: Dict[int, str],
    per_orientation_bboxes: Dict[int, List]
) -> None:
    """Save full metadata for reproducibility (TTA version)."""
    metadata = {
        "wsi_path": os.path.abspath(wsi_path),
        "wsi_dimensions": {"width": wsi_w, "height": wsi_h},
        "thumbnail_dimensions": {"width": thumbnail.size[0], "height": thumbnail.size[1]},
        "model": args.model,
        "backend": args.backend,
        "vertex_location": args.vertex_location if args.backend == "vertex" else None,
        "vertex_credentials": args.vertex_credentials if args.backend == "vertex" else None,
        "wsi_reader": wsi_reader,
        "bbox_coord_order": args.coord_order,
        "resolved_bbox_coord_order": args._resolved_coord_order if hasattr(args, "_resolved_coord_order") else args.coord_order,
        "prompt": prompt,
        "max_dim": args.max_dim,
        "padding": args.padding,
        "merge_overlap_threshold": args.merge_overlap_threshold,
        "rotations": args.rotations,
        "detected_regions": processed_bboxes,
        "regions_count": len(processed_bboxes),
        "per_orientation_bbox_counts": {str(r): len(b) for r, b in per_orientation_bboxes.items()},
        "total_pre_merge_bboxes": sum(len(b) for b in per_orientation_bboxes.values()),
        "created_at": datetime.now().isoformat(),
        "git_hash": state_info.get("git_hash", "unknown"),
        "reproducibility_bypassed": state_info.get("bypassed", False)
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata: {metadata_path}")


def print_next_stage_commands(wsi_path: str, bboxes: List[Dict]) -> None:
    """Print CLI commands for next pipeline stage (find_icl_regions.py)."""
    print("\n" + "=" * 60)
    print("CLI COMMANDS FOR find_icl_regions.py (Stage 4)")
    print("=" * 60)

    if not bboxes:
        print("No regions detected - cannot generate commands.")
        return

    # Single command with all bboxes
    cmd_parts = [f'python find_icl_regions.py --wsi "{wsi_path}"']
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox["bbox_level0"]
        cmd_parts.append(f'--bbox {x1} {y1} {x2} {y2}')
    cmd_parts.append('--point-grounding')

    print("\n# All bboxes in one command:")
    print(" \\\n    ".join(cmd_parts))

    # Individual commands
    print("\n# Or run individually:")
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox["bbox_level0"]
        label = bbox["label"]
        print(f'# {label}:')
        print(f'python find_icl_regions.py --wsi "{wsi_path}" --bbox {x1} {y1} {x2} {y2} --point-grounding')
    print()


# =============================================================================
# Orientation TTA Orchestration
# =============================================================================

def draw_bboxes_on_pil(img: Image.Image, bboxes: List, colors: Optional[List[str]] = None) -> Image.Image:
    """Draw bounding boxes on PIL Image. Returns new image with overlays."""
    from PIL import ImageDraw

    overlay = img.copy()
    draw = ImageDraw.Draw(overlay)

    if colors is None:
        colors = ["red", "blue", "green", "orange", "purple", "cyan", "magenta", "yellow"]

    w, h = img.size
    for i, bbox_item in enumerate(bboxes):
        # Handle multiple formats
        if isinstance(bbox_item, dict):
            bbox = bbox_item.get("box_2d") or bbox_item.get("bbox_2d") or bbox_item.get("bbox")
        elif isinstance(bbox_item, (list, tuple)):
            bbox = bbox_item
        else:
            continue

        if not bbox or len(bbox) != 4:
            continue

        # Internal format: [y_min, x_min, y_max, x_max] in 0-1000 coords
        y1, x1, y2, x2 = bbox
        # Convert to pixel coords
        px1 = int(x1 / 1000 * w)
        py1 = int(y1 / 1000 * h)
        px2 = int(x2 / 1000 * w)
        py2 = int(y2 / 1000 * h)

        color = colors[i % len(colors)]
        draw.rectangle([px1, py1, px2, py2], outline=color, width=3)

    return overlay


def _query_single_orientation(
    rotation: int,
    thumbnail: Image.Image,
    prompt: str,
    model: str,
    coord_order: str,
    backend: str,
    openrouter_url: str,
    vllm_url: str,
    api_key: Optional[str],
    vertex_credentials: Optional[str],
    vertex_location: str,
    repair_model: str,
    max_retries: int
) -> Tuple[int, Optional[List], str, Image.Image, bool]:
    """
    Query VLM for a single orientation. Thread-safe helper for parallel execution.

    Returns:
        (rotation, bboxes_parsed, response, rotated_thumb, was_rejected)
        was_rejected: True if orientation was rejected due to giant bbox after all retries
    """
    # Rotate thumbnail
    rotated_thumb = rotate_image(thumbnail, rotation)

    # Query VLM (with retries)
    response = None
    bboxes_parsed = None
    was_rejected = False

    for attempt in range(1, max_retries + 1):
        try:
            # Direct call without progress indicator (we're in a thread)
            response = query_vlm_pil(
                img=rotated_thumb,
                prompt=prompt,
                model=model,
                backend=backend,
                openrouter_url=openrouter_url,
                vllm_url=vllm_url,
                api_key=api_key,
                vertex_credentials=vertex_credentials,
                vertex_location=vertex_location,
            )

            # Try to parse
            bboxes_parsed = parse_bboxes_response(response, coord_order=coord_order)

            if bboxes_parsed is not None:
                # Check for giant bbox (single bbox covering >60% of thumbnail)
                if is_giant_bbox(bboxes_parsed, threshold=0.6):
                    # Giant bbox detected - retry if attempts remain
                    if attempt < max_retries:
                        continue  # Retry
                    else:
                        # All retries exhausted, reject this orientation
                        was_rejected = True
                        bboxes_parsed = None
                        break
                else:
                    # Valid bboxes, accept
                    break

        except Exception as e:
            if attempt == max_retries:
                response = f"ERROR: {e}"

    # If all retries failed to parse, try JSON repair
    if bboxes_parsed is None and not was_rejected and response and not response.startswith("ERROR:"):
        repaired = repair_json_with_llm(
            malformed_json=response,
            repair_model=repair_model,
            coord_order=coord_order,
            backend=backend,
            openrouter_url=openrouter_url,
            vllm_url=vllm_url,
            api_key=api_key,
            vertex_credentials=vertex_credentials,
            vertex_location=vertex_location,
        )
        if repaired:
            bboxes_parsed = parse_bboxes_response(repaired, coord_order=coord_order)
            # Check repaired result for giant bbox too
            if bboxes_parsed and is_giant_bbox(bboxes_parsed, threshold=0.6):
                was_rejected = True
                bboxes_parsed = None

    return (rotation, bboxes_parsed, response or "NO RESPONSE", rotated_thumb, was_rejected)


def run_orientation_tta(
    thumbnail: Image.Image,
    output_dir: Path,
    prompt: str,
    model: str,
    coord_order: str,
    backend: str,
    openrouter_url: str,
    vllm_url: str,
    api_key: Optional[str],
    vertex_credentials: Optional[str],
    vertex_location: str,
    repair_model: str,
    rotations: List[int],
    max_retries: int = 3,
    save_intermediate: bool = False
) -> Tuple[Dict[int, List], List[Dict], Dict[int, str]]:
    """
    Run VLM bbox detection under multiple orientations with TTA.
    Uses parallel execution for all orientations.

    Returns:
        per_orientation_bboxes: Dict mapping rotation -> list of raw bboxes in internal [y,x,y,x]
        transformed_bboxes: List of bboxes transformed to 0° space
        vlm_responses: Dict of raw VLM response strings
    """
    per_orientation_bboxes = {}  # rotation -> list of bboxes
    per_orientation_responses = {}  # rotation -> raw response string
    all_transformed = []  # all bboxes in 0° space

    # Create vlm_responses directory
    vlm_responses_dir = output_dir / "vlm_responses"
    vlm_responses_dir.mkdir(parents=True, exist_ok=True)

    # Optional intermediate directory
    intermediate_dir = None
    if save_intermediate:
        intermediate_dir = output_dir / "intermediate"
        intermediate_dir.mkdir(parents=True, exist_ok=True)

    print(f"Querying {len(rotations)} orientations in parallel...")

    # Run all orientations in parallel
    with ThreadPoolExecutor(max_workers=len(rotations)) as executor:
        futures = {
            executor.submit(
                _query_single_orientation,
                rotation,
                thumbnail,
                prompt,
                model,
                coord_order,
                backend,
                openrouter_url,
                vllm_url,
                api_key,
                vertex_credentials,
                vertex_location,
                repair_model,
                max_retries,
            ): rotation
            for rotation in rotations
        }

        for future in as_completed(futures):
            rotation = futures[future]
            try:
                rot, bboxes_parsed, response, rotated_thumb, was_rejected = future.result()

                # Save raw response
                response_path = vlm_responses_dir / f"rot{rot}_response.txt"
                response_path.write_text(response)
                per_orientation_responses[rot] = response

                # Save intermediate outputs if requested
                if save_intermediate:
                    rotated_thumb.save(intermediate_dir / f"rot{rot}_thumbnail.png")

                # Handle rejected orientation (giant bbox after all retries)
                if was_rejected:
                    per_orientation_bboxes[rot] = []
                    print(f"  {rot}°: REJECTED (giant bbox >80% coverage after retries)")
                    continue

                # Store parsed bboxes
                if bboxes_parsed:
                    per_orientation_bboxes[rot] = bboxes_parsed
                    print(f"  {rot}°: {len(bboxes_parsed)} bbox(es)")

                    # Save intermediate overlay if requested
                    if save_intermediate:
                        overlay = draw_bboxes_on_pil(rotated_thumb, bboxes_parsed)
                        overlay.save(intermediate_dir / f"rot{rot}_bbox_overlay.png")

                    # Transform to 0° space
                    for bbox_item in bboxes_parsed:
                        if isinstance(bbox_item, dict):
                            bbox = bbox_item.get("box_2d") or bbox_item.get("bbox_2d") or bbox_item.get("bbox")
                        else:
                            bbox = bbox_item
                        if bbox and len(bbox) == 4:
                            transformed = transform_bbox_to_rot0(bbox, rot)
                            all_transformed.append(transformed)
                else:
                    per_orientation_bboxes[rot] = []
                    print(f"  {rot}°: No bboxes detected")

            except Exception as e:
                print(f"  {rotation}°: ERROR - {e}")
                per_orientation_bboxes[rotation] = []
                per_orientation_responses[rotation] = f"ERROR: {e}"

    # Save aggregated pre-merge overlay if intermediate
    if save_intermediate and all_transformed:
        # Draw all transformed bboxes on original thumbnail
        agg_overlay = draw_bboxes_on_pil(thumbnail, all_transformed)
        agg_overlay.save(intermediate_dir / "aggregated_pre_merge_overlay.png")

    return per_orientation_bboxes, all_transformed, per_orientation_responses


# =============================================================================
# CLI
# =============================================================================

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Stage 1: Detect tissue bounding boxes from WSI thumbnail using VLM + orientation TTA',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Output structure:
    stage1_output/{wsi_id}/{model}/{timestamp}/
        thumbnail.png          - Original 0° thumbnail
        vlm_responses/         - Raw VLM responses per orientation
            rot0_response.txt
            rot90_response.txt
            rot180_response.txt
            rot270_response.txt
        bboxes.json            - Merged bboxes with all coord systems
        bbox_overlay.png       - Final merged bboxes on thumbnail
        metadata.json          - Full reproducibility metadata
        reproduce.txt          - Command to replicate
        [bbox_regions/]        - Optional L0 bbox crops (with --save-bbox-region)
        [intermediate/]        - Debug outputs (with --save-intermediate)

Examples:
    # Default: run all 4 orientations (0°, 90°, 180°, 270°)
    python detect_foreground_regions_from_wsi_thumbnail.py --wsi anon_6b692277.svs

    # Single orientation (legacy behavior)
    python detect_foreground_regions_from_wsi_thumbnail.py --wsi slide.svs --rotations 0

    # Save intermediate debug outputs
    python detect_foreground_regions_from_wsi_thumbnail.py --wsi slide.svs --save-intermediate

    # Export per-bbox region PNGs from level 0
    python detect_foreground_regions_from_wsi_thumbnail.py --wsi slide.svs --save-bbox-region

    # Custom model
    python detect_foreground_regions_from_wsi_thumbnail.py --wsi slide.svs --model google/gemini-3-pro-preview

    # Vertex backend (Google GenAI SDK)
    python detect_foreground_regions_from_wsi_thumbnail.py --wsi slide.svs --backend vertex --model gemini-3-flash-preview
"""
    )

    parser.add_argument(
        '--wsi',
        type=str,
        required=True,
        help='WSI path (absolute or relative)'
    )
    parser.add_argument(
        '--max-dim',
        type=int,
        default=DEFAULT_MAX_DIM,
        help=f'Maximum thumbnail dimension, maintains aspect ratio (default: {DEFAULT_MAX_DIM})'
    )
    parser.add_argument(
        "--wsi-reader",
        choices=["auto", "openslide", "cucim"],
        default="cucim",
        help="WSI reader backend (default: cucim). Use openslide for .ndpi compatibility.",
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
        help='Optional API key override for OpenRouter/vLLM (not used by Vertex backend).'
    )
    parser.add_argument(
        '--vertex-credentials',
        type=str,
        default=DEFAULT_VERTEX_CREDENTIALS,
        help='Optional Vertex service account JSON path. If unset, use GOOGLE_APPLICATION_CREDENTIALS.'
    )
    parser.add_argument(
        '--vertex-location',
        type=str,
        default=DEFAULT_VERTEX_LOCATION,
        help=f'Google Cloud location for Vertex backend (default: {DEFAULT_VERTEX_LOCATION})'
    )
    parser.add_argument(
        '--repair-model',
        type=str,
        default=None,
        help='Optional model for JSON repair. Defaults to --model.'
    )
    parser.add_argument(
        '--coord-order',
        choices=['auto', 'yxxy', 'xyxy'],
        default='auto',
        help='Input bbox coordinate order from model output (default: auto).'
    )
    parser.add_argument(
        '--prompt',
        type=str,
        default=None,
        help='Custom prompt (string or path to .txt file). Uses default if not specified.'
    )
    parser.add_argument(
        '--padding',
        type=float,
        default=0.25,
        help='Bbox padding fraction applied to detected regions (default: 0.25 = 25%%)'
    )
    parser.add_argument(
        '--merge-overlap-threshold',
        type=float,
        default=DEFAULT_MERGE_OVERLAP_THRESHOLD,
        help='Min overlap (intersection/min-area) required to merge bboxes (default: 0.20)'
    )
    parser.add_argument(
        '--max-retries',
        type=int,
        default=3,
        help='Max VLM retries on parse failure (default: 3)'
    )
    parser.add_argument(
        '--skip-dvc-check',
        action='store_true',
        help='Skip DVC clean-state check (git check still enforced unless bypassed)'
    )
    parser.add_argument(
        '--rotations',
        nargs='+',
        type=int,
        default=[0, 90, 180, 270],
        choices=[0, 90, 180, 270],
        help='Orientations to run VLM detection on (default: 0 90 180 270)'
    )
    parser.add_argument(
        '--save-intermediate',
        action='store_true',
        help='Save intermediate debug outputs (rotated thumbnails, per-orientation overlays)'
    )
    parser.add_argument(
        '--save-bbox-region',
        action='store_true',
        help='Save per-bbox region PNGs from level 0 under bbox_regions/ (downsampled to --max-dim).'
    )

    return parser


# =============================================================================
# Main
# =============================================================================

def main():
    parser = create_parser()
    args = parser.parse_args()
    if args.backend == "vertex" and args.model == DEFAULT_MODEL:
        args.model = DEFAULT_VERTEX_MODEL

    print("=" * 60)
    print("STAGE 1: DETECT FOREGROUND REGIONS FROM WSI THUMBNAIL")
    print(f"Orientations: {args.rotations}")
    print("=" * 60)

    # === STEP 1: Resolve WSI path ===
    from utils.wsi_paths import resolve_wsi_path
    try:
        wsi_path = resolve_wsi_path(args.wsi)
        print(f"WSI resolved: {wsi_path}")
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # === STEP 2: Check git/DVC state ===
    from utils.reproducibility import require_clean_state, create_reproduce_command
    state_info = require_clean_state([wsi_path], skip_dvc_check=args.skip_dvc_check)
    if state_info.get("bypassed"):
        print(f"Warning: Reproducibility check bypassed: {state_info.get('reason')}")

    # === STEP 3: Generate output directory ===
    output_dir = generate_output_dir(wsi_path, args.model)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {output_dir}")
    print()

    # === STEP 4: Create thumbnail ===
    print("Creating thumbnail...")
    thumbnail, wsi_w, wsi_h, resolved_wsi_reader = create_thumbnail(
        wsi_path,
        args.max_dim,
        args.wsi_reader,
    )
    thumbnail_path = output_dir / "thumbnail.png"
    thumbnail.save(thumbnail_path)
    print(f"Saved thumbnail: {thumbnail_path}")
    print(f"WSI reader: {resolved_wsi_reader}")
    print()

    # === STEP 5: Load prompt + resolve coordinate order ===
    prompt = load_prompt(args.prompt, args.model)
    resolved_coord_order = infer_bbox_coord_order(args.coord_order, args.model, prompt)
    args._resolved_coord_order = resolved_coord_order
    repair_model = args.repair_model or args.model
    print(f"Backend: {args.backend}")
    print(f"Model: {args.model}")
    print(f"BBox coord order: {resolved_coord_order}")
    print(f"Prompt: {prompt[:80]}...")

    # === STEP 6: Run Orientation TTA ===
    print("\n" + "=" * 60)
    print("RUNNING ORIENTATION TTA")
    print("=" * 60)

    per_orientation_bboxes, all_transformed, per_orientation_responses = run_orientation_tta(
        thumbnail=thumbnail,
        output_dir=output_dir,
        prompt=prompt,
        model=args.model,
        coord_order=resolved_coord_order,
        backend=args.backend,
        openrouter_url=args.openrouter_url,
        vllm_url=args.vllm_url,
        api_key=args.api_key,
        vertex_credentials=args.vertex_credentials,
        vertex_location=args.vertex_location,
        repair_model=repair_model,
        rotations=args.rotations,
        max_retries=args.max_retries,
        save_intermediate=args.save_intermediate
    )

    # === STEP 7: Pad bboxes THEN merge (padding before merge ensures no overlaps in output) ===
    print("\n" + "=" * 60)
    print("PADDING & MERGING BBOXES")
    print("=" * 60)
    print(f"Merge overlap threshold (intersection/min-area): {args.merge_overlap_threshold:.2f}")

    # Apply padding to each raw bbox BEFORE merging
    if args.padding > 0:
        padded_bboxes = []
        for bbox in all_transformed:
            # all_transformed is in Gemini format (y1, x1, y2, x2)
            y1, x1, y2, x2 = bbox
            # Convert to (x1, y1, x2, y2) for pad function
            bbox_xy = [x1, y1, x2, y2]
            padded_xy = pad_bbox_normalized(bbox_xy, args.padding)
            # Convert back to (y1, x1, y2, x2) for merge
            padded_bboxes.append((padded_xy[1], padded_xy[0], padded_xy[3], padded_xy[2]))
        all_transformed = padded_bboxes
        print(f"Applied {args.padding * 100:.0f}% padding to {len(all_transformed)} raw bboxes")

    total_pre_merge = len(all_transformed)
    merged_bboxes = merge_overlapping_bboxes(all_transformed, args.merge_overlap_threshold)
    print(f"Pre-merge: {total_pre_merge} bboxes from {len(args.rotations)} orientations")
    print(f"Post-merge: {len(merged_bboxes)} merged regions")

    if not merged_bboxes:
        print("\nWARNING: No bounding boxes detected from any orientation!")
        # Save metadata indicating no detection
        save_metadata_tta(output_dir, args, wsi_path, wsi_w, wsi_h, thumbnail, resolved_wsi_reader, state_info,
                          [], prompt, per_orientation_responses, per_orientation_bboxes)
        create_reproduce_command(parser, str(output_dir / "reproduce.txt"), git_hash=state_info.get("git_hash"))
        sys.exit(0)

    # === STEP 8: Convert merged bboxes to output coords (padding already applied in step 7) ===
    processed_bboxes = []
    thumb_w, thumb_h = thumbnail.size

    for i, bbox_tuple in enumerate(merged_bboxes):
        # Merged bbox is in Gemini format (y1, x1, y2, x2), already padded
        y1, x1, y2, x2 = bbox_tuple

        # Convert to (x1, y1, x2, y2) for downstream consistency
        bbox_norm = [x1, y1, x2, y2]

        # Convert to L0 coordinates
        bbox_l0 = normalized_to_level0(bbox_norm, wsi_w, wsi_h)

        # Convert to thumbnail coords for visualization
        bbox_thumb = normalized_to_thumbnail(bbox_norm, thumb_w, thumb_h)

        processed_bboxes.append({
            "label": f"tissue_{i+1}",
            "bbox_normalized": bbox_norm,
            "bbox_thumbnail": list(bbox_thumb),
            "bbox_level0": list(bbox_l0)
        })

    # Print results
    print("\n" + "=" * 60)
    print(f"FINAL MERGED BBOXES - LEVEL 0 COORDINATES (padded {args.padding * 100:.0f}%)")
    print("=" * 60)
    print()

    for bbox in processed_bboxes:
        x1, y1, x2, y2 = bbox["bbox_level0"]
        width = x2 - x1
        height = y2 - y1

        print(f"{bbox['label']}:")
        print(f"  Level 0 coords: ({x1}, {y1}) to ({x2}, {y2})")
        print(f"  Size: {width} x {height} px")
        print(f"  --bbox {x1} {y1} {x2} {y2}")
        print()

    # === STEP 9: Draw final overlay ===
    overlay_path = output_dir / "bbox_overlay.png"
    draw_bboxes_overlay(thumbnail, processed_bboxes, str(overlay_path))

    # === STEP 10: Optional bbox region export from L0 ===
    if args.save_bbox_region:
        save_bbox_regions_from_level0(
            wsi_path=wsi_path,
            processed_bboxes=processed_bboxes,
            output_dir=output_dir,
            max_dim=args.max_dim,
            wsi_reader=resolved_wsi_reader,
        )

    # === STEP 11: Save bboxes.json with new format ===
    bboxes_path = output_dir / "bboxes.json"

    # Convert per_orientation_bboxes to serializable format
    per_orientation_raw = {}
    for rot, bboxes in per_orientation_bboxes.items():
        per_orientation_raw[str(rot)] = bboxes

    with open(bboxes_path, 'w') as f:
        json.dump({
            "detected_regions": processed_bboxes,
            "regions_count": len(processed_bboxes),
            "per_orientation_raw": per_orientation_raw
        }, f, indent=2)
    print(f"Saved bboxes: {bboxes_path}")

    # === STEP 12: Save metadata.json ===
    save_metadata_tta(output_dir, args, wsi_path, wsi_w, wsi_h, thumbnail, resolved_wsi_reader, state_info,
                      processed_bboxes, prompt, per_orientation_responses, per_orientation_bboxes)

    # === STEP 13: Generate reproduce.txt ===
    create_reproduce_command(parser, str(output_dir / "reproduce.txt"), git_hash=state_info.get("git_hash"))
    print(f"Saved reproduce: {output_dir / 'reproduce.txt'}")

    # === STEP 14: Print CLI commands for next stage ===
    print_next_stage_commands(wsi_path, processed_bboxes)

    print("=" * 60)
    print("STAGE 1 COMPLETE")
    print(f"Output: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
