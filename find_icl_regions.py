#!/usr/bin/env python3
# ABOUTME: Point grounding for ICL patch sampling from stage2 QC outputs.
# ABOUTME: Takes bbox_region.png from stage2, queries VLM for tissue/background/artifact points.
"""
Find ICL Regions - Point Grounding from Stage2 Output

Reads bbox_region.png from stage2 QC pipeline outputs and uses VLM point grounding
to identify tissue, background, and optional artifact classes. Pen/paraffin classes
are auto-included from stage2 verdicts; ambiguous is optional via CLI.

TTA (test-time augmentation) is enabled by default - runs 4 rotations (0, 90, 180, 270°)
and aggregates points into points_overlay_all.png.

Output structure:
    stage4_output/{case_name}/{bbox_str}/{model}/{YYYYMMDD_HHMMSS}/
        rot_0/
            points.json           - Parsed points for this rotation
            points_overlay.png    - Visualization
            vlm_response.txt      - Raw VLM response
        rot_90/
        rot_180/
        rot_270/
        region_thumbnail.png      - Copy of input bbox_region.png
        points_overlay_all.png    - AGGREGATED: all rotations combined (TTA only)
        metadata.json             - Full metadata for reproducibility
        reproduce.txt             - Command to reproduce this run

Usage:
    # Single stage2 bbox directory
    python find_icl_regions.py --stage2-dir stage2_output/anon_xxx/.../10_20_30_40/

    # Auto-include pen/paraffin based on stage2 verdicts
    python find_icl_regions.py --stage2-dir stage2_output/anon_xxx/.../10_20_30_40/

    # Disable TTA (single rotation)
    python find_icl_regions.py --stage2-dir stage2_output/anon_xxx/.../10_20_30_40/ --no-tta

    # Batch mode
    python find_icl_regions.py --batch stage2_output/anon_*/*/*/*/ --workers 4
"""

import argparse
import ast
import base64
import glob
import json
import os
import re
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from openai import OpenAI
from PIL import Image

# Import reproducibility utilities
from utils.reproducibility import require_clean_state, create_reproduce_command


# =============================================================================
# Configuration
# =============================================================================

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_VLLM_BASE_URL = "http://localhost:8000/v1"
DEFAULT_BACKEND = "openrouter"
DEFAULT_MODEL = "google/gemini-3-flash-preview"
DEFAULT_MAX_DIM = 1024
OUTPUT_BASE_DIR = "stage4_output"
DEFAULT_TEMPERATURE = 0.5
DEFAULT_MAX_ITEMS = 10
DEFAULT_MAX_TOKENS = 512
DEFAULT_VERTEX_LOCATION = "global"
DEFAULT_VERTEX_CREDENTIALS: Optional[str] = None
DEFAULT_PEN_DESC = (
    "Large opaque coloured ink or pen marking on the slide, "
    "clearly distinct from the visual appearance of tissue or background"
)
DEFAULT_PARAFFIN_DESC = (
    "Pale, translucent, waxy mounting medium that is clearly distinct from the "
    "white / off white / grey slide background"
)
AUTO_AMBIGUOUS_CLASSES = ["debris", "air_bubbles", "tissue_folds", "cracks"]
PROMPT_TEMPLATE_DIR = Path(__file__).resolve().parent / "prompts" / "stage4"
DEFAULT_QWEN_PROMPT_TEMPLATE_PATH = PROMPT_TEMPLATE_DIR / "qwen_default.txt"
DEFAULT_GEMINI_PROMPT_TEMPLATE_PATH = PROMPT_TEMPLATE_DIR / "gemini_default.txt"
DEFAULT_PROMPT_PROFILE = "auto"
REQUIRED_PROMPT_TEMPLATE_PLACEHOLDERS = (
    "class_definitions",
    "max_items_instruction",
    "point_key",
    "point_order_desc",
)

_VERTEX_CLIENT = None
_VERTEX_CLIENT_CACHE_KEY: Optional[Tuple[Optional[str], str]] = None
_VERTEX_CLIENT_LOCK = threading.Lock()
VERTEX_CREDENTIALS: Optional[str] = DEFAULT_VERTEX_CREDENTIALS
VERTEX_LOCATION: str = DEFAULT_VERTEX_LOCATION


# Base prompt template - classes are inserted dynamically
PROMPT_TEMPLATE = """Look at this medium magnification thumbnail of a tissue core biopsy. Point to regions clearly belonging to the following categories:
{class_definitions}
{max_items_instruction}The answer should follow the json format: [{{"{point_key}": <point>, "label": <label1>}}, ...]. The points are in [{point_order_desc}] format normalized to 0-1000."""


def _build_class_definitions(
    pen_desc: Optional[str],
    paraffin_desc: Optional[str],
    ambiguous_desc: Optional[str],
    tissue_desc: Optional[str],
    background_desc: Optional[str],
) -> List[str]:
    # Defaults for core classes
    tissue_default = "Areas with tissue, cells, or histological structures"
    background_default = "Empty/glass areas with consistent white/off-white color"

    class_num = 1
    class_defs = []

    # Always include tissue and background
    class_defs.append(f"{class_num}. TISSUE - {tissue_desc or tissue_default}")
    class_num += 1

    class_defs.append(f"{class_num}. BACKGROUND - {background_desc or background_default}")
    class_num += 1

    # Add optional artifact classes
    if pen_desc:
        class_defs.append(f"{class_num}. PEN_INK_MARKS - {pen_desc}")
        class_num += 1

    if paraffin_desc:
        class_defs.append(f"{class_num}. PARAFFIN_MOUNTING_MEDIUM - {paraffin_desc}")
        class_num += 1

    if ambiguous_desc:
        class_defs.append(f"{class_num}. OTHER_ARTIFACTS - {ambiguous_desc}")
        class_num += 1

    return class_defs


def build_point_grounding_prompt(
    pen_desc: Optional[str] = None,
    paraffin_desc: Optional[str] = None,
    ambiguous_desc: Optional[str] = None,
    tissue_desc: Optional[str] = None,
    background_desc: Optional[str] = None,
    max_items: int = DEFAULT_MAX_ITEMS,
    point_key: str = "point",
    point_coord_order: str = "yx",
    prompt_template: str = PROMPT_TEMPLATE,
    include_max_items_instruction: bool = True,
) -> Tuple[str, int, int]:
    """
    Build dynamic point grounding prompt with specified artifact classes.

    Args:
        pen_desc: If provided, adds pen_ink_marks class with this description
        paraffin_desc: If provided, adds paraffin_mounting_medium class with this description
        ambiguous_desc: If provided, adds ambiguous class with this description
        tissue_desc: Custom tissue class description (or default)
        background_desc: Custom background class description (or default)
        prompt_template: Prompt template string with placeholders
        include_max_items_instruction: Whether to include point-count cap instruction

    Returns:
        (prompt, class_count, max_items_total)
    """
    class_defs = _build_class_definitions(
        pen_desc, paraffin_desc, ambiguous_desc, tissue_desc, background_desc
    )
    class_count = len(class_defs)
    max_items_total = max_items * class_count if class_count > 0 else max_items
    class_definitions = "\n\n".join(class_defs)
    max_items_instruction = ""
    if include_max_items_instruction:
        max_items_instruction = (
            f"Include at least one point for each category and return no more than "
            f"{max_items_total} points in the image across all categories.\n"
        )
    prompt = prompt_template.format(
        class_definitions=class_definitions,
        max_items_instruction=max_items_instruction,
        max_items_per_class=max_items,
        point_key=point_key,
        point_order_desc="y, x" if point_coord_order == "yx" else "x, y",
    )
    return prompt, class_count, max_items_total


# =============================================================================
# Helper Functions
# =============================================================================

def parse_json(json_output: str) -> str:
    """Remove markdown fencing from JSON output."""
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith("```"):
            json_output = "\n".join(lines[i+1:])
            json_output = json_output.split("```")[0]
            break
    return json_output.strip()


def resolve_prompt_override(prompt_arg: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """Resolve CLI --prompt argument as either inline text or file content."""
    if not prompt_arg:
        return None, None
    candidate = Path(prompt_arg)
    if candidate.is_file():
        return candidate.read_text(encoding="utf-8").strip(), str(candidate.resolve())
    return prompt_arg, "inline"


def normalize_thinking_level(thinking_level: Optional[str]) -> Optional[str]:
    """Normalize optional Gemini thinking level to accepted API values."""
    if thinking_level is None:
        return None
    cleaned = thinking_level.strip()
    if not cleaned:
        return None
    lowered = cleaned.lower()
    if lowered == "low":
        return "Low"
    if lowered == "high":
        return "High"
    raise ValueError("--thinking-level must be one of: Low, High")


def infer_prompt_profile_from_model(model: str) -> str:
    """Infer default prompt profile from model name."""
    return "qwen" if "qwen" in (model or "").lower() else "gemini"


def validate_prompt_template(template_text: str, template_source: str) -> None:
    """Validate prompt template placeholder contract and formatting."""
    missing = [
        f"{{{name}}}" for name in REQUIRED_PROMPT_TEMPLATE_PLACEHOLDERS
        if f"{{{name}}}" not in template_text
    ]
    if missing:
        raise ValueError(
            f"Prompt template {template_source} is missing required placeholders: {', '.join(missing)}"
        )

    preview_ctx = {
        "class_definitions": "<class_definitions>",
        "max_items_instruction": "<max_items_instruction>",
        "point_key": "point",
        "point_order_desc": "x, y",
    }
    try:
        template_text.format(**preview_ctx)
    except KeyError as e:
        key = e.args[0]
        raise ValueError(
            f"Prompt template {template_source} has unknown placeholder '{{{key}}}'. "
            f"Allowed placeholders: {', '.join('{' + p + '}' for p in REQUIRED_PROMPT_TEMPLATE_PLACEHOLDERS)}"
        ) from e
    except ValueError as e:
        raise ValueError(
            f"Prompt template {template_source} has invalid brace formatting: {e}. "
            "Use '{{' and '}}' for literal braces."
        ) from e


def load_prompt_template(path: Path) -> str:
    """Load prompt template text from disk with basic validation."""
    if not path.exists():
        raise FileNotFoundError(f"Prompt template file not found: {path}")
    template_text = path.read_text(encoding="utf-8").strip()
    if not template_text:
        raise ValueError(f"Prompt template file is empty: {path}")
    return template_text


def resolve_prompt_template_for_model(
    model: str,
    prompt_profile: str = DEFAULT_PROMPT_PROFILE,
    prompt_template_file: Optional[str] = None,
) -> Tuple[str, str, str]:
    """
    Resolve prompt template text for a model.

    Returns:
        (template_text, profile_used, template_source)
    """
    if prompt_template_file:
        template_path = Path(prompt_template_file)
        template_text = load_prompt_template(template_path)
        template_source = str(template_path.resolve())
        validate_prompt_template(template_text, template_source)
        return template_text, "custom", template_source

    profile_used = prompt_profile
    if profile_used == "auto":
        profile_used = infer_prompt_profile_from_model(model)

    if profile_used == "qwen":
        template_path = DEFAULT_QWEN_PROMPT_TEMPLATE_PATH
    elif profile_used == "gemini":
        template_path = DEFAULT_GEMINI_PROMPT_TEMPLATE_PATH
    else:
        raise ValueError(f"Unsupported prompt profile: {prompt_profile}")

    template_text = load_prompt_template(template_path)
    template_source = str(template_path.resolve())
    validate_prompt_template(template_text, template_source)
    return template_text, profile_used, template_source


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

    key = api_key or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError("Missing OpenRouter API key. Set OPENROUTER_API_KEY/OPENAI_API_KEY or pass --api-key.")
    return openrouter_url, key


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


def query_vertex_image(
    image_path: str,
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    vertex_credentials: Optional[str],
    vertex_location: str,
    thinking_level: Optional[str] = None,
    include_thoughts: bool = False,
) -> str:
    """Send image + prompt to Vertex Gemini."""
    from google.genai import types

    with Image.open(image_path) as img:
        image_rgb = img.convert("RGB")

    client = get_vertex_client(vertex_credentials, vertex_location)
    vertex_model = normalize_model_name_for_backend(model, backend="vertex")
    config_kwargs = dict(
        temperature=temperature,
        max_output_tokens=max_tokens if max_tokens else None,
    )
    if thinking_level:
        config_kwargs["thinking_config"] = types.ThinkingConfig(
            thinking_level=thinking_level,
            include_thoughts=include_thoughts,
        )
    config = types.GenerateContentConfig(**config_kwargs)
    response = client.models.generate_content(
        model=vertex_model,
        contents=[image_rgb, prompt],
        config=config,
    )
    return (response.text or "").strip()


def query_vertex_text(
    prompt: str,
    model: str,
    max_tokens: int,
    vertex_credentials: Optional[str],
    vertex_location: str,
) -> str:
    """Send text-only prompt to Vertex Gemini."""
    from google.genai import types

    client = get_vertex_client(vertex_credentials, vertex_location)
    vertex_model = normalize_model_name_for_backend(model, backend="vertex")
    config = types.GenerateContentConfig(
        temperature=0,
        max_output_tokens=max_tokens if max_tokens else None,
    )
    response = client.models.generate_content(
        model=vertex_model,
        contents=[prompt],
        config=config,
    )
    return (response.text or "").strip()


def infer_point_schema(
    model: str,
    prompt: str,
    point_order_arg: str,
    point_key_arg: str,
) -> Tuple[str, str]:
    """
    Infer point output schema.

    Returns:
        (point_coord_order, point_key)
        point_coord_order: 'xy' or 'yx'
        point_key: 'point_2d' or 'point'
    """
    prompt_l = (prompt or "").lower()

    if point_order_arg in {"xy", "yx"}:
        point_coord_order = point_order_arg
    elif re.search(r"\[\s*x\s*,\s*y\s*\]", prompt_l):
        point_coord_order = "xy"
    elif re.search(r"\[\s*y\s*,\s*x\s*\]", prompt_l):
        point_coord_order = "yx"
    elif "qwen" in (model or "").lower():
        point_coord_order = "xy"
    else:
        point_coord_order = "yx"

    if point_key_arg in {"point_2d", "point"}:
        point_key = point_key_arg
    elif '"point_2d"' in prompt_l:
        point_key = "point_2d"
    elif '"point"' in prompt_l:
        point_key = "point"
    elif "qwen" in (model or "").lower():
        point_key = "point_2d"
    else:
        point_key = "point"

    return point_coord_order, point_key


def normalize_point_to_xy(point: List[float], point_coord_order: str) -> List[float]:
    """Normalize a point to internal [x, y] format."""
    if point_coord_order == "yx":
        return [point[1], point[0]]
    return [point[0], point[1]]


def normalized_to_thumbnail(bbox_norm: List[float], thumb_w: int, thumb_h: int) -> Tuple[int, int, int, int]:
    """Convert normalized (0-1000) bbox to thumbnail pixel coords."""
    x1 = int(bbox_norm[0] / 1000 * thumb_w)
    y1 = int(bbox_norm[1] / 1000 * thumb_h)
    x2 = int(bbox_norm[2] / 1000 * thumb_w)
    y2 = int(bbox_norm[3] / 1000 * thumb_h)

    # Ensure correct order
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1

    return (x1, y1, x2, y2)


def normalized_to_l0(bbox_norm: List[float], input_bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    """Convert normalized (0-1000) bbox to L0 coords within input bbox."""
    ix1, iy1, ix2, iy2 = input_bbox
    iw, ih = ix2 - ix1, iy2 - iy1

    x1 = ix1 + int(bbox_norm[0] / 1000 * iw)
    y1 = iy1 + int(bbox_norm[1] / 1000 * ih)
    x2 = ix1 + int(bbox_norm[2] / 1000 * iw)
    y2 = iy1 + int(bbox_norm[3] / 1000 * ih)

    # Ensure correct order
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1

    return (x1, y1, x2, y2)


def normalized_point_to_l0(point_norm: List[float], input_bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
    """Convert normalized (0-1000) point to L0 coords within input bbox."""
    ix1, iy1, ix2, iy2 = input_bbox
    iw, ih = ix2 - ix1, iy2 - iy1

    x = ix1 + int(point_norm[0] / 1000 * iw)
    y = iy1 + int(point_norm[1] / 1000 * ih)

    return (x, y)


def encode_image_base64(image: Image.Image) -> str:
    """Encode PIL Image to base64 string."""
    import io
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


# =============================================================================
# Stage2 Input Functions
# =============================================================================

def load_stage2_input(stage2_dir: Path) -> Tuple[Image.Image, Tuple[int, int, int, int], str]:
    """
    Load bbox_region.png and parse metadata from stage2 output directory.

    Args:
        stage2_dir: Path to stage2 bbox directory (e.g., stage2_output/anon_xxx/.../10_20_30_40/)

    Returns:
        (image, bbox_l0, case_name)

    Directory structure expected:
        stage2_output/{wsi_id}/{model}/{timestamp}/{x1_y1_x2_y2}/
            bbox_region.png
            stage4_verdicts.json (optional)
            ...
    """
    stage2_dir = Path(stage2_dir)

    # Load bbox_region.png
    image_path = stage2_dir / "bbox_region.png"
    if not image_path.exists():
        raise FileNotFoundError(f"bbox_region.png not found in {stage2_dir}")

    image = Image.open(image_path).convert("RGB")

    # Parse bbox from directory name (format: x1_y1_x2_y2)
    bbox_str = stage2_dir.name
    try:
        coords = bbox_str.split("_")
        if len(coords) != 4:
            raise ValueError(f"Expected 4 coordinates, got {len(coords)}")
        bbox = tuple(int(c) for c in coords)
    except Exception as e:
        raise ValueError(f"Could not parse bbox from directory name '{bbox_str}': {e}")

    # Extract case name from path (structure: stage2_output/{wsi_id}/{model}/{timestamp}/{bbox}/)
    # wsi_id is 4 levels up from bbox dir
    try:
        case_name = stage2_dir.parent.parent.parent.name
    except Exception:
        case_name = "unknown"

    # Read wsi_path from stage2 run-level metadata (parent = timestamp dir)
    wsi_path = None
    for candidate in [stage2_dir.parent / "metadata.json", stage2_dir / "metadata.json"]:
        if candidate.exists():
            try:
                with open(candidate) as _f:
                    _s2meta = json.load(_f)
                if _s2meta.get("wsi_path"):
                    wsi_path = _s2meta["wsi_path"]
                    break
            except (json.JSONDecodeError, KeyError):
                pass

    print(f"Loaded stage2 input:")
    print(f"  Image: {image_path} ({image.size[0]}x{image.size[1]})")
    print(f"  Bbox: {bbox}")
    print(f"  Case: {case_name}")

    return image, bbox, case_name, wsi_path


def _collect_bbox_dirs(stage2_path: Path) -> List[Path]:
    """
    Given a stage2 path, return a list of bbox directories to process.

    - If stage2_path itself contains bbox_region.png, treat it as a bbox dir.
    - Otherwise, scan immediate children for bbox_region.png.
    """
    stage2_path = Path(stage2_path)
    if stage2_path.is_dir() and (stage2_path / "bbox_region.png").exists():
        return [stage2_path]

    bbox_dirs = []
    if stage2_path.is_dir():
        for child in stage2_path.iterdir():
            if child.is_dir() and (child / "bbox_region.png").exists():
                bbox_dirs.append(child)

    return sorted(bbox_dirs)


def load_verdicts(verdict_path: Path) -> Dict:
    """Load stage4_verdicts.json."""
    with verdict_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Unexpected verdicts format in {verdict_path}")
    return data


def verdict_includes(verdicts: Dict, key: str) -> bool:
    """Check if verdict is INCLUDE (conflicts are skipped)."""
    return verdicts.get(key, {}).get("verdict") == "INCLUDE"


def load_visual_descriptions(desc_path: Path) -> Optional[Dict[str, str]]:
    """Load visual_descriptions.json if it exists."""
    if not desc_path.exists():
        return None
    with desc_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else None


def _build_auto_ambiguous_desc(
    verdicts: Dict,
    visual_desc: Optional[Dict[str, str]]
) -> Optional[str]:
    if not visual_desc:
        return None
    parts: List[str] = []
    for key in AUTO_AMBIGUOUS_CLASSES:
        if verdict_includes(verdicts, key):
            desc = visual_desc.get(key)
            if isinstance(desc, str) and desc.strip():
                parts.append(f"{key} - {desc.strip()}")
    return "; ".join(parts) if parts else None


def _merge_descriptions(base: Optional[str], extra: Optional[str]) -> Optional[str]:
    if base and extra:
        return f"{base}; {extra}"
    return base or extra


def maybe_generate_visual_descriptions(stage2_dir: Path) -> bool:
    """Generate visual_descriptions.json for a bbox dir via helper script."""
    repo_root = Path(__file__).resolve().parent
    script_path = repo_root / "generate_visual_descriptions.py"
    if not script_path.exists():
        print(f"Warning: {script_path} not found; cannot generate visual descriptions.")
        return False

    cmd = [
        sys.executable,
        str(script_path),
        "--stage2-dir",
        str(stage2_dir),
    ]
    try:
        subprocess.run(cmd, check=True, cwd=repo_root)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Warning: Failed to generate visual descriptions for {stage2_dir.name}: {e}")
        return False


def resolve_class_descriptions(
    stage2_dir: Path,
    use_visual_descriptions: bool,
    ambiguous_desc: Optional[str],
    tissue_desc_override: Optional[str],
    background_desc_override: Optional[str],
    pen_desc_default: str,
    paraffin_desc_default: str,
) -> Dict:
    """
    Resolve class descriptions and auto-included classes for a stage2 bbox directory.
    """
    verdict_path = stage2_dir / "stage4_verdicts.json"
    verdicts = {}
    if verdict_path.exists():
        verdicts = load_verdicts(verdict_path)
    else:
        print(f"Warning: Missing stage4_verdicts.json in {stage2_dir}")

    pen_included = verdict_includes(verdicts, "ink_or_pen_marks")
    paraffin_included = verdict_includes(verdicts, "paraffin_mounting_medium")

    visual_desc = None
    generated = False
    desc_path = stage2_dir / "visual_descriptions.json"
    if use_visual_descriptions:
        visual_desc = load_visual_descriptions(desc_path)
        if visual_desc is None:
            generated = maybe_generate_visual_descriptions(stage2_dir)
            visual_desc = load_visual_descriptions(desc_path)

    tissue_desc = tissue_desc_override or (visual_desc.get("tissue") if visual_desc else None)
    background_desc = background_desc_override or (visual_desc.get("background") if visual_desc else None)

    pen_desc = None
    if pen_included:
        pen_desc = (visual_desc.get("pen_ink_marks") if visual_desc else None) or pen_desc_default

    paraffin_desc = None
    if paraffin_included:
        paraffin_desc = (
            (visual_desc.get("paraffin_mounting_medium") if visual_desc else None)
            or paraffin_desc_default
        )

    auto_ambiguous_desc = _build_auto_ambiguous_desc(verdicts, visual_desc)
    ambiguous_desc = _merge_descriptions(ambiguous_desc, auto_ambiguous_desc)

    return {
        "tissue_desc": tissue_desc,
        "background_desc": background_desc,
        "pen_desc": pen_desc,
        "paraffin_desc": paraffin_desc,
        "ambiguous_desc": ambiguous_desc,
        "pen_included": pen_included,
        "paraffin_included": paraffin_included,
        "visual_descriptions_used": bool(visual_desc),
        "visual_descriptions_generated": generated,
        "visual_descriptions_path": str(desc_path) if desc_path.exists() else None,
    }


def expand_batch_dirs(patterns: List[str]) -> List[Path]:
    """
    Expand glob patterns to list of stage2 bbox directories.

    Args:
        patterns: List of glob patterns (e.g., ["stage2_output/anon_*/*/*/*/"])

    Returns:
        List of valid stage2 bbox directories (containing bbox_region.png)
    """
    dirs = []
    for pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        for match in matches:
            match_path = Path(match)
            if match_path.is_dir():
                dirs.extend(_collect_bbox_dirs(match_path))

    # Remove duplicates and sort
    dirs = sorted(set(dirs))
    print(f"Expanded {len(patterns)} pattern(s) to {len(dirs)} stage2 directories")
    return dirs


# =============================================================================
# Core Functions
# =============================================================================

def query_vlm(
    image_path: str,
    prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    backend: str = DEFAULT_BACKEND,
    openrouter_url: str = OPENROUTER_BASE_URL,
    vllm_url: str = DEFAULT_VLLM_BASE_URL,
    api_key: Optional[str] = None,
    thinking_level: Optional[str] = None,
    include_thoughts: bool = False,
) -> str:
    """Send image + prompt to VLM, return response text."""
    backend = (backend or DEFAULT_BACKEND).lower()
    if backend == "vertex":
        print(f"Calling {backend} API ({model})...")
        return query_vertex_image(
            image_path=image_path,
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            vertex_credentials=VERTEX_CREDENTIALS,
            vertex_location=VERTEX_LOCATION,
            thinking_level=thinking_level,
            include_thoughts=include_thoughts,
        )

    with open(image_path, "rb") as f:
        base64_image = base64.b64encode(f.read()).decode("utf-8")

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

    print(f"Calling {backend} API ({model})...")
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return completion.choices[0].message.content


def parse_points_response(response_text: str, point_coord_order: str = "xy") -> Optional[Dict]:
    """
    Parse multi-class points from VLM response.

    Expected format:
    [
      {"point" or "point_2d": [.., ..], "label": "<class_name>"},
      ...
    ]

    Returns dict with class labels as keys and point lists as values.
    Each point is stored as (coords, original_index) tuple to preserve VLM output order.
    Supports arbitrary labels (not just foreground/background).
    Returns None if parsing fails or no valid points found.
    """
    try:
        json_str = parse_json(response_text)
        data = ast.literal_eval(json_str)

        result = {}  # Dynamic dict keyed by label

        if isinstance(data, list):
            for idx, item in enumerate(data):
                # Support both "point_2d" (Qwen) and "point" (Gemini) keys
                if isinstance(item, dict) and ("point_2d" in item or "point" in item):
                    point = item.get("point_2d") or item.get("point")
                    label = item.get("label", "unknown").lower().strip()

                    if isinstance(point, list) and len(point) == 2:
                        point_xy = normalize_point_to_xy(point, point_coord_order)

                        if not all(isinstance(v, (int, float)) for v in point_xy):
                            continue
                        if not all(0 <= float(v) <= 1000 for v in point_xy):
                            point_xy = [max(0, min(1000, float(v))) for v in point_xy]

                        if label not in result:
                            result[label] = []
                        # Store as (point, original_index) to preserve order for visualization
                        result[label].append((point_xy, idx))

        return result if result else None

    except Exception as e:
        print(f"Warning: Could not parse points response: {e}")
        return None


def repair_json_with_llm(
    malformed_json: str,
    model: str,
    point_coord_order: str = "xy",
    point_key: str = "point_2d",
    max_tokens: int = DEFAULT_MAX_TOKENS,
    backend: str = DEFAULT_BACKEND,
    openrouter_url: str = OPENROUTER_BASE_URL,
    vllm_url: str = DEFAULT_VLLM_BASE_URL,
    api_key: Optional[str] = None,
    max_repair_attempts: int = 3
) -> Optional[str]:
    """
    Use a lightweight LLM to repair malformed JSON, with retry loop.

    Args:
        malformed_json: The malformed JSON string to repair
        model: Model to use for repair
        point_coord_order: Expected point order ("xy" or "yx")
        point_key: Preferred point key ("point_2d" or "point")
        max_repair_attempts: Max repair attempts before giving up

    Returns:
        Repaired JSON string, or None if repair fails
    """
    current_json = malformed_json

    for attempt in range(1, max_repair_attempts + 1):
        print(f"  JSON repair attempt {attempt}/{max_repair_attempts}...")

        # Truncate if too long (keep first/last portions for context)
        max_len = 2000
        if len(current_json) > max_len:
            half = max_len // 2
            truncated = current_json[:half] + "\n... [TRUNCATED] ...\n" + current_json[-half:]
        else:
            truncated = current_json

        expected_point = "[x, y]" if point_coord_order == "xy" else "[y, x]"
        repair_prompt = f"""Fix ONLY the STRUCTURE of this malformed JSON. DO NOT change any numerical values or label strings.

Expected format is an array of point objects with arbitrary class labels:
[
  {{"{point_key}": {expected_point}, "label": "<any_class_name>"}},
  ...
]

IMPORTANT:
- Preserve ALL original coordinate numbers exactly as they appear
- Preserve ALL original label strings exactly as they appear (any class name is valid)

Structure fixes to apply:
- Fix missing/extra brackets, quotes, commas
- Remove malformed entries that cannot be recovered

Malformed JSON:
{truncated}

Output ONLY the repaired JSON array, no explanation."""

        try:
            if (backend or DEFAULT_BACKEND).lower() == "vertex":
                repaired = query_vertex_text(
                    prompt=repair_prompt,
                    model=model,
                    max_tokens=max_tokens,
                    vertex_credentials=VERTEX_CREDENTIALS,
                    vertex_location=VERTEX_LOCATION,
                )
            else:
                base_url, resolved_api_key = resolve_api_settings(backend, openrouter_url, vllm_url, api_key)
                client = OpenAI(api_key=resolved_api_key, base_url=base_url)

                completion = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": repair_prompt}],
                    temperature=0,
                    max_tokens=max_tokens,
                )
                repaired = completion.choices[0].message.content

            # Clean up any markdown fencing
            repaired = parse_json(repaired)

            # Try to parse as JSON to verify it's valid
            try:
                import ast
                ast.literal_eval(repaired)
                print(f"  Repair attempt {attempt} produced valid JSON")
                return repaired
            except Exception as parse_err:
                print(f"  Repair attempt {attempt} still invalid: {parse_err}")
                # Use the repaired output as input for next attempt
                current_json = repaired

        except Exception as e:
            print(f"  JSON repair attempt {attempt} failed: {e}")

    print(f"  All {max_repair_attempts} repair attempts failed")
    return None


def draw_points_on_thumbnail(
    thumbnail: Image.Image,
    points_data: Dict[str, List],
    save_path: str
) -> None:
    """Draw multi-class points on thumbnail with dynamic color palette and legend."""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.imshow(thumbnail)

    thumb_w, thumb_h = thumbnail.size

    # Get sorted class labels for consistent coloring
    class_labels = sorted(points_data.keys())
    num_classes = len(class_labels)

    # Generate color palette using tab10 (supports up to 10 distinct colors)
    cmap = plt.cm.get_cmap('tab10')
    colors = {label: cmap(i % 10) for i, label in enumerate(class_labels)}

    # Draw points for each class
    legend_handles = []
    for label in class_labels:
        color = colors[label]
        points = points_data[label]

        for point_entry in points:
            # Handle both old format (just coords) and new format (coords, original_index)
            if isinstance(point_entry, tuple) and len(point_entry) == 2 and isinstance(point_entry[1], int):
                point_norm, orig_idx = point_entry
            else:
                point_norm = point_entry
                orig_idx = None

            x = int(point_norm[0] / 1000 * thumb_w)
            y = int(point_norm[1] / 1000 * thumb_h)
            # Draw circle marker
            circle = plt.Circle((x, y), radius=8, color=color, fill=True, alpha=0.8)
            ax.add_patch(circle)
            # Add index label (use original VLM response index if available)
            idx_str = str(orig_idx) if orig_idx is not None else '?'
            ax.text(
                x + 12, y + 4, idx_str,
                color=color, fontsize=8, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7)
            )

        # Create legend entry
        legend_handles.append(
            mpatches.Patch(color=color, label=f'{label} ({len(points)})')
        )

    # Build title with class counts
    total_points = sum(len(pts) for pts in points_data.values())
    ax.set_title(f'Detected Points: {total_points} total across {num_classes} classes',
                 fontsize=12, fontweight='bold')
    ax.axis('off')

    # Add legend outside plot
    ax.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1.02, 1),
              fontsize=10, title='Classes')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved points overlay: {save_path}")


def sanitize_model_name(model: str) -> str:
    """Sanitize model name for use in file paths."""
    return model.replace("/", "_").replace(":", "_").replace("-", "_")


def generate_output_dir(case_name: str, bbox: Tuple[int, int, int, int], model: str = None, timestamp: str = None) -> Path:
    """
    Generate output directory path.

    Structure: stage4_output/{case_name}/{bbox_str}/{model_sanitized}/{YYYYMMDD_HHMMSS}/

    Args:
        case_name: WSI case name (without extension)
        bbox: Bounding box tuple (x1, y1, x2, y2)
        model: Model name for directory
        timestamp: Optional timestamp string. If None, generates new one.
                   Pass same timestamp for TTA runs to group rotations together.
    """
    bbox_str = f"{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}"
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if model:
        model_dir = sanitize_model_name(model)
        output_dir = Path(OUTPUT_BASE_DIR) / case_name / bbox_str / model_dir / timestamp
    else:
        output_dir = Path(OUTPUT_BASE_DIR) / case_name / bbox_str / timestamp
    return output_dir


# =============================================================================
# CLI
# =============================================================================

def create_parser():
    """Create argument parser (separate for reproduce.txt generation)."""
    parser = argparse.ArgumentParser(
        description='Point grounding for ICL patch sampling from stage2 QC outputs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single stage2 bbox directory
  python find_icl_regions.py --stage2-dir stage2_output/anon_xxx/.../10_20_30_40/

  # Auto-include pen/paraffin based on stage2 verdicts
  python find_icl_regions.py --stage2-dir stage2_output/anon_xxx/.../10_20_30_40/

  # Disable TTA (single rotation only)
  python find_icl_regions.py --stage2-dir stage2_output/anon_xxx/.../10_20_30_40/ --no-tta

  # Stage2 run dir (timestamp) - process all bbox dirs
  python find_icl_regions.py --stage2-dir stage2_output/anon_xxx/.../20260131_225556/ --workers 4

  # Batch mode
  python find_icl_regions.py --batch stage2_output/anon_*/*/*/*/ --workers 4
"""
    )

    # Input mode: either single dir or batch
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--stage2-dir',
        type=str,
        help='Path to stage2 bbox directory (bbox_region.png) OR stage2 run dir (timestamp folder)'
    )
    input_group.add_argument(
        '--batch',
        type=str,
        nargs='+',
        help='Glob patterns for batch processing stage2 directories (e.g., stage2_output/anon_*/*/*/*/)'
    )

    # Optional custom class
    parser.add_argument(
        '--ambiguous',
        type=str,
        metavar='DESC',
        help='Add ambiguous class with this description (e.g., "unclear regions that could confuse models")'
    )

    # Core class descriptions (VLM-generated or custom)
    parser.add_argument(
        '--tissue',
        type=str,
        metavar='DESC',
        help='Custom tissue class description (overrides default)'
    )
    parser.add_argument(
        '--background',
        type=str,
        metavar='DESC',
        help='Custom background class description (overrides default)'
    )

    # Model and prompt
    parser.add_argument(
        '--prompt',
        type=str,
        default=None,
        help='Custom VLM prompt (string or path to .txt file). Overrides default point grounding prompt.'
    )
    parser.add_argument(
        '--prompt-profile',
        choices=['auto', 'qwen', 'gemini'],
        default=DEFAULT_PROMPT_PROFILE,
        help='Default prompt template profile when --prompt is not provided (default: auto).'
    )
    parser.add_argument(
        '--prompt-template-file',
        type=str,
        default=None,
        help='Path to custom prompt template file with placeholders; used when --prompt is not provided.'
    )
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        default=[DEFAULT_MODEL],
        help=f'Model(s) to run on selected backend (default: {DEFAULT_MODEL})'
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
        '--thinking-level',
        type=str,
        default=None,
        help='Gemini thinking level for --backend vertex: Low/High (default: disabled).',
    )
    parser.add_argument(
        '--include-thoughts',
        action='store_true',
        default=False,
        help='Request Gemini thought summaries for --backend vertex.',
    )
    parser.add_argument(
        '--no-include-thoughts',
        dest='include_thoughts',
        action='store_false',
        help='Disable Gemini thought summaries for --backend vertex.',
    )
    parser.add_argument(
        '--point-order',
        choices=['auto', 'xy', 'yx'],
        default='auto',
        help='Expected point coordinate order in model output (default: auto).'
    )
    parser.add_argument(
        '--point-key',
        choices=['auto', 'point', 'point_2d'],
        default='auto',
        help='Preferred point JSON key in prompt/output (default: auto).'
    )
    parser.add_argument(
        '--repair-model',
        type=str,
        default=None,
        help='Optional model for JSON repair. Defaults to current --models entry.'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f'VLM temperature (default: {DEFAULT_TEMPERATURE})'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f'Max output tokens per VLM call (default: {DEFAULT_MAX_TOKENS})'
    )
    parser.add_argument(
        '--max-items',
        type=int,
        default=DEFAULT_MAX_ITEMS,
        help=f'Base max items per class; total cap scales with number of classes (default: {DEFAULT_MAX_ITEMS})'
    )
    parser.add_argument(
        '--max-items-per-class',
        type=int,
        dest='max_items',
        help='Alias for --max-items'
    )
    parser.add_argument(
        '--use-visual-descriptions',
        action='store_true',
        help='Use visual_descriptions.json if present, otherwise generate it'
    )
    parser.add_argument(
        '--default-pen-desc',
        type=str,
        default=DEFAULT_PEN_DESC,
        help='Default description for pen_ink_marks when verdict includes it'
    )
    parser.add_argument(
        '--default-paraffin-desc',
        type=str,
        default=DEFAULT_PARAFFIN_DESC,
        help='Default description for paraffin_mounting_medium when verdict includes it'
    )
    parser.add_argument(
        '--output-base',
        type=str,
        default=OUTPUT_BASE_DIR,
        help=f'Base output directory (default: {OUTPUT_BASE_DIR})'
    )

    # TTA (default ON)
    parser.add_argument(
        '--no-tta',
        action='store_true',
        help='Disable TTA (test-time augmentation). By default, runs 4 rotations (0, 90, 180, 270°).'
    )

    # Batch mode options
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='Number of parallel workers for batch mode (default: 1)'
    )

    # Retry options
    parser.add_argument(
        '--max-retries',
        type=int,
        default=3,
        help='Max VLM retries on parse failure before attempting JSON repair (default: 3)'
    )
    parser.add_argument(
        '--max-repair-attempts',
        type=int,
        default=3,
        help='Max LLM repair attempts for malformed JSON (default: 3)'
    )

    parser.add_argument(
        '--skip-dvc-check',
        action='store_true',
        help='Bypass DVC clean-state check (still checks git)'
    )

    return parser


def transform_points_for_rotation(points_data: Dict, rotation_deg: int, thumb_w: int, thumb_h: int) -> Dict:
    """
    Transform points from rotated coordinate space back to original (0°) space.

    Args:
        points_data: Dict of {label: [(point_norm, orig_idx), ...]}
        rotation_deg: The rotation that was applied (0, 90, 180, 270 CCW)
        thumb_w: Original thumbnail width
        thumb_h: Original thumbnail height

    Returns:
        Dict with points transformed to original orientation
    """
    if rotation_deg == 0:
        return points_data

    transformed = {}
    for label, points in points_data.items():
        transformed[label] = []
        for point_entry in points:
            if isinstance(point_entry, tuple) and len(point_entry) == 2 and isinstance(point_entry[1], int):
                point_norm, orig_idx = point_entry
            else:
                point_norm = point_entry
                orig_idx = None

            x, y = point_norm[0], point_norm[1]

            # Transform based on rotation (reverse the CCW rotation)
            if rotation_deg == 90:
                # 90° CCW: (x, y) -> (y, 1000-x)
                # Reverse: (x, y) -> (1000-y, x)
                new_x, new_y = 1000 - y, x
            elif rotation_deg == 180:
                # 180°: (x, y) -> (1000-x, 1000-y)
                new_x, new_y = 1000 - x, 1000 - y
            elif rotation_deg == 270:
                # 270° CCW: (x, y) -> (1000-y, x)
                # Reverse: (x, y) -> (y, 1000-x)
                new_x, new_y = y, 1000 - x
            else:
                new_x, new_y = x, y

            transformed[label].append(([new_x, new_y], orig_idx))

    return transformed


def create_aggregated_overlay(
    original_thumbnail: Image.Image,
    rotation_results: Dict[int, Dict],
    save_path: str
) -> None:
    """
    Create aggregated overlay showing points from all rotations on the original thumbnail.

    Args:
        original_thumbnail: The unrotated thumbnail image
        rotation_results: Dict of {rotation_deg: points_data} where points_data is already
                         transformed to original coordinate space
        save_path: Path to save the overlay image
    """
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.imshow(original_thumbnail)

    thumb_w, thumb_h = original_thumbnail.size

    # Collect all unique class labels across rotations
    all_labels = set()
    for points_data in rotation_results.values():
        if points_data:
            all_labels.update(points_data.keys())
    class_labels = sorted(all_labels)

    # Color map for classes
    cmap = plt.cm.get_cmap('tab10')
    class_colors = {label: cmap(i % 10) for i, label in enumerate(class_labels)}

    # Marker styles for different rotations
    rotation_markers = {0: 'o', 90: 's', 180: '^', 270: 'D'}
    rotation_names = {0: '0°', 90: '90°', 180: '180°', 270: '270°'}

    # Draw points for each rotation
    legend_handles = []
    for rotation_deg, points_data in sorted(rotation_results.items()):
        if not points_data:
            continue

        marker = rotation_markers.get(rotation_deg, 'o')

        for label in class_labels:
            if label not in points_data:
                continue

            color = class_colors[label]
            points = points_data[label]

            for point_entry in points:
                if isinstance(point_entry, tuple) and len(point_entry) == 2 and isinstance(point_entry[1], int):
                    point_norm, _ = point_entry
                else:
                    point_norm = point_entry

                x = int(point_norm[0] / 1000 * thumb_w)
                y = int(point_norm[1] / 1000 * thumb_h)

                ax.scatter(x, y, c=[color], marker=marker, s=100, alpha=0.7, edgecolors='white', linewidth=0.5)

    # Build legend
    # First add class colors
    for label in class_labels:
        legend_handles.append(
            mpatches.Patch(color=class_colors[label], label=f'{label}')
        )

    # Add rotation markers
    legend_handles.append(mpatches.Patch(color='none', label=''))  # Spacer
    for rot_deg, marker in rotation_markers.items():
        if rot_deg in rotation_results and rotation_results[rot_deg]:
            legend_handles.append(
                plt.Line2D([0], [0], marker=marker, color='gray', linestyle='None',
                          markersize=10, label=f'Rotation: {rotation_names[rot_deg]}')
            )

    # Count total points
    total_points = sum(
        sum(len(pts) for pts in points_data.values())
        for points_data in rotation_results.values()
        if points_data
    )
    num_rotations = sum(1 for pd in rotation_results.values() if pd)

    ax.set_title(f'Aggregated Points: {total_points} total from {num_rotations} rotations',
                 fontsize=14, fontweight='bold')
    ax.axis('off')

    ax.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1.02, 1),
              fontsize=10, title='Classes & Rotations')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved aggregated overlay: {save_path}")


def run_single_rotation(
    thumbnail: Image.Image,
    bbox: Tuple[int, int, int, int],
    model: str,
    prompt: str,
    output_dir: Path,
    rotation_deg: int,
    temperature: float,
    max_tokens: int,
    point_coord_order: str,
    point_key: str,
    backend: str,
    openrouter_url: str,
    vllm_url: str,
    api_key: Optional[str],
    thinking_level: Optional[str],
    include_thoughts: bool,
    repair_model: str,
    max_retries: int = 3,
    max_repair_attempts: int = 3
) -> Tuple[Optional[Dict], Path]:
    """
    Run point grounding for a single rotation.

    Args:
        thumbnail: Original (unrotated) thumbnail
        bbox: Bounding box tuple
        model: Model to use
        prompt: VLM prompt
        output_dir: Base output directory (rotation subdir will be added)
        rotation_deg: Rotation to apply (0, 90, 180, 270)
        max_retries: Max VLM retries
        max_repair_attempts: Max JSON repair attempts

    Returns:
        (points_data dict or None, rotation output directory)
    """
    # Create rotation subdirectory
    rot_dir = output_dir / f"rot_{rotation_deg}"
    rot_dir.mkdir(parents=True, exist_ok=True)

    # Apply rotation
    rotated_thumb = thumbnail
    if rotation_deg != 0:
        rotated_thumb = thumbnail.rotate(rotation_deg, expand=True)

    # Save rotated thumbnail
    thumb_path = rot_dir / "region_thumbnail.png"
    rotated_thumb.save(thumb_path)

    print(f"\n  [ROT {rotation_deg}°] Querying VLM...")

    # Query VLM with retries
    response = None
    points_data = None
    parse_succeeded = False

    for attempt in range(1, max_retries + 1):
        try:
            response = query_vlm(
                str(thumb_path),
                prompt,
                model,
                temperature,
                max_tokens=max_tokens,
                backend=backend,
                openrouter_url=openrouter_url,
                vllm_url=vllm_url,
                api_key=api_key,
                thinking_level=thinking_level,
                include_thoughts=include_thoughts,
            )

            # Save response
            response_path = rot_dir / f"vlm_response{'_' + str(attempt) if attempt > 1 else ''}.txt"
            response_path.write_text(response)

            # Parse response
            points_data = parse_points_response(response, point_coord_order=point_coord_order)
            if points_data:
                parse_succeeded = True
                break
            else:
                print(f"    Parse failed on attempt {attempt}")

        except Exception as e:
            print(f"    VLM error on attempt {attempt}: {e}")

    # JSON repair if needed
    if not parse_succeeded and response:
        print(f"    Attempting JSON repair...")
        repaired_json = repair_json_with_llm(
            malformed_json=response,
            model=repair_model,
            point_coord_order=point_coord_order,
            point_key=point_key,
            max_tokens=max_tokens,
            backend=backend,
            openrouter_url=openrouter_url,
            vllm_url=vllm_url,
            api_key=api_key,
            max_repair_attempts=max_repair_attempts,
        )
        if repaired_json:
            repaired_path = rot_dir / "vlm_response_repaired.txt"
            repaired_path.write_text(repaired_json)
            points_data = parse_points_response(repaired_json, point_coord_order=point_coord_order)

    # Ensure final response is saved
    final_response_path = rot_dir / "vlm_response.txt"
    if not final_response_path.exists() and response:
        final_response_path.write_text(response)

    # Save points JSON
    if points_data:
        # Convert rotated points to canonical (un-rotated) coordinates for storage
        canonical_points = transform_points_for_rotation(
            points_data,
            rotation_deg,
            thumbnail.size[0],
            thumbnail.size[1]
        )

        points_l0 = {
            "input_bbox": list(bbox),
            "rotation_deg": rotation_deg,
            "classes": {}
        }
        for label, points in points_data.items():
            canon_points = canonical_points.get(label, [])
            entries = []
            for idx, point_entry in enumerate(points):
                if isinstance(point_entry, tuple) and len(point_entry) == 2 and isinstance(point_entry[1], int):
                    point_norm_rot, rot_idx = point_entry
                else:
                    point_norm_rot, rot_idx = point_entry, None

                if idx < len(canon_points):
                    canon_entry = canon_points[idx]
                    if isinstance(canon_entry, tuple) and len(canon_entry) == 2 and isinstance(canon_entry[1], int):
                        point_norm_canon, canon_idx = canon_entry
                    else:
                        point_norm_canon, canon_idx = canon_entry, None
                else:
                    point_norm_canon, canon_idx = point_norm_rot, None

                point_dict = {
                    "point_normalized": point_norm_canon,
                    "point_l0": normalized_point_to_l0(point_norm_canon, bbox),
                }
                if rot_idx is not None or canon_idx is not None:
                    point_dict["vlm_index"] = rot_idx if rot_idx is not None else canon_idx
                if rotation_deg != 0:
                    point_dict["point_normalized_rotated"] = point_norm_rot

                entries.append(point_dict)

            points_l0["classes"][label] = entries

        points_path = rot_dir / "points.json"
        with open(points_path, 'w') as f:
            json.dump(points_l0, f, indent=2)

        total_pts = sum(len(pts) for pts in points_data.values())
        print(f"    [ROT {rotation_deg}°] Detected {total_pts} points across {len(points_data)} classes")
    else:
        print(f"    [ROT {rotation_deg}°] No points detected")

    return points_data, rot_dir


def process_stage2_dir(
    stage2_dir: Path,
    model: str,
    prompt_override: Optional[str],
    prompt_override_source: Optional[str],
    prompt_profile: str,
    prompt_template_file: Optional[str],
    rotations: List[int],
    state_info: Dict,
    backend: str,
    openrouter_url: str,
    vllm_url: str,
    api_key: Optional[str],
    thinking_level: Optional[str],
    include_thoughts: bool,
    temperature: float,
    max_tokens: int,
    max_items: int,
    point_order_arg: str,
    point_key_arg: str,
    use_visual_descriptions: bool,
    ambiguous_desc: Optional[str],
    tissue_desc_override: Optional[str],
    background_desc_override: Optional[str],
    pen_desc_default: str,
    paraffin_desc_default: str,
    repair_model: Optional[str],
    max_retries: int = 3,
    max_repair_attempts: int = 3
) -> Tuple[Path, str]:
    """
    Process a single stage2 directory - run all rotations and create aggregated overlay.

    Returns:
        (output_dir, status) where status is "success" or error message
    """
    try:
        # Load stage2 input
        thumbnail, bbox, case_name, wsi_path = load_stage2_input(stage2_dir)

        class_info = resolve_class_descriptions(
            stage2_dir=stage2_dir,
            use_visual_descriptions=use_visual_descriptions,
            ambiguous_desc=ambiguous_desc,
            tissue_desc_override=tissue_desc_override,
            background_desc_override=background_desc_override,
            pen_desc_default=pen_desc_default,
            paraffin_desc_default=paraffin_desc_default,
        )
        if class_info.get("ambiguous_desc"):
            print(f"Ambiguous (auto/cli): {class_info['ambiguous_desc']}")

        # Model-specific native schema defaults:
        # - Gemini: point + [y, x]
        # - Qwen: point_2d + [x, y]
        model_lower = (model or "").lower()
        model_is_qwen = "qwen" in model_lower
        default_point_key = "point_2d" if model_is_qwen else "point"
        default_point_order = "xy" if model_is_qwen else "yx"
        prompt_source = "prompt_override" if prompt_override else "template"
        prompt_profile_used = "override" if prompt_override else prompt_profile
        prompt_template_source = prompt_override_source if prompt_override else None

        if prompt_override:
            prompt = prompt_override
            class_defs = _build_class_definitions(
                class_info["pen_desc"],
                class_info["paraffin_desc"],
                class_info["ambiguous_desc"],
                class_info["tissue_desc"],
                class_info["background_desc"],
            )
            class_count = len(class_defs)
            max_items_total = max_items * class_count if class_count > 0 else max_items
        else:
            requested_point_order = point_order_arg if point_order_arg != "auto" else default_point_order
            requested_point_key = point_key_arg if point_key_arg != "auto" else default_point_key
            prompt_template_text, prompt_profile_used, prompt_template_source = resolve_prompt_template_for_model(
                model=model,
                prompt_profile=prompt_profile,
                prompt_template_file=prompt_template_file,
            )
            prompt, class_count, max_items_total = build_point_grounding_prompt(
                pen_desc=class_info["pen_desc"],
                paraffin_desc=class_info["paraffin_desc"],
                ambiguous_desc=class_info["ambiguous_desc"],
                tissue_desc=class_info["tissue_desc"],
                background_desc=class_info["background_desc"],
                max_items=max_items,
                point_key=requested_point_key,
                point_coord_order=requested_point_order,
                prompt_template=prompt_template_text,
            )

        point_coord_order, point_key = infer_point_schema(
            model=model,
            prompt=prompt,
            point_order_arg=point_order_arg,
            point_key_arg=point_key_arg,
        )
        prompt_point_order = "y,x" if point_coord_order == "yx" else "x,y"
        repair_model_resolved = repair_model or model

        # Generate output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = generate_output_dir(case_name, bbox, model, timestamp=timestamp)
        output_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 60)
        print(f"POINT GROUNDING - {case_name}")
        print("=" * 60)
        print(f"Stage2 input: {stage2_dir}")
        print(f"Bbox: {bbox}")
        print(f"Backend: {backend}")
        print(f"Model: {model}")
        if backend == "vertex":
            print(f"Gemini thinking level: {thinking_level or 'disabled'}")
            print(f"Gemini include thoughts: {include_thoughts}")
        print(f"Prompt source: {prompt_source}")
        if prompt_template_source:
            print(f"Prompt template: {prompt_template_source}")
        print(f"Prompt profile: {prompt_profile_used}")
        print(f"Max tokens: {max_tokens}")
        print(f"Point schema: key={point_key}, order={prompt_point_order}")
        print(f"Rotations: {rotations}")
        print(f"Output: {output_dir}")
        print(f"Classes: {class_count}, Max items total: {max_items_total}")
        print("=" * 60)

        # Copy original thumbnail to output dir
        orig_thumb_path = output_dir / "region_thumbnail.png"
        thumbnail.save(orig_thumb_path)

        # Run each rotation
        rotation_results = {}
        for rotation_deg in rotations:
            points_data, rot_dir = run_single_rotation(
                thumbnail=thumbnail,
                bbox=bbox,
                model=model,
                prompt=prompt,
                output_dir=output_dir,
                rotation_deg=rotation_deg,
                temperature=temperature,
                max_tokens=max_tokens,
                point_coord_order=point_coord_order,
                point_key=point_key,
                backend=backend,
                openrouter_url=openrouter_url,
                vllm_url=vllm_url,
                api_key=api_key,
                thinking_level=thinking_level,
                include_thoughts=include_thoughts,
                repair_model=repair_model_resolved,
                max_retries=max_retries,
                max_repair_attempts=max_repair_attempts
            )

            # Transform points back to original orientation
            if points_data:
                transformed = transform_points_for_rotation(
                    points_data, rotation_deg,
                    thumbnail.size[0], thumbnail.size[1]
                )
                rotation_results[rotation_deg] = transformed
            else:
                rotation_results[rotation_deg] = None

        # Aggregated overlay is rendered after all VLM calls complete

        # Save metadata
        class_counts = {}
        total_count = 0
        for rot_deg, points_data in rotation_results.items():
            if points_data:
                for label, pts in points_data.items():
                    class_counts[label] = class_counts.get(label, 0) + len(pts)
                    total_count += len(pts)

        metadata = {
            "stage2_input": str(stage2_dir),
            "case_name": case_name,
            "wsi_path": wsi_path,
            "input_bbox": list(bbox),
            "thumbnail_dimensions": {"width": thumbnail.size[0], "height": thumbnail.size[1]},
            "backend": backend,
            "model": model,
            "gemini_thinking_level": thinking_level,
            "gemini_include_thoughts": include_thoughts,
            "prompt": prompt,
            "prompt_rendered": prompt,
            "prompt_source": prompt_source,
            "prompt_profile": prompt_profile_used,
            "prompt_template_file": prompt_template_source,
            "prompt_override_source": prompt_override_source if prompt_override else None,
            "point_key": point_key,
            "prompt_point_order": prompt_point_order,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "max_items_per_class": max_items,
            "max_items_total": max_items_total,
            "auto_classes": {
                "pen_ink_marks": class_info["pen_included"],
                "paraffin_mounting_medium": class_info["paraffin_included"],
                "ambiguous": bool(class_info["ambiguous_desc"]),
            },
            "visual_descriptions": {
                "used": class_info["visual_descriptions_used"],
                "generated": class_info["visual_descriptions_generated"],
                "path": class_info["visual_descriptions_path"],
            },
            "rotations": rotations,
            "tta_enabled": len(rotations) > 1,
            "class_counts": class_counts,
            "total_count": total_count,
            "created_at": datetime.now().isoformat(),
            "git_hash": state_info.get("git_hash", "unknown"),
            "reproducibility_bypassed": state_info.get("bypassed", False)
        }

        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\n{'=' * 60}")
        print(f"Complete! Output: {output_dir}")
        print(f"{'=' * 60}")

        return output_dir, "success"

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, str(e)


def _load_points_for_overlay(
    rot_dir: Path,
    point_key: str = "point_normalized",
    fallback_key: str = "point_normalized"
) -> Optional[Dict[str, List]]:
    points_path = rot_dir / "points.json"
    if not points_path.exists():
        return None
    with open(points_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    classes = data.get("classes") if isinstance(data, dict) else None
    if not isinstance(classes, dict):
        return None
    # Flatten back to list of point_norm entries with optional vlm_index
    points_data: Dict[str, List] = {}
    for label, entries in classes.items():
        if not isinstance(entries, list):
            continue
        pts = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            point_norm = entry.get(point_key)
            if point_norm is None and fallback_key:
                point_norm = entry.get(fallback_key)
            if not (isinstance(point_norm, list) and len(point_norm) == 2):
                continue
            idx = entry.get("vlm_index")
            if isinstance(idx, int):
                pts.append((point_norm, idx))
            else:
                pts.append(point_norm)
        if pts:
            points_data[label] = pts
    return points_data if points_data else None


def render_overlays_for_output(output_dir: Path, rotations: List[int]) -> None:
    """
    Render per-rotation overlays and aggregated overlay sequentially.
    Assumes points.json already exists for each rotation.
    """
    rotation_results: Dict[int, Optional[Dict]] = {}
    thumbnail_path = output_dir / "region_thumbnail.png"
    if not thumbnail_path.exists():
        print(f"Warning: Missing region_thumbnail.png in {output_dir}")
        return
    base_thumb = Image.open(thumbnail_path).convert("RGB")

    for rotation_deg in rotations:
        rot_dir = output_dir / f"rot_{rotation_deg}"
        if not rot_dir.exists():
            rotation_results[rotation_deg] = None
            continue
        points_canon = _load_points_for_overlay(rot_dir, point_key="point_normalized")
        if not points_canon:
            rotation_results[rotation_deg] = None
            continue
        points_rot = points_canon
        if rotation_deg != 0:
            points_rot = _load_points_for_overlay(
                rot_dir,
                point_key="point_normalized_rotated",
                fallback_key="point_normalized"
            ) or points_canon

        # Use rotation-specific thumbnail if available; otherwise rotate base
        rot_thumb_path = rot_dir / "region_thumbnail.png"
        if rot_thumb_path.exists():
            thumb = Image.open(rot_thumb_path).convert("RGB")
        else:
            thumb = base_thumb.rotate(rotation_deg, expand=True) if rotation_deg != 0 else base_thumb

        overlay_path = rot_dir / "points_overlay.png"
        draw_points_on_thumbnail(thumb, points_rot, str(overlay_path))

        # Aggregation uses canonical (0°) points
        rotation_results[rotation_deg] = points_canon

    # Aggregated overlay (TTA only)
    if len(rotations) > 1:
        aggregated_path = output_dir / "points_overlay_all.png"
        create_aggregated_overlay(base_thumb, rotation_results, str(aggregated_path))


def main():
    global VERTEX_CREDENTIALS, VERTEX_LOCATION

    parser = create_parser()
    args = parser.parse_args()
    args.thinking_level = normalize_thinking_level(args.thinking_level)
    VERTEX_CREDENTIALS = args.vertex_credentials
    VERTEX_LOCATION = args.vertex_location

    # Apply output base override
    global OUTPUT_BASE_DIR
    OUTPUT_BASE_DIR = args.output_base

    # === REPRODUCIBILITY CHECK ===
    state_info = require_clean_state([], skip_dvc_check=args.skip_dvc_check)
    if state_info.get("bypassed"):
        print(f"Warning: Reproducibility check bypassed: {state_info.get('reason')}")

    # Determine rotations
    rotations = [0] if args.no_tta else [0, 90, 180, 270]

    # Build prompt override (if provided)
    prompt_override, prompt_override_source = resolve_prompt_override(args.prompt)

    if prompt_override:
        print(
            f"\nPrompt override ({prompt_override_source}): {prompt_override[:100]}..."
            if len(prompt_override) > 100
            else f"\nPrompt override ({prompt_override_source}): {prompt_override}"
        )
        if args.prompt_template_file:
            print("Note: --prompt-template-file is ignored because --prompt was provided.")
        if args.prompt_profile != DEFAULT_PROMPT_PROFILE:
            print("Note: --prompt-profile is ignored because --prompt was provided.")
    elif args.prompt_template_file:
        print(f"\nPrompt template file: {Path(args.prompt_template_file).resolve()}")
    else:
        profile_msg = args.prompt_profile
        if args.prompt_profile == "auto":
            profile_msg = "auto (qwen for Qwen models, gemini otherwise)"
        print(f"\nPrompt template profile: {profile_msg}")
    print(f"Backend: {args.backend}")
    if args.backend == "vertex":
        print(f"Gemini thinking level: {args.thinking_level or 'disabled'}")
        print(f"Gemini include thoughts: {args.include_thoughts}")
    elif args.thinking_level or args.include_thoughts:
        print("Note: --thinking-level/--include-thoughts are only used with --backend vertex.")
    print(f"Max tokens per call: {args.max_tokens}")
    print(f"Point order setting: {args.point_order}")
    print(f"Point key setting: {args.point_key}")
    print(f"TTA: {'Enabled' if len(rotations) > 1 else 'Disabled'} ({rotations})")

    # Get stage2 directories to process
    if args.stage2_dir:
        stage2_dirs = _collect_bbox_dirs(Path(args.stage2_dir))
    else:
        stage2_dirs = expand_batch_dirs(args.batch)

    if not stage2_dirs:
        print("Error: No valid stage2 directories found", file=sys.stderr)
        sys.exit(1)

    print(f"\nProcessing {len(stage2_dirs)} stage2 directory(ies)")
    print(f"Models: {', '.join(args.models)}")

    # Process each directory
    results = []
    total_tasks = len(stage2_dirs) * len(args.models)
    task_idx = 0

    if args.workers > 1 and len(stage2_dirs) > 1:
        # Parallel batch mode
        print(f"Using {args.workers} parallel workers")

        def process_task(task):
            stage2_dir, model = task
            return (stage2_dir, model, *process_stage2_dir(
                stage2_dir=stage2_dir,
                model=model,
                prompt_override=prompt_override,
                prompt_override_source=prompt_override_source,
                prompt_profile=args.prompt_profile,
                prompt_template_file=args.prompt_template_file,
                rotations=rotations,
                state_info=state_info,
                backend=args.backend,
                openrouter_url=args.openrouter_url,
                vllm_url=args.vllm_url,
                api_key=args.api_key,
                thinking_level=args.thinking_level,
                include_thoughts=args.include_thoughts,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                max_items=args.max_items,
                point_order_arg=args.point_order,
                point_key_arg=args.point_key,
                use_visual_descriptions=args.use_visual_descriptions,
                ambiguous_desc=args.ambiguous,
                tissue_desc_override=args.tissue,
                background_desc_override=args.background,
                pen_desc_default=args.default_pen_desc,
                paraffin_desc_default=args.default_paraffin_desc,
                repair_model=args.repair_model,
                max_retries=args.max_retries,
                max_repair_attempts=args.max_repair_attempts
            ))

        tasks = [(d, m) for d in stage2_dirs for m in args.models]

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_task, task): task for task in tasks}
            for future in as_completed(futures):
                task_idx += 1
                try:
                    stage2_dir, model, output_dir, status = future.result()
                    results.append((stage2_dir, model, output_dir, status))
                    print(f"[{task_idx}/{total_tasks}] {stage2_dir.name}: {status}")
                except Exception as e:
                    task = futures[future]
                    results.append((task[0], task[1], None, str(e)))
                    print(f"[{task_idx}/{total_tasks}] {task[0].name}: ERROR - {e}")
    else:
        # Sequential mode
        for stage2_dir in stage2_dirs:
            for model in args.models:
                task_idx += 1
                print(f"\n[{task_idx}/{total_tasks}] Processing {stage2_dir.name} with {model}")
                output_dir, status = process_stage2_dir(
                    stage2_dir=stage2_dir,
                    model=model,
                    prompt_override=prompt_override,
                    prompt_override_source=prompt_override_source,
                    prompt_profile=args.prompt_profile,
                    prompt_template_file=args.prompt_template_file,
                    rotations=rotations,
                    state_info=state_info,
                    backend=args.backend,
                    openrouter_url=args.openrouter_url,
                    vllm_url=args.vllm_url,
                    api_key=args.api_key,
                    thinking_level=args.thinking_level,
                    include_thoughts=args.include_thoughts,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    max_items=args.max_items,
                    point_order_arg=args.point_order,
                    point_key_arg=args.point_key,
                    use_visual_descriptions=args.use_visual_descriptions,
                    ambiguous_desc=args.ambiguous,
                    tissue_desc_override=args.tissue,
                    background_desc_override=args.background,
                    pen_desc_default=args.default_pen_desc,
                    paraffin_desc_default=args.default_paraffin_desc,
                    repair_model=args.repair_model,
                    max_retries=args.max_retries,
                    max_repair_attempts=args.max_repair_attempts
                )
                results.append((stage2_dir, model, output_dir, status))

    # Render overlays sequentially after VLM calls complete
    for stage2_dir, model, output_dir, status in results:
        if status != "success" or not output_dir:
            continue
        try:
            render_overlays_for_output(Path(output_dir), rotations)
        except Exception as e:
            print(f"Warning: Overlay rendering failed for {output_dir}: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    success_count = sum(1 for r in results if r[3] == "success")
    print(f"Completed: {success_count}/{len(results)} successful")
    for stage2_dir, model, output_dir, status in results:
        status_str = f"✓ {output_dir}" if status == "success" else f"✗ {status}"
        print(f"  {stage2_dir.name} [{model.split('/')[-1]}]: {status_str}")
    print("=" * 60)


if __name__ == '__main__':
    main()
