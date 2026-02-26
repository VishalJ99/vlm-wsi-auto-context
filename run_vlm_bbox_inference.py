#!/usr/bin/env python3
# ABOUTME: Stage 6-style patch classification for Stage 5 bboxes with Gemini SDK or vLLM.
# ABOUTME: Supports prompt templating, ICL shuffles, rotation TTA, and Stage 3 gating overlays.
"""
VLM Patch Classification for Stage 5 Bboxes

This script runs patch-level VLM classification over Stage 5 bbox regions,
with configurable in-context examples, prompt templates, and backends.

Input modes:
  - --stage5-run: single Stage 5 run dir
  - --stage5-list: text file with Stage 5 run dirs (one per line)
  - --single-patch: classify one or more patch PNGs (prints to stdout)
  - --rerun-from: Stage 6 output dir or patch PNG (loads metadata defaults)

Backends:
  - gemini: Google genai SDK (Vertex AI)
  - vllm: OpenAI-compatible vLLM server
  - openrouter: OpenRouter API (Gemini, GPT-4V, Claude, etc.)
"""

import argparse
import csv
import hashlib
import json
import math
import os
import random
import string
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from utils.patch_blur import compute_blur_from_patch
from utils.reproducibility import require_clean_state, create_reproduce_command
from utils.vlm_utils import (
    encode_image_base64,
    normalize_class_label,
    parse_vlm_output,
)
from utils.wsi_backend import (
    close_wsi,
    get_pyramid_info,
    load_wsi as load_wsi_backend,
    read_region_rgb,
)
from utils.wsi_paths import resolve_wsi_path

# =============================================================================
# Constants
# =============================================================================

QUALITY_LABELS = [
    "Sharp",
    "Somewhat Blurred",
    "Out of Focus",
    "NA",
]

DEFAULT_TISSUE_QUALITY = "Sharp"

CANONICAL_CLASS_ORDER = [
    "background",
    "tissue",
    "paraffin_mounting_medium",
    "pen_ink_marks",
]

DEFAULT_PROMPT_TEMPLATE = "prompts/vlm_patch_classify.txt"

DEFAULT_VLLM_URL = "http://localhost:8000/v1"
DEFAULT_VLLM_MODEL = "Qwen/Qwen3-VL-8B-Instruct-FP8"
DEFAULT_GEMINI_MODEL = "gemini-3-flash-preview"
DEFAULT_OPENROUTER_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_MODEL = "google/gemini-3-flash-preview"
DEFAULT_OPENROUTER_REFERER = "https://github.com/wsi-agents"

DEFAULT_CLASSICAL_BLUR_THRESHOLD = 0.1
DEFAULT_CLASSICAL_BLUR_SIGMA = 0.5
DEFAULT_CLASSICAL_BLUR_PIXEL_THRESHOLD = 0.005

PATCHES_CSV_BASE_HEADER = [
    "patch_id",
    "row",
    "col",
    "wsi_x",
    "wsi_y",
    "patch_w",
    "patch_h",
    "stage3_fg_ratio",
    "stage3_kept",
    "pred_label",
    "pred_label_canonical",
    "pred_blur",
    "classical_blur_score",
    "classical_sharp_score",
    "classical_blur_pass",
    "class_votes",
    "blur_votes",
    "vlm_runs",
    "batch_group_id",
    "batch_query_index",
    "batch_query_size",
    "batch_mode",
]


# =============================================================================
# Utility Helpers
# =============================================================================

def compute_config_hash(config: dict) -> str:
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()[:8]


def sanitize_model_name(model: str) -> str:
    return model.replace("/", "_").replace(":", "_").replace("-", "_")


def load_text(path: str) -> str:
    with open(path, "r") as f:
        return f.read()


def safe_format(text: str, mapping: dict) -> str:
    missing = set()

    class DefaultDict(dict):
        def __missing__(self, key):
            missing.add(key)
            return ""

    out = text.format_map(DefaultDict(mapping))
    if missing:
        print(f"Warning: Missing template keys: {sorted(missing)}", file=sys.stderr)
    return out


def parse_rotations(value) -> List[int]:
    if isinstance(value, list):
        return [int(v) for v in value]
    if not value:
        return [0]
    items = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        items.append(int(part))
    if not items:
        return [0]
    return items


def load_class_definitions(path: Optional[str]) -> Dict[str, str]:
    if not path:
        return {}
    with open(path, "r") as f:
        data = json.load(f)
    out: Dict[str, str] = {}
    if isinstance(data, dict):
        # Allow nested formats like {"descriptions": {...}}
        if isinstance(data.get("descriptions"), dict):
            data = data.get("descriptions")
        elif isinstance(data.get("class_descriptions"), dict):
            data = data.get("class_descriptions")
        if isinstance(data, dict):
            for k, v in data.items():
                key = normalize_class_label(str(k))
                out[key] = str(v).strip()
    return out


def build_class_def_block(
    class_defs: Dict[str, str],
    label_map: Dict[str, str],
    class_order: List[str]
) -> str:
    lines = []
    for canonical in class_order:
        label = label_map.get(canonical, canonical)
        desc = class_defs.get(canonical, "").strip()
        if desc:
            lines.append(f"- {label}: {desc}")
        else:
            lines.append(f"- {label}")
    return "\n".join(lines)


def format_example_label(model_label: str, quality: str) -> str:
    return f"Class: {model_label}\nQuality: {quality}"


def prepare_image_for_vlm(image: Image.Image, size: Optional[int]) -> Image.Image:
    img = image.convert("RGB")
    if size:
        img = img.resize((size, size), resample=Image.BICUBIC)
    return img


def build_patch_id(info: dict) -> str:
    return f"r{info['row']}_c{info['col']}_x{info['x1']}_y{info['y1']}"


def build_patches_csv_header(save_variants: bool) -> List[str]:
    header = list(PATCHES_CSV_BASE_HEADER)
    if save_variants:
        header.append("variants_json")
    return header


def result_to_csv_row(res: dict, save_variants: bool) -> List[Any]:
    info = res["patch"]
    classical_blur_score, classical_sharp_score, classical_blur_pass = get_result_classical_blur_fields(res)
    batch_group_id = info.get("batch_group_id")
    batch_query_index = info.get("batch_query_index")
    batch_query_size = info.get("batch_query_size")
    batch_mode = info.get("batch_mode", "single")
    row = [
        build_patch_id(info),
        info["row"],
        info["col"],
        info["x1"],
        info["y1"],
        info["x2"] - info["x1"],
        info["y2"] - info["y1"],
        info.get("stage3_fg_ratio"),
        bool(info.get("stage3_keep")),
        res.get("pred_label"),
        res.get("pred_label_canonical"),
        res.get("pred_blur"),
        classical_blur_score,
        classical_sharp_score,
        "" if classical_blur_pass is None else int(classical_blur_pass),
        json.dumps(res.get("class_votes", {})),
        json.dumps(res.get("blur_votes", {})),
        len(res.get("runs", [])),
        "" if batch_group_id is None else str(batch_group_id),
        "" if batch_query_index is None else int(batch_query_index),
        "" if batch_query_size is None else int(batch_query_size),
        str(batch_mode),
    ]
    if save_variants:
        row.append(json.dumps(res.get("runs", [])))
    return row


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "t", "yes", "y"}


def _parse_json_object(value: Any) -> Dict[str, Any]:
    if value in (None, ""):
        return {}
    try:
        parsed = json.loads(value)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _parse_optional_float(value: Any) -> Optional[float]:
    if value in (None, "", "None", "none", "nan", "NaN"):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _parse_optional_int(value: Any) -> Optional[int]:
    if value in (None, "", "None", "none"):
        return None
    try:
        return int(value)
    except Exception:
        return None


def _parse_optional_bool(value: Any) -> Optional[bool]:
    if value in (None, "", "None", "none"):
        return None
    return _parse_bool(value)


def compute_classical_blur_metrics(
    patch_img: Image.Image,
    sigma: float,
    pixel_threshold: float,
    patch_threshold: float,
) -> Tuple[Optional[float], Optional[float], Optional[bool]]:
    try:
        blur_result = compute_blur_from_patch(
            patch_rgb=np.asarray(patch_img.convert("RGB")),
            sigma=sigma,
            pixel_threshold=pixel_threshold,
        )
    except Exception:
        return None, None, None

    raw_blur = getattr(blur_result, "blur_score", None)
    raw_sharp = getattr(blur_result, "sharp_score", None)
    blur_score = float(raw_blur) if raw_blur is not None else None
    sharp_score = float(raw_sharp) if raw_sharp is not None else None
    if blur_score is not None and not np.isfinite(blur_score):
        blur_score = None
    if sharp_score is not None and not np.isfinite(sharp_score):
        sharp_score = None
    blur_pass = (blur_score <= patch_threshold) if blur_score is not None else None
    return blur_score, sharp_score, blur_pass


def get_result_classical_blur_fields(
    res: dict,
) -> Tuple[Optional[float], Optional[float], Optional[bool]]:
    patch = res.get("patch", {})
    if not isinstance(patch, dict):
        patch = {}

    blur_score = _parse_optional_float(res.get("classical_blur_score"))
    sharp_score = _parse_optional_float(res.get("classical_sharp_score"))
    blur_pass = _parse_optional_bool(res.get("classical_blur_pass"))

    if blur_score is None:
        blur_score = _parse_optional_float(patch.get("classical_blur_score"))
    if sharp_score is None:
        sharp_score = _parse_optional_float(patch.get("classical_sharp_score"))
    if blur_pass is None:
        blur_pass = _parse_optional_bool(patch.get("classical_blur_pass"))

    return blur_score, sharp_score, blur_pass


def set_result_classical_blur_fields(
    res: dict,
    blur_score: Optional[float],
    sharp_score: Optional[float],
    blur_pass: Optional[bool],
) -> None:
    res["classical_blur_score"] = blur_score
    res["classical_sharp_score"] = sharp_score
    res["classical_blur_pass"] = blur_pass

    patch = res.get("patch")
    if isinstance(patch, dict):
        patch["classical_blur_score"] = blur_score
        patch["classical_sharp_score"] = sharp_score
        patch["classical_blur_pass"] = blur_pass


def load_existing_patch_results(csv_path: Path) -> Tuple[Dict[str, dict], int, bool]:
    """Load completed patch rows from patches.csv for resume."""
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return {}, 0, False

    results: Dict[str, dict] = {}
    skipped_rows = 0
    has_variants_col = False

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if "patch_id" not in fieldnames:
            return {}, 0, False
        has_variants_col = "variants_json" in fieldnames

        for row in reader:
            try:
                patch_id = (row.get("patch_id") or "").strip()
                if not patch_id:
                    skipped_rows += 1
                    continue

                row_idx = int(row.get("row"))
                col_idx = int(row.get("col"))
                x1 = int(row.get("wsi_x"))
                y1 = int(row.get("wsi_y"))
                patch_w = int(row.get("patch_w"))
                patch_h = int(row.get("patch_h"))
                x2 = x1 + patch_w
                y2 = y1 + patch_h

                stage3_fg_ratio = _parse_optional_float(row.get("stage3_fg_ratio"))
                stage3_keep = _parse_bool(row.get("stage3_kept"))
                pred_label = row.get("pred_label") or row.get("pred_label_canonical") or "background"
                pred_label_canonical = row.get("pred_label_canonical") or normalize_class_label(pred_label) or "background"
                pred_blur = row.get("pred_blur") or "NA"
                classical_blur_score = _parse_optional_float(row.get("classical_blur_score"))
                classical_sharp_score = _parse_optional_float(row.get("classical_sharp_score"))
                classical_blur_pass = _parse_optional_bool(row.get("classical_blur_pass"))
                class_votes = _parse_json_object(row.get("class_votes"))
                blur_votes = _parse_json_object(row.get("blur_votes"))
                vlm_runs = int(row.get("vlm_runs") or 0)
                batch_group_id = (row.get("batch_group_id") or "").strip() or None
                batch_query_index = _parse_optional_int(row.get("batch_query_index"))
                batch_query_size = _parse_optional_int(row.get("batch_query_size"))
                batch_mode = (row.get("batch_mode") or "").strip() or "single"
                variants = []
                if has_variants_col:
                    raw_variants = row.get("variants_json")
                    if raw_variants not in (None, ""):
                        try:
                            parsed_variants = json.loads(raw_variants)
                            if isinstance(parsed_variants, list):
                                variants = parsed_variants
                        except Exception:
                            variants = []
                if not variants and vlm_runs > 0:
                    variants = [None] * vlm_runs

                results[patch_id] = {
                    "patch": {
                        "patch_id": patch_id,
                        "row": row_idx,
                        "col": col_idx,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "stage3_fg_ratio": stage3_fg_ratio,
                        "stage3_keep": stage3_keep,
                        "batch_group_id": batch_group_id,
                        "batch_query_index": batch_query_index,
                        "batch_query_size": batch_query_size,
                        "batch_mode": batch_mode,
                        "classical_blur_score": classical_blur_score,
                        "classical_sharp_score": classical_sharp_score,
                        "classical_blur_pass": classical_blur_pass,
                    },
                    "pred_label": pred_label,
                    "pred_label_canonical": pred_label_canonical,
                    "pred_blur": pred_blur,
                    "classical_blur_score": classical_blur_score,
                    "classical_sharp_score": classical_sharp_score,
                    "classical_blur_pass": classical_blur_pass,
                    "class_votes": class_votes,
                    "blur_votes": blur_votes,
                    "runs": variants,
                    "raw_responses": [],
                }
            except Exception:
                skipped_rows += 1

    return results, skipped_rows, has_variants_col


def find_existing_output_dir(
    out_root: Path,
    wsi_id: str,
    model_dir: str,
    config_hash: str,
) -> Optional[Path]:
    """Find latest existing Stage 6 output dir for this config hash."""
    candidates: List[Path] = []

    # Current layout: {out_root}/{wsi_id}/{model}/{timestamp_hash}/
    direct_root = out_root / wsi_id / model_dir
    if direct_root.exists():
        for p in direct_root.glob(f"*_{config_hash}"):
            if p.is_dir():
                candidates.append(p)

    # Legacy pipeline temp layout:
    # {out_root}/attempt_*/{wsi_id}/{model}/{timestamp_hash}/
    for p in out_root.glob(f"attempt_*/{wsi_id}/{model_dir}/*_{config_hash}"):
        if p.is_dir():
            candidates.append(p)

    if not candidates:
        return None
    candidates.sort(key=lambda p: p.name)
    return candidates[-1]


# =============================================================================
# Stage 5 ICL Loading
# =============================================================================

def load_stage5_metadata(stage5_run: str) -> dict:
    meta_path = os.path.join(stage5_run, "metadata.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Stage 5 metadata not found: {meta_path}")
    with open(meta_path, "r") as f:
        return json.load(f)


def infer_classes_from_stage5(meta: dict, icl_pool: Dict[str, List[str]]) -> List[str]:
    classes = meta.get("classes_present") or meta.get("classes_ranked") or list(icl_pool.keys())
    classes = [normalize_class_label(c) for c in classes if c]
    present = [c for c in CANONICAL_CLASS_ORDER if c in classes]
    if not present:
        present = [c for c in CANONICAL_CLASS_ORDER if c in icl_pool]
    return present


def load_icl_pool(stage5_run: str) -> Dict[str, List[str]]:
    meta_path = os.path.join(stage5_run, "metadata.json")
    pool: Dict[str, List[str]] = {}
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            output = meta.get("output")
            if isinstance(output, dict):
                for class_name, rel_paths in output.items():
                    label = normalize_class_label(class_name)
                    if not label:
                        continue
                    pool[label] = [os.path.join(stage5_run, p) for p in rel_paths]
                if pool:
                    return pool
        except Exception:
            pass

    # Fallback: scan subdirs
    for name in sorted(os.listdir(stage5_run)):
        dir_path = os.path.join(stage5_run, name)
        if not os.path.isdir(dir_path):
            continue
        if name in ("intermediate", "__pycache__"):
            continue
        label = normalize_class_label(name)
        if not label:
            continue
        files = sorted([
            os.path.join(dir_path, f)
            for f in os.listdir(dir_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])
        if files:
            pool[label] = files
    return pool


def sample_icl_examples(
    icl_pool: Dict[str, List[str]],
    class_order: List[str],
    per_class: int,
    seed: int
) -> List[dict]:
    rng = random.Random(seed)
    examples = []
    for cls in class_order:
        candidates = list(icl_pool.get(cls, []))
        rng.shuffle(candidates)
        selected = candidates[:per_class] if per_class and per_class > 0 else []
        for path in selected:
            examples.append({"label": cls, "path": path})
    return examples


# =============================================================================
# Stage 3 gating helpers
# =============================================================================

def resolve_stage3_bbox_dir(stage3_run: str, bbox: Tuple[int, int, int, int]) -> str:
    if not stage3_run:
        raise ValueError("stage3_run is required")

    mask_path = os.path.join(stage3_run, "mask.png")
    meta_path = os.path.join(stage3_run, "metadata.json")
    if os.path.isfile(mask_path) and os.path.isfile(meta_path):
        return stage3_run

    bbox_str = f"{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}"
    direct_dir = os.path.join(stage3_run, bbox_str)
    if os.path.isfile(os.path.join(direct_dir, "mask.png")) and os.path.isfile(os.path.join(direct_dir, "metadata.json")):
        return direct_dir

    matches = []
    base_depth = stage3_run.rstrip(os.sep).count(os.sep)
    for root, dirs, files in os.walk(stage3_run):
        depth = root.count(os.sep) - base_depth
        if depth > 4:
            dirs[:] = []
            continue
        if os.path.basename(root) != bbox_str:
            continue
        if "mask.png" in files and "metadata.json" in files:
            matches.append(root)

    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"Stage 3 bbox directory not found for {bbox_str} under {stage3_run}")
    raise FileNotFoundError(f"Multiple Stage 3 bbox directories found for {bbox_str} under {stage3_run}")


def load_stage3_info(stage3_run: str, bbox: Tuple[int, int, int, int]) -> dict:
    bbox_dir = resolve_stage3_bbox_dir(stage3_run, bbox)
    mask_path = os.path.join(bbox_dir, "mask.png")
    meta_path = os.path.join(bbox_dir, "metadata.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)

    scale = meta.get("scale_factor") or {}
    scale_x = float(scale.get("x", 1.0))
    scale_y = float(scale.get("y", 1.0))
    bbox_level0 = meta.get("bbox_level0") or meta.get("bbox") or list(bbox)

    mask_img = Image.open(mask_path).convert("L")
    mask = np.array(mask_img) > 0

    return {
        "bbox_dir": bbox_dir,
        "mask": mask,
        "scale_x": scale_x,
        "scale_y": scale_y,
        "bbox_level0": bbox_level0,
        "meta": meta,
    }


def compute_stage3_fg_ratio(
    stage3_info: dict,
    patch_x1: int,
    patch_y1: int,
    patch_x2: int,
    patch_y2: int
) -> float:
    mask = stage3_info["mask"]
    bbox_x1, bbox_y1, _, _ = stage3_info["bbox_level0"]
    scale_x = stage3_info["scale_x"]
    scale_y = stage3_info["scale_y"]

    mx1 = int((patch_x1 - bbox_x1) / scale_x)
    my1 = int((patch_y1 - bbox_y1) / scale_y)
    mx2 = int(math.ceil((patch_x2 - bbox_x1) / scale_x))
    my2 = int(math.ceil((patch_y2 - bbox_y1) / scale_y))

    h, w = mask.shape[:2]
    mx1 = max(0, min(mx1, w))
    mx2 = max(0, min(mx2, w))
    my1 = max(0, min(my1, h))
    my2 = max(0, min(my2, h))

    if mx2 <= mx1 or my2 <= my1:
        return 0.0

    region = mask[my1:my2, mx1:mx2]
    return float(region.mean())


# =============================================================================
# WSI Helpers
# =============================================================================

def load_wsi(wsi_path: str, wsi_reader: str = "cucim"):
    return load_wsi_backend(wsi_path, wsi_reader)


def extract_patch(
    wsi,
    wsi_backend: str,
    x: int,
    y: int,
    width: int,
    height: int
) -> Image.Image:
    patch_np = read_region_rgb(
        wsi,
        wsi_backend,
        x=x,
        y=y,
        width=width,
        height=height,
        level=0,
    )
    return Image.fromarray(patch_np)


def _compute_thumb_size(width: int, height: int, max_dim: int) -> Tuple[int, int]:
    scale = min(max_dim / width, max_dim / height, 1.0)
    thumb_w = max(1, int(width * scale))
    thumb_h = max(1, int(height * scale))
    return thumb_w, thumb_h


def _read_bbox_at_level(
    wsi,
    wsi_backend: str,
    bbox: Tuple[int, int, int, int],
    level: int,
    downsample: float,
) -> Image.Image:
    x1, y1, x2, y2 = bbox
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    ds = max(float(downsample), 1e-6)
    read_w = max(1, int(math.ceil(width / ds)))
    read_h = max(1, int(math.ceil(height / ds)))
    region = read_region_rgb(
        wsi,
        wsi_backend,
        x=x1,
        y=y1,
        width=read_w,
        height=read_h,
        level=level,
    )
    return Image.fromarray(region).convert("RGB")


def _extract_bbox_thumbnail_tiled_l0(
    wsi,
    wsi_backend: str,
    bbox: Tuple[int, int, int, int],
    max_dim: int,
    src_tile_size: int = 4096,
) -> Image.Image:
    x1, y1, x2, y2 = bbox
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    thumb_w, thumb_h = _compute_thumb_size(width, height, max_dim)
    scale_x = thumb_w / width
    scale_y = thumb_h / height
    thumb = Image.new("RGB", (thumb_w, thumb_h), color=(255, 255, 255))

    for sy in range(y1, y2, src_tile_size):
        sh = min(src_tile_size, y2 - sy)
        if sh <= 0:
            continue
        for sx in range(x1, x2, src_tile_size):
            sw = min(src_tile_size, x2 - sx)
            if sw <= 0:
                continue

            tile_np = read_region_rgb(
                wsi,
                wsi_backend,
                x=sx,
                y=sy,
                width=sw,
                height=sh,
                level=0,
            )
            tile_img = Image.fromarray(tile_np).convert("RGB")

            ox1 = int(math.floor((sx - x1) * scale_x))
            oy1 = int(math.floor((sy - y1) * scale_y))
            ox2 = int(math.ceil((sx + sw - x1) * scale_x))
            oy2 = int(math.ceil((sy + sh - y1) * scale_y))

            ox1 = max(0, min(ox1, thumb_w))
            oy1 = max(0, min(oy1, thumb_h))
            ox2 = max(0, min(ox2, thumb_w))
            oy2 = max(0, min(oy2, thumb_h))
            if ox2 <= ox1 or oy2 <= oy1:
                continue

            out_w = ox2 - ox1
            out_h = oy2 - oy1
            tile_img = tile_img.resize((out_w, out_h), Image.LANCZOS)
            thumb.paste(tile_img, (ox1, oy1))

    return thumb


def _load_cached_bbox_thumbnail(
    stage2_input: Optional[str],
    bbox: Tuple[int, int, int, int],
    max_dim: int,
) -> Tuple[Optional[Image.Image], Optional[Dict[str, Any]], Optional[str]]:
    if not stage2_input:
        return None, None, None

    x1, y1, x2, y2 = bbox
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    thumb_w, thumb_h = _compute_thumb_size(width, height, max_dim)

    stage2_path = Path(stage2_input)
    bbox_str = "_".join(str(int(v)) for v in bbox)
    candidates: List[Path] = []
    if stage2_path.is_file():
        candidates.append(stage2_path)
    else:
        candidates.append(stage2_path / "bbox_region.png")
        candidates.append(stage2_path / bbox_str / "bbox_region.png")

    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            with Image.open(candidate) as img:
                cached = img.convert("RGB")
            if cached.size != (thumb_w, thumb_h):
                cached = cached.resize((thumb_w, thumb_h), Image.LANCZOS)
            effective_ds = max(width / max(1, cached.size[0]), height / max(1, cached.size[1]))
            overlay_read = {
                "strategy": "stage2_bbox_region_cache",
                "level": 0,
                "downsample": float(effective_ds),
                "reason": f"loaded cached bbox thumbnail from {candidate}",
                "error": None,
                "target_max_dim": int(max_dim),
                "bbox_size_level0": {"width": int(width), "height": int(height)},
                "thumbnail_size": {"width": int(thumb_w), "height": int(thumb_h)},
            }
            return cached, overlay_read, None
        except Exception as exc:
            return None, None, f"stage2_cache_error: {type(exc).__name__}: {exc}"

    return None, None, "stage2_cache_missing: bbox_region.png not found"


def extract_bbox_thumbnail(
    wsi,
    wsi_backend: str,
    bbox: Tuple[int, int, int, int],
    max_dim: int = 1024,
    stage2_input: Optional[str] = None,
) -> Tuple[Image.Image, Dict[str, Any]]:
    x1, y1, x2, y2 = bbox
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    thumb_w, thumb_h = _compute_thumb_size(width, height, max_dim)

    overlay_read: Dict[str, Any] = {
        "strategy": None,
        "level": None,
        "downsample": None,
        "reason": None,
        "error": None,
        "target_max_dim": int(max_dim),
        "bbox_size_level0": {"width": int(width), "height": int(height)},
        "thumbnail_size": {"width": int(thumb_w), "height": int(thumb_h)},
    }

    errors: List[str] = []
    cached_img, cached_read, cache_error = _load_cached_bbox_thumbnail(stage2_input, bbox, max_dim)
    if cached_img is not None and cached_read is not None:
        return cached_img, cached_read
    if cache_error:
        errors.append(cache_error)

    pyramid: Optional[dict] = None
    try:
        pyramid = get_pyramid_info(wsi, wsi_backend)
    except Exception as exc:
        errors.append(f"pyramid_info_error: {type(exc).__name__}: {exc}")

    # Primary strategy: best-fit level for max_dim.
    if pyramid:
        try:
            level_count = int(pyramid.get("level_count", 0))
            downsamples = list(pyramid.get("level_downsamples", []))
            best_level = 0
            best_downsample = 1.0
            best_diff = float("inf")
            for level in range(level_count):
                ds = float(downsamples[level]) if level < len(downsamples) else float(2**level)
                ds = max(ds, 1e-6)
                projected_max = max(width / ds, height / ds)
                diff = abs(projected_max - max_dim)
                if diff < best_diff:
                    best_diff = diff
                    best_level = level
                    best_downsample = ds
            img = _read_bbox_at_level(wsi, wsi_backend, bbox, best_level, best_downsample)
            if img.size != (thumb_w, thumb_h):
                img = img.resize((thumb_w, thumb_h), Image.LANCZOS)
            overlay_read.update({
                "strategy": "best_fit_level",
                "level": int(best_level),
                "downsample": float(best_downsample),
                "reason": "selected pyramid level with bbox size closest to overlay_max_dim",
                "error": None,
            })
            return img, overlay_read
        except Exception as exc:
            errors.append(f"best_fit_error: {type(exc).__name__}: {exc}")

    # Secondary strategy: deterministic hard rule by level count.
    if pyramid:
        try:
            level_count = int(pyramid.get("level_count", 0))
            downsamples = list(pyramid.get("level_downsamples", []))
            if level_count <= 1:
                fallback_level = 0
            elif level_count == 2:
                fallback_level = level_count - 1
            else:
                fallback_level = level_count - 2
            fallback_level = max(0, min(fallback_level, max(0, level_count - 1)))
            fallback_ds = float(downsamples[fallback_level]) if fallback_level < len(downsamples) else float(2**fallback_level)
            fallback_ds = max(fallback_ds, 1e-6)
            img = _read_bbox_at_level(wsi, wsi_backend, bbox, fallback_level, fallback_ds)
            if img.size != (thumb_w, thumb_h):
                img = img.resize((thumb_w, thumb_h), Image.LANCZOS)
            overlay_read.update({
                "strategy": "hard_rule_level_fallback",
                "level": int(fallback_level),
                "downsample": float(fallback_ds),
                "reason": "best-fit failed or unavailable; used deterministic level-count fallback",
                "error": "; ".join(errors) if errors else None,
            })
            return img, overlay_read
        except Exception as exc:
            errors.append(f"hard_rule_error: {type(exc).__name__}: {exc}")

    # Final fallback: tiled L0 read + downsample (never full-bbox allocation).
    img = _extract_bbox_thumbnail_tiled_l0(wsi, wsi_backend, bbox, max_dim=max_dim)
    effective_ds = max(width / max(1, img.size[0]), height / max(1, img.size[1]))
    overlay_read.update({
        "strategy": "tiled_l0_downsample_fallback",
        "level": 0,
        "downsample": float(effective_ds),
        "reason": "level-based overlay extraction failed; used memory-bounded tiled L0 fallback",
        "error": "; ".join(errors) if errors else None,
    })
    return img, overlay_read


# =============================================================================
# Prompt Template Building
# =============================================================================

def build_examples_block(examples: List[dict]) -> List[dict]:
    parts: List[dict] = []
    for idx, ex in enumerate(examples):
        parts.append({"type": "text", "text": f"Example {idx + 1}:\n"})
        parts.append({"type": "image", "image": ex.get("image_payload", ex["image"])})
        parts.append({"type": "text", "text": ex["label_text"] + "\n"})
    return parts


def build_examples_text(examples: List[dict]) -> str:
    lines: List[str] = []
    for idx, ex in enumerate(examples):
        lines.append(f"Example {idx + 1}:")
        lines.append("[EXAMPLE_IMAGE]")
        lines.append(ex["label_text"])
        lines.append("")
    return "\n".join(lines).strip()


def build_prompt_parts(
    template: str,
    text_map: dict,
    image_map: dict,
    examples_block: List[dict]
) -> List[dict]:
    parts: List[dict] = []
    formatter = string.Formatter()
    buffer = ""

    for literal_text, field_name, format_spec, conversion in formatter.parse(template):
        buffer += literal_text.replace("{", "{{").replace("}", "}}")
        if field_name is None:
            continue

        if field_name == "EXAMPLES_BLOCK":
            text = safe_format(buffer, text_map)
            if text.strip():
                parts.append({"type": "text", "text": text})
            buffer = ""
            parts.extend(examples_block)
            continue

        if field_name in image_map:
            text = safe_format(buffer, text_map)
            if text.strip():
                parts.append({"type": "text", "text": text})
            buffer = ""
            parts.append({"type": "image", "image": image_map[field_name]})
            continue

        # Keep as a placeholder for text formatting later
        buffer += "{" + field_name + "}"

    # Flush remaining text
    text = safe_format(buffer, text_map)
    if text.strip():
        parts.append({"type": "text", "text": text})

    return parts


# =============================================================================
# Backend Runners
# =============================================================================

class GeminiRunner:
    def __init__(
        self,
        model: str,
        temperature: float,
        max_tokens: int,
        max_retries: int,
        use_vertex: bool,
        credentials_path: Optional[str],
        location: str,
        thinking_level: Optional[str] = None,
        include_thoughts: bool = False,
    ):
        from google import genai
        from google.genai import types

        if use_vertex:
            creds_path = Path(credentials_path) if credentials_path else None
            if creds_path and creds_path.exists():
                with open(creds_path, "r") as f:
                    creds = json.load(f)
                project_id = creds.get("project_id")
                if project_id:
                    os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(creds_path.absolute())
            elif creds_path and not creds_path.exists():
                if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
                    raise FileNotFoundError(f"Credentials file not found: {credentials_path}")
                print(f"Warning: Credentials file not found at {credentials_path}, using env var", file=sys.stderr)
            elif not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
                raise ValueError(
                    "Gemini Vertex mode requires credentials. Provide --gemini-credentials "
                    "or set GOOGLE_APPLICATION_CREDENTIALS."
                )
            os.environ["GOOGLE_CLOUD_LOCATION"] = location
            os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"

        self.client = genai.Client()
        self.model = model
        self.max_retries = max_retries
        self.include_thoughts = include_thoughts
        self.thinking_level = thinking_level

        config_kwargs = dict(
            temperature=temperature,
            max_output_tokens=max_tokens if max_tokens else None,
        )
        if thinking_level:
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_level=thinking_level,
                include_thoughts=include_thoughts,
            )
        self.config = types.GenerateContentConfig(**config_kwargs)

    def run(self, parts: List[dict]) -> str:
        contents = []
        for part in parts:
            if part["type"] == "text":
                contents.append(part["text"])
            else:
                img = part["image"]
                if isinstance(img, dict):
                    img = img.get("pil", img)
                contents.append(img)

        for attempt in range(self.max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=self.config,
                )
                return (response.text or "").strip()
            except Exception as e:
                if attempt == self.max_retries - 1:
                    print(f"Warning: Gemini call failed: {e}", file=sys.stderr)
                    return ""
                time.sleep(2 ** attempt)

        return ""

    def run_full(self, parts: List[dict]) -> dict:
        """Run and return full response with thought parts and usage metadata."""
        contents = []
        for part in parts:
            if part["type"] == "text":
                contents.append(part["text"])
            else:
                img = part["image"]
                if isinstance(img, dict):
                    img = img.get("pil", img)
                contents.append(img)

        for attempt in range(self.max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=self.config,
                )
                finish_reason = None
                candidates = getattr(response, "candidates", None) or []
                if candidates:
                    first_finish_reason = getattr(candidates[0], "finish_reason", None)
                    if first_finish_reason is not None:
                        finish_reason = getattr(first_finish_reason, "name", str(first_finish_reason))

                result = {
                    "text": (response.text or "").strip(),
                    "thoughts": [],
                    "usage": {},
                    "finish_reason": finish_reason,
                    "error": None,
                    "attempts": attempt + 1,
                }
                if hasattr(response, "candidates") and response.candidates:
                    for candidate in response.candidates:
                        if hasattr(candidate, "content") and candidate.content:
                            for p in (getattr(candidate.content, "parts", None) or []):
                                if getattr(p, "thought", False):
                                    thought_text = getattr(p, "text", None)
                                    if thought_text:
                                        result["thoughts"].append(thought_text)
                if hasattr(response, "usage_metadata") and response.usage_metadata:
                    um = response.usage_metadata
                    result["usage"] = {
                        "prompt_tokens": getattr(um, "prompt_token_count", None),
                        "output_tokens": getattr(um, "candidates_token_count", None),
                        "thoughts_tokens": getattr(um, "thoughts_token_count", None),
                        "total_tokens": getattr(um, "total_token_count", None),
                    }
                return result
            except Exception as e:
                if attempt == self.max_retries - 1:
                    print(f"Warning: Gemini call failed: {e}", file=sys.stderr)
                    return {
                        "text": "",
                        "thoughts": [],
                        "usage": {},
                        "finish_reason": None,
                        "error": str(e),
                        "attempts": attempt + 1,
                    }
                time.sleep(2 ** attempt)

        return {
            "text": "",
            "thoughts": [],
            "usage": {},
            "finish_reason": None,
            "error": "unknown_error",
            "attempts": self.max_retries,
        }


class VLLMRunner:
    def __init__(
        self,
        model: str,
        url: str,
        timeout: int,
        temperature: float,
        max_tokens: int,
        max_retries: int,
    ):
        self.model = model
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries

    def run(self, parts: List[dict]) -> str:
        content = []
        for part in parts:
            if part["type"] == "text":
                content.append({"type": "text", "text": part["text"]})
            else:
                img = part["image"]
                if isinstance(img, dict) and "b64" in img:
                    b64 = img["b64"]
                else:
                    if isinstance(img, dict):
                        img = img.get("pil", img)
                    b64 = encode_image_base64(img, resize=False)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"}
                })

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        for attempt in range(self.max_retries):
            try:
                resp = requests.post(
                    f"{self.url}/chat/completions",
                    json=payload,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"].strip()
            except Exception as e:
                if attempt == self.max_retries - 1:
                    print(f"Warning: vLLM call failed: {e}", file=sys.stderr)
                    return ""
                time.sleep(2 ** attempt)
        return ""


class OpenRouterRunner:
    def __init__(
        self,
        model: str,
        api_key: Optional[str],
        url: str,
        timeout: int,
        temperature: float,
        max_tokens: int,
        max_retries: int,
        referer: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key required. Set OPENROUTER_API_KEY or OPENAI_API_KEY, "
                "or pass --openrouter-api-key."
            )
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.referer = referer
        self.reasoning_effort = reasoning_effort

    def _headers(self) -> dict:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.referer:
            headers["HTTP-Referer"] = self.referer
        return headers

    def run(self, parts: List[dict]) -> str:
        content = []
        for part in parts:
            if part["type"] == "text":
                content.append({"type": "text", "text": part["text"]})
            else:
                img = part["image"]
                if isinstance(img, dict) and "b64" in img:
                    b64 = img["b64"]
                else:
                    if isinstance(img, dict):
                        img = img.get("pil", img)
                    b64 = encode_image_base64(img, resize=False)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"}
                })

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if self.reasoning_effort:
            payload["reasoning"] = {"effort": self.reasoning_effort}

        for attempt in range(self.max_retries):
            try:
                resp = requests.post(
                    f"{self.url}/chat/completions",
                    json=payload,
                    headers=self._headers(),
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"].strip()
            except Exception as e:
                if attempt == self.max_retries - 1:
                    print(f"Warning: OpenRouter call failed: {e}", file=sys.stderr)
                    return ""
                time.sleep(2 ** attempt)
        return ""

    def run_full(self, parts: List[dict]) -> dict:
        """Run and return full response with reasoning details."""
        content = []
        for part in parts:
            if part["type"] == "text":
                content.append({"type": "text", "text": part["text"]})
            else:
                img = part["image"]
                if isinstance(img, dict) and "b64" in img:
                    b64 = img["b64"]
                else:
                    if isinstance(img, dict):
                        img = img.get("pil", img)
                    b64 = encode_image_base64(img, resize=False)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"}
                })

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if self.reasoning_effort:
            payload["reasoning"] = {"effort": self.reasoning_effort}

        for attempt in range(self.max_retries):
            try:
                resp = requests.post(
                    f"{self.url}/chat/completions",
                    json=payload,
                    headers=self._headers(),
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                msg = data["choices"][0]["message"]
                finish_reason = None
                choices = data.get("choices") or []
                if choices:
                    finish_reason = choices[0].get("finish_reason")
                result = {
                    "text": (msg.get("content") or "").strip(),
                    "thoughts": [],
                    "usage": {},
                    "finish_reason": finish_reason,
                    "error": None,
                    "attempts": attempt + 1,
                }
                reasoning = msg.get("reasoning_details") or msg.get("reasoning_content")
                if reasoning:
                    if isinstance(reasoning, list):
                        result["thoughts"] = [r.get("text", str(r)) for r in reasoning if isinstance(r, dict)]
                    elif isinstance(reasoning, str):
                        result["thoughts"] = [reasoning]
                usage = data.get("usage", {})
                if usage:
                    result["usage"] = {
                        "prompt_tokens": usage.get("prompt_tokens"),
                        "output_tokens": usage.get("completion_tokens"),
                        "thoughts_tokens": usage.get("reasoning_tokens"),
                        "total_tokens": usage.get("total_tokens"),
                    }
                return result
            except Exception as e:
                if attempt == self.max_retries - 1:
                    print(f"Warning: OpenRouter call failed: {e}", file=sys.stderr)
                    return {
                        "text": "",
                        "thoughts": [],
                        "usage": {},
                        "finish_reason": None,
                        "error": str(e),
                        "attempts": attempt + 1,
                    }
                time.sleep(2 ** attempt)
        return {
            "text": "",
            "thoughts": [],
            "usage": {},
            "finish_reason": None,
            "error": "unknown_error",
            "attempts": self.max_retries,
        }


# =============================================================================
# Overlay Rendering
# =============================================================================

def overlay_grid(
    base_img: Image.Image,
    bbox: Tuple[int, int, int, int],
    patch_infos: List[dict],
    color_fn,
    alpha: int = 120
) -> Image.Image:
    base_rgba = base_img.convert("RGBA")
    overlay = np.zeros((base_rgba.height, base_rgba.width, 4), dtype=np.uint8)

    x1, y1, x2, y2 = bbox
    bbox_w = x2 - x1
    bbox_h = y2 - y1
    scale_x = base_rgba.width / bbox_w if bbox_w else 1.0
    scale_y = base_rgba.height / bbox_h if bbox_h else 1.0

    for info in patch_infos:
        color = color_fn(info)
        if color is None:
            continue
        px1 = int((info["x1"] - x1) * scale_x)
        py1 = int((info["y1"] - y1) * scale_y)
        px2 = int((info["x2"] - x1) * scale_x)
        py2 = int((info["y2"] - y1) * scale_y)
        if px2 <= px1 or py2 <= py1:
            continue
        r, g, b = color
        overlay[py1:py2, px1:px2] = [r, g, b, alpha]

    overlay_img = Image.fromarray(overlay, mode="RGBA")
    return Image.alpha_composite(base_rgba, overlay_img)


def save_bbox_grid_overlay(
    base_overlay: Image.Image,
    bbox: Tuple[int, int, int, int],
    patch_size: int,
    rows: int,
    cols: int,
    out_path: Path,
) -> None:
    """Save class overlay with row/col grid labels for patch CSV lookup."""
    x1, y1, x2, y2 = bbox
    bbox_w = max(1, x2 - x1)
    bbox_h = max(1, y2 - y1)
    img_w, img_h = base_overlay.size
    scale_x = img_w / bbox_w
    scale_y = img_h / bbox_h

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(np.array(base_overlay.convert("RGB")))

    x_ticks = []
    x_labels = []
    for c in range(cols + 1):
        x_pos = c * patch_size * scale_x
        if 0 <= x_pos <= img_w:
            x_ticks.append(x_pos)
            x_labels.append(str(c))

    y_ticks = []
    y_labels = []
    for r in range(rows + 1):
        y_pos = r * patch_size * scale_y
        if 0 <= y_pos <= img_h:
            y_ticks.append(y_pos)
            y_labels.append(str(r))

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.grid(True, color="white", linewidth=0.5, alpha=0.7)
    ax.set_xlabel("col (patch column index)")
    ax.set_ylabel("row (patch row index)")
    ax.set_title("Bbox: row/col grid for CSV lookup (high-res)")

    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


# =============================================================================
# Core Processing
# =============================================================================

def build_label_map(
    class_order: List[str],
    label_mode: str
) -> Tuple[Dict[str, str], Dict[str, str], List[str]]:
    if label_mode == "semantic":
        mapping = {c: c for c in class_order}
        reverse = {normalize_class_label(c): c for c in class_order}
        return mapping, reverse, class_order

    # neutral
    ordered = []
    if "background" in class_order:
        ordered.append("background")
    if "tissue" in class_order:
        ordered.append("tissue")
    for c in class_order:
        if c not in ordered:
            ordered.append(c)

    mapping = {c: f"CLASS_{i}" for i, c in enumerate(ordered)}
    reverse = {normalize_class_label(v): k for k, v in mapping.items()}
    return mapping, reverse, ordered


def prepare_examples(
    examples: List[dict],
    label_map: Dict[str, str],
    image_size: Optional[int]
) -> List[dict]:
    prepared = []
    for ex in examples:
        img = Image.open(ex["path"]).convert("RGB")
        img = prepare_image_for_vlm(img, image_size)
        model_label = label_map.get(ex["label"], ex["label"])
        quality = DEFAULT_TISSUE_QUALITY if ex["label"] == "tissue" else "NA"
        label_text = format_example_label(model_label, quality)
        image_payload = {
            "pil": img,
            "b64": encode_image_base64(img, resize=False),
        }
        prepared.append({
            "label": ex["label"],
            "model_label": model_label,
            "path": ex["path"],
            "image": img,
            "image_b64": image_payload["b64"],
            "image_payload": image_payload,
            "label_text": label_text,
            "quality": quality,
        })
    return prepared


def build_variants(
    examples: List[dict],
    shuffle_n: int,
    rotations: List[int],
    seed: int
) -> List[dict]:
    variants = []
    shuffle_n = max(1, shuffle_n)
    for s in range(shuffle_n):
        rng = random.Random(seed + s)
        ordered = list(examples)
        rng.shuffle(ordered)
        for rot in rotations:
            variants.append({
                "shuffle_idx": s,
                "rotation": rot,
                "examples": ordered,
            })
    return variants


def iter_chunks(items: List[dict], chunk_size: int):
    chunk_size = max(1, int(chunk_size))
    for i in range(0, len(items), chunk_size):
        yield items[i:i + chunk_size]


def parse_vlm_batch_output(
    answer: str,
    expected_ids: List[str],
    allowed_labels: Optional[List[str]] = None,
) -> Dict[str, Tuple[str, str]]:
    if not answer:
        raise ValueError("Empty VLM response")

    text = answer.strip().replace("```json", "").replace("```", "").strip()
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Batch response must be a JSON object")

    preds = data.get("predictions")
    if not isinstance(preds, list):
        raise ValueError("Missing predictions list")
    if len(preds) != len(expected_ids):
        raise ValueError(f"Expected {len(expected_ids)} predictions, got {len(preds)}")

    parsed: Dict[str, Tuple[str, str]] = {}
    expected_set = set(expected_ids)
    for item in preds:
        if not isinstance(item, dict):
            raise ValueError("Each prediction must be a JSON object")
        pred_id = str(item.get("id", "")).strip()
        if not pred_id:
            raise ValueError("Each prediction must include non-empty id")
        if pred_id in parsed:
            raise ValueError(f"Duplicate prediction id: {pred_id}")

        # Reuse single-item parser for canonical class/quality handling.
        class_label, quality = parse_vlm_output(
            json.dumps({
                "class": item.get("class", ""),
                "quality": item.get("quality", "NA"),
            }),
            allowed_labels,
        )
        parsed[pred_id] = (class_label, quality)

    missing = [qid for qid in expected_ids if qid not in parsed]
    extras = [qid for qid in parsed.keys() if qid not in expected_set]
    if missing or extras:
        raise ValueError(f"Prediction IDs mismatch (missing={missing}, extras={extras})")

    return parsed


def dump_batch_failure_response(
    answer: str,
    query_ids: List[str],
    error: Exception,
    debug_dir: Optional[str],
) -> Optional[str]:
    if not debug_dir:
        return None
    try:
        out_dir = Path(debug_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        suffix = "".join(random.choices(string.hexdigits.lower(), k=8))
        out_path = out_dir / f"batch_parse_failure_{stamp}_{os.getpid()}_{suffix}.json"
        payload = {
            "error": str(error),
            "query_ids": list(query_ids),
            "response_text": answer,
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        return str(out_path)
    except Exception:
        return None


def classify_patch_batch(
    patch_infos: List[dict],
    variants: List[dict],
    template: str,
    class_list: str,
    class_defs_block: str,
    label_map: Dict[str, str],
    reverse_label_map: Dict[str, str],
    allowed_labels: List[str],
    runner,
    image_size: Optional[int],
    save_variants: bool,
    save_raw: bool,
    profile: bool = False,
    batch_failure_debug_dir: Optional[str] = None,
    query_shuffle_n: int = 1,
    query_shuffle_seed: Optional[int] = None,
) -> List[dict]:
    def _fallback_single(mark_mode: Optional[str] = None) -> List[dict]:
        if mark_mode:
            for info in patch_infos:
                info["batch_mode"] = mark_mode
        return [
            classify_patch(
                info,
                variants,
                template,
                class_list,
                class_defs_block,
                label_map,
                reverse_label_map,
                allowed_labels,
                runner,
                image_size,
                save_variants,
                save_raw,
                profile,
            )
            for info in patch_infos
        ]

    if len(patch_infos) <= 1:
        return _fallback_single()

    if any(not info.get("stage3_keep", True) for info in patch_infos):
        return _fallback_single()

    if len(variants) != 1 or int(variants[0].get("rotation", 0)) != 0:
        return _fallback_single()

    variant = variants[0]
    query_shuffle_n = max(1, int(query_shuffle_n))
    if query_shuffle_seed is None:
        query_shuffle_seed = 0
    profile_timings = None
    if profile:
        profile_timings = {
            "image_prep": 0.0,
            "prompt_build": 0.0,
            "vlm_request": 0.0,
            "parse": 0.0,
        }

    patch_count = len(patch_infos)
    runs_by_patch: List[List[dict]] = [[] for _ in range(patch_count)]
    raw_by_patch: List[List[str]] = [[] for _ in range(patch_count)] if save_raw else []
    query_ids: List[str] = []
    answer = ""
    try:
        for query_shuffle_idx in range(query_shuffle_n):
            if profile:
                t_step = time.time()

            query_order = list(range(patch_count))
            if query_shuffle_idx > 0:
                rng = random.Random(int(query_shuffle_seed) + query_shuffle_idx)
                rng.shuffle(query_order)

            query_ids = []
            query_images: List[Tuple[str, int, int, Image.Image]] = []
            for query_position, patch_idx in enumerate(query_order):
                info = patch_infos[patch_idx]
                if "image" not in info:
                    patch_id = info.get("patch_id") or build_patch_id(info)
                    raise KeyError(f"Missing in-memory patch image for batch query: {patch_id}")
                qid = f"q{query_position}"
                query_ids.append(qid)
                query_images.append((
                    qid,
                    patch_idx,
                    query_position,
                    prepare_image_for_vlm(info["image"], image_size),
                ))

            if profile:
                profile_timings["image_prep"] += time.time() - t_step
                t_step = time.time()

            batch_prompt = (
                "Classify each high-magnification WSI query patch independently.\n"
                "Do not compare queries against each other.\n\n"
                f"Classes (choose exactly one per query): {class_list}\n"
                f"Class definitions:\n{class_defs_block}\n\n"
                "Quality score criteria:\n"
                "- Sharp: Crisp cellular details and distinct boundaries.\n"
                "- Somewhat Blurred: Identifiable structures but lacking fine edge definition.\n"
                "- Out of Focus: Little to no recognizable structure beyond color blobs or haze.\n"
                "If class is not tissue, set quality to NA.\n\n"
                "Use the labeled examples as in-context references.\n"
                f"Return ONLY valid JSON with exactly {len(query_ids)} predictions.\n"
                f"Use IDs exactly as provided: {', '.join(query_ids)}.\n"
                "Format: "
                "{\"predictions\":[{\"id\":\"q0\",\"class\":\"<label>\","
                "\"quality\":\"<Sharp|Somewhat Blurred|Out of Focus|NA>\"}]}"
            )

            parts: List[dict] = [{"type": "text", "text": batch_prompt + "\n\nExamples:\n"}]
            parts.extend(build_examples_block(variant["examples"]))
            parts.append({"type": "text", "text": "\nQueries:\n"})
            for qid, _, _, query_img in query_images:
                parts.append({"type": "text", "text": f"{qid}:\n"})
                parts.append({"type": "image", "image": query_img})
            parts.append({"type": "text", "text": "\nJSON response:"})

            if profile:
                profile_timings["prompt_build"] += time.time() - t_step
                t_step = time.time()

            answer = runner.run(parts)

            if profile:
                profile_timings["vlm_request"] += time.time() - t_step
                t_step = time.time()

            parsed = parse_vlm_batch_output(answer, query_ids, allowed_labels)

            if profile:
                profile_timings["parse"] += time.time() - t_step

            for qid, patch_idx, query_position, _ in query_images:
                class_label, quality = parsed[qid]
                canonical = reverse_label_map.get(normalize_class_label(class_label), class_label)
                run = {
                    "shuffle_idx": variant["shuffle_idx"],
                    "rotation": 0,
                    "query_shuffle_idx": query_shuffle_idx,
                    "query_position": query_position,
                    "class_label": class_label,
                    "class_canonical": canonical,
                    "quality": quality,
                }
                runs_by_patch[patch_idx].append(run)
                if save_raw:
                    raw_by_patch[patch_idx].append(answer)

        per_patch_timings = None
        if profile and profile_timings:
            n = float(len(patch_infos))
            per_patch_timings = {k: v / n for k, v in profile_timings.items()}

        results: List[dict] = []
        for patch_idx, info in enumerate(patch_infos):
            runs = runs_by_patch[patch_idx]
            class_votes = Counter([r["class_canonical"] for r in runs]) if runs else Counter()
            blur_votes = Counter([r["quality"] for r in runs]) if runs else Counter()
            pred_label = class_votes.most_common(1)[0][0] if class_votes else "background"
            pred_blur = blur_votes.most_common(1)[0][0] if blur_votes else "NA"
            result = {
                "patch": info,
                "pred_label": label_map.get(pred_label, pred_label) if pred_label in label_map else pred_label,
                "pred_label_canonical": pred_label,
                "pred_blur": pred_blur,
                "class_votes": dict(class_votes),
                "blur_votes": dict(blur_votes),
                "runs": runs if save_variants else [],
                "raw_responses": raw_by_patch[patch_idx] if save_raw else [],
            }
            if per_patch_timings is not None:
                result["timings"] = per_patch_timings
                result["timing_counts"] = {"runs": query_shuffle_n}
            results.append(result)
        return results
    except Exception as e:
        debug_path = dump_batch_failure_response(
            answer=answer,
            query_ids=query_ids,
            error=e,
            debug_dir=batch_failure_debug_dir,
        )
        if debug_path:
            print(f"Debug: saved batched failure response to {debug_path}", file=sys.stderr)
        print(f"Warning: batched classify failed ({e}); falling back to single-patch calls", file=sys.stderr)
        return _fallback_single(mark_mode="fallback_single")


def classify_patch(
    patch_info: dict,
    variants: List[dict],
    template: str,
    class_list: str,
    class_defs_block: str,
    label_map: Dict[str, str],
    reverse_label_map: Dict[str, str],
    allowed_labels: List[str],
    runner,
    image_size: Optional[int],
    save_variants: bool,
    save_raw: bool,
    profile: bool = False,
) -> dict:
    profile_timings = None
    run_count = 0
    if profile:
        profile_timings = {
            "image_prep": 0.0,
            "prompt_build": 0.0,
            "vlm_request": 0.0,
            "parse": 0.0,
        }

    if not patch_info.get("stage3_keep", True):
        bg_label = label_map.get("background", "background")
        result = {
            "patch": patch_info,
            "pred_label": bg_label,
            "pred_label_canonical": "background",
            "pred_blur": "NA",
            "class_votes": {"background": 1},
            "blur_votes": {"NA": 1},
            "runs": [],
            "raw_responses": [],
        }
        if profile:
            result["timings"] = profile_timings
            result["timing_counts"] = {"runs": 0}
        return result

    runs = []
    raw_responses = []
    if "image" not in patch_info:
        patch_id = patch_info.get("patch_id") or build_patch_id(patch_info)
        raise KeyError(f"Missing in-memory patch image for classification: {patch_id}")

    for variant in variants:
        if profile:
            t_step = time.time()
        rot = variant["rotation"]
        query_img = patch_info["image"]
        if rot:
            query_img = query_img.rotate(rot, expand=False)
        query_img = prepare_image_for_vlm(query_img, image_size)
        if profile:
            profile_timings["image_prep"] += time.time() - t_step
            t_step = time.time()

        text_map = {
            "CLASS_LIST": class_list,
            "CLASS_DEFS": class_defs_block,
        }

        image_map = {
            "QUERY_IMAGE": query_img,
        }

        # Add per-example placeholders
        for idx, ex in enumerate(variant["examples"]):
            text_map[f"EX{idx}_LABEL"] = ex["label_text"]
            image_map[f"EX{idx}_IMAGE"] = ex.get("image_payload", ex["image"])

        examples_block = build_examples_block(variant["examples"])
        parts = build_prompt_parts(template, text_map, image_map, examples_block)
        if profile:
            profile_timings["prompt_build"] += time.time() - t_step
            t_step = time.time()

        answer = runner.run(parts)
        if profile:
            profile_timings["vlm_request"] += time.time() - t_step
            t_step = time.time()
        class_label, quality = parse_vlm_output(answer, allowed_labels)
        canonical = reverse_label_map.get(normalize_class_label(class_label), class_label)
        if profile:
            profile_timings["parse"] += time.time() - t_step
            run_count += 1
        runs.append({
            "shuffle_idx": variant["shuffle_idx"],
            "rotation": rot,
            "class_label": class_label,
            "class_canonical": canonical,
            "quality": quality,
        })
        if save_raw:
            raw_responses.append(answer)

    class_votes = Counter([r["class_canonical"] for r in runs]) if runs else Counter()
    pred_label = class_votes.most_common(1)[0][0] if class_votes else "background"

    blur_votes = Counter([r["quality"] for r in runs]) if runs else Counter()
    pred_blur = blur_votes.most_common(1)[0][0] if blur_votes else "NA"

    result = {
        "patch": patch_info,
        "pred_label": label_map.get(pred_label, pred_label) if pred_label in label_map else pred_label,
        "pred_label_canonical": pred_label,
        "pred_blur": pred_blur,
        "class_votes": dict(class_votes),
        "blur_votes": dict(blur_votes),
        "runs": runs if save_variants else [],
        "raw_responses": raw_responses if save_raw else [],
    }
    if profile:
        result["timings"] = profile_timings
        result["timing_counts"] = {"runs": run_count}
    return result


def process_stage5_run(
    stage5_run: str,
    args,
    template_text: str,
    class_defs: Dict[str, str],
) -> str:
    timings = {}
    t_all = time.time()
    meta = load_stage5_metadata(stage5_run)
    bbox = tuple(meta.get("bbox", []))
    if len(bbox) != 4:
        raise ValueError(f"Invalid bbox in stage5 metadata: {bbox}")

    wsi_path = resolve_wsi_path(meta.get("wsi_path"))
    wsi_id = Path(wsi_path).stem

    if args.patch_size is None:
        patch_size = None
        patch_meta = meta.get("patch_extraction") or {}
        if isinstance(patch_meta, dict):
            patch_size = patch_meta.get("patch_size")
        if patch_size is None:
            raise ValueError("patch_size is required; supply --patch-size or ensure stage5 metadata has patch_extraction.patch_size")
        args.patch_size = int(patch_size)

    t = time.time()
    icl_pool = load_icl_pool(stage5_run)
    timings["load_icl_pool"] = time.time() - t
    class_order = infer_classes_from_stage5(meta, icl_pool)
    label_map, reverse_map, ordered_classes = build_label_map(class_order, args.label_mode)

    class_defs_block = build_class_def_block(class_defs, label_map, ordered_classes)
    class_list = ", ".join([label_map.get(c, c) for c in ordered_classes])
    allowed_labels = [label_map.get(c, c) for c in ordered_classes]

    t = time.time()
    icl_examples = sample_icl_examples(
        icl_pool=icl_pool,
        class_order=ordered_classes,
        per_class=args.icl_k,
        seed=args.seed,
    )
    prepared_examples = prepare_examples(icl_examples, label_map, args.vlm_image_size)
    timings["prepare_icl_examples"] = time.time() - t

    t = time.time()
    variants = build_variants(
        examples=prepared_examples,
        shuffle_n=args.icl_shuffle_n,
        rotations=args.rotations,
        seed=args.seed,
    )
    timings["build_variants"] = time.time() - t
    if args.query_batch_size > 1 and (
        len(variants) != 1 or int(variants[0].get("rotation", 0)) != 0
    ):
        raise ValueError(
            "--query-batch-size > 1 currently requires --rotations 0 and --icl-shuffle-n 1"
        )

    # Render a text-only prompt preview (images replaced with placeholders)
    examples_for_prompt = variants[0]["examples"] if variants else []
    examples_text = build_examples_text(examples_for_prompt)
    prompt_rendered_text = safe_format(template_text, {
        "CLASS_DEFS": class_defs_block,
        "EXAMPLES_BLOCK": examples_text,
        "QUERY_IMAGE": "[QUERY_IMAGE]",
    })

    # Output dir
    model_name = args.model
    model_dir = sanitize_model_name(model_name)
    run_config = {
        "backend": args.backend,
        "model": model_name,
        "wsi_reader": args.wsi_reader,
        "patch_size": args.patch_size,
        "vlm_image_size": args.vlm_image_size,
        "classical_blur_threshold": args.classical_blur_threshold,
        "classical_blur_sigma": args.classical_blur_sigma,
        "classical_blur_pixel_threshold": args.classical_blur_pixel_threshold,
        "label_mode": args.label_mode,
        "label_map": label_map,
        "class_order": ordered_classes,
        "class_definitions": class_defs,
        "icl_k": args.icl_k,
        "icl_shuffle_n": args.icl_shuffle_n,
        "rotations": args.rotations,
        "query_batch_size": args.query_batch_size,
        "query_shuffle_n": args.query_shuffle_n,
        "query_shuffle_seed": args.query_shuffle_seed,
        "seed": args.seed,
        "stage5_run": stage5_run,
        "stage3_run": args.stage3_run,
        "stage3_fg_threshold": args.stage3_fg_threshold,
        "prompt_template_path": args.prompt_template,
    }
    config_hash = compute_config_hash(run_config)
    out_root = Path(args.output_dir)
    resumed_output_dir = False
    if args.resume:
        existing_dir = find_existing_output_dir(
            out_root=out_root,
            wsi_id=wsi_id,
            model_dir=model_dir,
            config_hash=config_hash,
        )
        if existing_dir is not None:
            out_dir = existing_dir
            resumed_output_dir = True
            print(f"Resume checkpoint: using existing output dir {out_dir}")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir = out_root / wsi_id / model_dir / f"{timestamp}_{config_hash}"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = out_root / wsi_id / model_dir / f"{timestamp}_{config_hash}"
    os.makedirs(out_dir, exist_ok=True)

    # Prepare backend
    t = time.time()
    if args.backend == "gemini":
        runner = GeminiRunner(
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            max_retries=args.max_retries,
            use_vertex=args.gemini_use_vertex,
            credentials_path=args.gemini_credentials,
            location=args.gemini_location,
            thinking_level=args.thinking_level,
            include_thoughts=args.include_thoughts,
        )
    elif args.backend == "openrouter":
        runner = OpenRouterRunner(
            model=args.model,
            api_key=args.openrouter_api_key,
            url=args.openrouter_url,
            timeout=args.timeout,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            max_retries=args.max_retries,
            referer=args.openrouter_referer,
            reasoning_effort=args.reasoning_effort,
        )
    else:
        runner = VLLMRunner(
            model=args.model,
            url=args.vllm_url,
            timeout=args.timeout,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            max_retries=args.max_retries,
        )
    timings["init_backend"] = time.time() - t

    # Load WSI and prepare patch grid
    t = time.time()
    wsi, resolved_wsi_reader = load_wsi(wsi_path, args.wsi_reader)
    timings["load_wsi"] = time.time() - t
    effective_max_workers = int(args.max_workers)
    if resolved_wsi_reader == "isyntax" and effective_max_workers > 1:
        print(
            f"Resolved WSI reader is '{resolved_wsi_reader}'; "
            f"forcing --max-workers 1 (requested {args.max_workers}) for stability."
        )
        effective_max_workers = 1

    x1, y1, x2, y2 = bbox
    bbox_w = x2 - x1
    bbox_h = y2 - y1
    cols = math.ceil(bbox_w / args.patch_size)
    rows = math.ceil(bbox_h / args.patch_size)

    stage3_info = None
    if args.stage3_run:
        t = time.time()
        stage3_info = load_stage3_info(args.stage3_run, bbox)
        timings["load_stage3_info"] = time.time() - t

    t = time.time()
    patch_infos = []
    patch_info_by_id: Dict[str, dict] = {}
    for row in range(rows):
        for col in range(cols):
            px1 = x1 + col * args.patch_size
            py1 = y1 + row * args.patch_size
            px2 = min(px1 + args.patch_size, x2)
            py2 = min(py1 + args.patch_size, y2)
            if px2 <= px1 or py2 <= py1:
                continue

            fg_ratio = None
            keep = True
            if stage3_info:
                fg_ratio = compute_stage3_fg_ratio(stage3_info, px1, py1, px2, py2)
                keep = fg_ratio >= args.stage3_fg_threshold

            patch_info = {
                "row": row,
                "col": col,
                "x1": px1,
                "y1": py1,
                "x2": px2,
                "y2": py2,
                "stage3_fg_ratio": fg_ratio,
                "stage3_keep": keep,
                "classical_blur_score": None,
                "classical_sharp_score": None,
                "classical_blur_pass": None,
            }
            patch_info["patch_id"] = build_patch_id(patch_info)
            patch_infos.append(patch_info)
            patch_info_by_id[patch_info["patch_id"]] = patch_info
    timings["build_patch_grid"] = time.time() - t
    total_patches = len(patch_infos)
    kept_patches = sum(1 for info in patch_infos if info.get("stage3_keep", True))

    csv_path = out_dir / "patches.csv"
    csv_header_valid = True
    if csv_path.exists() and csv_path.stat().st_size > 0:
        try:
            with open(csv_path, "r", newline="") as f:
                header_row = next(csv.reader(f), [])
            csv_header_valid = set(PATCHES_CSV_BASE_HEADER).issubset(set(header_row))
        except Exception:
            csv_header_valid = False

    existing_results_raw: Dict[str, dict] = {}
    malformed_resume_rows = 0
    csv_has_variants_col = args.save_variants
    if args.resume and csv_path.exists() and csv_header_valid:
        existing_results_raw, malformed_resume_rows, csv_has_variants_col = load_existing_patch_results(csv_path)
        if existing_results_raw:
            print(f"Resume checkpoint: loaded {len(existing_results_raw)} patch rows from {csv_path}")
        if malformed_resume_rows:
            print(
                f"Warning: skipped {malformed_resume_rows} malformed rows in {csv_path}",
                file=sys.stderr,
            )
    elif args.resume and csv_path.exists() and not csv_header_valid:
        print(
            f"Warning: {csv_path} is missing required columns for current schema; "
            "starting a fresh checkpoint file",
            file=sys.stderr,
        )

    existing_results: Dict[str, dict] = {}
    stale_resume_rows = 0
    for patch_id, res in existing_results_raw.items():
        base_info = patch_info_by_id.get(patch_id)
        if base_info is None:
            stale_resume_rows += 1
            continue
        loaded = dict(res)
        loaded_patch = dict(base_info)
        loaded_patch["classical_blur_score"] = res.get("classical_blur_score")
        loaded_patch["classical_sharp_score"] = res.get("classical_sharp_score")
        loaded_patch["classical_blur_pass"] = res.get("classical_blur_pass")
        loaded["patch"] = loaded_patch
        existing_results[patch_id] = loaded
    if stale_resume_rows:
        print(
            f"Warning: ignored {stale_resume_rows} resume rows not in current patch grid",
            file=sys.stderr,
        )

    resumed_patch_count = len(existing_results)
    resumed_vlm_count = sum(
        1 for patch_id in existing_results if patch_info_by_id[patch_id].get("stage3_keep", True)
    )

    pending_patch_infos = [
        info for info in patch_infos if info["patch_id"] not in existing_results
    ]
    pending_vlm_tasks = sum(1 for info in pending_patch_infos if info.get("stage3_keep", True))

    patch_extract_seconds = 0.0
    patch_save_seconds = 0.0
    patch_dir = None
    if args.save_patches:
        patch_dir = out_dir / "patches"
        patch_dir.mkdir(parents=True, exist_ok=True)

    def load_patch_image_and_metrics(info: dict) -> None:
        nonlocal patch_extract_seconds, patch_save_seconds
        if not info.get("stage3_keep", True):
            return
        if "image" in info:
            return

        t_patch = time.time()
        patch = extract_patch(
            wsi,
            resolved_wsi_reader,
            info["x1"],
            info["y1"],
            info["x2"] - info["x1"],
            info["y2"] - info["y1"],
        )
        patch_extract_seconds += time.time() - t_patch
        info["image"] = patch

        blur_score, sharp_score, blur_pass = compute_classical_blur_metrics(
            patch_img=patch,
            sigma=args.classical_blur_sigma,
            pixel_threshold=args.classical_blur_pixel_threshold,
            patch_threshold=args.classical_blur_threshold,
        )
        info["classical_blur_score"] = blur_score
        info["classical_sharp_score"] = sharp_score
        info["classical_blur_pass"] = blur_pass

        if patch_dir is not None:
            t_save = time.time()
            img = prepare_image_for_vlm(patch, args.vlm_image_size)
            img.save(patch_dir / f"{info['patch_id']}.png")
            patch_save_seconds += time.time() - t_save

    def release_patch_image(info: dict) -> None:
        if isinstance(info, dict):
            info.pop("image", None)

    # Classify patches
    new_results: List[dict] = []
    detail_sums = Counter() if args.profile else None
    detail_runs = 0
    start_time = time.time()
    last_report = start_time
    completed = 0
    total_tasks = len(pending_patch_infos)
    completed_vlm = 0
    total_vlm_tasks = pending_vlm_tasks
    csv_include_variants = args.save_variants
    if args.resume and csv_path.exists():
        csv_include_variants = csv_has_variants_col
        if csv_has_variants_col != args.save_variants:
            print(
                "Warning: existing patches.csv variant column differs from current --save-variants; "
                "using existing CSV layout for compatibility",
                file=sys.stderr,
            )
    csv_open_mode = "a"
    if (not csv_path.exists()) or csv_path.stat().st_size == 0 or not csv_header_valid:
        csv_open_mode = "w"
    csv_needs_header = (csv_open_mode == "w")

    if total_tasks == 0 and resumed_patch_count > 0:
        print("Resume checkpoint: all patch rows already present; skipping VLM calls")

    def consume_batch_results(
        batch_results: List[dict],
        csv_writer: csv.writer,
        csv_file,
    ) -> None:
        nonlocal completed, completed_vlm, last_report, detail_runs
        wrote_rows = False
        for res in batch_results:
            if not isinstance(res, dict):
                continue
            blur_score, sharp_score, blur_pass = get_result_classical_blur_fields(res)
            set_result_classical_blur_fields(res, blur_score, sharp_score, blur_pass)
            patch_meta = res.get("patch", {})
            if isinstance(patch_meta, dict):
                patch_meta.pop("image", None)
            new_results.append(res)
            completed += 1
            if patch_meta.get("stage3_keep", True):
                completed_vlm += 1
            csv_writer.writerow(result_to_csv_row(res, csv_include_variants))
            wrote_rows = True

            if args.profile:
                timing_detail = res.get("timings")
                if timing_detail:
                    for k, v in timing_detail.items():
                        detail_sums[k] += v
                counts = res.get("timing_counts")
                if counts:
                    detail_runs += counts.get("runs", 0)

        if wrote_rows:
            csv_file.flush()

        if args.progress:
            now = time.time()
            if completed == total_tasks or (now - last_report) >= args.progress_interval:
                elapsed = now - start_time
                rate = (completed_vlm / elapsed) if elapsed > 0 else 0.0
                remaining = ((total_vlm_tasks - completed_vlm) / rate) if rate > 0 else 0.0
                overall_done = resumed_vlm_count + completed_vlm
                print(
                    f"\r  [{overall_done}/{kept_patches}] {rate:.2f} patches/s, ETA: {remaining:.0f}s",
                    end="",
                    flush=True,
                )
                last_report = now

    with open(csv_path, csv_open_mode, newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        if csv_needs_header:
            csv_writer.writerow(build_patches_csv_header(csv_include_variants))
            csv_file.flush()

        if total_tasks > 0:
            if args.query_batch_size > 1:
                kept_infos = [info for info in pending_patch_infos if info.get("stage3_keep", True)]
                skipped_infos = [info for info in pending_patch_infos if not info.get("stage3_keep", True)]
                batch_group_counter = 0

                for info in skipped_infos:
                    info["batch_group_id"] = None
                    info["batch_query_index"] = 0
                    info["batch_query_size"] = 1
                    info["batch_mode"] = "single"
                    out = classify_patch(
                        info,
                        variants,
                        template_text,
                        class_list,
                        class_defs_block,
                        label_map,
                        reverse_map,
                        allowed_labels,
                        runner,
                        args.vlm_image_size,
                        args.save_variants,
                        args.save_raw_responses,
                        args.profile,
                    )
                    consume_batch_results([out], csv_writer, csv_file)

                for chunk in iter_chunks(kept_infos, args.query_batch_size):
                    chunk = list(chunk)
                    batch_group_id = f"bg{batch_group_counter}"
                    batch_group_counter += 1
                    chunk_size = len(chunk)
                    try:
                        for idx, info in enumerate(chunk):
                            info["batch_group_id"] = batch_group_id
                            info["batch_query_index"] = idx
                            info["batch_query_size"] = chunk_size
                            info["batch_mode"] = "batch"
                            load_patch_image_and_metrics(info)
                        out = classify_patch_batch(
                            chunk,
                            variants,
                            template_text,
                            class_list,
                            class_defs_block,
                            label_map,
                            reverse_map,
                            allowed_labels,
                            runner,
                            args.vlm_image_size,
                            args.save_variants,
                            args.save_raw_responses,
                            args.profile,
                            args.batch_failure_debug_dir,
                            args.query_shuffle_n,
                            args.query_shuffle_seed,
                        )
                        batch_results = out if isinstance(out, list) else [out]
                        consume_batch_results(batch_results, csv_writer, csv_file)
                    finally:
                        for info in chunk:
                            release_patch_image(info)
            else:
                worker_chunk_size = max(1, int(effective_max_workers))
                with ThreadPoolExecutor(max_workers=effective_max_workers) as ex:
                    for chunk in iter_chunks(pending_patch_infos, worker_chunk_size):
                        chunk = list(chunk)
                        future_to_info = {}
                        try:
                            for info in chunk:
                                info["batch_group_id"] = None
                                info["batch_query_index"] = 0
                                info["batch_query_size"] = 1
                                info["batch_mode"] = "single"
                                load_patch_image_and_metrics(info)
                                fut = ex.submit(
                                    classify_patch,
                                    info,
                                    variants,
                                    template_text,
                                    class_list,
                                    class_defs_block,
                                    label_map,
                                    reverse_map,
                                    allowed_labels,
                                    runner,
                                    args.vlm_image_size,
                                    args.save_variants,
                                    args.save_raw_responses,
                                    args.profile,
                                )
                                future_to_info[fut] = info

                            for fut in as_completed(future_to_info):
                                out = fut.result()
                                batch_results = out if isinstance(out, list) else [out]
                                consume_batch_results(batch_results, csv_writer, csv_file)
                        finally:
                            for info in chunk:
                                release_patch_image(info)

    timings["extract_patches"] = patch_extract_seconds
    if args.save_patches:
        timings["save_patches"] = patch_save_seconds

    elapsed = time.time() - start_time
    final_rate = (completed_vlm / elapsed) if elapsed > 0 else 0.0
    final_remaining = ((total_vlm_tasks - completed_vlm) / final_rate) if final_rate > 0 else 0.0
    overall_completed_vlm = resumed_vlm_count + completed_vlm
    overall_completed_tasks = resumed_patch_count + completed
    progress_line = f"[{overall_completed_vlm}/{kept_patches}] {final_rate:.2f} patches/s, ETA: {final_remaining:.0f}s"
    progress_summary = {
        "completed": int(overall_completed_vlm),
        "total": int(kept_patches),
        "completed_all_tasks": int(overall_completed_tasks),
        "total_all_tasks": int(total_patches),
        "kept_after_stage3": int(kept_patches),
        "skipped_by_stage3": int(total_patches - kept_patches),
        "percent_complete": float((overall_completed_vlm / kept_patches) * 100.0) if kept_patches > 0 else 100.0,
        "resumed_patch_rows": int(resumed_patch_count),
        "resumed_vlm_rows": int(resumed_vlm_count),
        "new_patch_rows": int(completed),
        "new_vlm_rows": int(completed_vlm),
        "patches_per_second": float(final_rate),
        "eta_seconds": float(final_remaining),
        "display": progress_line,
    }
    if args.progress and total_tasks > 0:
        print()
    # Stable final line for logs/grep (not carriage-return progress).
    print(f"Progress summary: {progress_line}")
    timings["vlm_classify"] = elapsed
    if args.profile and detail_sums is not None:
        timings["vlm_classify_image_prep_sum"] = detail_sums.get("image_prep", 0.0)
        timings["vlm_classify_prompt_build_sum"] = detail_sums.get("prompt_build", 0.0)
        timings["vlm_classify_request_sum"] = detail_sums.get("vlm_request", 0.0)
        timings["vlm_classify_parse_sum"] = detail_sums.get("parse", 0.0)
        if detail_runs:
            timings["vlm_classify_image_prep_mean"] = detail_sums.get("image_prep", 0.0) / detail_runs
            timings["vlm_classify_prompt_build_mean"] = detail_sums.get("prompt_build", 0.0) / detail_runs
            timings["vlm_classify_request_mean"] = detail_sums.get("vlm_request", 0.0) / detail_runs
            timings["vlm_classify_parse_mean"] = detail_sums.get("parse", 0.0) / detail_runs
        vlm_requests = detail_runs
        if elapsed > 0:
            timings["patches_total"] = float(total_patches)
            timings["patches_kept"] = float(kept_patches)
            timings["patches_skipped"] = float(total_patches - kept_patches)
            timings["vlm_requests"] = float(vlm_requests)
            timings["patches_per_s"] = total_vlm_tasks / elapsed
            timings["requests_per_s"] = vlm_requests / elapsed
    if elapsed > 0 and total_vlm_tasks > 0:
        print(
            f"VLM throughput: {total_vlm_tasks} new patches in {elapsed:.1f}s "
            f"({total_vlm_tasks / elapsed:.2f} patches/s)"
        )

    results_by_patch: Dict[str, dict] = dict(existing_results)
    for res in new_results:
        patch_id = res["patch"].get("patch_id")
        if not patch_id:
            patch_id = build_patch_id(res["patch"])
            res["patch"]["patch_id"] = patch_id
        results_by_patch[patch_id] = res

    missing_patch_ids = [
        info["patch_id"] for info in patch_infos if info["patch_id"] not in results_by_patch
    ]
    if missing_patch_ids:
        raise RuntimeError(
            f"Missing {len(missing_patch_ids)} patch rows after processing; "
            "cannot finalize outputs safely"
        )
    all_results = [results_by_patch[info["patch_id"]] for info in patch_infos]

    t = time.time()
    classical_blur_backfilled = 0
    for res in all_results:
        blur_score, sharp_score, blur_pass = get_result_classical_blur_fields(res)
        patch_meta = res.get("patch", {})
        if not isinstance(patch_meta, dict):
            patch_meta = {}
        if (
            blur_score is None
            and patch_meta.get("stage3_keep", True)
        ):
            patch_img = extract_patch(
                wsi,
                resolved_wsi_reader,
                patch_meta["x1"],
                patch_meta["y1"],
                patch_meta["x2"] - patch_meta["x1"],
                patch_meta["y2"] - patch_meta["y1"],
            )
            blur_score, sharp_score, blur_pass = compute_classical_blur_metrics(
                patch_img=patch_img,
                sigma=args.classical_blur_sigma,
                pixel_threshold=args.classical_blur_pixel_threshold,
                patch_threshold=args.classical_blur_threshold,
            )
            classical_blur_backfilled += 1
        set_result_classical_blur_fields(res, blur_score, sharp_score, blur_pass)
    timings["classical_blur"] = time.time() - t
    classical_blur_scored = 0
    classical_blur_passed = 0
    for res in all_results:
        blur_score, _, blur_pass = get_result_classical_blur_fields(res)
        if blur_score is not None:
            classical_blur_scored += 1
        if blur_pass is True:
            classical_blur_passed += 1

    # Build maps
    t = time.time()
    class_to_id = {c: i for i, c in enumerate(ordered_classes)}
    quality_to_id = {q: i for i, q in enumerate(QUALITY_LABELS)}
    class_map = np.full((rows, cols), len(class_to_id), dtype=np.int16)
    quality_map = np.full((rows, cols), quality_to_id.get("NA", 0), dtype=np.int8)

    for res in all_results:
        info = res["patch"]
        r, c = info["row"], info["col"]
        label = res["pred_label_canonical"]
        class_map[r, c] = class_to_id.get(label, len(class_to_id))
        if label == "tissue":
            quality_map[r, c] = quality_to_id.get(res["pred_blur"], quality_to_id.get("NA", 0))
    timings["build_maps"] = time.time() - t

    # Overlays
    t = time.time()
    # Recreate in case a concurrent retry touched the temp root.
    out_dir.mkdir(parents=True, exist_ok=True)
    bbox_thumb, overlay_read = extract_bbox_thumbnail(
        wsi,
        resolved_wsi_reader,
        bbox,
        max_dim=args.overlay_max_dim,
        stage2_input=meta.get("stage2_input"),
    )
    if overlay_read.get("strategy") not in {"best_fit_level", "stage2_bbox_region_cache"}:
        print(
            "Overlay read fallback: "
            f"strategy={overlay_read.get('strategy')} "
            f"level={overlay_read.get('level')} "
            f"reason={overlay_read.get('reason')}",
            file=sys.stderr,
        )

    class_colors = {
        "tissue": (0, 200, 0),
        "paraffin_mounting_medium": (255, 165, 0),
        "pen_ink_marks": (220, 20, 60),
    }
    blur_colors = {
        "Sharp": (0, 200, 0),
        "Somewhat Blurred": (255, 215, 0),
        "Out of Focus": (255, 0, 0),
    }
    classical_blur_colors = {
        True: (0, 200, 0),
        False: (255, 0, 0),
    }

    class_patch_infos = []
    for res in all_results:
        info = dict(res["patch"])
        info["pred_label_canonical"] = res["pred_label_canonical"]
        class_patch_infos.append(info)

    class_overlay = overlay_grid(
        bbox_thumb,
        bbox,
        class_patch_infos,
        color_fn=lambda info: class_colors.get(info.get("pred_label_canonical")),
        alpha=120,
    )
    class_overlay.save(out_dir / "class_overlay.png")
    save_bbox_grid_overlay(
        class_overlay,
        bbox,
        args.patch_size,
        rows,
        cols,
        out_dir / "bbox_grid_overlay.png",
    )

    vlm_blur_patch_infos = []
    for res in all_results:
        info = dict(res["patch"])
        info["pred_blur"] = res["pred_blur"]
        info["pred_label_canonical"] = res["pred_label_canonical"]
        vlm_blur_patch_infos.append(info)

    vlm_blur_overlay = overlay_grid(
        bbox_thumb,
        bbox,
        vlm_blur_patch_infos,
        color_fn=lambda info: blur_colors.get(info.get("pred_blur")) if info.get("pred_label_canonical") == "tissue" else None,
        alpha=120,
    )
    vlm_blur_overlay.save(out_dir / "vlm_blur_overlay.png")
    # Compatibility alias for older report tools that still read blur_overlay.png.
    vlm_blur_overlay.save(out_dir / "blur_overlay.png")

    classical_blur_patch_infos = []
    for res in all_results:
        info = dict(res["patch"])
        info["pred_label_canonical"] = res["pred_label_canonical"]
        _, _, classical_blur_pass = get_result_classical_blur_fields(res)
        info["classical_blur_pass"] = classical_blur_pass
        classical_blur_patch_infos.append(info)

    classical_blur_overlay = overlay_grid(
        bbox_thumb,
        bbox,
        classical_blur_patch_infos,
        color_fn=lambda info: classical_blur_colors.get(info.get("classical_blur_pass")),
        alpha=120,
    )
    classical_blur_overlay.save(out_dir / "classical_blur_overlay.png")

    if stage3_info:
        gating_overlay = overlay_grid(
            bbox_thumb,
            bbox,
            patch_infos,
            color_fn=lambda info: (0, 120, 255) if info.get("stage3_keep") else None,
            alpha=90,
        )
        gating_overlay.save(out_dir / "stage3_gating_overlay.png")
    timings["render_overlays"] = time.time() - t

    # Save maps
    t = time.time()
    np.save(out_dir / "class_map.npy", class_map.astype(np.int16))
    np.save(out_dir / "quality_map.npy", quality_map.astype(np.int8))
    with open(out_dir / "class_palette.json", "w") as f:
        json.dump({
            "class_labels": ordered_classes,
            "quality_labels": QUALITY_LABELS,
        }, f, indent=2)

    # Save raw responses if requested
    if args.save_raw_responses:
        raw_path = out_dir / "raw_responses.jsonl"
        raw_mode = "a" if args.resume and raw_path.exists() else "w"
        with open(raw_path, raw_mode) as f:
            for res in new_results:
                info = res["patch"]
                patch_id = info.get("patch_id") or build_patch_id(info)
                for idx, raw in enumerate(res["raw_responses"]):
                    f.write(json.dumps({
                        "patch_id": patch_id,
                        "variant_idx": idx,
                        "response": raw,
                    }) + "\n")
    timings["save_outputs"] = time.time() - t

    # Save metadata
    metadata = {
        "stage5_run": stage5_run,
        "wsi_path": wsi_path,
        "wsi_reader_requested": args.wsi_reader,
        "wsi_reader": resolved_wsi_reader,
        "bbox_level0": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        "overlay_read": overlay_read,
        "class_order": ordered_classes,
        "label_mode": args.label_mode,
        "label_map": label_map,
        "class_definitions": class_defs,
        "prompt_template_path": args.prompt_template,
        "prompt_template_text": template_text,
        "prompt_rendered_text": prompt_rendered_text,
        "icl_k": args.icl_k,
        "icl_selected_paths": [ex["path"] for ex in icl_examples],
        "icl_shuffle_n": args.icl_shuffle_n,
        "rotations": args.rotations,
        "query_batch_size": args.query_batch_size,
        "query_shuffle_n": args.query_shuffle_n,
        "query_shuffle_seed": args.query_shuffle_seed,
        "seed": args.seed,
        "vlm_image_size": args.vlm_image_size,
        "patch_size": args.patch_size,
        "classical_blur_threshold": args.classical_blur_threshold,
        "classical_blur_sigma": args.classical_blur_sigma,
        "classical_blur_pixel_threshold": args.classical_blur_pixel_threshold,
        "stage3_run": args.stage3_run,
        "stage3_fg_threshold": args.stage3_fg_threshold,
        "backend": args.backend,
        "model": args.model,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "max_retries": args.max_retries,
        "max_workers_requested": int(args.max_workers),
        "max_workers": int(effective_max_workers),
        "thinking_level": args.thinking_level,
        "include_thoughts": args.include_thoughts,
        "reasoning_effort": args.reasoning_effort,
        "resume_enabled": bool(args.resume),
        "resumed_output_dir": resumed_output_dir,
        "resume_checkpoint_csv": str(csv_path),
        "resumed_patch_rows": int(resumed_patch_count),
        "resumed_vlm_rows": int(resumed_vlm_count),
        "new_patch_rows": int(completed),
        "new_vlm_rows": int(completed_vlm),
        "malformed_resume_rows": int(malformed_resume_rows),
        "stale_resume_rows": int(stale_resume_rows),
        "csv_variants_column": bool(csv_include_variants),
        "classical_blur": {
            "threshold": args.classical_blur_threshold,
            "sigma": args.classical_blur_sigma,
            "pixel_threshold": args.classical_blur_pixel_threshold,
            "scored_patch_count": int(classical_blur_scored),
            "passed_patch_count": int(classical_blur_passed),
            "backfilled_patch_count": int(classical_blur_backfilled),
        },
        "progress_summary": progress_summary,
        "processing_time_seconds": elapsed,
        "timings_seconds": timings,
        "created_at": datetime.now().isoformat(),
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    create_reproduce_command(args._parser, str(out_dir / "reproduce.txt"))

    timings["total"] = time.time() - t_all
    if args.profile:
        print("Timing breakdown (s):")
        for k, v in timings.items():
            print(f"  {k}: {v:.2f}")

    close_wsi(wsi, resolved_wsi_reader)
    return str(out_dir)


def find_metadata_for_path(path: str) -> Tuple[Optional[dict], Optional[str]]:
    p = Path(path)
    if p.is_file():
        cur = p.parent
    else:
        cur = p
    while cur != cur.parent:
        meta = cur / "metadata.json"
        if meta.exists():
            with open(meta, "r") as f:
                return json.load(f), str(cur)
        cur = cur.parent
    return None, None


def apply_rerun_defaults(args, meta: dict, defaults: dict, specified: set):
    cfg = meta or {}

    def default(name, value):
        if value is not None and name not in specified:
            setattr(args, name, value)

    default("backend", cfg.get("backend"))
    default("model", cfg.get("model"))
    default("wsi_reader", cfg.get("wsi_reader_requested") or cfg.get("wsi_reader"))
    default("prompt_template", cfg.get("prompt_template_path"))
    default("icl_k", cfg.get("icl_k"))
    default("icl_shuffle_n", cfg.get("icl_shuffle_n"))
    default("rotations", cfg.get("rotations"))
    default("query_batch_size", cfg.get("query_batch_size"))
    default("query_shuffle_n", cfg.get("query_shuffle_n"))
    default("query_shuffle_seed", cfg.get("query_shuffle_seed"))
    default("vlm_image_size", cfg.get("vlm_image_size"))
    default("patch_size", cfg.get("patch_size"))
    default("classical_blur_threshold", cfg.get("classical_blur_threshold"))
    default("classical_blur_sigma", cfg.get("classical_blur_sigma"))
    default("classical_blur_pixel_threshold", cfg.get("classical_blur_pixel_threshold"))
    default("label_mode", cfg.get("label_mode"))
    default("stage3_run", cfg.get("stage3_run"))
    default("stage3_fg_threshold", cfg.get("stage3_fg_threshold"))
    default("seed", cfg.get("seed"))


def resolve_auto_batch_replay_paths(
    rerun_output_dir: Optional[str],
    target_patch_path: str,
) -> Tuple[Optional[List[str]], Optional[str], Optional[int]]:
    if not rerun_output_dir:
        return None, None, None

    csv_path = Path(rerun_output_dir) / "patches.csv"
    if not csv_path.exists():
        return None, None, None

    target_patch_id = Path(target_patch_path).stem
    rows_by_patch: Dict[str, dict] = {}
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            patch_id = (row.get("patch_id") or "").strip()
            if patch_id:
                rows_by_patch[patch_id] = row

    if not rows_by_patch:
        return None, None, None

    target_row = rows_by_patch.get(target_patch_id)
    if not target_row:
        return None, None, None

    batch_group_id = (target_row.get("batch_group_id") or "").strip()
    if not batch_group_id:
        return None, None, None

    group_rows = [
        row for row in rows_by_patch.values()
        if (row.get("batch_group_id") or "").strip() == batch_group_id
    ]
    if len(group_rows) <= 1:
        return None, None, None

    def sort_key(row: dict) -> Tuple[int, str]:
        idx = _parse_optional_int(row.get("batch_query_index"))
        return (idx if idx is not None else 10 ** 9, (row.get("patch_id") or "").strip())

    group_rows.sort(key=sort_key)
    target_index = None
    ordered_patch_ids = []
    for idx, row in enumerate(group_rows):
        patch_id = (row.get("patch_id") or "").strip()
        if not patch_id:
            continue
        ordered_patch_ids.append(patch_id)
        if patch_id == target_patch_id:
            target_index = idx

    if not ordered_patch_ids:
        return None, None, None

    candidate_patch_dirs: List[Path] = []
    target_parent = Path(target_patch_path).parent
    replay_patch_dir = Path(rerun_output_dir) / "patches"
    for d in (target_parent, replay_patch_dir):
        if d not in candidate_patch_dirs:
            candidate_patch_dirs.append(d)

    ordered_paths: List[str] = []
    missing_patch_ids: List[str] = []
    for patch_id in ordered_patch_ids:
        found = None
        for patch_dir in candidate_patch_dirs:
            candidate = patch_dir / f"{patch_id}.png"
            if candidate.exists():
                found = str(candidate)
                break
        if found is None:
            missing_patch_ids.append(patch_id)
        else:
            ordered_paths.append(found)

    if missing_patch_ids:
        missing_preview = ", ".join(missing_patch_ids[:5])
        raise FileNotFoundError(
            f"Missing {len(missing_patch_ids)} patch PNG(s) needed for batch replay "
            f"(examples: {missing_preview})"
        )

    return ordered_paths, batch_group_id, target_index


def run_single_patch(args, template_text: str, class_defs: Dict[str, str], meta: Optional[dict]):
    if not args.single_patch:
        raise ValueError("single_patch path required")
    patch_paths = list(args.single_patch) if isinstance(args.single_patch, list) else [args.single_patch]
    auto_replay_target_patch_id = None
    effective_query_batch_size = int(args.query_batch_size)

    # Infer stage5 run from metadata if available
    stage5_run = args.stage5_run
    if not stage5_run and meta:
        stage5_run = meta.get("stage5_run")
    if not stage5_run:
        raise ValueError("stage5_run is required for single_patch mode (for ICL context)")

    if (
        (not args.disable_auto_batch_replay)
        and args.rerun_from
        and len(patch_paths) == 1
    ):
        try:
            replay_paths, replay_group_id, replay_target_index = resolve_auto_batch_replay_paths(
                getattr(args, "_rerun_meta_dir", None),
                patch_paths[0],
            )
        except FileNotFoundError as e:
            replay_paths, replay_group_id, replay_target_index = None, None, None
            print(f"Warning: auto batch replay disabled for this patch: {e}", file=sys.stderr)
        if replay_paths:
            auto_replay_target_patch_id = Path(patch_paths[0]).stem
            patch_paths = replay_paths
            effective_query_batch_size = len(patch_paths)
            print(
                "Auto batch replay: "
                f"group={replay_group_id}, size={len(patch_paths)}, "
                f"target_patch={auto_replay_target_patch_id}, target_index={replay_target_index}",
                file=sys.stderr,
            )

    stage5_meta = load_stage5_metadata(stage5_run)
    icl_pool = load_icl_pool(stage5_run)
    class_order = infer_classes_from_stage5(stage5_meta, icl_pool)
    label_map, reverse_map, ordered_classes = build_label_map(class_order, args.label_mode)
    class_defs_block = build_class_def_block(class_defs, label_map, ordered_classes)
    class_list = ", ".join([label_map.get(c, c) for c in ordered_classes])
    allowed_labels = [label_map.get(c, c) for c in ordered_classes]

    # Use selected paths from metadata if available
    if meta and meta.get("icl_selected_paths"):
        icl_examples = [{"label": normalize_class_label(Path(p).parent.name), "path": p} for p in meta["icl_selected_paths"]]
    else:
        icl_examples = sample_icl_examples(
            icl_pool=icl_pool,
            class_order=ordered_classes,
            per_class=args.icl_k,
            seed=args.seed,
        )

    prepared_examples = prepare_examples(icl_examples, label_map, args.vlm_image_size)
    variants = build_variants(
        examples=prepared_examples,
        shuffle_n=args.icl_shuffle_n,
        rotations=args.rotations,
        seed=args.seed,
    )

    # Backend
    if args.backend == "gemini":
        runner = GeminiRunner(
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            max_retries=args.max_retries,
            use_vertex=args.gemini_use_vertex,
            credentials_path=args.gemini_credentials,
            location=args.gemini_location,
            thinking_level=args.thinking_level,
            include_thoughts=args.include_thoughts,
        )
    elif args.backend == "openrouter":
        runner = OpenRouterRunner(
            model=args.model,
            api_key=args.openrouter_api_key,
            url=args.openrouter_url,
            timeout=args.timeout,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            max_retries=args.max_retries,
            referer=args.openrouter_referer,
            reasoning_effort=args.reasoning_effort,
        )
    else:
        runner = VLLMRunner(
            model=args.model,
            url=args.vllm_url,
            timeout=args.timeout,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            max_retries=args.max_retries,
        )

    patch_infos: List[dict] = []
    for idx, patch_path in enumerate(patch_paths):
        patch_img = Image.open(patch_path).convert("RGB")
        patch_id = Path(patch_path).stem
        patch_infos.append({
            "row": idx,
            "col": 0,
            "x1": 0,
            "y1": 0,
            "x2": patch_img.width,
            "y2": patch_img.height,
            "stage3_keep": True,
            "image": patch_img,
            "patch_path": patch_path,
            "patch_id": patch_id,
        })

    if effective_query_batch_size > 1 and (
        len(variants) != 1 or int(variants[0].get("rotation", 0)) != 0
    ):
        raise ValueError(
            "--query-batch-size > 1 currently requires --rotations 0 and --icl-shuffle-n 1"
        )

    results: List[dict] = []
    if effective_query_batch_size > 1 and len(patch_infos) > 1:
        batch_group_counter = 0
        for chunk in iter_chunks(patch_infos, effective_query_batch_size):
            chunk = list(chunk)
            batch_group_id = f"single_bg{batch_group_counter}"
            batch_group_counter += 1
            chunk_size = len(chunk)
            for idx, info in enumerate(chunk):
                info["batch_group_id"] = batch_group_id
                info["batch_query_index"] = idx
                info["batch_query_size"] = chunk_size
                info["batch_mode"] = "batch"
            results.extend(classify_patch_batch(
                chunk,
                variants,
                template_text,
                class_list,
                class_defs_block,
                label_map,
                reverse_map,
                allowed_labels,
                runner,
                args.vlm_image_size,
                save_variants=True,
                save_raw=True,
                profile=args.profile,
                batch_failure_debug_dir=args.batch_failure_debug_dir,
                query_shuffle_n=args.query_shuffle_n,
                query_shuffle_seed=args.query_shuffle_seed,
            ))
    else:
        for info in patch_infos:
            info["batch_group_id"] = None
            info["batch_query_index"] = 0
            info["batch_query_size"] = 1
            info["batch_mode"] = "single"
            results.append(classify_patch(
                info,
                variants,
                template_text,
                class_list,
                class_defs_block,
                label_map,
                reverse_map,
                allowed_labels,
                runner,
                args.vlm_image_size,
                save_variants=True,
                save_raw=True,
                profile=args.profile,
            ))

    if len(results) == 1:
        res = results[0]
        for idx, run in enumerate(res["runs"]):
            raw = res["raw_responses"][idx] if idx < len(res["raw_responses"]) else ""
            q_shuffle = run.get("query_shuffle_idx")
            q_extra = "" if q_shuffle is None else f" query_shuffle={q_shuffle}"
            print(f"[run {idx}] shuffle={run['shuffle_idx']} rot={run['rotation']}{q_extra}")
            print(raw)
            print("-" * 40)
        print(f"Aggregated Class: {res['pred_label_canonical']}")
        print(f"Aggregated Blur: {res['pred_blur']}")
        return

    for idx, res in enumerate(results):
        patch_path = res.get("patch", {}).get("patch_path", f"index_{idx}")
        print(f"[patch {idx}] {patch_path}")
        for run_idx, run in enumerate(res["runs"]):
            raw = res["raw_responses"][run_idx] if run_idx < len(res["raw_responses"]) else ""
            q_shuffle = run.get("query_shuffle_idx")
            q_extra = "" if q_shuffle is None else f" query_shuffle={q_shuffle}"
            print(f"[run {run_idx}] shuffle={run['shuffle_idx']} rot={run['rotation']}{q_extra}")
            print(raw)
            print("-" * 40)
        print(f"Aggregated Class: {res['pred_label_canonical']}")
        print(f"Aggregated Blur: {res['pred_blur']}")

    if auto_replay_target_patch_id:
        for res in results:
            patch_id = res.get("patch", {}).get("patch_id")
            if patch_id == auto_replay_target_patch_id:
                print("Target Patch Summary:")
                print(f"Patch ID: {patch_id}")
                print(f"Aggregated Class: {res['pred_label_canonical']}")
                print(f"Aggregated Blur: {res['pred_blur']}")
                break


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="VLM patch classification for Stage 5 bboxes",
    )

    io = parser.add_argument_group("Inputs")
    io.add_argument("--stage5-run", help="Stage5 run dir (single bbox)")
    io.add_argument("--stage5-list", help="Text file with Stage5 run dirs (one per line)")
    io.add_argument("--single-patch", nargs="+", help="Classify one or more patch PNGs (stdout only)")
    io.add_argument("--rerun-from", help="Stage6 output dir or patch PNG to reuse metadata")
    io.add_argument(
        "--wsi-reader",
        choices=["auto", "openslide", "cucim"],
        default="cucim",
        help="WSI reader backend for patch extraction (default: cucim).",
    )

    cfg = parser.add_argument_group("VLM")
    cfg.add_argument("--backend", choices=["gemini", "vllm", "openrouter"], default=None)
    cfg.add_argument("--model", default=None)
    cfg.add_argument("--temperature", type=float, default=0.0)
    cfg.add_argument("--max-tokens", type=int, default=128)
    cfg.add_argument("--max-retries", type=int, default=3)
    cfg.add_argument("--timeout", type=int, default=120)
    cfg.add_argument("--max-workers", type=int, default=8)
    cfg.add_argument("--query-batch-size", type=int, default=1, help="Number of query patches per VLM call (default: 1)")
    cfg.add_argument("--query-shuffle-n", "--query-shuffle", dest="query_shuffle_n", type=int, default=1, help="Number of query-order shuffles per batch call (default: 1)")
    cfg.add_argument("--query-shuffle-seed", type=int, default=None, help="Seed for query-order shuffles (default: --seed)")
    cfg.add_argument("--thinking-level", type=str, default=None, help="Thinking level for Gemini 3: Low/High (gemini backend)")
    cfg.add_argument("--include-thoughts", action="store_true", default=False, help="Include thought summaries in raw responses (gemini backend)")
    cfg.add_argument("--reasoning-effort", type=str, default=None, help="OpenRouter reasoning effort: low/medium/high (openrouter backend)")

    vllm = parser.add_argument_group("vLLM")
    vllm.add_argument("--vllm-url", default=DEFAULT_VLLM_URL)

    openrouter = parser.add_argument_group("OpenRouter")
    openrouter.add_argument("--openrouter-api-key", default=None)
    openrouter.add_argument("--openrouter-url", default=DEFAULT_OPENROUTER_URL)
    openrouter.add_argument("--openrouter-referer", default=DEFAULT_OPENROUTER_REFERER)

    gemini = parser.add_argument_group("Gemini SDK")
    gemini.add_argument("--gemini-use-vertex", dest="gemini_use_vertex", action="store_true")
    gemini.add_argument("--gemini-no-vertex", dest="gemini_use_vertex", action="store_false")
    gemini.add_argument(
        "--gemini-credentials",
        default=None,
        help="Optional Vertex service account JSON path. If unset, use GOOGLE_APPLICATION_CREDENTIALS.",
    )
    gemini.add_argument("--gemini-location", default="global")

    prompt = parser.add_argument_group("Prompt")
    prompt.add_argument("--prompt-template", default=DEFAULT_PROMPT_TEMPLATE)
    prompt.add_argument("--class-defs", help="JSON file with class definitions")
    prompt.add_argument("--label-mode", choices=["semantic", "neutral"], default="semantic")

    icl = parser.add_argument_group("ICL")
    icl.add_argument("--icl-k", type=int, default=1)
    icl.add_argument("--icl-shuffle-n", type=int, default=1)
    icl.add_argument("--seed", type=int, default=0)

    aug = parser.add_argument_group("Augmentation")
    aug.add_argument("--rotations", type=str, default="0")
    aug.add_argument("--vlm-image-size", type=int, default=512)
    aug.add_argument("--patch-size", type=int, default=None)
    aug.add_argument(
        "--classical-blur-threshold",
        type=float,
        default=DEFAULT_CLASSICAL_BLUR_THRESHOLD,
        help="Classical blur pass threshold (pass when blur_score <= threshold).",
    )
    aug.add_argument(
        "--classical-blur-sigma",
        type=float,
        default=DEFAULT_CLASSICAL_BLUR_SIGMA,
        help="Gaussian sigma for classical blur scoring.",
    )
    aug.add_argument(
        "--classical-blur-pixel-threshold",
        type=float,
        default=DEFAULT_CLASSICAL_BLUR_PIXEL_THRESHOLD,
        help="Per-pixel sharpness threshold for classical blur scoring.",
    )

    stage3 = parser.add_argument_group("Stage3")
    stage3.add_argument("--stage3-run", help="Stage3 run dir (root or bbox dir)")
    stage3.add_argument("--stage3-fg-threshold", type=float, default=0.05)

    out = parser.add_argument_group("Output")
    out.add_argument("--output-dir", default="stage6_output")
    out.add_argument(
        "--resume",
        action="store_true",
        help="Resume in-place from matching config hash output and skip patch_ids already in patches.csv",
    )
    out.add_argument("--save-patches", action="store_true")
    out.add_argument("--save-variants", action="store_true")
    out.add_argument("--save-raw-responses", action="store_true")
    out.add_argument(
        "--batch-failure-debug-dir",
        default=None,
        help="If set, save raw batched responses when batch parse fails",
    )
    out.add_argument(
        "--disable-auto-batch-replay",
        action="store_true",
        help="In --single-patch mode with --rerun-from, disable replaying the original batch context from patches.csv",
    )
    out.add_argument("--overlay-max-dim", type=int, default=1024)
    out.add_argument("--profile", action="store_true", help="Print timing breakdown")
    out.add_argument("--skip-dvc-check", action="store_true", help="Bypass DVC clean-state check")
    out.add_argument("--progress", dest="progress", action="store_true", help="Show throughput progress during VLM requests (default: on)")
    out.add_argument("--no-progress", dest="progress", action="store_false", help="Disable throughput progress output")
    out.add_argument("--progress-interval", type=float, default=2.0, help="Seconds between progress updates (default: 2.0)")

    parser.set_defaults(gemini_use_vertex=True)
    parser.set_defaults(progress=True)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args._parser = parser
    args._rerun_meta_dir = None
    defaults = {action.dest: action.default for action in parser._actions}

    # Detect which args were explicitly set by the user on the command line
    specified = set()
    for action in parser._actions:
        for opt in action.option_strings:
            if opt in sys.argv:
                specified.add(action.dest)
                break

    # Handle rerun-from metadata
    meta = None
    if args.rerun_from:
        meta, meta_dir = find_metadata_for_path(args.rerun_from)
        if meta is None:
            raise FileNotFoundError(f"metadata.json not found near {args.rerun_from}")
        args._rerun_meta_dir = meta_dir
        apply_rerun_defaults(args, meta, defaults, specified)
        if os.path.isfile(args.rerun_from) and not args.single_patch:
            args.single_patch = [args.rerun_from]
        if os.path.isdir(args.rerun_from) and not args.stage5_run:
            args.stage5_run = meta.get("stage5_run")

    if args.query_batch_size < 1:
        parser.error("--query-batch-size must be >= 1")
    if args.query_shuffle_n < 1:
        parser.error("--query-shuffle-n must be >= 1")
    if args.query_shuffle_seed is None:
        args.query_shuffle_seed = args.seed
    if not (0.0 <= args.classical_blur_threshold <= 1.0):
        parser.error("--classical-blur-threshold must be in [0, 1]")
    if args.classical_blur_sigma <= 0.0:
        parser.error("--classical-blur-sigma must be > 0")
    if args.classical_blur_pixel_threshold < 0.0:
        parser.error("--classical-blur-pixel-threshold must be >= 0")

    # Defaults for backend/model
    if args.backend is None:
        args.backend = "vllm"
    if args.model is None:
        if args.backend == "gemini":
            args.model = DEFAULT_GEMINI_MODEL
        elif args.backend == "openrouter":
            args.model = DEFAULT_OPENROUTER_MODEL
        else:
            args.model = DEFAULT_VLLM_MODEL

    # Parse rotations
    args.rotations = parse_rotations(args.rotations)

    # Load template and class defs
    template_text = load_text(args.prompt_template)
    class_defs = load_class_definitions(args.class_defs)
    if meta and meta.get("class_definitions") and not args.class_defs:
        class_defs = meta.get("class_definitions")

    # Single patch mode (stdout only)
    if args.single_patch:
        run_single_patch(args, template_text, class_defs, meta)
        return

    # Batch mode requires stage5 inputs
    stage5_runs = []
    if args.stage5_run:
        stage5_runs.append(args.stage5_run)
    if args.stage5_list:
        with open(args.stage5_list, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                stage5_runs.append(line)

    if not stage5_runs:
        parser.error("Must provide --stage5-run or --stage5-list (or --single-patch)")

    # Require clean state for reproducible outputs
    require_clean_state(stage5_runs, skip_dvc_check=args.skip_dvc_check)

    for run_dir in stage5_runs:
        out_dir = process_stage5_run(
            stage5_run=run_dir,
            args=args,
            template_text=template_text,
            class_defs=class_defs,
        )
        print(f"Saved outputs: {out_dir}")


if __name__ == "__main__":
    main()
