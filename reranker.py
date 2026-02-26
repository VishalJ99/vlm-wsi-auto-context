#!/usr/bin/env python3
# ABOUTME: Stage 5 - Re-ranker pipeline for ICL patch curation.
# ABOUTME: Extracts patches from stage4 output and ranks per class with a VLM.
"""
Re-Ranker Pipeline for WSI ICL Patch Curation (Stage 5)

Input:
    --stage4-dir path/to/stage4_output/{case}/{bbox}/{model}/{timestamp}
    --case <case_name> --model <model_name> [--stage4-root stage4_output]

Output structure:
    stage5_output/{case}/{bbox_str}/{timestamp}_{config_hash}/
        {class_name}/         - Selected patches per class
        metadata.json         - Full config + reasoning + provenance
        reproduce.txt         - Exact CLI to replicate
        intermediate/         - Debug artifacts

Usage:
    python reranker.py \
      --stage4-dir stage4_output/anon_xxx/.../20260131_225556 \
      --top-k 3

    python reranker.py \
      --stage4-dir stage4_output/anon_xxx/.../20260131_225556 \
      --top-k 3 3 2 1 \
      --use-descriptors

    python reranker.py \
      --case anon_xxx \
      --model google_gemini_3_flash_preview \
      --top-k 3
"""

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from utils.reproducibility import require_clean_state, create_reproduce_command
from utils.patch_blur import compute_blur_from_patch
from utils.wsi_backend import close_wsi, get_level0_dimensions, load_wsi, read_region_rgb
from utils.wsi_paths import resolve_wsi_path

from ablation.rankers import get_ranker


# =============================================================================
# Configuration
# =============================================================================

OUTPUT_BASE_DIR = "stage5_output"
DEFAULT_VLM_MODEL = "google/gemini-2.0-flash-001"

ICL_CLASSES = [
    "tissue",
    "background",
    "paraffin_mounting_medium",
    "pen_ink_marks",
]

CLASS_ALIASES = {
    "foreground": "tissue",
    "bg": "background",
    "paraffin": "paraffin_mounting_medium",
}

DEFAULT_CLASS_DESCRIPTIONS = {
    "tissue": "Representative histopathology tissue with clear structure and staining.",
    "background": "Glass/empty background with no tissue.",
    "paraffin_mounting_medium": "Paraffin/mounting medium artifact areas.",
    "pen_ink_marks": "Dark pen/ink markings on the slide.",
}


# =============================================================================
# Reproducibility Functions
# =============================================================================

def compute_config_hash(config: dict) -> str:
    """Generate 8-char hash from config for deterministic directory naming."""
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()[:8]


def generate_output_dir(case_name: str, bbox: Tuple[int, int, int, int], config: dict) -> Tuple[Path, str]:
    """
    Generate output directory path.

    Structure: stage5_output/{case_name}/{bbox_str}/{timestamp}_{config_hash}/

    Returns:
        (output_dir_path, config_hash)
    """
    bbox_str = f"{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}"
    config_hash = compute_config_hash(config)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir = Path(OUTPUT_BASE_DIR) / case_name / bbox_str / f"{timestamp}_{config_hash}"
    return output_dir, config_hash


# =============================================================================
# Stage4 Input Loading
# =============================================================================

def normalize_class_name(name: str) -> str:
    """Normalize class labels for consistency."""
    if not name:
        return name
    lower = name.strip().lower()
    return CLASS_ALIASES.get(lower, lower)


def infer_case_name(stage4_dir: Path) -> Optional[str]:
    """Infer case name from a stage4 directory path."""
    parts = list(stage4_dir.parts)
    if "stage4_output" in parts:
        idx = parts.index("stage4_output")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    # Fallback: try two levels up (case/bbox/...) if structure is standard
    try:
        return stage4_dir.parents[2].name
    except IndexError:
        return None


def infer_bbox_from_stage4_dir(stage4_dir: Path) -> Optional[Tuple[int, int, int, int]]:
    """Infer bbox from any parent directory named x1_y1_x2_y2."""
    for parent in [stage4_dir] + list(stage4_dir.parents):
        parts = parent.name.split("_")
        if len(parts) == 4 and all(p.lstrip("-").isdigit() for p in parts):
            return tuple(int(p) for p in parts)  # type: ignore[return-value]
    return None


def _find_latest_subdir(parent: Path) -> Optional[Path]:
    """Return newest subdir by name (timestamp-prefixed)."""
    if not parent.exists():
        return None
    candidates = sorted([p for p in parent.iterdir() if p.is_dir()], key=lambda p: p.name)
    return candidates[-1] if candidates else None


def resolve_latest_stage4_dirs(stage4_root: Path, case_name: str, model_name: str) -> List[Path]:
    """Resolve latest stage4 timestamp dir per bbox for a case + model."""
    case_dir = stage4_root / case_name
    if not stage4_root.exists():
        raise ValueError(f"stage4 root not found: {stage4_root}")
    if not case_dir.exists():
        raise ValueError(f"case not found under stage4 root: {case_dir}")

    stage4_dirs: List[Path] = []
    for bbox_dir in sorted(case_dir.iterdir()):
        if not bbox_dir.is_dir():
            continue
        model_dir = bbox_dir / model_name
        if not model_dir.is_dir():
            continue
        latest = _find_latest_subdir(model_dir)
        if latest is None:
            continue
        stage4_dirs.append(latest)
    return stage4_dirs


def resolve_stage2_dir(stage2_input: Optional[str], repo_root: Path) -> Optional[Path]:
    if not stage2_input:
        return None
    stage2_path = Path(stage2_input)
    if stage2_path.exists():
        return stage2_path
    # Common container path mapping: /workspace -> repo root
    stage2_str = str(stage2_path)
    if stage2_str.startswith("/workspace/"):
        mapped = repo_root / stage2_str[len("/workspace/"):]
        if mapped.exists():
            return mapped
    return stage2_path


def load_visual_descriptors(stage2_dir: Optional[Path]) -> Optional[Dict[str, str]]:
    if stage2_dir is None:
        return None
    desc_path = stage2_dir / "visual_descriptions.json"
    if not desc_path.exists():
        return None
    with open(desc_path) as f:
        data = json.load(f)
    descriptors: Dict[str, str] = {}
    for key, value in data.items():
        if isinstance(value, str):
            descriptors[normalize_class_name(key)] = value
    return descriptors


def _collect_points_from_json(
    points_json: Dict[str, Any],
    source_tag: str,
    points_by_class: Dict[str, List[Dict[str, Any]]],
) -> Optional[Tuple[int, int, int, int]]:
    """Collect points from a points.json, appending into points_by_class."""
    bbox = None
    if isinstance(points_json.get("input_bbox"), list) and len(points_json["input_bbox"]) == 4:
        bbox = tuple(points_json["input_bbox"])  # type: ignore[assignment]

    if "classes" in points_json and isinstance(points_json["classes"], dict):
        classes_dict = points_json["classes"]
    else:
        classes_dict = {}
        if "foreground" in points_json:
            classes_dict["foreground"] = points_json["foreground"]
        if "background" in points_json:
            classes_dict["background"] = points_json["background"]

    for class_name, points in classes_dict.items():
        norm_class = normalize_class_name(class_name)
        if not isinstance(points, list):
            continue
        points_by_class.setdefault(norm_class, [])
        for point in points:
            if not isinstance(point, dict) or "point_l0" not in point:
                continue
            point_copy = dict(point)
            point_copy["_source"] = source_tag
            points_by_class[norm_class].append(point_copy)

    return bbox


def load_points_data(stage4_dir: Path) -> Tuple[Dict[str, List[Dict[str, Any]]], Optional[Tuple[int, int, int, int]], List[str]]:
    """
    Load points from stage4 output.

    Supports:
        - points.json at root
        - rot_*/points.json for TTA runs
    """
    points_by_class: Dict[str, List[Dict[str, Any]]] = {}
    sources: List[str] = []
    bbox: Optional[Tuple[int, int, int, int]] = None

    root_points = stage4_dir / "points.json"
    if root_points.exists():
        with open(root_points) as f:
            points_json = json.load(f)
        bbox = _collect_points_from_json(points_json, "root", points_by_class)
        sources.append(str(root_points))
        return points_by_class, bbox, sources

    rot_dirs = sorted([p for p in stage4_dir.iterdir() if p.is_dir() and p.name.startswith("rot_")])
    for rot_dir in rot_dirs:
        points_path = rot_dir / "points.json"
        if not points_path.exists():
            continue
        with open(points_path) as f:
            points_json = json.load(f)
        bbox_from_points = _collect_points_from_json(points_json, rot_dir.name, points_by_class)
        if bbox is None and bbox_from_points:
            bbox = bbox_from_points
        sources.append(str(points_path))

    return points_by_class, bbox, sources


def load_stage4_input(stage4_dir: Path, use_descriptors: bool) -> Dict[str, Any]:
    """Load stage4 metadata, points, and optional visual descriptors."""
    metadata_path = stage4_dir / "metadata.json"
    metadata: Dict[str, Any] = {}
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)

    case_name = metadata.get("case_name") or infer_case_name(stage4_dir)
    if not case_name:
        raise ValueError(f"Could not infer case name from {stage4_dir}")

    bbox = None
    if isinstance(metadata.get("input_bbox"), list) and len(metadata["input_bbox"]) == 4:
        bbox = tuple(metadata["input_bbox"])  # type: ignore[assignment]
    if bbox is None:
        bbox = infer_bbox_from_stage4_dir(stage4_dir)

    stage2_dir = resolve_stage2_dir(metadata.get("stage2_input"), repo_root=Path(__file__).resolve().parent)
    visual_descriptors = load_visual_descriptors(stage2_dir) if use_descriptors else None

    points_by_class, bbox_from_points, points_sources = load_points_data(stage4_dir)
    if bbox is None and bbox_from_points is not None:
        bbox = bbox_from_points

    if bbox is None:
        raise ValueError("Could not infer bbox from stage4 metadata or points.json")

    wsi_path = metadata.get("wsi_path")
    if wsi_path and os.path.exists(wsi_path):
        wsi_path = os.path.abspath(wsi_path)
    else:
        wsi_path = resolve_wsi_path(case_name)

    return {
        "wsi_path": wsi_path,
        "bbox": bbox,
        "case_name": case_name,
        "stage2_dir": stage2_dir,
        "visual_descriptors": visual_descriptors,
        "points_by_class": points_by_class,
        "points_sources": points_sources,
        "metadata": metadata,
    }


# =============================================================================
# Patch Extraction
# =============================================================================

def extract_patch_at_point(
    wsi: Any,
    wsi_backend: str,
    point: Tuple[int, int],
    patch_size: int,
    k: int
) -> List[Tuple[Image.Image, str]]:
    """
    Extract patch(es) centered on a point.

    Args:
        wsi: WSI object
        wsi_backend: WSI backend ("cucim" or "openslide")
        point: (x, y) L0 coordinates
        patch_size: Size of each output patch
        k: Neighborhood mode (1=central, 4=2x2 quadrants)

    Returns:
        List of (patch_image, suffix) tuples.
        suffix is 'central' for k=1, or 'q0'/'q1'/'q2'/'q3' for k=4
    """
    wsi_width, wsi_height = get_level0_dimensions(wsi, wsi_backend)

    x, y = point
    patches = []

    if k == 1:
        # Central patch only
        half = patch_size // 2
        x1 = max(0, x - half)
        y1 = max(0, y - half)
        x2 = min(wsi_width, x1 + patch_size)
        y2 = min(wsi_height, y1 + patch_size)

        patch = _read_patch(wsi, wsi_backend, x1, y1, x2 - x1, y2 - y1)
        patches.append((patch, "central"))

    elif k == 4:
        # 2x2 quadrants around point
        full_size = patch_size * 2
        half_full = full_size // 2

        x1 = max(0, x - half_full)
        y1 = max(0, y - half_full)
        x2 = min(wsi_width, x1 + full_size)
        y2 = min(wsi_height, y1 + full_size)

        full_patch = _read_patch(wsi, wsi_backend, x1, y1, x2 - x1, y2 - y1)
        full_w, full_h = full_patch.size

        mid_x = full_w // 2
        mid_y = full_h // 2

        quadrants = [
            (0, 0, mid_x, mid_y, "q0"),
            (mid_x, 0, full_w, mid_y, "q1"),
            (0, mid_y, mid_x, full_h, "q2"),
            (mid_x, mid_y, full_w, full_h, "q3"),
        ]

        for qx1, qy1, qx2, qy2, suffix in quadrants:
            if qx2 > qx1 and qy2 > qy1:
                quadrant = full_patch.crop((qx1, qy1, qx2, qy2))
                patches.append((quadrant, suffix))

    return patches


def _read_patch(wsi: Any, wsi_backend: str, x: int, y: int, w: int, h: int) -> Image.Image:
    """Read a patch from WSI at level 0."""
    if w < 1 or h < 1:
        return Image.new('RGB', (1, 1))

    patch_np = read_region_rgb(
        wsi,
        wsi_backend,
        x=x,
        y=y,
        width=w,
        height=h,
        level=0,
    )
    return Image.fromarray(patch_np)


def extract_all_patches_multiclass(
    wsi_path: str,
    points_by_class: Dict[str, List[Dict[str, Any]]],
    patch_size: int,
    k: int,
    intermediate_dir: Path,
    wsi_reader: str = "cucim",
    tissue_blur_filter: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any], str]:
    """
    Extract patches for all classes present in points_by_class.

    Returns:
        (candidates_by_class, blur_filter_info, resolved_wsi_reader)
    """

    print(f"\nExtracting patches for {len(points_by_class)} class(es)...")

    blur_enabled = bool(tissue_blur_filter and tissue_blur_filter.get("enabled"))
    blur_target_class = str(tissue_blur_filter.get("target_class", "tissue")) if tissue_blur_filter else "tissue"
    blur_apply_filter = bool(tissue_blur_filter.get("apply_filter", False)) if tissue_blur_filter else False
    blur_sigma = float(tissue_blur_filter.get("sigma", 0.5)) if tissue_blur_filter else 0.5
    blur_pixel_threshold = float(tissue_blur_filter.get("pixel_threshold", 0.005)) if tissue_blur_filter else 0.005
    blur_patch_threshold = float(tissue_blur_filter.get("patch_threshold", 0.1)) if tissue_blur_filter else 0.1
    blur_compute_fn = tissue_blur_filter.get("compute_fn") if tissue_blur_filter else None
    if blur_enabled and not callable(blur_compute_fn):
        raise RuntimeError("Tissue blur filter is enabled but compute_fn is missing")

    blur_rows: List[Dict[str, Any]] = []
    blur_filter_info: Dict[str, Any] = {
        "enabled": blur_enabled,
        "target_class": blur_target_class,
        "apply_filter": blur_apply_filter,
        "sigma": blur_sigma,
        "pixel_threshold": blur_pixel_threshold,
        "patch_threshold": blur_patch_threshold,
        "candidates_before": None,
        "candidates_after": None,
        "removed_count": 0,
        "score_csv": None,
    }

    wsi, resolved_wsi_reader = load_wsi(wsi_path, wsi_reader)
    candidates_by_class: Dict[str, List[Dict[str, Any]]] = {}

    try:
        for class_name, points in points_by_class.items():
            class_candidates: List[Dict[str, Any]] = []
            class_dir = intermediate_dir / "all_candidates" / class_name
            class_dir.mkdir(parents=True, exist_ok=True)

            for point_idx, point_info in enumerate(points):
                point_l0 = tuple(point_info["point_l0"])
                patches = extract_patch_at_point(
                    wsi,
                    resolved_wsi_reader,
                    point_l0,
                    patch_size,
                    k,
                )

                point_id = point_info.get("vlm_index", point_idx)
                source_tag = point_info.get("_source", "root")
                point_normalized = point_info.get("point_normalized")
                point_normalized_rotated = point_info.get("point_normalized_rotated")

                for patch_img, suffix in patches:
                    name = f"{source_tag}_p{point_id}_{point_l0[0]}_{point_l0[1]}_{suffix}.png"
                    patch_img.save(class_dir / name)

                    class_candidates.append({
                        "image": patch_img,
                        "name": name,
                        "point_idx": point_idx,
                        "point_l0": point_l0,
                        "point_normalized": point_normalized,
                        "point_normalized_rotated": point_normalized_rotated,
                        "vlm_index": point_info.get("vlm_index"),
                        "label": class_name,
                        "source": source_tag,
                    })

            if blur_enabled and class_name == blur_target_class:
                before_count = len(class_candidates)
                all_candidates = list(class_candidates)
                kept_candidates: List[Dict[str, Any]] = []
                for candidate in class_candidates:
                    patch_rgb = np.asarray(candidate["image"])
                    blur_result = blur_compute_fn(
                        patch_rgb=patch_rgb,
                        sigma=blur_sigma,
                        pixel_threshold=blur_pixel_threshold,
                    )
                    raw_blur_score = getattr(blur_result, "blur_score", None)
                    raw_sharp_score = getattr(blur_result, "sharp_score", None)
                    blur_score = float(raw_blur_score) if raw_blur_score is not None else None
                    sharp_score = float(raw_sharp_score) if raw_sharp_score is not None else None
                    keep = (blur_score is not None) and (blur_score <= blur_patch_threshold)

                    candidate["blur_score"] = blur_score
                    candidate["sharp_score"] = sharp_score
                    candidate["blur_filter_pass"] = keep

                    blur_rows.append({
                        "candidate_name": candidate["name"],
                        "point_l0_x": candidate["point_l0"][0],
                        "point_l0_y": candidate["point_l0"][1],
                        "source": candidate.get("source"),
                        "blur_score": blur_score,
                        "sharp_score": sharp_score,
                        "pass_blur_filter": int(keep),
                    })
                    if keep:
                        kept_candidates.append(candidate)

                if blur_apply_filter:
                    class_candidates = kept_candidates
                    after_count = len(class_candidates)
                else:
                    class_candidates = all_candidates
                    after_count = before_count
                blur_filter_info["candidates_before"] = before_count
                blur_filter_info["candidates_after"] = after_count
                blur_filter_info["removed_count"] = before_count - len(kept_candidates)

            candidates_by_class[class_name] = class_candidates
            print(f"  {class_name}: {len(points)} points -> {len(class_candidates)} patches")

        if blur_rows:
            blur_csv_path = intermediate_dir / "tissue_blur_scores.csv"
            with blur_csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "candidate_name",
                        "point_l0_x",
                        "point_l0_y",
                        "source",
                        "blur_score",
                        "sharp_score",
                        "pass_blur_filter",
                    ],
                )
                writer.writeheader()
                writer.writerows(blur_rows)
            blur_filter_info["score_csv"] = str(blur_csv_path)
            print(
                "  tissue blur filter: "
                f"kept {blur_filter_info['candidates_after']} / {blur_filter_info['candidates_before']} "
                f"(threshold={blur_patch_threshold}, apply_filter={blur_apply_filter})"
            )
    finally:
        close_wsi(wsi, resolved_wsi_reader)

    return candidates_by_class, blur_filter_info, resolved_wsi_reader


# =============================================================================
# Point Overlay Visualization
# =============================================================================

def transform_point_to_original(point_norm: List[int], rotation_deg: int) -> List[int]:
    """
    Transform point from rotated coordinate space back to original (0°) space.

    Args:
        point_norm: [x, y] in 0-1000 normalized coordinates (rotated space)
        rotation_deg: The rotation that was applied (0, 90, 180, 270 CCW)

    Returns:
        [x, y] transformed to original orientation
    """
    x, y = point_norm[0], point_norm[1]
    if rotation_deg == 90:
        # 90° CCW reverse: (x, y) -> (1000-y, x)
        return [1000 - y, x]
    elif rotation_deg == 180:
        # 180° reverse: (x, y) -> (1000-x, 1000-y)
        return [1000 - x, 1000 - y]
    elif rotation_deg == 270:
        # 270° CCW reverse: (x, y) -> (y, 1000-x)
        return [y, 1000 - x]
    return [x, y]


def create_per_rotation_overlays(
    stage4_dir: Path,
    points_by_class: Dict[str, List[Dict[str, Any]]],
    output_dir: Path,
) -> None:
    """
    Create separate point overlay images for each rotation to avoid crowding.

    Generates 4 images: points_overlay_rot_0.png, points_overlay_rot_90.png, etc.
    Each shows only points from that rotation, with enumerated labels.

    Args:
        stage4_dir: Path to stage4 output directory
        points_by_class: Dict of class_name -> list of point dicts with _source and vlm_index
        output_dir: Directory to save the overlay images
    """
    # Find the thumbnail
    thumbnail_path = stage4_dir / "region_thumbnail.png"
    if not thumbnail_path.exists():
        thumbnail_path = stage4_dir / "rot_0" / "region_thumbnail.png"

    if not thumbnail_path.exists():
        print(f"Warning: No thumbnail found in {stage4_dir}, skipping overlays")
        return

    thumbnail = Image.open(thumbnail_path)
    thumb_w, thumb_h = thumbnail.size

    # Get all class labels and colors (consistent across all overlays)
    class_labels = sorted(points_by_class.keys())
    cmap = plt.colormaps.get_cmap('tab10')
    class_colors = {label: cmap(i % 10) for i, label in enumerate(class_labels)}

    # Group points by rotation source
    rotations = ['rot_0', 'rot_90', 'rot_180', 'rot_270']
    rotation_names = {'rot_0': '0°', 'rot_90': '90°', 'rot_180': '180°', 'rot_270': '270°'}

    for rotation in rotations:
        # Filter points for this rotation
        points_this_rot: Dict[str, List[Dict]] = {}
        for class_name in class_labels:
            filtered = [p for p in points_by_class[class_name] if p.get("_source") == rotation]
            if filtered:
                points_this_rot[class_name] = filtered

        if not points_this_rot:
            continue  # Skip rotations with no points

        # Create figure
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.imshow(thumbnail)

        # Parse rotation degree
        rotation_deg = int(rotation.split("_")[1])

        legend_handles = []
        total_points = 0

        for class_name in class_labels:
            if class_name not in points_this_rot:
                continue

            points = points_this_rot[class_name]
            color = class_colors[class_name]

            for point in points:
                vlm_idx = point.get("vlm_index", "?")

                # Simpler label for per-rotation view: just p{idx}
                label = f"p{vlm_idx}"

                point_norm_rotated = point.get("point_normalized_rotated")
                point_norm = point.get("point_normalized")
                if point_norm_rotated:
                    point_norm_original = transform_point_to_original(point_norm_rotated, rotation_deg)
                elif point_norm:
                    if rotation_deg != 0:
                        point_norm_original = transform_point_to_original(point_norm, rotation_deg)
                    else:
                        point_norm_original = point_norm
                else:
                    continue

                x = int(point_norm_original[0] / 1000 * thumb_w)
                y = int(point_norm_original[1] / 1000 * thumb_h)

                # Draw point
                ax.scatter(x, y, c=[color], marker='o', s=150, alpha=0.9,
                          edgecolors='white', linewidth=1.5)

                # Draw label
                ax.annotate(
                    label,
                    (x, y),
                    xytext=(10, 5),
                    textcoords='offset points',
                    fontsize=9,
                    fontweight='bold',
                    color='black',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor=color),
                )

                total_points += 1

            # Legend entry
            legend_handles.append(
                mpatches.Patch(color=color, label=f'{class_name} ({len(points)})')
            )

        ax.set_title(
            f'Points Overlay - Rotation {rotation_names[rotation]} ({total_points} points)\n'
            f'Patch filenames: {rotation}_p{{idx}}_{{x}}_{{y}}_{{suffix}}.png',
            fontsize=12, fontweight='bold'
        )
        ax.axis('off')

        ax.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1.02, 1),
                  fontsize=10, title='Classes')

        plt.tight_layout()
        output_path = output_dir / f"points_overlay_{rotation}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")


# =============================================================================
# CLI
# =============================================================================

def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description='Stage 5 Re-ranker: Curate ICL patches from stage4 output',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python reranker.py --stage4-dir stage4_output/anon_xxx/.../20260131_225556 --top-k 3
  python reranker.py --stage4-dir stage4_output/anon_xxx/.../20260131_225556 --top-k 3 3 2 1 --use-descriptors
  python reranker.py --case anon_xxx --model google_gemini_3_flash_preview --top-k 3
"""
    )

    # Input arguments
    input_group = parser.add_argument_group('Input')
    input_group.add_argument(
        '--stage4-dir', type=str, default=None,
        help='Path to stage4 output directory (timestamp folder)'
    )
    input_group.add_argument(
        '--case', type=str, default=None,
        help='Case name under stage4_output/ (used with --model)'
    )
    input_group.add_argument(
        '--model', type=str, default=None,
        help='Stage4 model directory name (used with --case)'
    )
    input_group.add_argument(
        '--stage4-root', type=str, default='stage4_output',
        help='Root directory for stage4 outputs (default: stage4_output)'
    )
    input_group.add_argument(
        '--top-k', type=int, nargs='+', default=[3],
        help='Top-k patches per class. Single value applies to all classes; list applies per-class in order.'
    )
    input_group.add_argument(
        '--use-descriptors', action='store_true',
        help='Include visual descriptors from stage2 in the VLM prompt'
    )

    # Patch extraction
    extract_group = parser.add_argument_group('Patch Extraction')
    extract_group.add_argument(
        '--k', type=int, default=1, choices=[1, 4],
        help='Neighborhood: 1=central patch, 4=2x2 quadrants (default: 1)'
    )
    extract_group.add_argument(
        '--patch-size', type=int, default=512,
        help='Patch size at L0 in pixels (default: 512)'
    )
    extract_group.add_argument(
        "--wsi-reader",
        choices=["auto", "openslide", "cucim"],
        default="cucim",
        help="WSI reader backend for patch extraction (default: cucim).",
    )
    extract_group.add_argument(
        '--max-candidates-per-class', type=int, default=None,
        help=(
            'Legacy random cap per class before extraction. '
            'Ignored when --max-total-candidates is active.'
        )
    )
    extract_group.add_argument(
        '--sample-seed', type=int, default=42,
        help='Random seed for candidate sampling (default: 42)'
    )
    extract_group.add_argument(
        '--max-total-candidates', type=int, default=25,
        help='Hard cap on candidates per VLM call for tournament reranking (default: 25).'
    )
    extract_group.add_argument(
        '--tournament-round1-k', type=int, default=None,
        help=(
            'Optional override for per-class top-k in the first tournament reduction round. '
            'If unset, computed automatically from pool size and --max-total-candidates.'
        ),
    )
    extract_group.add_argument(
        '--disable-tissue-blur-filter',
        action='store_true',
        help=(
            'Disable post-selection blur gating for tissue ICL picks. '
            'By default, selected tissue patches are blur-checked with HistoQC scoring.'
        ),
    )
    extract_group.add_argument(
        '--tissue-blur-threshold',
        type=float,
        default=0.1,
        help='Keep tissue candidates with blur_score <= threshold (default: 0.1).',
    )
    extract_group.add_argument(
        '--tissue-blur-sigma',
        type=float,
        default=0.5,
        help='Gaussian sigma for HistoQC blur scoring (default: 0.5).',
    )
    extract_group.add_argument(
        '--tissue-blur-pixel-threshold',
        type=float,
        default=0.005,
        help='Per-pixel sharpness threshold for HistoQC blur scoring (default: 0.005).',
    )

    # Ranking - VLM-specific
    vlm_group = parser.add_argument_group('VLM Ranking Options')
    vlm_group.add_argument(
        '--vlm-backend', type=str, default='openrouter', choices=['vllm', 'openrouter', 'gemini'],
        help='VLM backend for ranking (default: openrouter)'
    )
    vlm_group.add_argument(
        '--vlm-model', type=str, default=DEFAULT_VLM_MODEL,
        help=f'VLM model for ranking (default: {DEFAULT_VLM_MODEL})'
    )
    vlm_group.add_argument(
        '--vlm-port', type=int, default=8000,
        help='vLLM server port (default: 8000, for --vlm-backend vllm)'
    )
    vlm_group.add_argument(
        '--vlm-max-tokens', type=int, default=512,
        help='Max completion tokens for VLM ranking responses (default: 512)'
    )
    vlm_group.add_argument(
        '--vlm-max-retries', type=int, default=3,
        help='Max retry attempts for VLM ranking calls (default: 3)'
    )
    vlm_group.add_argument(
        '--selection-mode',
        choices=['auto', 'tournament', 'single_pass'],
        default='auto',
        help=(
            'Ranking reduction strategy: auto chooses single_pass for gemini and '
            'tournament for other backends; tournament uses overlapping reduction windows.'
        ),
    )
    vlm_group.add_argument(
        '--vlm-image-size',
        type=int,
        default=None,
        help=(
            'Optional square image size used only for VLM ranking payloads '
            '(e.g., 256 to downsample 512x512 patches for context efficiency).'
        ),
    )

    gemini_group = parser.add_argument_group('Gemini SDK Options')
    gemini_group.add_argument(
        '--gemini-use-vertex', dest='gemini_use_vertex', action='store_true',
        help='Use Gemini via Vertex AI (default: enabled)'
    )
    gemini_group.add_argument(
        '--gemini-no-vertex', dest='gemini_use_vertex', action='store_false',
        help='Use Gemini SDK without Vertex mode'
    )
    gemini_group.add_argument(
        '--gemini-credentials', type=str, default=None,
        help='Optional Gemini Vertex credentials JSON. If unset, use GOOGLE_APPLICATION_CREDENTIALS.'
    )
    gemini_group.add_argument(
        '--gemini-location', type=str, default='global',
        help='Gemini Vertex location (default: global)'
    )
    gemini_group.add_argument(
        '--gemini-thinking-level', type=str, default='High',
        help='Gemini thinking level (Low/High, default: High)'
    )
    gemini_group.add_argument(
        '--gemini-include-thoughts', action='store_true',
        help='Include Gemini thought summaries in responses (if supported)'
    )

    openrouter_group = parser.add_argument_group('OpenRouter Options')
    openrouter_group.add_argument(
        '--reasoning-effort', type=str, default='high',
        help='OpenRouter reasoning effort: low/medium/high (default: high)'
    )

    # Output
    output_group = parser.add_argument_group('Output')
    output_group.add_argument(
        '--output-dir', type=str, default=None,
        help='Output directory (default: auto-generated in stage5_output/)'
    )

    # Reproducibility
    repro_group = parser.add_argument_group('Reproducibility')
    repro_group.add_argument(
        '--skip-dvc-check',
        action='store_true',
        help='Bypass DVC clean-state check (still checks git)'
    )

    parser.set_defaults(gemini_use_vertex=True)

    return parser


def _parse_top_k(values: List[int], classes: List[str]) -> Dict[str, int]:
    if not values:
        raise ValueError("--top-k must have at least one value")
    if len(values) == 1:
        return {c: values[0] for c in classes}
    if len(values) != len(classes):
        raise ValueError(
            f"--top-k expects 1 value or {len(classes)} values for classes {classes}. "
            f"Got {len(values)} value(s)."
    )
    return {c: v for c, v in zip(classes, values)}


def _resolve_selection_mode(selection_mode: str, vlm_backend: str) -> str:
    """Resolve final ranking mode from CLI choice + backend."""
    mode = (selection_mode or "auto").strip().lower()
    if mode in {"tournament", "single_pass"}:
        return mode
    if mode != "auto":
        raise ValueError(f"Unsupported selection mode: {selection_mode}")

    backend = (vlm_backend or "").strip().lower()
    if backend == "gemini":
        return "single_pass"
    return "tournament"


def _dedupe_preserve_order(values: List[int]) -> List[int]:
    """Deduplicate while preserving first occurrence order."""
    seen = set()
    deduped: List[int] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def compute_tournament_windows(
    total_points: int,
    max_total: int,
) -> Tuple[List[Tuple[int, int]], List[int]]:
    """
    Build fixed-size overlapping windows over [0, total_points).

    Uses deterministic overlap distribution:
      K = ceil(N / M), x = K*M - N
      distribute x overlaps across K-1 boundaries as evenly as possible.

    Returns:
      - windows: [(start, end), ...] with half-open ranges [start, end)
      - boundary_overlaps: overlap size for each boundary i->i+1 (len K-1)
    """
    if total_points <= 0:
        return [], []
    if max_total <= 0:
        raise ValueError("max_total must be >= 1")

    if total_points <= max_total:
        return [(0, total_points)], []

    n_calls = int(math.ceil(total_points / max_total))
    total_capacity = n_calls * max_total
    overlap_total = total_capacity - total_points

    n_boundaries = n_calls - 1
    base_overlap, extra = divmod(overlap_total, n_boundaries)
    boundary_overlaps = [
        base_overlap + (1 if idx < extra else 0)
        for idx in range(n_boundaries)
    ]

    windows: List[Tuple[int, int]] = []
    start = 0
    for call_idx in range(n_calls):
        end = min(total_points, start + max_total)
        windows.append((start, end))
        if call_idx < n_boundaries:
            start = start + max_total - boundary_overlaps[call_idx]

    if windows and windows[-1][1] != total_points:
        windows[-1] = (windows[-1][0], total_points)

    return windows, boundary_overlaps


def _compute_reduction_k_per_class(
    final_k_per_class: Dict[str, int],
    pool_size: int,
    max_total_candidates: int,
) -> Dict[str, int]:
    """
    Compute per-class k for a reduction round before the final round.

    Keeps at least 1 per active class to preserve class coverage in early rounds.
    """
    active_classes = [c for c, k in final_k_per_class.items() if k > 0]
    if not active_classes:
        return {c: 0 for c in final_k_per_class}

    if pool_size <= max_total_candidates:
        return dict(final_k_per_class)

    n_calls = int(math.ceil(pool_size / max_total_candidates))
    base = max_total_candidates // (n_calls * len(active_classes))
    if base < 1:
        base = 1

    out: Dict[str, int] = {}
    for class_name, final_k in final_k_per_class.items():
        if final_k <= 0:
            out[class_name] = 0
        else:
            out[class_name] = min(final_k, base)
    return out


def _build_call_inputs(
    pool_candidate_ids: List[int],
    classes_ranked: List[str],
    candidate_lookup: Dict[int, Dict[str, Any]],
) -> Tuple[Dict[str, Tuple[List[Image.Image], List[str]]], List[int]]:
    """
    Build rank_multiclass inputs for one call.

    Returns:
      - patches_by_class keyed by suggested/source class
      - call_manifest mapping call-local global index -> candidate_manifest global_index
    """
    patches_by_class: Dict[str, Tuple[List[Image.Image], List[str]]] = {}
    call_manifest: List[int] = []

    for class_name in classes_ranked:
        class_patches: List[Image.Image] = []
        class_names: List[str] = []
        for candidate_id in pool_candidate_ids:
            info = candidate_lookup.get(candidate_id)
            if not info:
                continue
            if info.get("source_class") != class_name:
                continue
            candidate = info.get("candidate")
            if not isinstance(candidate, dict):
                continue
            image = candidate.get("image")
            name = candidate.get("name")
            if image is None or not isinstance(name, str):
                continue
            class_patches.append(image)
            class_names.append(name)
            call_manifest.append(candidate_id)
        patches_by_class[class_name] = (class_patches, class_names)

    return patches_by_class, call_manifest


def _run_rerank_for_stage4_dir(
    stage4_dir: Path,
    stage4_input: Dict[str, Any],
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    state_info: Dict[str, Any],
) -> None:
    wsi_path = stage4_input["wsi_path"]
    bbox = stage4_input["bbox"]
    case_name = stage4_input["case_name"]
    points_by_class = stage4_input["points_by_class"]

    if args.max_total_candidates < 1:
        print("Error: --max-total-candidates must be >= 1", file=sys.stderr)
        sys.exit(1)
    if args.tournament_round1_k is not None and args.tournament_round1_k < 1:
        print("Error: --tournament-round1-k must be >= 1", file=sys.stderr)
        sys.exit(1)
    if args.vlm_image_size is not None and args.vlm_image_size < 1:
        print("Error: --vlm-image-size must be >= 1", file=sys.stderr)
        sys.exit(1)
    if args.gemini_thinking_level is not None:
        thinking_level_normalized = args.gemini_thinking_level.strip().lower()
        if not thinking_level_normalized:
            args.gemini_thinking_level = None
        elif thinking_level_normalized not in {"low", "high"}:
            print("Error: --gemini-thinking-level must be one of: Low, High", file=sys.stderr)
            sys.exit(1)
        else:
            args.gemini_thinking_level = "Low" if thinking_level_normalized == "low" else "High"
    if args.reasoning_effort is not None:
        reasoning_effort_normalized = args.reasoning_effort.strip().lower()
        if not reasoning_effort_normalized:
            args.reasoning_effort = None
        elif reasoning_effort_normalized not in {"low", "medium", "high"}:
            print("Error: --reasoning-effort must be one of: low, medium, high", file=sys.stderr)
            sys.exit(1)
        else:
            args.reasoning_effort = reasoning_effort_normalized

    # Normalize class list
    classes_present = sorted(points_by_class.keys())
    classes_ranked = [c for c in ICL_CLASSES if c in classes_present]
    skipped_classes = [c for c in classes_present if c not in classes_ranked]

    if not classes_ranked:
        print("Error: No ICL-worthy classes found in points.json", file=sys.stderr)
        sys.exit(1)

    # Parse top-k values
    try:
        k_per_class = _parse_top_k(args.top_k, classes_ranked)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    selection_mode_resolved = _resolve_selection_mode(args.selection_mode, args.vlm_backend)
    effective_vlm_image_size = args.vlm_image_size
    if (
        effective_vlm_image_size is None
        and selection_mode_resolved == "single_pass"
        and (args.vlm_backend or "").lower() == "gemini"
    ):
        # Gemini single-pass mode can typically fit all candidates at this resolution.
        effective_vlm_image_size = 256

    # Build config for hashing
    run_config = {
        "k": args.k,
        "patch_size": args.patch_size,
        "wsi_reader": args.wsi_reader,
        "max_total_candidates": args.max_total_candidates,
        "tournament_round1_k": args.tournament_round1_k,
        "max_candidates_per_class": args.max_candidates_per_class,
        "sample_seed": args.sample_seed,
        "disable_tissue_blur_filter": args.disable_tissue_blur_filter,
        "tissue_blur_threshold": args.tissue_blur_threshold,
        "tissue_blur_sigma": args.tissue_blur_sigma,
        "tissue_blur_pixel_threshold": args.tissue_blur_pixel_threshold,
        "top_k": args.top_k,
        "use_descriptors": args.use_descriptors,
        "vlm_backend": args.vlm_backend,
        "vlm_model": args.vlm_model,
        "vlm_max_tokens": args.vlm_max_tokens,
        "vlm_max_retries": args.vlm_max_retries,
        "selection_mode": args.selection_mode,
        "selection_mode_resolved": selection_mode_resolved,
        "vlm_image_size": effective_vlm_image_size,
        "openrouter_reasoning_effort": args.reasoning_effort if args.vlm_backend == "openrouter" else None,
        "gemini_use_vertex": args.gemini_use_vertex if args.vlm_backend == "gemini" else None,
        "gemini_location": args.gemini_location if args.vlm_backend == "gemini" else None,
        "gemini_thinking_level": args.gemini_thinking_level if args.vlm_backend == "gemini" else None,
        "gemini_include_thoughts": args.gemini_include_thoughts if args.vlm_backend == "gemini" else None,
    }

    # Generate output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
        config_hash = compute_config_hash(run_config)
    else:
        output_dir, config_hash = generate_output_dir(case_name, bbox, run_config)

    output_dir.mkdir(parents=True, exist_ok=True)
    intermediate_dir = output_dir / "intermediate"
    intermediate_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("STAGE 5: RE-RANKER PIPELINE (MULTI-CLASS)")
    print("=" * 60)
    print(f"Stage4 dir: {stage4_dir}")
    print(f"WSI: {wsi_path}")
    print(f"Bbox: {bbox}")
    print(f"Classes present: {classes_present}")
    print(f"Classes ranked: {classes_ranked}")
    if skipped_classes:
        print(f"Classes skipped (not ICL-worthy): {skipped_classes}")
    print(f"Patch size: {args.patch_size}, k={args.k}")
    print(f"Selection mode: {selection_mode_resolved} (requested: {args.selection_mode})")
    print(f"VLM image size: {effective_vlm_image_size if effective_vlm_image_size else 'original'}")
    if selection_mode_resolved == "tournament":
        print(f"Max total candidates per VLM call: {args.max_total_candidates}")
        if args.tournament_round1_k is not None:
            print(f"Tournament round-1 k override: {args.tournament_round1_k}")
    print(f"Top-k per class: {k_per_class}")
    print(f"Output: {output_dir}")
    print(f"Config hash: {config_hash}")
    print("=" * 60)

    # Save stage4 input reference
    stage4_input_ref = {
        "stage4_dir": str(stage4_dir),
        "points_sources": stage4_input.get("points_sources", []),
    }
    with open(intermediate_dir / "stage4_input.json", 'w') as f:
        json.dump(stage4_input_ref, f, indent=2)

    # Filter points to ICL-worthy classes
    points_for_ranking = {c: points_by_class[c] for c in classes_ranked}

    # Keep legacy flag in metadata for traceability; current ranking modes ignore this cap.
    sampling_info: Optional[Dict[str, Any]] = None
    if args.max_candidates_per_class is not None:
        sampling_info = {
            "mode": "ignored",
            "requested_max_candidates_per_class": args.max_candidates_per_class,
            "sample_seed": args.sample_seed,
        }

    # Create per-rotation point overlays for traceability (avoids crowding)
    create_per_rotation_overlays(stage4_dir, points_for_ranking, output_dir)

    # === STAGE 1: PATCH EXTRACTION ===
    print("\n[Stage 1] Patch Extraction")
    print("-" * 40)

    tissue_blur_filter_cfg = {
        "enabled": not args.disable_tissue_blur_filter,
        "target_class": "tissue",
        "apply_filter": False,
        "sigma": args.tissue_blur_sigma,
        "pixel_threshold": args.tissue_blur_pixel_threshold,
        "patch_threshold": args.tissue_blur_threshold,
        "compute_fn": compute_blur_from_patch,
    }

    candidates_by_class, blur_filter_info, resolved_wsi_reader = extract_all_patches_multiclass(
        wsi_path=wsi_path,
        points_by_class=points_for_ranking,
        patch_size=args.patch_size,
        k=args.k,
        intermediate_dir=intermediate_dir,
        wsi_reader=args.wsi_reader,
        tissue_blur_filter=tissue_blur_filter_cfg,
    )
    print(f"WSI reader (resolved): {resolved_wsi_reader}")

    fallback_reason: Optional[str] = None
    classes_present_output = list(classes_present)
    classes_ranked_output = list(classes_ranked)

    # Build flat candidate manifest (global id space for all tournament rounds)
    candidate_manifest: List[Dict[str, Any]] = []
    candidate_lookup: Dict[int, Dict[str, Any]] = {}
    for class_name in classes_ranked:
        for local_idx, candidate in enumerate(candidates_by_class.get(class_name, [])):
            global_idx = len(candidate_manifest)
            candidate_manifest.append({
                "global_index": global_idx,
                "class_name": class_name,
                "class_local_index": local_idx,
                "candidate_name": candidate.get("name"),
                "source": candidate.get("source"),
                "point_l0": list(candidate.get("point_l0")) if candidate.get("point_l0") else None,
                "point_normalized": candidate.get("point_normalized"),
                "point_normalized_rotated": candidate.get("point_normalized_rotated"),
                "vlm_index": candidate.get("vlm_index"),
            })
            candidate_lookup[global_idx] = {
                "source_class": class_name,
                "class_local_index": local_idx,
                "candidate": candidate,
            }

    manifest_path = intermediate_dir / "candidate_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(candidate_manifest, f, indent=2)

    # === STAGE 2: MULTI-CLASS RANKING ===
    print("\n[Stage 2] Multi-class Ranking")
    print("-" * 40)

    ranker = get_ranker(
        "vlm",
        backend=args.vlm_backend,
        model=args.vlm_model,
        port=args.vlm_port,
        max_tokens=args.vlm_max_tokens,
        max_retries=args.vlm_max_retries,
        vlm_image_size=effective_vlm_image_size,
        openrouter_reasoning_effort=args.reasoning_effort,
        gemini_use_vertex=args.gemini_use_vertex,
        gemini_credentials=args.gemini_credentials,
        gemini_location=args.gemini_location,
        gemini_thinking_level=args.gemini_thinking_level,
        gemini_include_thoughts=args.gemini_include_thoughts,
    )

    class_descriptions = None
    if args.use_descriptors:
        class_descriptions = stage4_input.get("visual_descriptors") or {}
        # Fill in fallbacks if missing
        for class_name in classes_ranked:
            if class_name not in class_descriptions and class_name in DEFAULT_CLASS_DESCRIPTIONS:
                class_descriptions[class_name] = DEFAULT_CLASS_DESCRIPTIONS[class_name]

    selected_indices_by_class: Dict[str, List[int]] = {c: [] for c in classes_ranked}
    reasoning_by_class: Dict[str, str] = {c: "" for c in classes_ranked}
    ranker_name = "vlm"
    ranker_config: Dict[str, Any] = ranker.get_config()

    total_vlm_calls = 0
    pool_candidate_ids = [entry["global_index"] for entry in candidate_manifest]
    reduction_rounds: List[Dict[str, Any]] = []
    final_manifest_path: Optional[Path] = None

    if selection_mode_resolved == "single_pass":
        ranker_config["mode"] = "single_pass"
        print(f"Single-pass ranking on {len(pool_candidate_ids)} candidate(s)")

        selections = ranker.rank_multiclass(
            patches_by_class={
                class_name: (
                    [c["image"] for c in candidates_by_class.get(class_name, [])],
                    [c["name"] for c in candidates_by_class.get(class_name, [])],
                )
                for class_name in classes_ranked
            },
            k_per_class=k_per_class,
            class_descriptions=class_descriptions,
            icl_dir=output_dir,
        )

        model_call_expected = any(
            len(candidates_by_class.get(class_name, [])) > k_per_class.get(class_name, 0)
            for class_name in classes_ranked
        )
        total_vlm_calls = 1 if (candidate_manifest and model_call_expected) else 0

        for class_name in classes_ranked:
            raw_indices = selections.get(class_name, ([], ""))[0]
            mapped = [
                idx
                for idx in raw_indices
                if isinstance(idx, int) and 0 <= idx < len(candidate_manifest)
            ]
            selected_indices_by_class[class_name] = _dedupe_preserve_order(mapped)
            reasoning_by_class[class_name] = selections.get(class_name, ([], ""))[1]

        if hasattr(ranker, 'get_raw_response'):
            raw_response = ranker.get_raw_response()
            if raw_response:
                with open(intermediate_dir / "vlm_multiclass_ranking_response.txt", 'w') as f:
                    f.write(raw_response)

        tournament_info: Dict[str, Any] = {
            "enabled": False,
            "selection_mode": "single_pass",
            "max_total_candidates": args.max_total_candidates,
            "round1_k_override": args.tournament_round1_k,
            "initial_pool_size": len(candidate_manifest),
            "reduction_rounds": [],
            "final_pool_size": len(candidate_manifest),
            "final_round_manifest": None,
            "total_vlm_calls": total_vlm_calls,
        }
    else:
        ranker_config["mode"] = "tournament_overlapping_windows"
        tournament_dir = intermediate_dir / "tournament"
        tournament_dir.mkdir(parents=True, exist_ok=True)

        # Reduction rounds until finalists fit in one VLM call.
        while len(pool_candidate_ids) > args.max_total_candidates:
            round_idx = len(reduction_rounds) + 1
            round_k_per_class = _compute_reduction_k_per_class(
                final_k_per_class=k_per_class,
                pool_size=len(pool_candidate_ids),
                max_total_candidates=args.max_total_candidates,
            )
            if round_idx == 1 and args.tournament_round1_k is not None:
                round_k_per_class = {
                    class_name: (
                        min(k_per_class.get(class_name, 0), args.tournament_round1_k)
                        if k_per_class.get(class_name, 0) > 0
                        else 0
                    )
                    for class_name in classes_ranked
                }

            windows, boundary_overlaps = compute_tournament_windows(
                total_points=len(pool_candidate_ids),
                max_total=args.max_total_candidates,
            )
            round_dir = tournament_dir / f"round_{round_idx:02d}"
            round_dir.mkdir(parents=True, exist_ok=True)

            print(
                f"Reduction round {round_idx}: pool={len(pool_candidate_ids)}, "
                f"calls={len(windows)}, k_per_class={round_k_per_class}"
            )

            round_selected_union: set = set()
            call_summaries: List[Dict[str, Any]] = []

            for call_idx, (start, end) in enumerate(windows):
                call_candidate_ids = pool_candidate_ids[start:end]
                call_patches_by_class, call_manifest = _build_call_inputs(
                    pool_candidate_ids=call_candidate_ids,
                    classes_ranked=classes_ranked,
                    candidate_lookup=candidate_lookup,
                )
                call_dir = round_dir / f"call_{call_idx:02d}"
                call_dir.mkdir(parents=True, exist_ok=True)

                selections = ranker.rank_multiclass(
                    patches_by_class=call_patches_by_class,
                    k_per_class=round_k_per_class,
                    class_descriptions=class_descriptions,
                    icl_dir=call_dir,
                )
                total_vlm_calls += 1

                call_selected_counts: Dict[str, int] = {}
                call_selected_ids_by_class: Dict[str, List[int]] = {}
                for class_name in classes_ranked:
                    raw_indices = selections.get(class_name, ([], ""))[0]
                    mapped = [
                        call_manifest[idx]
                        for idx in raw_indices
                        if isinstance(idx, int) and 0 <= idx < len(call_manifest)
                    ]
                    mapped = _dedupe_preserve_order(mapped)
                    call_selected_ids_by_class[class_name] = mapped
                    call_selected_counts[class_name] = len(mapped)
                    round_selected_union.update(mapped)

                    # Keep latest reasoning for metadata visibility.
                    reasoning_by_class[class_name] = selections.get(class_name, ([], ""))[1]

                if hasattr(ranker, 'get_raw_response'):
                    raw_response = ranker.get_raw_response()
                    if raw_response:
                        with open(call_dir / "vlm_multiclass_ranking_response.txt", 'w') as f:
                            f.write(raw_response)

                call_summaries.append({
                    "call_index": call_idx,
                    "window_start": start,
                    "window_end": end,
                    "window_size": end - start,
                    "selected_counts": call_selected_counts,
                    "selected_ids_by_class": call_selected_ids_by_class,
                })

            next_pool_candidate_ids = [
                candidate_id
                for candidate_id in pool_candidate_ids
                if candidate_id in round_selected_union
            ]

            round_summary: Dict[str, Any] = {
                "round_index": round_idx,
                "pool_in": len(pool_candidate_ids),
                "pool_out": len(next_pool_candidate_ids),
                "k_per_class": round_k_per_class,
                "n_calls": len(windows),
                "windows": [{"start": s, "end": e} for s, e in windows],
                "boundary_overlaps": boundary_overlaps,
                "calls": call_summaries,
            }

            if not next_pool_candidate_ids and pool_candidate_ids:
                round_summary["warning"] = "no_selections_in_round"
                reduction_rounds.append(round_summary)
                pool_candidate_ids = []
                break

            # Safety valve: ensure progress even if model repeatedly returns broad selections.
            if (
                len(next_pool_candidate_ids) >= len(pool_candidate_ids)
                and len(next_pool_candidate_ids) > args.max_total_candidates
            ):
                next_pool_candidate_ids = next_pool_candidate_ids[: args.max_total_candidates]
                round_summary["forced_truncation_to_max_total"] = True

            reduction_rounds.append(round_summary)
            pool_candidate_ids = next_pool_candidate_ids

        # Final call on reduced pool writes actual Stage 5 outputs.
        if pool_candidate_ids:
            final_patches_by_class, final_call_manifest = _build_call_inputs(
                pool_candidate_ids=pool_candidate_ids,
                classes_ranked=classes_ranked,
                candidate_lookup=candidate_lookup,
            )
            final_manifest = [
                {
                    "final_local_index": local_idx,
                    "global_index": candidate_id,
                }
                for local_idx, candidate_id in enumerate(final_call_manifest)
            ]
            final_manifest_path = intermediate_dir / "final_round_manifest.json"
            with open(final_manifest_path, 'w') as f:
                json.dump(final_manifest, f, indent=2)

            selections = ranker.rank_multiclass(
                patches_by_class=final_patches_by_class,
                k_per_class=k_per_class,
                class_descriptions=class_descriptions,
                icl_dir=output_dir,
            )
            total_vlm_calls += 1

            for class_name in classes_ranked:
                raw_indices = selections.get(class_name, ([], ""))[0]
                mapped = [
                    final_call_manifest[idx]
                    for idx in raw_indices
                    if isinstance(idx, int) and 0 <= idx < len(final_call_manifest)
                ]
                selected_indices_by_class[class_name] = _dedupe_preserve_order(mapped)
                reasoning_by_class[class_name] = selections.get(class_name, ([], ""))[1]

            # Save final raw VLM response if available.
            if hasattr(ranker, 'get_raw_response'):
                raw_response = ranker.get_raw_response()
                if raw_response:
                    with open(intermediate_dir / "vlm_multiclass_ranking_response.txt", 'w') as f:
                        f.write(raw_response)
        else:
            for class_name in classes_ranked:
                selected_indices_by_class[class_name] = []
                reasoning_by_class[class_name] = "No candidates available after tournament reduction."

        tournament_info = {
            "enabled": True,
            "selection_mode": "tournament",
            "max_total_candidates": args.max_total_candidates,
            "round1_k_override": args.tournament_round1_k,
            "initial_pool_size": len(candidate_manifest),
            "reduction_rounds": reduction_rounds,
            "final_pool_size": len(pool_candidate_ids),
            "final_round_manifest": str(final_manifest_path) if final_manifest_path else None,
            "total_vlm_calls": total_vlm_calls,
        }

    # === STAGE 3: SAVE METADATA ===
    print("\n[Stage 3] Saving Metadata")
    print("-" * 40)

    manifest_by_index = {entry["global_index"]: entry for entry in candidate_manifest}

    selected_tissue_indices_pre = list(selected_indices_by_class.get("tissue", []))
    selected_tissue_indices_post = list(selected_tissue_indices_pre)
    dropped_tissue_indices: List[int] = []
    tissue_selected_blur_rows: List[Dict[str, Any]] = []
    tissue_selected_blur_csv: Optional[Path] = None

    if not args.disable_tissue_blur_filter and selected_tissue_indices_pre:
        for global_idx in selected_tissue_indices_pre:
            entry = manifest_by_index.get(global_idx)
            if not entry:
                continue
            source_class = entry.get("class_name")
            local_idx = entry.get("class_local_index")
            if (
                not isinstance(source_class, str)
                or source_class not in candidates_by_class
                or not isinstance(local_idx, int)
                or local_idx < 0
                or local_idx >= len(candidates_by_class[source_class])
            ):
                continue
            candidate = candidates_by_class[source_class][local_idx]
            patch_rgb = np.asarray(candidate["image"])
            blur_result = compute_blur_from_patch(
                patch_rgb=patch_rgb,
                sigma=args.tissue_blur_sigma,
                pixel_threshold=args.tissue_blur_pixel_threshold,
            )
            blur_score = float(getattr(blur_result, "blur_score", np.nan))
            sharp_score = float(getattr(blur_result, "sharp_score", np.nan))
            passed = blur_score <= args.tissue_blur_threshold
            tissue_selected_blur_rows.append({
                "selected_class": "tissue",
                "global_index": int(global_idx),
                "source_class": source_class,
                "source_local_index": int(local_idx),
                "candidate_name": entry.get("candidate_name"),
                "source": entry.get("source"),
                "blur_score": blur_score,
                "sharp_score": sharp_score,
                "pass_blur_filter": int(passed),
            })
            if not passed:
                dropped_tissue_indices.append(global_idx)

        if tissue_selected_blur_rows:
            tissue_selected_blur_csv = intermediate_dir / "tissue_selected_blur_audit.csv"
            with tissue_selected_blur_csv.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "selected_class",
                        "global_index",
                        "source_class",
                        "source_local_index",
                        "candidate_name",
                        "source",
                        "blur_score",
                        "sharp_score",
                        "pass_blur_filter",
                    ],
                )
                writer.writeheader()
                writer.writerows(tissue_selected_blur_rows)

        if dropped_tissue_indices:
            dropped_set = set(dropped_tissue_indices)
            selected_tissue_indices_post = [
                idx for idx in selected_tissue_indices_pre if idx not in dropped_set
            ]
            tissue_dir = output_dir / "tissue"
            for idx in dropped_set:
                entry = manifest_by_index.get(idx)
                if not entry:
                    continue
                name = entry.get("candidate_name")
                if not isinstance(name, str):
                    continue
                path = tissue_dir / name
                if path.exists():
                    path.unlink()
            print(
                "Post-selection tissue blur filter removed "
                f"{len(dropped_set)} / {len(selected_tissue_indices_pre)} tissue selection(s)."
            )

    selected_indices_by_class["tissue"] = selected_tissue_indices_post

    if len(selected_tissue_indices_post) <= 0:
        if selected_tissue_indices_pre and not args.disable_tissue_blur_filter:
            fallback_reason = "stage5_no_tissue_after_blur"
        else:
            fallback_reason = "stage5_no_tissue_selected"
        print(
            "Stage 5 fallback triggered after ranking: no valid tissue selections remain "
            f"(reason={fallback_reason})."
        )

    tissue_selected_blur_info: Dict[str, Any] = {
        "enabled": not args.disable_tissue_blur_filter,
        "sigma": args.tissue_blur_sigma,
        "pixel_threshold": args.tissue_blur_pixel_threshold,
        "patch_threshold": args.tissue_blur_threshold,
        "selected_tissue_indices_pre": selected_tissue_indices_pre,
        "selected_tissue_indices_post": selected_tissue_indices_post,
        "dropped_tissue_indices": dropped_tissue_indices,
        "audit_csv": str(tissue_selected_blur_csv) if tissue_selected_blur_csv else None,
    }

    if fallback_reason:
        present_set = {c for c in classes_present if c in ICL_CLASSES}
        present_set.update({"background", "tissue"})
        classes_present_output = [c for c in ICL_CLASSES if c in present_set]
        classes_ranked_output = list(classes_present_output)
        selected_indices_by_class = {class_name: [] for class_name in classes_ranked_output}
        reasoning_by_class = {
            class_name: (
                "Stage 5 fallback: no valid tissue ICL examples remain after reranking "
                "and blur gating; disabling ICL for Stage 6."
            )
            for class_name in classes_ranked_output
        }
        ranker_config = dict(ranker_config)
        ranker_config["mode"] = "stage5_disable_icl_fallback"

    output_paths = {}
    for class_name in classes_ranked_output:
        selected = selected_indices_by_class.get(class_name, [])
        paths = []
        for idx in selected:
            entry = manifest_by_index.get(idx)
            if entry and entry.get("candidate_name"):
                paths.append(f"{class_name}/{entry['candidate_name']}")
        output_paths[class_name] = paths

    output_k_per_class = (
        {class_name: 0 for class_name in classes_ranked_output}
        if fallback_reason
        else k_per_class
    )
    metadata = {
        "stage4_input": str(stage4_dir),
        "stage2_input": str(stage4_input.get("stage2_dir")) if stage4_input.get("stage2_dir") else None,
        "wsi_path": os.path.abspath(wsi_path),
        "wsi_reader_requested": args.wsi_reader,
        "wsi_reader": resolved_wsi_reader,
        "bbox": list(bbox),
        "classes_present": classes_present_output,
        "classes_ranked": classes_ranked_output,
        "classes_skipped": skipped_classes,
        "k_per_class": output_k_per_class,
        "patch_extraction": {
            "k": args.k,
            "patch_size": args.patch_size,
            "level": 0,
            "candidates_per_class": {c: len(candidates_by_class.get(c, [])) for c in classes_ranked_output},
            "candidate_manifest": str(manifest_path),
            "sampling": sampling_info if sampling_info else None,
            "tissue_blur_filter": blur_filter_info if blur_filter_info.get("enabled") else None,
        },
        "tissue_selection_blur": tissue_selected_blur_info,
        "ranking": {
            "ranker": ranker_name,
            "config": ranker_config,
            "selected_indices": selected_indices_by_class,
            "index_space": "candidate_manifest_global_index",
            "reasoning": reasoning_by_class,
            "tournament": tournament_info,
        },
        "output": output_paths,
        "reproducibility": {
            "git_hash": state_info.get("git_hash", "unknown"),
            "created_at": datetime.now().isoformat(),
            "bypassed": state_info.get("bypassed", False),
            "run_config": run_config,
        },
    }
    if fallback_reason:
        metadata["fallback"] = {
            "used": True,
            "reason": fallback_reason,
            "disable_icl": True,
            "selected_tissue_count_pre_blur": len(selected_tissue_indices_pre),
            "selected_tissue_count_post_blur": len(selected_tissue_indices_post),
            "created_by": "reranker.py",
            "created_at": datetime.now().isoformat(),
        }

    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved: {output_dir / 'metadata.json'}")

    reproduce_path = output_dir / "reproduce.txt"
    create_reproduce_command(parser, str(reproduce_path), git_hash=state_info.get("git_hash"))
    print(f"Saved: {reproduce_path}")

    # === SUMMARY ===
    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"Output directory: {output_dir.absolute()}")
    for class_name, indices in selected_indices_by_class.items():
        print(f"{class_name}: {len(indices)} patches selected")
    print("=" * 60)


def main():
    parser = create_parser()
    args = parser.parse_args()
    if args.max_candidates_per_class is not None and args.max_candidates_per_class < 1:
        print("Error: --max-candidates-per-class must be >= 1", file=sys.stderr)
        sys.exit(1)
    if args.max_total_candidates < 1:
        print("Error: --max-total-candidates must be >= 1", file=sys.stderr)
        sys.exit(1)
    if args.tournament_round1_k is not None and args.tournament_round1_k < 1:
        print("Error: --tournament-round1-k must be >= 1", file=sys.stderr)
        sys.exit(1)

    # Resolve stage4 input directories
    if args.stage4_dir and (args.case or args.model):
        print("Error: Use either --stage4-dir OR (--case and --model), not both.", file=sys.stderr)
        sys.exit(1)

    if args.stage4_dir:
        stage4_dirs = [Path(args.stage4_dir)]
    else:
        if not (args.case and args.model):
            print("Error: Must provide --stage4-dir or both --case and --model.", file=sys.stderr)
            sys.exit(1)
        stage4_root = Path(args.stage4_root)
        try:
            stage4_dirs = resolve_latest_stage4_dirs(stage4_root, args.case, args.model)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        if not stage4_dirs:
            print(
                f"Error: No stage4 outputs found for case '{args.case}' and model '{args.model}'.",
                file=sys.stderr,
            )
            sys.exit(1)

    if args.output_dir and len(stage4_dirs) > 1:
        print("Error: --output-dir can only be used with a single --stage4-dir.", file=sys.stderr)
        sys.exit(1)

    # Load stage4 inputs (preflight)
    stage4_inputs: List[Tuple[Path, Dict[str, Any]]] = []
    for stage4_dir in stage4_dirs:
        if not stage4_dir.exists():
            print(f"Error: stage4-dir not found: {stage4_dir}", file=sys.stderr)
            sys.exit(1)
        try:
            stage4_input = load_stage4_input(stage4_dir, use_descriptors=args.use_descriptors)
        except Exception as e:
            print(f"Error loading stage4 input from {stage4_dir}: {e}", file=sys.stderr)
            sys.exit(1)
        stage4_inputs.append((stage4_dir, stage4_input))

    # Reproducibility check (once, before outputs)
    wsi_paths = [entry[1]["wsi_path"] for entry in stage4_inputs]
    state_info = require_clean_state(wsi_paths, skip_dvc_check=args.skip_dvc_check)
    if state_info.get("bypassed"):
        print(f"Warning: Reproducibility check bypassed: {state_info.get('reason')}")

    # Run reranker per stage4 dir
    if len(stage4_inputs) > 1:
        print(f"Found {len(stage4_inputs)} stage4 dirs to rerank.")
    for stage4_dir, stage4_input in stage4_inputs:
        _run_rerank_for_stage4_dir(stage4_dir, stage4_input, args, parser, state_info)


if __name__ == '__main__':
    main()
