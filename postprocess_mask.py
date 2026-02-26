#!/usr/bin/env python3
# ABOUTME: Stage 7 post-processing for Stage 6 class maps (run_vlm_bbox_inference outputs).
# ABOUTME: Builds tissue-only mask from class_map.npy, applies morphology, and saves previews/outputs.

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage as ndi

from utils.vlm_utils import normalize_class_label


# -----------------------------------------------------------------------------
# Data models
# -----------------------------------------------------------------------------

@dataclass
class RunData:
    run_dir: Path
    case_id: str
    run_id: str
    class_map: np.ndarray
    metadata: dict
    class_labels: List[str]
    tissue_id: int
    background_id: Optional[int]
    quality_map: Optional[np.ndarray]
    quality_na_id: Optional[int]
    base_image: Image.Image
    base_image_source: str
    base_image_colorized: bool


# -----------------------------------------------------------------------------
# Discovery / loading
# -----------------------------------------------------------------------------

def is_run_dir(path: Path) -> bool:
    return (path / "class_map.npy").is_file() and (path / "metadata.json").is_file()


def discover_run_dirs(inputs: Sequence[Path]) -> List[Path]:
    found: List[Path] = []
    for input_path in inputs:
        if not input_path.exists():
            continue
        if is_run_dir(input_path):
            found.append(input_path)
            continue
        for meta_path in input_path.rglob("metadata.json"):
            run_dir = meta_path.parent
            if is_run_dir(run_dir):
                found.append(run_dir)

    # Deduplicate while preserving deterministic order.
    unique = sorted({p.resolve() for p in found})
    return unique


def select_runs(run_dirs: Sequence[Path], selection: str) -> List[Path]:
    if selection == "all":
        return list(run_dirs)

    # quick_iter: latest run per case (case/model/run layout expected)
    by_case: Dict[str, Path] = {}
    for run_dir in run_dirs:
        # Expected .../{case}/{model}/{run}
        case_id = run_dir.parent.parent.name if run_dir.parent.parent else run_dir.name
        prev = by_case.get(case_id)
        if prev is None or run_dir.name > prev.name:
            by_case[case_id] = run_dir
    return [by_case[k] for k in sorted(by_case.keys())]


def load_class_labels(meta: dict, run_dir: Path) -> List[str]:
    labels = meta.get("class_order") or meta.get("class_labels")
    if labels:
        return [str(x) for x in labels]

    palette_path = run_dir / "class_palette.json"
    if palette_path.exists():
        with palette_path.open("r") as f:
            palette = json.load(f)
        labels = palette.get("class_labels") or []
        if labels:
            return [str(x) for x in labels]

    raise ValueError(f"Could not infer class label ordering for run: {run_dir}")


def resolve_tissue_background_ids(meta: dict, class_labels: Sequence[str]) -> Tuple[int, Optional[int]]:
    normalized = [normalize_class_label(x) for x in class_labels]

    # Prefer explicit mapping when present.
    class_to_id = meta.get("class_to_id")
    if isinstance(class_to_id, dict):
        norm_map = {normalize_class_label(k): int(v) for k, v in class_to_id.items()}
        if "tissue" in norm_map:
            tissue_id = norm_map["tissue"]
            background_id = norm_map.get("background")
            return tissue_id, background_id

    if "tissue" not in normalized:
        raise ValueError("'tissue' class not present in class label ordering")

    tissue_id = normalized.index("tissue")
    background_id = normalized.index("background") if "background" in normalized else None
    return tissue_id, background_id


def resolve_quality_map(run_dir: Path) -> Tuple[Optional[np.ndarray], Optional[int]]:
    quality_path = run_dir / "quality_map.npy"
    if not quality_path.exists():
        return None, None

    quality_map = np.load(quality_path)

    palette_path = run_dir / "class_palette.json"
    if not palette_path.exists():
        return quality_map, None

    with palette_path.open("r") as f:
        palette = json.load(f)
    quality_labels = [str(x) for x in (palette.get("quality_labels") or [])]
    if "NA" in quality_labels:
        return quality_map, quality_labels.index("NA")

    return quality_map, None


def _resolve_ref_path(path_value: str, anchor_dir: Optional[Path] = None) -> Optional[Path]:
    if not path_value:
        return None
    p = Path(path_value)
    candidates = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append(Path.cwd() / p)
        if anchor_dir is not None:
            candidates.append(anchor_dir / p)
    for c in candidates:
        if c.exists():
            return c.resolve()
    return None


def load_base_image(run_dir: Path, metadata: dict) -> Tuple[Image.Image, str, bool]:
    # Preferred path: Stage 6 metadata -> Stage 5 metadata -> Stage 4 raw thumbnail
    stage5_run = metadata.get("stage5_run")
    stage5_path = _resolve_ref_path(stage5_run) if stage5_run else None
    if stage5_path is not None:
        stage5_meta_path = stage5_path / "metadata.json"
        if stage5_meta_path.exists():
            try:
                with stage5_meta_path.open("r") as f:
                    s5_meta = json.load(f)
            except Exception:
                s5_meta = {}

            # 1) Stage 4 region thumbnail (best raw base)
            stage4_input = s5_meta.get("stage4_input")
            stage4_path = _resolve_ref_path(stage4_input, anchor_dir=stage5_path.parent) if stage4_input else None
            if stage4_path is not None:
                p = stage4_path / "region_thumbnail.png"
                if p.exists():
                    return Image.open(p).convert("RGB"), "stage4_region_thumbnail", False

            # 2) Stage 2 bbox region (raw fallback)
            stage2_input = s5_meta.get("stage2_input")
            stage2_path = _resolve_ref_path(stage2_input, anchor_dir=stage5_path.parent) if stage2_input else None
            if stage2_path is not None:
                p = stage2_path / "bbox_region.png"
                if p.exists():
                    return Image.open(p).convert("RGB"), "stage2_bbox_region", False

    # Final fallback: Stage 6 overlays (already colorized).
    candidates = [
        (run_dir / "class_overlay.png", "stage6_class_overlay", True),
        (run_dir / "bbox_grid_overlay.png", "stage6_bbox_grid_overlay", True),
        (run_dir / "overlay.png", "stage6_overlay", True),
    ]
    for path, source, is_colorized in candidates:
        if path.exists():
            return Image.open(path).convert("RGB"), source, is_colorized

    raise ValueError(f"No usable base image found for {run_dir}")


def load_run(run_dir: Path) -> RunData:
    with (run_dir / "metadata.json").open("r") as f:
        metadata = json.load(f)

    class_map = np.load(run_dir / "class_map.npy")
    class_labels = load_class_labels(metadata, run_dir)
    tissue_id, background_id = resolve_tissue_background_ids(metadata, class_labels)
    quality_map, quality_na_id = resolve_quality_map(run_dir)
    base_image, base_image_source, base_image_colorized = load_base_image(run_dir, metadata)

    case_id = run_dir.parent.parent.name
    run_id = run_dir.name

    return RunData(
        run_dir=run_dir,
        case_id=case_id,
        run_id=run_id,
        class_map=class_map,
        metadata=metadata,
        class_labels=class_labels,
        tissue_id=tissue_id,
        background_id=background_id,
        quality_map=quality_map,
        quality_na_id=quality_na_id,
        base_image=base_image,
        base_image_source=base_image_source,
        base_image_colorized=base_image_colorized,
    )


# -----------------------------------------------------------------------------
# Morphology helpers
# -----------------------------------------------------------------------------

def connectivity_to_rank(connectivity: int) -> int:
    return 1 if connectivity == 4 else 2


def count_components(mask: np.ndarray, connectivity: int) -> int:
    structure = ndi.generate_binary_structure(2, connectivity_to_rank(connectivity))
    _, n = ndi.label(mask, structure=structure)
    return int(n)


def remove_small_components(mask: np.ndarray, min_size: int, connectivity: int) -> Tuple[np.ndarray, int, int]:
    if min_size <= 1:
        return mask.copy(), 0, 0

    structure = ndi.generate_binary_structure(2, connectivity_to_rank(connectivity))
    labeled, n = ndi.label(mask, structure=structure)
    if n == 0:
        return mask.copy(), 0, 0

    sizes = np.bincount(labeled.ravel())
    # label 0 is background
    remove_ids = np.where((sizes < min_size) & (np.arange(sizes.size) != 0))[0]
    if remove_ids.size == 0:
        return mask.copy(), 0, 0

    out = mask.copy()
    remove_mask = np.isin(labeled, remove_ids)
    removed_pixels = int(remove_mask.sum())
    out[remove_mask] = False
    return out, int(remove_ids.size), removed_pixels


def build_close_structure(kernel: int) -> np.ndarray:
    k = max(1, int(kernel))
    if k % 2 == 0:
        k += 1
    return np.ones((k, k), dtype=bool)


def apply_closing(mask: np.ndarray, kernel: int) -> np.ndarray:
    structure = build_close_structure(kernel)
    # Preserve edge-connected tissue for tight bboxes by padding with foreground.
    # Without this, closing treats outside-image as background and can erode border tissue.
    pad = structure.shape[0] // 2
    if pad <= 0:
        return ndi.binary_closing(mask, structure=structure)
    padded = np.pad(mask, pad_width=pad, mode="constant", constant_values=True)
    closed = ndi.binary_closing(padded, structure=structure)
    return closed[pad:-pad, pad:-pad]


def apply_binary_fill_holes(mask: np.ndarray, connectivity: int) -> np.ndarray:
    structure = ndi.generate_binary_structure(2, connectivity_to_rank(connectivity))
    return ndi.binary_fill_holes(mask, structure=structure)


def fill_small_holes(mask: np.ndarray, max_hole_size: int, connectivity: int) -> np.ndarray:
    """Fill enclosed background components up to max_hole_size cells."""
    if max_hole_size <= 0:
        return apply_binary_fill_holes(mask, connectivity=connectivity)

    bg = ~mask.astype(bool)
    structure = ndi.generate_binary_structure(2, connectivity_to_rank(connectivity))
    labeled, num = ndi.label(bg, structure=structure)
    if num == 0:
        return mask.astype(bool)

    sizes = np.bincount(labeled.ravel())
    hole_ids = np.arange(sizes.size)

    border_ids = np.unique(
        np.concatenate((labeled[0, :], labeled[-1, :], labeled[:, 0], labeled[:, -1]))
    )
    border_ids = border_ids[border_ids != 0]

    fill_ids = hole_ids[(hole_ids != 0) & (sizes <= int(max_hole_size))]
    if border_ids.size > 0 and fill_ids.size > 0:
        fill_ids = fill_ids[~np.isin(fill_ids, border_ids)]
    if fill_ids.size == 0:
        return mask.astype(bool)

    out = mask.astype(bool).copy()
    out[np.isin(labeled, fill_ids)] = True
    return out


# -----------------------------------------------------------------------------
# Overlay rendering
# -----------------------------------------------------------------------------

def _cell_edges(length: int, cells: int) -> np.ndarray:
    # Fallback only (when bbox metadata is missing).
    return np.linspace(0, length, cells + 1, dtype=int)


def _int_bbox(d: dict) -> Optional[Dict[str, int]]:
    if not isinstance(d, dict):
        return None
    keys = ("x1", "y1", "x2", "y2")
    if not all(k in d for k in keys):
        return None
    try:
        x1, y1, x2, y2 = int(d["x1"]), int(d["y1"]), int(d["x2"]), int(d["y2"])
    except Exception:
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def _resolve_overlay_geometry(metadata: dict) -> Optional[Tuple[int, Dict[str, int], List[Dict[str, int]]]]:
    patch_size = metadata.get("patch_size_level0", None)
    if patch_size is None:
        patch_size = metadata.get("patch_size", None)
    try:
        patch_size = int(patch_size) if patch_size is not None else 0
    except Exception:
        patch_size = 0
    if patch_size <= 0:
        return None

    union_bbox = _int_bbox(metadata.get("union_bbox_level0", {}))
    if union_bbox is None:
        union_bbox = _int_bbox(metadata.get("bbox_level0", {}))
    if union_bbox is None:
        return None

    raw_bboxes = metadata.get("bboxes_level0")
    bboxes: List[Dict[str, int]] = []
    if isinstance(raw_bboxes, list):
        for b in raw_bboxes:
            bb = _int_bbox(b)
            if bb is not None:
                bboxes.append(bb)

    if not bboxes:
        single_bbox = _int_bbox(metadata.get("bbox_level0", {}))
        bboxes = [single_bbox] if single_bbox is not None else [union_bbox]

    return patch_size, union_bbox, bboxes


def overlay_mask(
    base_img: Image.Image,
    mask: np.ndarray,
    metadata: dict,
    color: Tuple[int, int, int],
    alpha: int,
    grayscale_base: bool = True,
) -> Image.Image:
    base_rgb = base_img.convert("RGB")
    if grayscale_base:
        base_rgb = ImageOps.grayscale(base_rgb).convert("RGB")
    base = base_rgb.convert("RGBA")
    arr = np.array(base)
    h, w = arr.shape[0], arr.shape[1]

    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    geom = _resolve_overlay_geometry(metadata)
    if geom is not None:
        patch_size_l0, union_bbox, bboxes_l0 = geom
        union_x1, union_y1 = union_bbox["x1"], union_bbox["y1"]
        union_w = union_bbox["x2"] - union_bbox["x1"]
        union_h = union_bbox["y2"] - union_bbox["y1"]
        rows, cols = mask.shape
        scale_x = w / union_w if union_w > 0 else 1.0
        scale_y = h / union_h if union_h > 0 else 1.0

        for bbox in bboxes_l0:
            bbox_x1, bbox_y1 = bbox["x1"], bbox["y1"]
            bbox_x2, bbox_y2 = bbox["x2"], bbox["y2"]
            col_start = (bbox_x1 - union_x1) // patch_size_l0
            row_start = (bbox_y1 - union_y1) // patch_size_l0
            col_end = int(np.ceil((bbox_x2 - union_x1) / patch_size_l0))
            row_end = int(np.ceil((bbox_y2 - union_y1) / patch_size_l0))

            for r in range(row_start, row_end):
                for c in range(col_start, col_end):
                    if r < 0 or c < 0 or r >= rows or c >= cols or not mask[r, c]:
                        continue
                    wsi_x = bbox_x1 + (c - col_start) * patch_size_l0
                    wsi_y = bbox_y1 + (r - row_start) * patch_size_l0
                    offset_x = wsi_x - union_x1
                    offset_y = wsi_y - union_y1
                    cell_x2_l0 = min(offset_x + patch_size_l0, union_w)
                    cell_y2_l0 = min(offset_y + patch_size_l0, union_h)

                    x1 = max(0, int(offset_x * scale_x))
                    y1 = max(0, int(offset_y * scale_y))
                    x2 = min(w, int(cell_x2_l0 * scale_x))
                    y2 = min(h, int(cell_y2_l0 * scale_y))
                    if x2 > x1 and y2 > y1:
                        overlay[y1:y2, x1:x2] = [color[0], color[1], color[2], alpha]
    else:
        rows, cols = mask.shape
        y_edges = _cell_edges(h, rows)
        x_edges = _cell_edges(w, cols)
        for r in range(rows):
            y1, y2 = y_edges[r], y_edges[r + 1]
            if y2 <= y1:
                continue
            for c in range(cols):
                if not mask[r, c]:
                    continue
                x1, x2 = x_edges[c], x_edges[c + 1]
                if x2 <= x1:
                    continue
                overlay[y1:y2, x1:x2] = [color[0], color[1], color[2], alpha]

    out = Image.alpha_composite(base, Image.fromarray(overlay, mode="RGBA"))
    return out.convert("RGB")


def overlay_diff(base_img: Image.Image, added: np.ndarray, removed: np.ndarray, metadata: dict) -> Image.Image:
    base_rgb = ImageOps.grayscale(base_img.convert("RGB")).convert("RGB")
    base = base_rgb.convert("RGBA")
    arr = np.array(base)
    h, w = arr.shape[0], arr.shape[1]

    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    geom = _resolve_overlay_geometry(metadata)
    if geom is not None:
        patch_size_l0, union_bbox, bboxes_l0 = geom
        union_x1, union_y1 = union_bbox["x1"], union_bbox["y1"]
        union_w = union_bbox["x2"] - union_bbox["x1"]
        union_h = union_bbox["y2"] - union_bbox["y1"]
        rows, cols = added.shape
        scale_x = w / union_w if union_w > 0 else 1.0
        scale_y = h / union_h if union_h > 0 else 1.0

        for bbox in bboxes_l0:
            bbox_x1, bbox_y1 = bbox["x1"], bbox["y1"]
            bbox_x2, bbox_y2 = bbox["x2"], bbox["y2"]
            col_start = (bbox_x1 - union_x1) // patch_size_l0
            row_start = (bbox_y1 - union_y1) // patch_size_l0
            col_end = int(np.ceil((bbox_x2 - union_x1) / patch_size_l0))
            row_end = int(np.ceil((bbox_y2 - union_y1) / patch_size_l0))

            for r in range(row_start, row_end):
                for c in range(col_start, col_end):
                    if r < 0 or c < 0 or r >= rows or c >= cols:
                        continue
                    if not (added[r, c] or removed[r, c]):
                        continue
                    wsi_x = bbox_x1 + (c - col_start) * patch_size_l0
                    wsi_y = bbox_y1 + (r - row_start) * patch_size_l0
                    offset_x = wsi_x - union_x1
                    offset_y = wsi_y - union_y1
                    cell_x2_l0 = min(offset_x + patch_size_l0, union_w)
                    cell_y2_l0 = min(offset_y + patch_size_l0, union_h)

                    x1 = max(0, int(offset_x * scale_x))
                    y1 = max(0, int(offset_y * scale_y))
                    x2 = min(w, int(cell_x2_l0 * scale_x))
                    y2 = min(h, int(cell_y2_l0 * scale_y))
                    if x2 <= x1 or y2 <= y1:
                        continue
                    if added[r, c]:
                        overlay[y1:y2, x1:x2] = [0, 200, 0, 125]
                    elif removed[r, c]:
                        overlay[y1:y2, x1:x2] = [220, 30, 30, 125]
    else:
        rows, cols = added.shape
        y_edges = _cell_edges(h, rows)
        x_edges = _cell_edges(w, cols)
        for r in range(rows):
            y1, y2 = y_edges[r], y_edges[r + 1]
            if y2 <= y1:
                continue
            for c in range(cols):
                x1, x2 = x_edges[c], x_edges[c + 1]
                if x2 <= x1:
                    continue
                if added[r, c]:
                    overlay[y1:y2, x1:x2] = [0, 200, 0, 125]
                elif removed[r, c]:
                    overlay[y1:y2, x1:x2] = [220, 30, 30, 125]

    out = Image.alpha_composite(base, Image.fromarray(overlay, mode="RGBA"))
    return out.convert("RGB")


# -----------------------------------------------------------------------------
# Core processing
# -----------------------------------------------------------------------------

def process_run(
    run: RunData,
    min_component_size: int,
    connectivity: int,
    close_kernel: int,
    skip_remove_small: bool,
    skip_close: bool,
    skip_fill_holes: bool,
    protect_non_bg_tissue: bool,
) -> dict:
    max_hole_size = 10
    class_map = run.class_map
    tissue_mask_before = class_map == run.tissue_id

    if run.background_id is not None and protect_non_bg_tissue:
        editable = (class_map == run.tissue_id) | (class_map == run.background_id)
    elif run.background_id is not None:
        editable = np.ones_like(tissue_mask_before, dtype=bool)
    else:
        editable = class_map == run.tissue_id

    mask = tissue_mask_before.copy()
    removed_components = 0
    removed_pixels_small = 0

    if not skip_remove_small:
        candidate, removed_components, removed_pixels_small = remove_small_components(
            mask, min_size=min_component_size, connectivity=connectivity
        )
        mask = np.where(editable, candidate, mask)

    if not skip_close:
        candidate = apply_closing(mask, kernel=close_kernel)
        mask = np.where(editable, candidate, mask)

    if not skip_fill_holes:
        candidate = fill_small_holes(mask, max_hole_size=max_hole_size, connectivity=connectivity)
        mask = np.where(editable, candidate, mask)

    tissue_mask_after = mask.astype(bool)

    added = (~tissue_mask_before) & tissue_mask_after
    removed = tissue_mask_before & (~tissue_mask_after)

    class_map_post = class_map.copy()
    if run.background_id is not None:
        class_map_post[editable & tissue_mask_after] = run.tissue_id
        class_map_post[editable & (~tissue_mask_after)] = run.background_id
    else:
        class_map_post[editable & tissue_mask_after] = run.tissue_id

    quality_map_post = None
    if run.quality_map is not None:
        quality_map_post = run.quality_map.copy()
        if run.quality_na_id is not None:
            quality_map_post[class_map_post != run.tissue_id] = run.quality_na_id
            quality_map_post[added] = run.quality_na_id

    cc_before = count_components(tissue_mask_before, connectivity)
    cc_after = count_components(tissue_mask_after, connectivity)

    result = {
        "case_id": run.case_id,
        "run_id": run.run_id,
        "run_dir": str(run.run_dir),
        "shape_rows": int(class_map.shape[0]),
        "shape_cols": int(class_map.shape[1]),
        "class_labels": run.class_labels,
        "tissue_id": int(run.tissue_id),
        "background_id": None if run.background_id is None else int(run.background_id),
        "tissue_before": int(tissue_mask_before.sum()),
        "tissue_after": int(tissue_mask_after.sum()),
        "delta_tissue": int(tissue_mask_after.sum() - tissue_mask_before.sum()),
        "components_before": int(cc_before),
        "components_after": int(cc_after),
        "delta_components": int(cc_after - cc_before),
        "added_pixels": int(added.sum()),
        "removed_pixels": int(removed.sum()),
        "removed_components_small": int(removed_components),
        "removed_pixels_small": int(removed_pixels_small),
        "params": {
            "min_component_size": int(min_component_size),
            "connectivity": int(connectivity),
            "close_kernel": int(close_kernel),
            "max_hole_size": int(max_hole_size),
            "skip_remove_small": bool(skip_remove_small),
            "skip_close": bool(skip_close),
            "skip_fill_holes": bool(skip_fill_holes),
            "protect_non_bg_tissue": bool(protect_non_bg_tissue),
        },
        "tissue_mask_before": tissue_mask_before,
        "tissue_mask_after": tissue_mask_after,
        "class_map_post": class_map_post,
        "quality_map_post": quality_map_post,
        "added_mask": added,
        "removed_mask": removed,
        "base_image_source": run.base_image_source,
        "overlay_before": overlay_mask(
            run.base_image,
            tissue_mask_before,
            metadata=run.metadata,
            color=(30, 144, 255),
            alpha=140,
            grayscale_base=run.base_image_colorized,
        ),
        "overlay_after": overlay_mask(
            run.base_image,
            tissue_mask_after,
            metadata=run.metadata,
            color=(255, 140, 0),
            alpha=140,
            grayscale_base=run.base_image_colorized,
        ),
        "overlay_diff": overlay_diff(run.base_image, added=added, removed=removed, metadata=run.metadata),
    }
    return result


# -----------------------------------------------------------------------------
# Output writers
# -----------------------------------------------------------------------------

def write_preview_bundle(
    out_dir: Path,
    results: Sequence[dict],
    argv: Sequence[str],
    selection: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = out_dir / "postprocess_preview.pdf"
    csv_path = out_dir / "summary.csv"
    meta_path = out_dir / "preview_meta.json"

    with PdfPages(pdf_path) as pdf:
        for res in results:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            axes[0].imshow(np.asarray(res["overlay_before"]))
            axes[0].set_title(f"Before (tissue={res['tissue_before']})")
            axes[0].axis("off")

            axes[1].imshow(np.asarray(res["overlay_after"]))
            axes[1].set_title(f"After (tissue={res['tissue_after']})")
            axes[1].axis("off")

            axes[2].imshow(np.asarray(res["overlay_diff"]))
            axes[2].set_title("Diff (green=added, red=removed)")
            axes[2].axis("off")

            title = f"{res['case_id']} / {res['run_id']}"
            fig.suptitle(title, fontsize=12)

            footer = (
                f"base={res['base_image_source']} | "
                f"tissue: {res['tissue_before']} -> {res['tissue_after']} (delta {res['delta_tissue']:+d}) | "
                f"CC: {res['components_before']} -> {res['components_after']} (delta {res['delta_components']:+d}) | "
                f"removed_small_cc: {res['removed_components_small']} ({res['removed_pixels_small']} px) | "
                f"added: {res['added_pixels']} | removed: {res['removed_pixels']}"
            )
            fig.text(0.02, 0.02, footer, fontsize=9)
            fig.tight_layout(rect=[0, 0.05, 1, 0.95])
            pdf.savefig(fig)
            plt.close(fig)

    fieldnames = [
        "case_id",
        "run_id",
        "run_dir",
        "base_image_source",
        "shape_rows",
        "shape_cols",
        "tissue_id",
        "background_id",
        "tissue_before",
        "tissue_after",
        "delta_tissue",
        "components_before",
        "components_after",
        "delta_components",
        "added_pixels",
        "removed_pixels",
        "removed_components_small",
        "removed_pixels_small",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for res in results:
            writer.writerow({k: res.get(k) for k in fieldnames})

    payload = {
        "created_at": datetime.now().isoformat(),
        "selection": selection,
        "num_runs": len(results),
        "argv": list(argv),
        "files": {
            "pdf": str(pdf_path),
            "csv": str(csv_path),
        },
    }
    with meta_path.open("w") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved preview PDF: {pdf_path}")
    print(f"Saved summary CSV: {csv_path}")
    print(f"Saved preview metadata: {meta_path}")


def write_run_outputs(run_dir: Path, res: dict) -> None:
    out_dir = run_dir / "stage7_postprocess"
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "tissue_mask_post.npy", res["tissue_mask_after"].astype(bool))
    np.save(out_dir / "class_map_postprocessed.npy", res["class_map_post"].astype(np.int16))

    if res["quality_map_post"] is not None:
        np.save(out_dir / "quality_map_postprocessed.npy", res["quality_map_post"].astype(np.int8))

    res["overlay_before"].save(out_dir / "postprocess_before.png")
    res["overlay_after"].save(out_dir / "postprocess_after.png")
    res["overlay_diff"].save(out_dir / "postprocess_diff.png")
    side_by_side = Image.new("RGB", (res["overlay_before"].width * 2, res["overlay_before"].height))
    side_by_side.paste(res["overlay_before"], (0, 0))
    side_by_side.paste(res["overlay_after"], (res["overlay_before"].width, 0))
    side_by_side.save(out_dir / "postprocess_overlay_before_after.png")

    meta = {
        "created_at": datetime.now().isoformat(),
        "run_dir": str(run_dir),
        "case_id": res["case_id"],
        "run_id": res["run_id"],
        "class_labels": res["class_labels"],
        "base_image_source": res["base_image_source"],
        "tissue_id": res["tissue_id"],
        "background_id": res["background_id"],
        "stats": {
            "tissue_before": res["tissue_before"],
            "tissue_after": res["tissue_after"],
            "delta_tissue": res["delta_tissue"],
            "components_before": res["components_before"],
            "components_after": res["components_after"],
            "delta_components": res["delta_components"],
            "added_pixels": res["added_pixels"],
            "removed_pixels": res["removed_pixels"],
            "removed_components_small": res["removed_components_small"],
            "removed_pixels_small": res["removed_pixels_small"],
        },
        "params": res["params"],
        "files": {
            "tissue_mask_post": "tissue_mask_post.npy",
            "class_map_postprocessed": "class_map_postprocessed.npy",
            "quality_map_postprocessed": "quality_map_postprocessed.npy" if res["quality_map_post"] is not None else None,
            "before": "postprocess_before.png",
            "after": "postprocess_after.png",
            "diff": "postprocess_diff.png",
            "before_after": "postprocess_overlay_before_after.png",
        },
    }
    with (out_dir / "postprocess_metadata.json").open("w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved Stage 7 outputs: {out_dir}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 7 post-processing for run_vlm_bbox_inference outputs. "
            "Builds tissue mask from class_map.npy, applies morphology, and outputs previews/files."
        )
    )

    parser.add_argument(
        "inputs",
        nargs="+",
        help=(
            "One or more run directories or roots to scan recursively "
            "(e.g., stage6_output/phaseB/qwen/sem_classes_desc_images)"
        ),
    )
    parser.add_argument(
        "--selection",
        choices=["quick_iter", "all"],
        default="quick_iter",
        help="Run selection policy when roots contain multiple runs (default: quick_iter = latest per case)",
    )

    parser.add_argument(
        "--min-component-size",
        type=int,
        default=3,
        help="Remove connected tissue components smaller than this many cells (default: 3)",
    )
    parser.add_argument(
        "--connectivity",
        type=int,
        choices=[4, 8],
        default=4,
        help="Connectivity for CC and hole operations (4=no diagonals, 8=with diagonals; default: 4)",
    )
    parser.add_argument(
        "--close-kernel",
        type=int,
        default=3,
        help="Odd kernel size for binary closing footprint (full kxk; default: 3)",
    )
    parser.add_argument(
        "--skip-remove-small",
        action="store_true",
        help="Disable small connected-component removal",
    )
    parser.add_argument(
        "--skip-close",
        action="store_true",
        help="Disable binary closing",
    )
    parser.add_argument(
        "--skip-fill-holes",
        action="store_true",
        help="Disable binary_fill_holes",
    )
    parser.add_argument(
        "--allow-artifact-overwrite",
        action="store_true",
        help=(
            "Allow morphology to overwrite non-background classes in class_map updates. "
            "Default behavior protects artifact/other classes."
        ),
    )

    parser.add_argument(
        "--preview",
        action="store_true",
        help="Preview mode: write only PDF/CSV/JSON bundle, do not write per-run npy updates",
    )
    parser.add_argument(
        "--preview-dir",
        default="stage7_preview",
        help="Preview output base directory (default: stage7_preview)",
    )
    parser.add_argument(
        "--preview-name",
        default=None,
        help="Optional fixed preview folder name. Default: timestamp.",
    )

    return parser


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    parser = create_parser()
    args = parser.parse_args()

    input_paths = [Path(p) for p in args.inputs]
    run_dirs = discover_run_dirs(input_paths)
    if not run_dirs:
        raise SystemExit("No valid run directories found (need class_map.npy + metadata.json)")

    selected_run_dirs = select_runs(run_dirs, args.selection)
    print(f"Discovered {len(run_dirs)} run dirs; selected {len(selected_run_dirs)} using '{args.selection}'")

    results: List[dict] = []
    for run_dir in selected_run_dirs:
        run = load_run(run_dir)
        res = process_run(
            run=run,
            min_component_size=max(1, args.min_component_size),
            connectivity=args.connectivity,
            close_kernel=max(1, args.close_kernel),
            skip_remove_small=args.skip_remove_small,
            skip_close=args.skip_close,
            skip_fill_holes=args.skip_fill_holes,
            protect_non_bg_tissue=not args.allow_artifact_overwrite,
        )
        results.append(res)
        print(
            f"[{res['case_id']}] {res['run_id']}: "
            f"tissue {res['tissue_before']}->{res['tissue_after']} "
            f"(delta {res['delta_tissue']:+d}), CC {res['components_before']}->{res['components_after']}"
        )

    if args.preview:
        ts = args.preview_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(args.preview_dir) / ts
        write_preview_bundle(out_dir, results, argv=sys.argv, selection=args.selection)
        return

    for res in results:
        write_run_outputs(Path(res["run_dir"]), res)


if __name__ == "__main__":
    main()
