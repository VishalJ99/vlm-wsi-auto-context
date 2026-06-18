#!/usr/bin/env python3
"""Build stress32 PDFs for qwen/GT sampling-pool bbox overlays."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GT_ROOT = (
    Path("/data2/vj724/wsi-agents")
    / "vlm_gt_seg_comparison_experiment/dataset_thumbnails_harder_jones_leica/leica/jones"
)
DEFAULT_QWEN_ROOT = (
    Path("/data2/vj724/wsi-agents")
    / "new_vlm_gt_preds/leica_hard_jones_evg_qwen_2b_zero_shot"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "runs/stress32_qwen2b_hard_negative_probe_v1/visuals"
DEFAULT_LINEAR_PROBE_PREDICTIONS = REPO_ROOT / "runs/stress32_yolo_dinov3_probe_v1/patch_predictions.csv"
DEFAULT_RUN_ID = "harder_jones_leica_manual"
LANCZOS = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
NEAREST = getattr(getattr(Image, "Resampling", Image), "NEAREST")
POOL_COLORS = {
    "true_positive": (46, 204, 113),
    "hard_negative": (230, 126, 34),
    "easy_negative": (74, 144, 226),
}
POOL_LABELS = {
    "true_positive": "true positives: qwen FG, GT FG",
    "hard_negative": "hard negatives: qwen FG, GT BG",
    "easy_negative": "easy negatives: qwen BG, GT BG",
}


class LinearProbeCell:
    def __init__(self, row: int, col: int, prob_fg: float, pred_fg: int) -> None:
        self.row = row
        self.col = col
        self.prob_fg = prob_fg
        self.pred_fg = pred_fg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-root", type=Path, default=DEFAULT_GT_ROOT)
    parser.add_argument("--qwen-root", type=Path, default=DEFAULT_QWEN_ROOT)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--linear-probe-predictions", type=Path, default=DEFAULT_LINEAR_PROBE_PREDICTIONS)
    parser.add_argument("--pdf-name", default="stress32_qwen_stage1_bbox_verification.pdf")
    parser.add_argument("--pool-pdf-name", default="stress32_qwen_stage1_bbox_pool_sampling_review.pdf")
    parser.add_argument("--case-ids", default="", help="Comma-separated case IDs; default uses all cases with gt_bboxes.json.")
    parser.add_argument("--case-limit", type=int, default=None)
    parser.add_argument("--panel-width", type=int, default=440)
    parser.add_argument("--samples-per-region", type=int, default=3)
    parser.add_argument("--seed", type=int, default=250)
    parser.add_argument("--grid-cell-size-level0", type=int, default=512)
    parser.add_argument("--bbox-source", choices=("qwen-stage1", "gt-bboxes"), default="qwen-stage1")
    parser.add_argument("--crop-source", choices=("auto", "crop-file", "wsi", "thumbnail"), default="auto")
    parser.add_argument("--wsi-crop-max-dim", type=int, default=2048)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def qwen_mask_path(qwen_root: Path, run_id: str, case_id: str) -> tuple[Path | None, str]:
    base = qwen_root / case_id / run_id
    for stage in ("stage7_new", "stage7"):
        path = base / stage / "mask.npy"
        if path.exists():
            return path, stage
    return None, "missing"


def qwen_stage1_bboxes_path(qwen_root: Path, run_id: str, case_id: str) -> Path:
    return qwen_root / case_id / run_id / "stage1" / "bboxes.json"


def load_anchor_bbox(
    case_dir: Path,
    args: argparse.Namespace,
) -> tuple[list[int], list[int], dict[str, Any]]:
    case_id = case_dir.name
    gt_bboxes_path = case_dir / "gt_bboxes.json"
    qwen_bboxes_path = qwen_stage1_bboxes_path(args.qwen_root, args.run_id, case_id)
    if args.bbox_source == "qwen-stage1" and qwen_bboxes_path.exists():
        path = qwen_bboxes_path
        source = "qwen-stage1"
    else:
        path = gt_bboxes_path
        source = "gt-bboxes"
    payload = read_json(path)
    region = payload["detected_regions"][0]
    return (
        [int(x) for x in region["bbox_thumbnail"]],
        [int(x) for x in region["bbox_level0"]],
        {
            "bbox_source": source,
            "bbox_path": str(path),
            "annotation_source": payload.get("source"),
            "updated_at": payload.get("updated_at"),
        },
    )


def load_linear_probe_predictions(path: Path) -> dict[str, dict[tuple[int, int], LinearProbeCell]]:
    by_case: dict[str, dict[tuple[int, int], LinearProbeCell]] = {}
    if not path.exists():
        return by_case
    import csv

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            case_id = row.get("case_id", "")
            if not case_id:
                continue
            cell = LinearProbeCell(
                row=int(float(row["row"])),
                col=int(float(row["col"])),
                prob_fg=float(row["prob_fg"]),
                pred_fg=int(float(row["pred_fg"])),
            )
            by_case.setdefault(case_id, {})[(cell.row, cell.col)] = cell
    return by_case


def get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def fit_image(image: Image.Image, width: int, height: int) -> Image.Image:
    image = image.convert("RGB")
    scale = min(width / image.width, height / image.height)
    resized = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), LANCZOS)
    canvas = Image.new("RGB", (width, height), "white")
    canvas.paste(resized, ((width - resized.width) // 2, (height - resized.height) // 2))
    return canvas


def mask_overlay_on_thumbnail(thumbnail: Image.Image, mask: np.ndarray, color: tuple[int, int, int]) -> Image.Image:
    base = thumbnail.convert("RGBA")
    mask_image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L").resize(base.size, NEAREST)
    overlay = Image.new("RGBA", base.size, (*color, 110))
    empty = Image.new("RGBA", base.size, (0, 0, 0, 0))
    return Image.alpha_composite(base, Image.composite(overlay, empty, mask_image)).convert("RGB")


def cell_rect_thumbnail(row: int, col: int, mask_shape: tuple[int, int], thumb_size: tuple[int, int]) -> tuple[int, int, int, int]:
    mask_rows, mask_cols = mask_shape
    width, height = thumb_size
    x0 = int(round(col / mask_cols * width))
    x1 = int(round((col + 1) / mask_cols * width))
    y0 = int(round(row / mask_rows * height))
    y1 = int(round((row + 1) / mask_rows * height))
    return x0, y0, max(x0 + 1, x1), max(y0 + 1, y1)


def rect_intersection(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[int, int, int, int] | None:
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def local_cell_rect(
    row: int,
    col: int,
    mask_shape: tuple[int, int],
    thumb_size: tuple[int, int],
    crop_box: tuple[int, int, int, int],
) -> tuple[int, int, int, int] | None:
    rect = cell_rect_thumbnail(row, col, mask_shape, thumb_size)
    clipped = rect_intersection(rect, crop_box)
    if clipped is None:
        return None
    return clipped[0] - crop_box[0], clipped[1] - crop_box[1], clipped[2] - crop_box[0], clipped[3] - crop_box[1]


def cell_rect_level0(row: int, col: int, cell_size_level0: int) -> tuple[int, int, int, int]:
    x0 = int(col) * int(cell_size_level0)
    y0 = int(row) * int(cell_size_level0)
    return x0, y0, x0 + int(cell_size_level0), y0 + int(cell_size_level0)


def local_cell_rect_level0(
    row: int,
    col: int,
    bbox_level0: tuple[int, int, int, int],
    image_size: tuple[int, int],
    cell_size_level0: int,
) -> tuple[int, int, int, int] | None:
    clipped = rect_intersection(cell_rect_level0(row, col, cell_size_level0), bbox_level0)
    if clipped is None:
        return None
    bbox_w = max(1, bbox_level0[2] - bbox_level0[0])
    bbox_h = max(1, bbox_level0[3] - bbox_level0[1])
    sx = image_size[0] / float(bbox_w)
    sy = image_size[1] / float(bbox_h)
    x0 = int(round((clipped[0] - bbox_level0[0]) * sx))
    y0 = int(round((clipped[1] - bbox_level0[1]) * sy))
    x1 = int(round((clipped[2] - bbox_level0[0]) * sx))
    y1 = int(round((clipped[3] - bbox_level0[1]) * sy))
    return (
        max(0, min(image_size[0] - 1, x0)),
        max(0, min(image_size[1] - 1, y0)),
        max(1, min(image_size[0], max(x0 + 1, x1))),
        max(1, min(image_size[1], max(y0 + 1, y1))),
    )


def cell_center_in_crop(
    row: int,
    col: int,
    mask_shape: tuple[int, int],
    thumb_size: tuple[int, int],
    crop_box: tuple[int, int, int, int],
) -> bool:
    x0, y0, x1, y1 = cell_rect_thumbnail(row, col, mask_shape, thumb_size)
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    return crop_box[0] <= cx < crop_box[2] and crop_box[1] <= cy < crop_box[3]


def cell_center_in_bbox_level0(
    row: int,
    col: int,
    bbox_level0: tuple[int, int, int, int],
    cell_size_level0: int,
) -> bool:
    x0, y0, x1, y1 = cell_rect_level0(row, col, cell_size_level0)
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    return bbox_level0[0] <= cx < bbox_level0[2] and bbox_level0[1] <= cy < bbox_level0[3]


def stable_rng(case_id: str, pool: str, seed: int) -> np.random.Generator:
    digest = hashlib.sha256(f"{case_id}:{pool}:{seed}".encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little") % (2**32))


def sample_pool_cells(
    case_id: str,
    pool_name: str,
    mask: np.ndarray,
    bbox_level0: tuple[int, int, int, int],
    cell_size_level0: int,
    samples_per_region: int,
    seed: int,
) -> list[tuple[int, int]]:
    coords = [
        (int(row), int(col))
        for row, col in np.argwhere(mask)
        if cell_center_in_bbox_level0(int(row), int(col), bbox_level0, cell_size_level0)
    ]
    if len(coords) <= samples_per_region:
        return coords
    rng = stable_rng(case_id, pool_name, seed)
    chosen = rng.choice(len(coords), size=samples_per_region, replace=False)
    return [coords[int(idx)] for idx in chosen]


def overlay_pool_on_crop(
    source_crop: Image.Image,
    pool_mask: np.ndarray,
    bbox_level0: tuple[int, int, int, int],
    color: tuple[int, int, int],
    sample_cells: list[tuple[int, int]],
    cell_size_level0: int,
) -> Image.Image:
    base = source_crop.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for row, col in np.argwhere(pool_mask):
        rect = local_cell_rect_level0(int(row), int(col), bbox_level0, base.size, cell_size_level0)
        if rect is None:
            continue
        draw.rectangle(rect, fill=(*color, 95))

    out = Image.alpha_composite(base, overlay).convert("RGB")
    draw = ImageDraw.Draw(out)
    for idx, (row, col) in enumerate(sample_cells, start=1):
        rect = local_cell_rect_level0(row, col, bbox_level0, out.size, cell_size_level0)
        if rect is None:
            continue
        for inset in range(3):
            draw.rectangle(
                (rect[0] - inset, rect[1] - inset, rect[2] + inset, rect[3] + inset),
                outline=color,
                width=1,
            )
        draw.text((rect[0] + 3, rect[1] + 2), str(idx), fill=(0, 0, 0), font=get_font(14, bold=True))
    return out


def overlay_linear_probe_on_crop(
    source_crop: Image.Image,
    predictions: dict[tuple[int, int], LinearProbeCell],
    bbox_level0: tuple[int, int, int, int],
    cell_size_level0: int,
) -> Image.Image:
    base = source_crop.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    scored = 0
    for (row, col), pred in predictions.items():
        rect = local_cell_rect_level0(row, col, bbox_level0, base.size, cell_size_level0)
        if rect is None:
            continue
        scored += 1
        prob = max(0.0, min(1.0, pred.prob_fg))
        if pred.pred_fg:
            draw.rectangle(rect, fill=(46, 204, 113, int(75 + 130 * prob)))
        else:
            draw.rectangle(rect, outline=(231, 76, 60, 190), width=2)
    out = Image.alpha_composite(base, overlay).convert("RGB")
    draw = ImageDraw.Draw(out)
    draw.text((8, 8), f"scored cells: {scored}", fill=(0, 0, 0), font=get_font(14, bold=True))
    return out


def sample_patch_panel(
    source_crop: Image.Image,
    pool_name: str,
    sample_cells: list[tuple[int, int]],
    sample_patch_images: dict[tuple[int, int], Image.Image],
    bbox_level0: tuple[int, int, int, int],
    cell_size_level0: int,
    panel_width: int,
    patch_size: int,
) -> Image.Image:
    color = POOL_COLORS[pool_name]
    panel = Image.new("RGB", (panel_width, patch_size + 54), "white")
    draw = ImageDraw.Draw(panel)
    draw.text((0, 0), POOL_LABELS[pool_name], fill=(0, 0, 0), font=get_font(15, bold=True))
    gap = 12
    for idx in range(3):
        x = idx * (patch_size + gap)
        y = 30
        if idx < len(sample_cells):
            row, col = sample_cells[idx]
            patch = sample_patch_images.get((row, col))
            if patch is None:
                rect = local_cell_rect_level0(row, col, bbox_level0, source_crop.size, cell_size_level0)
                patch = source_crop.crop(rect) if rect is not None else None
            if patch is not None:
                fitted = fit_image(patch, patch_size, patch_size)
                panel.paste(fitted, (x, y))
                draw.rectangle((x, y, x + patch_size - 1, y + patch_size - 1), outline=color, width=4)
                draw.text((x + 6, y + patch_size - 18), f"r{row} c{col}", fill=(0, 0, 0), font=get_font(11, bold=True))
                continue
        draw.rectangle((x, y, x + patch_size - 1, y + patch_size - 1), outline=(190, 190, 190), width=1)
        draw.text((x + 8, y + patch_size // 2 - 8), "none", fill=(100, 100, 100), font=get_font(13))
    return panel


def pool_masks(gt_mask: np.ndarray, qwen_mask: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "true_positive": gt_mask & qwen_mask,
        "hard_negative": (~gt_mask) & qwen_mask,
        "easy_negative": (~gt_mask) & (~qwen_mask),
    }


def case_wsi_path(case_dir: Path) -> Path | None:
    meta_path = case_dir / "case_meta.json"
    if not meta_path.exists():
        return None
    try:
        raw = read_json(meta_path).get("wsi_path")
    except Exception:
        return None
    if not raw:
        return None
    path = Path(raw)
    return path if path.exists() else None


def bbox_crop_file(case_dir: Path, bbox_level0: tuple[int, int, int, int]) -> Path | None:
    stem = "_".join(str(int(x)) for x in bbox_level0)
    path = case_dir / f"{stem}.png"
    return path if path.exists() else None


def choose_read_level(level_downsamples: tuple[float, ...] | list[float], long_edge_level0: int, max_dim: int) -> int:
    if not level_downsamples:
        return 0
    desired_downsample = max(1.0, long_edge_level0 / float(max(1, max_dim)))
    level = 0
    for idx, downsample in enumerate(level_downsamples):
        if float(downsample) <= desired_downsample:
            level = idx
    return level


def read_bbox_preview_from_wsi(
    case_dir: Path,
    bbox_level0: tuple[int, int, int, int],
    fallback_crop: Image.Image,
    max_dim: int,
    crop_source: str,
) -> tuple[Image.Image, dict[str, Any]]:
    crop_file = bbox_crop_file(case_dir, bbox_level0)
    if crop_source in {"auto", "crop-file"} and crop_file is not None:
        return Image.open(crop_file).convert("RGB"), {
            "source": "crop-file",
            "path": str(crop_file),
            "preview_size": list(Image.open(crop_file).size),
        }
    if crop_source == "crop-file":
        return fallback_crop, {"source": "thumbnail", "reason": "missing_bbox_crop_file"}
    if crop_source == "thumbnail":
        return fallback_crop, {"source": "thumbnail", "reason": "requested"}
    wsi_path = case_wsi_path(case_dir)
    if wsi_path is None:
        return fallback_crop, {"source": "thumbnail", "reason": "missing_case_wsi_path"}
    try:
        import openslide  # type: ignore
    except Exception as exc:  # pragma: no cover - optional local dependency
        return fallback_crop, {"source": "thumbnail", "reason": f"openslide_unavailable:{type(exc).__name__}"}

    x0, y0, x1, y1 = bbox_level0
    width = max(1, x1 - x0)
    height = max(1, y1 - y0)
    slide = openslide.OpenSlide(str(wsi_path))
    try:
        level_downsamples = [float(x) for x in slide.level_downsamples]
        level = choose_read_level(level_downsamples, max(width, height), max_dim)
        downsample = float(level_downsamples[level]) if level_downsamples else 1.0
        read_size = (max(1, int(math.ceil(width / downsample))), max(1, int(math.ceil(height / downsample))))
        crop = slide.read_region((x0, y0), level, read_size).convert("RGB")
    finally:
        slide.close()

    if max(crop.size) > max_dim:
        scale = max_dim / float(max(crop.size))
        crop = crop.resize(
            (max(1, int(round(crop.width * scale))), max(1, int(round(crop.height * scale)))),
            LANCZOS,
        )
    return crop, {
        "source": "wsi",
        "wsi_path": str(wsi_path),
        "read_level": level,
        "read_downsample": downsample,
        "preview_size": list(crop.size),
    }


def read_sample_patch_images(
    case_dir: Path,
    samples: dict[str, list[tuple[int, int]]],
    cell_size_level0: int,
) -> tuple[dict[tuple[int, int], Image.Image], dict[str, Any]]:
    wsi_path = case_wsi_path(case_dir)
    if wsi_path is None:
        return {}, {"source": "crop_preview", "reason": "missing_case_wsi_path"}
    try:
        import openslide  # type: ignore
    except Exception as exc:  # pragma: no cover - optional local dependency
        return {}, {"source": "crop_preview", "reason": f"openslide_unavailable:{type(exc).__name__}"}

    cells = sorted({cell for pool_cells in samples.values() for cell in pool_cells})
    out: dict[tuple[int, int], Image.Image] = {}
    slide = openslide.OpenSlide(str(wsi_path))
    try:
        for row, col in cells:
            x0 = int(col) * int(cell_size_level0)
            y0 = int(row) * int(cell_size_level0)
            out[(row, col)] = slide.read_region(
                (x0, y0),
                0,
                (int(cell_size_level0), int(cell_size_level0)),
            ).convert("RGB")
    finally:
        slide.close()
    return out, {"source": "wsi_level0", "wsi_path": str(wsi_path), "patch_count": len(out)}


def draw_bbox(image: Image.Image, bbox: list[int], label: str) -> Image.Image:
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    x0, y0, x1, y1 = bbox
    for inset in range(4):
        draw.rectangle((x0 - inset, y0 - inset, x1 + inset, y1 + inset), outline=(220, 30, 30), width=1)
    draw.text((x0 + 6, max(4, y0 + 6)), label, fill=(220, 30, 30), font=get_font(18, bold=True))
    return out


def page_for_case(case_dir: Path, args: argparse.Namespace) -> tuple[Image.Image, dict[str, Any]]:
    case_id = case_dir.name
    bbox_thumb, bbox_l0, anchor_meta = load_anchor_bbox(case_dir, args)
    bbox_l0_tuple = tuple(bbox_l0)
    thumbnail = Image.open(case_dir / "thumbnail.png").convert("RGB")
    thumb_with_bbox = draw_bbox(thumbnail, bbox_thumb, anchor_meta["bbox_source"])
    crop_box = tuple(bbox_thumb)
    thumbnail_crop = thumbnail.crop(crop_box)
    source_crop, crop_source_meta = read_bbox_preview_from_wsi(
        case_dir,
        bbox_l0_tuple,
        thumbnail_crop,
        args.wsi_crop_max_dim,
        args.crop_source,
    )

    gt_mask = np.load(case_dir / "mask.npy").astype(bool)
    gt_overlay_crop = overlay_pool_on_crop(
        source_crop,
        gt_mask,
        bbox_l0_tuple,
        (46, 204, 113),
        [],
        args.grid_cell_size_level0,
    )

    qwen_path, qwen_stage = qwen_mask_path(args.qwen_root, args.run_id, case_id)
    if qwen_path is not None:
        qwen_mask = np.load(qwen_path).astype(bool)
    else:
        qwen_mask = np.zeros_like(gt_mask)
    qwen_overlay_crop = overlay_pool_on_crop(
        source_crop,
        qwen_mask,
        bbox_l0_tuple,
        (155, 89, 182),
        [],
        args.grid_cell_size_level0,
    )

    existing_bbox_overlay = case_dir / "bbox_overlay.png"
    bbox_overlay = Image.open(existing_bbox_overlay).convert("RGB") if existing_bbox_overlay.exists() else thumb_with_bbox

    panels = [
        ("thumbnail + selected bbox", thumb_with_bbox),
        ("existing bbox_overlay.png", bbox_overlay),
        ("selected bbox tissue crop", source_crop),
        ("bbox crop + GT mask", gt_overlay_crop),
        ("bbox crop + qwen2b mask", qwen_overlay_crop),
    ]
    panel_w = args.panel_width
    panel_h = round(args.panel_width * 0.72)
    pad = 22
    title_h = 105
    label_h = 30
    cols = 2
    rows = 3
    width = cols * panel_w + (cols + 1) * pad
    height = title_h + rows * (panel_h + label_h) + (rows + 1) * pad
    page = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(page)
    draw.text((pad, 18), case_id, fill=(0, 0, 0), font=get_font(24, bold=True))
    draw.text((pad, 52), f"bbox_level0={bbox_l0} | bbox_thumbnail={bbox_thumb}", fill=(55, 55, 55), font=get_font(15))
    draw.text(
        (pad, 75),
        f"bbox={anchor_meta['bbox_source']} | qwen={qwen_stage} | crop={crop_source_meta.get('source')}",
        fill=(55, 55, 55),
        font=get_font(15),
    )

    for idx, (label, panel) in enumerate(panels):
        row = idx // cols
        col = idx % cols
        x = pad + col * (panel_w + pad)
        y = title_h + pad + row * (panel_h + label_h + pad)
        fitted = fit_image(panel, panel_w, panel_h)
        page.paste(fitted, (x, y))
        draw.rectangle((x, y, x + panel_w - 1, y + panel_h - 1), outline=(170, 170, 170), width=1)
        draw.text((x, y + panel_h + 7), label, fill=(0, 0, 0), font=get_font(15, bold=True))

    meta = {
        "case_id": case_id,
        "bbox_level0": bbox_l0,
        "bbox_thumbnail": bbox_thumb,
        "source": anchor_meta.get("annotation_source"),
        "anchor_bbox": anchor_meta,
        "qwen_stage": qwen_stage,
        "gt_bboxes_path": str(case_dir / "gt_bboxes.json"),
        "qwen_mask_path": str(qwen_path) if qwen_path else None,
        "crop_source": crop_source_meta,
    }
    return page, meta


def pool_sampling_page_for_case(
    case_dir: Path,
    args: argparse.Namespace,
    linear_predictions: dict[tuple[int, int], LinearProbeCell],
) -> tuple[Image.Image, dict[str, Any]]:
    case_id = case_dir.name
    bbox_thumb_list, bbox_l0, anchor_meta = load_anchor_bbox(case_dir, args)
    bbox_thumb = tuple(bbox_thumb_list)
    bbox_l0_tuple = tuple(bbox_l0)
    thumbnail = Image.open(case_dir / "thumbnail.png").convert("RGB")
    thumbnail_crop = thumbnail.crop(bbox_thumb)
    source_crop, crop_source_meta = read_bbox_preview_from_wsi(
        case_dir,
        bbox_l0_tuple,
        thumbnail_crop,
        args.wsi_crop_max_dim,
        args.crop_source,
    )
    gt_mask = np.load(case_dir / "mask.npy").astype(bool)
    qwen_path, qwen_stage = qwen_mask_path(args.qwen_root, args.run_id, case_id)
    if qwen_path is None:
        qwen_mask = np.zeros_like(gt_mask)
    else:
        qwen_mask = np.load(qwen_path).astype(bool)
    pools = pool_masks(gt_mask, qwen_mask)
    thumb_size = thumbnail.size
    samples = {
        pool_name: sample_pool_cells(
            case_id=case_id,
            pool_name=pool_name,
            mask=pool_mask,
            bbox_level0=bbox_l0_tuple,
            cell_size_level0=args.grid_cell_size_level0,
            samples_per_region=args.samples_per_region,
            seed=args.seed,
        )
        for pool_name, pool_mask in pools.items()
    }
    sample_patch_images, patch_source_meta = read_sample_patch_images(
        case_dir,
        samples,
        args.grid_cell_size_level0,
    )

    panel_w = args.panel_width
    panel_h = round(args.panel_width * 0.70)
    pad = 22
    title_h = 124
    label_h = 31
    width = 4 * panel_w + 5 * pad
    patch_size = min(130, (panel_w - 24) // 3)
    patch_panel_h = patch_size + 54
    height = title_h + 2 * (panel_h + label_h) + 3 * pad + 3 * patch_panel_h + 4 * pad
    page = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(page)
    draw.text((pad, 18), case_id, fill=(0, 0, 0), font=get_font(24, bold=True))
    draw.text((pad, 52), f"selected bbox_level0={bbox_l0} | bbox_thumbnail={list(bbox_thumb)}", fill=(55, 55, 55), font=get_font(15))
    draw.text(
        (pad, 76),
        f"bbox={anchor_meta['bbox_source']} | qwen={qwen_stage} | samples={args.samples_per_region}/region | crop={crop_source_meta.get('source')}",
        fill=(55, 55, 55),
        font=get_font(15),
    )
    draw.text(
        (pad, 98),
        "Pools use 512px level-0 mask-grid cells whose centers fall inside the selected bbox; patch thumbnails are level-0 reads.",
        fill=(80, 80, 80),
        font=get_font(13),
    )

    top_panels = [
        ("selected bbox tissue crop", source_crop),
        (
            "true positives: qwen FG, GT FG",
            overlay_pool_on_crop(
                source_crop,
                pools["true_positive"],
                bbox_l0_tuple,
                POOL_COLORS["true_positive"],
                samples["true_positive"],
                args.grid_cell_size_level0,
            ),
        ),
        (
            "hard negatives: qwen FG, GT BG",
            overlay_pool_on_crop(
                source_crop,
                pools["hard_negative"],
                bbox_l0_tuple,
                POOL_COLORS["hard_negative"],
                samples["hard_negative"],
                args.grid_cell_size_level0,
            ),
        ),
        (
            "easy negatives: qwen BG, GT BG",
            overlay_pool_on_crop(
                source_crop,
                pools["easy_negative"],
                bbox_l0_tuple,
                POOL_COLORS["easy_negative"],
                samples["easy_negative"],
                args.grid_cell_size_level0,
            ),
        ),
    ]
    y_top = title_h + pad
    for idx, (label, panel) in enumerate(top_panels):
        x = pad + idx * (panel_w + pad)
        fitted = fit_image(panel, panel_w, panel_h)
        page.paste(fitted, (x, y_top))
        draw.rectangle((x, y_top, x + panel_w - 1, y_top + panel_h - 1), outline=(170, 170, 170), width=1)
        draw.text((x, y_top + panel_h + 7), label, fill=(0, 0, 0), font=get_font(13, bold=True))

    second_row = [
        (
            "GT overlay",
            overlay_pool_on_crop(source_crop, gt_mask, bbox_l0_tuple, (46, 204, 113), [], args.grid_cell_size_level0),
        ),
        (
            "qwen2b zero-shot overlay",
            overlay_pool_on_crop(source_crop, qwen_mask, bbox_l0_tuple, (155, 89, 182), [], args.grid_cell_size_level0),
        ),
        (
            "scale500 linear-probe overlay",
            overlay_linear_probe_on_crop(source_crop, linear_predictions, bbox_l0_tuple, args.grid_cell_size_level0),
        ),
    ]
    y_second = y_top + panel_h + label_h + pad
    for idx, (label, panel) in enumerate(second_row):
        x = pad + idx * (panel_w + pad)
        fitted = fit_image(panel, panel_w, panel_h)
        page.paste(fitted, (x, y_second))
        draw.rectangle((x, y_second, x + panel_w - 1, y_second + panel_h - 1), outline=(170, 170, 170), width=1)
        draw.text((x, y_second + panel_h + 7), label, fill=(0, 0, 0), font=get_font(13, bold=True))

    y = y_second + panel_h + label_h + pad
    for pool_name in ("true_positive", "hard_negative", "easy_negative"):
        panel = sample_patch_panel(
            source_crop,
            pool_name,
            samples[pool_name],
            sample_patch_images,
            bbox_l0_tuple,
            args.grid_cell_size_level0,
            width - 2 * pad,
            patch_size,
        )
        page.paste(panel, (pad, y))
        y += patch_panel_h + pad

    meta = {
        "case_id": case_id,
        "bbox_level0": bbox_l0,
        "bbox_thumbnail": list(bbox_thumb),
        "anchor_bbox": anchor_meta,
        "qwen_stage": qwen_stage,
        "qwen_mask_path": str(qwen_path) if qwen_path else None,
        "pool_counts_inside_bbox": {
            pool_name: sum(
                1
                for row, col in np.argwhere(pool_mask)
                if cell_center_in_bbox_level0(int(row), int(col), bbox_l0_tuple, args.grid_cell_size_level0)
            )
            for pool_name, pool_mask in pools.items()
        },
        "sampled_cells": {
            pool_name: [{"row": row, "col": col} for row, col in cells]
            for pool_name, cells in samples.items()
        },
        "scale500_linear_probe_scored_cells_inside_bbox": sum(
            1
            for row, col in linear_predictions
            if cell_center_in_bbox_level0(row, col, bbox_l0_tuple, args.grid_cell_size_level0)
        ),
        "crop_source": crop_source_meta,
        "patch_source": patch_source_meta,
    }
    return page, meta


def capture_repo_command(command: list[str]) -> str:
    try:
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except Exception as exc:
        return f"unavailable: {type(exc).__name__}: {exc}"
    text = result.stdout.strip()
    return text if text else "<no output>"


def write_reproduction_txt(
    args: argparse.Namespace,
    pdf_path: Path,
    pool_pdf_path: Path,
    summary_path: Path,
    case_count: int,
) -> Path:
    command = " ".join(shlex.quote(part) for part in [sys.executable, *sys.argv])
    repro_path = args.output_dir / "reproduction.txt"
    content = f"""Generated timestamp: {datetime.now(timezone.utc).isoformat()}
Ticket: PER-250
Source repository: {REPO_ROOT}
Working directory: {Path.cwd()}
Git commit: {capture_repo_command(["git", "rev-parse", "HEAD"])}
Dirty-worktree status at creation time:
{capture_repo_command(["git", "status", "--short"])}
DVC status: not used in this repository checkout for this output

Exact command:
{command}

Inputs:
- GT dataset root: {args.gt_root.resolve()}
- qwen2b zero-shot root: {args.qwen_root.resolve()}
- qwen2b run id: {args.run_id}
- scale500 linear-probe predictions: {args.linear_probe_predictions.resolve() if args.linear_probe_predictions.exists() else "missing"}
- selected bbox source: {args.bbox_source}; qwen-stage1 uses per-case qwen run `stage1/bboxes.json`, gt-bboxes uses the GT dataset `gt_bboxes.json`
- mask grid: {args.grid_cell_size_level0}px level-0 cells

Outputs:
- selected bbox verification PDF: {pdf_path.resolve()}
- sampling-pool review PDF: {pool_pdf_path.resolve()}
- summary JSON: {summary_path.resolve()}

Run parameters:
- cases: {case_count}
- samples per region: {args.samples_per_region}
- sample seed: {args.seed}
- crop source: {args.crop_source}
- bbox source: {args.bbox_source}
- WSI crop max dim: {args.wsi_crop_max_dim}

Notes:
- Pool definitions are true positive = qwen2b foreground and GT foreground; hard negative = qwen2b foreground and GT background; easy negative = qwen2b background and GT background.
- Pool overlays and sample selection use 512px level-0 grid-cell geometry, not thumbnail pixel geometry.
- The tissue crop panels use existing bbox crop PNGs when available, then fall back to WSI or thumbnail reads; sampled patch thumbnails are direct level-0 WSI reads. If OpenSlide or a WSI path is unavailable, the script records the fallback in the summary JSON.
"""
    repro_path.write_text(content)
    return repro_path


def main() -> None:
    args = parse_args()
    linear_predictions = load_linear_probe_predictions(args.linear_probe_predictions)
    requested = [x.strip() for x in args.case_ids.split(",") if x.strip()]
    case_dirs = [args.gt_root / x for x in requested] if requested else sorted(p for p in args.gt_root.iterdir() if p.is_dir())
    case_dirs = [p for p in case_dirs if (p / "gt_bboxes.json").exists() and (p / "thumbnail.png").exists() and (p / "mask.npy").exists()]
    if args.case_limit is not None:
        case_dirs = case_dirs[: args.case_limit]
    pages: list[Image.Image] = []
    pool_pages: list[Image.Image] = []
    cases: list[dict[str, Any]] = []
    pool_cases: list[dict[str, Any]] = []
    for case_dir in case_dirs:
        page, meta = page_for_case(case_dir, args)
        pages.append(page)
        cases.append(meta)
        pool_page, pool_meta = pool_sampling_page_for_case(
            case_dir,
            args,
            linear_predictions.get(case_dir.name, {}),
        )
        pool_pages.append(pool_page)
        pool_cases.append(pool_meta)
    if not pages:
        raise SystemExit("No cases with gt_bboxes.json, thumbnail.png, and mask.npy found.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = args.output_dir / args.pdf_name
    pool_pdf_path = args.output_dir / args.pool_pdf_name
    pages[0].save(pdf_path, save_all=True, append_images=pages[1:])
    pool_pages[0].save(pool_pdf_path, save_all=True, append_images=pool_pages[1:])
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pdf_path": str(pdf_path.resolve()),
        "pool_sampling_pdf_path": str(pool_pdf_path.resolve()),
        "case_count": len(cases),
        "samples_per_region": args.samples_per_region,
        "seed": args.seed,
        "gt_root": str(args.gt_root.resolve()),
        "qwen_root": str(args.qwen_root.resolve()),
        "linear_probe_predictions": str(args.linear_probe_predictions.resolve()) if args.linear_probe_predictions.exists() else None,
        "cases": cases,
        "pool_sampling_cases": pool_cases,
    }
    summary_path = args.output_dir / "stress32_manual_gt_bbox_verification_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    repro_path = write_reproduction_txt(args, pdf_path, pool_pdf_path, summary_path, len(cases))
    print(f"Wrote {len(pages)} pages: {pdf_path}")
    print(f"Wrote {len(pool_pages)} pool-sampling pages: {pool_pdf_path}")
    print(f"Summary: {summary_path}")
    print(f"Reproduction: {repro_path}")


if __name__ == "__main__":
    main()
