#!/usr/bin/env python3
"""Build review visuals for the stress32 qwen2b hard-negative sampling pools."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import openslide
except ImportError:  # pragma: no cover - optional runtime dependency
    openslide = None


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "runs/stress32_qwen2b_hard_negative_probe_v1"
DEFAULT_GT_ROOT = (
    Path("/data2/vj724/wsi-agents")
    / "vlm_gt_seg_comparison_experiment/dataset_thumbnails_harder_jones_leica/leica/jones"
)
DEFAULT_QWEN_ROOT = (
    Path("/data2/vj724/wsi-agents")
    / "new_vlm_gt_preds/leica_hard_jones_evg_qwen_2b_zero_shot"
)
DEFAULT_RUN_ID = "harder_jones_leica_manual"
CELL_SIZE_LEVEL0 = 512
LANCZOS = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
NEAREST = getattr(getattr(Image, "Resampling", Image), "NEAREST")

POOL_COLORS = {
    "easy_negative": (74, 144, 226),
    "hard_negative": (230, 126, 34),
    "true_positive": (46, 204, 113),
    "other": (160, 160, 160),
}


@dataclass(frozen=True)
class CaseInputs:
    case_id: str
    case_dir: Path
    thumbnail_path: Path
    gt_mask_path: Path
    qwen_mask_path: Path
    qwen_stage: str
    crop_image_path: Path | None
    gt_crop_overlay_path: Path | None
    wsi_gt_overlay_path: Path | None
    meta: dict[str, Any]


@dataclass(frozen=True)
class SampleCell:
    case_id: str
    row: int
    col: int
    pool: str
    source: str


@dataclass(frozen=True)
class LinearProbeCell:
    case_id: str
    row: int
    col: int
    prob_fg: float
    pred_fg: int


@dataclass(frozen=True)
class CropWindow:
    row0: int
    row1: int
    col0: int
    col1: int
    anchor_pool: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--gt-root", type=Path, default=DEFAULT_GT_ROOT)
    parser.add_argument("--qwen-root", type=Path, default=DEFAULT_QWEN_ROOT)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--case-ids", default="", help="Comma-separated case IDs; default selects from manifest or mask intersections.")
    parser.add_argument("--case-limit", type=int, default=4)
    parser.add_argument("--crops-per-case", type=int, default=3)
    parser.add_argument("--crop-grid-radius", type=int, default=5)
    parser.add_argument("--dry-run-samples-per-pool", type=int, default=40)
    parser.add_argument("--seed", type=int, default=250)
    parser.add_argument("--max-panel-size", type=int, default=260)
    parser.add_argument(
        "--linear-probe-predictions",
        type=Path,
        default=REPO_ROOT / "runs/stress32_yolo_dinov3_probe_v1/patch_predictions.csv",
        help="Patch predictions from the scale500-trained DINOv3 linear probe applied to stress32.",
    )
    parser.add_argument(
        "--pdf-name",
        default="stress32_qwen2b_sampling_pool_tissue_overlay_review.pdf",
        help="Filename for the multipage PDF written under output-root/visuals.",
    )
    parser.add_argument(
        "--read-wsi-crops",
        action="store_true",
        help="Read crop panels from source WSIs instead of prepared crop PNGs; sharper but slower and SVS-reader dependent.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_mask(path: Path) -> np.ndarray:
    mask = np.load(path)
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask at {path}, got shape {mask.shape}")
    return mask.astype(bool)


def qwen_mask_path(qwen_root: Path, run_id: str, case_id: str) -> tuple[Path | None, str | None]:
    base = qwen_root / case_id / run_id
    for stage in ("stage7_new", "stage7"):
        candidate = base / stage / "mask.npy"
        if candidate.exists():
            return candidate, stage
    return None, None


def crop_image_path(case_dir: Path) -> Path | None:
    candidates = [
        p
        for p in case_dir.glob("*.png")
        if p.name
        not in {
            "thumbnail.png",
            "bbox_overlay.png",
            "wsi_mask_thumbnail_mask.png",
            "wsi_mask_overlay_green50.png",
        }
        and not p.name.endswith(".mask.png")
        and not p.name.endswith(".overlay_green50.png")
    ]
    return sorted(candidates)[0] if candidates else None


def paired_overlay_path(crop_path: Path | None) -> Path | None:
    if crop_path is None:
        return None
    candidate = crop_path.with_name(crop_path.stem + ".overlay_green50.png")
    return candidate if candidate.exists() else None


def parse_bbox_from_crop_path(path: Path | None) -> tuple[int, int, int, int] | None:
    if path is None:
        return None
    parts = path.stem.split("_")
    if len(parts) != 4:
        return None
    try:
        x0, y0, x1, y1 = (int(part) for part in parts)
    except ValueError:
        return None
    return x0, y0, x1, y1


def discover_cases(gt_root: Path, qwen_root: Path, run_id: str) -> list[CaseInputs]:
    cases: list[CaseInputs] = []
    for case_dir in sorted(p for p in gt_root.iterdir() if p.is_dir()):
        gt_mask = case_dir / "mask.npy"
        thumbnail = case_dir / "thumbnail.png"
        qwen_mask, qwen_stage = qwen_mask_path(qwen_root, run_id, case_dir.name)
        if not gt_mask.exists() or not thumbnail.exists() or qwen_mask is None or qwen_stage is None:
            continue
        crop_path = crop_image_path(case_dir)
        wsi_overlay = case_dir / "wsi_mask_overlay_green50.png"
        cases.append(
            CaseInputs(
                case_id=case_dir.name,
                case_dir=case_dir,
                thumbnail_path=thumbnail,
                gt_mask_path=gt_mask,
                qwen_mask_path=qwen_mask,
                qwen_stage=qwen_stage,
                crop_image_path=crop_path,
                gt_crop_overlay_path=paired_overlay_path(crop_path),
                wsi_gt_overlay_path=wsi_overlay if wsi_overlay.exists() else None,
                meta=read_json(case_dir / "case_meta.json"),
            )
        )
    return cases


def first_present(row: dict[str, str], names: tuple[str, ...]) -> str | None:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return value
    return None


def normalize_pool(value: str | None, qwen_fg: bool | None = None, gt_fg: bool | None = None) -> str:
    if value:
        text = value.strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "easy": "easy_negative",
            "easy_bg": "easy_negative",
            "easy_negative": "easy_negative",
            "hard": "hard_negative",
            "hard_bg": "hard_negative",
            "hard_negative": "hard_negative",
            "tp": "true_positive",
            "true_positive": "true_positive",
            "positive": "true_positive",
        }
        if text in aliases:
            return aliases[text]
    if qwen_fg is False and gt_fg is False:
        return "easy_negative"
    if qwen_fg is True and gt_fg is False:
        return "hard_negative"
    if qwen_fg is True and gt_fg is True:
        return "true_positive"
    return "other"


def parse_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "fg", "foreground"}:
        return True
    if text in {"0", "false", "no", "bg", "background"}:
        return False
    return None


def parse_manifest(path: Path) -> list[SampleCell]:
    cells: list[SampleCell] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            case_id = first_present(row, ("case_id", "wsi_id", "slide_id", "case"))
            if not case_id:
                continue
            row_value = first_present(row, ("row", "grid_row", "mask_row", "patch_row"))
            col_value = first_present(row, ("col", "grid_col", "mask_col", "patch_col"))
            if row_value is None or col_value is None:
                x_value = first_present(row, ("x", "x_level0", "patch_x", "x0_level0"))
                y_value = first_present(row, ("y", "y_level0", "patch_y", "y0_level0"))
                if x_value is None or y_value is None:
                    continue
                col = int(float(x_value)) // CELL_SIZE_LEVEL0
                row_idx = int(float(y_value)) // CELL_SIZE_LEVEL0
            else:
                row_idx = int(float(row_value))
                col = int(float(col_value))
            pool_value = first_present(row, ("pool", "sample_pool", "label", "class", "target_name", "sampling_pool"))
            qwen_fg = parse_bool(first_present(row, ("qwen_fg", "pred_fg", "vlm_fg", "qwen2b_fg")))
            gt_fg = parse_bool(first_present(row, ("gt_fg", "target", "label_fg", "mask_fg")))
            cells.append(
                SampleCell(
                    case_id=case_id,
                    row=row_idx,
                    col=col,
                    pool=normalize_pool(pool_value, qwen_fg, gt_fg),
                    source="sample_manifest",
                )
            )
    return cells


def parse_linear_probe_predictions(path: Path) -> dict[str, dict[tuple[int, int], LinearProbeCell]]:
    by_case: dict[str, dict[tuple[int, int], LinearProbeCell]] = defaultdict(dict)
    if not path.exists():
        return by_case
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            case_id = row.get("case_id", "")
            if not case_id:
                continue
            cell = LinearProbeCell(
                case_id=case_id,
                row=int(float(row["row"])),
                col=int(float(row["col"])),
                prob_fg=float(row["prob_fg"]),
                pred_fg=int(float(row["pred_fg"])),
            )
            by_case[case_id][(cell.row, cell.col)] = cell
    return by_case


def pool_masks(qwen: np.ndarray, gt: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "easy_negative": (~qwen) & (~gt),
        "hard_negative": qwen & (~gt),
        "true_positive": qwen & gt,
    }


def deterministic_cells(case_id: str, masks: dict[str, np.ndarray], per_pool: int, seed: int) -> list[SampleCell]:
    digest = hashlib.sha256(f"{case_id}:{seed}".encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little") % (2**32))
    cells: list[SampleCell] = []
    for pool, mask in masks.items():
        coords = np.argwhere(mask)
        if coords.size == 0:
            continue
        take = min(per_pool, len(coords))
        chosen = coords[rng.choice(len(coords), size=take, replace=False)]
        for row, col in chosen:
            cells.append(SampleCell(case_id=case_id, row=int(row), col=int(col), pool=pool, source="mask_dry_run"))
    return cells


def select_cases(cases: list[CaseInputs], samples: list[SampleCell], case_ids: str, limit: int) -> list[CaseInputs]:
    if case_ids:
        requested = [x.strip() for x in case_ids.split(",") if x.strip()]
        by_id = {case.case_id: case for case in cases}
        return [by_id[x] for x in requested if x in by_id]
    sample_counts = defaultdict(int)
    for sample in samples:
        sample_counts[sample.case_id] += 1
    if sample_counts:
        ordered_ids = [case_id for case_id, _ in sorted(sample_counts.items(), key=lambda item: (-item[1], item[0]))]
        by_id = {case.case_id: case for case in cases}
        return [by_id[x] for x in ordered_ids if x in by_id][:limit]

    scored: list[tuple[int, str, CaseInputs]] = []
    for case in cases:
        gt = load_mask(case.gt_mask_path)
        qwen = load_mask(case.qwen_mask_path)
        if gt.shape != qwen.shape:
            continue
        scored.append((int((qwen & ~gt).sum()), case.case_id, case))
    return [case for _, _, case in sorted(scored, key=lambda item: (-item[0], item[1]))[:limit]]


def choose_windows(
    shape: tuple[int, int],
    masks: dict[str, np.ndarray],
    samples: list[SampleCell],
    count: int,
    radius: int,
) -> list[CropWindow]:
    anchors: list[tuple[int, int, str]] = []
    for pool in ("hard_negative", "true_positive", "easy_negative"):
        pool_samples = [s for s in samples if s.pool == pool]
        anchors.extend((s.row, s.col, pool) for s in pool_samples[:count])
        if len(anchors) >= count:
            break
    for pool in ("hard_negative", "true_positive", "easy_negative"):
        if len(anchors) >= count:
            break
        coords = np.argwhere(masks[pool])
        if len(coords) == 0:
            continue
        mid = coords[len(coords) // 2]
        anchors.append((int(mid[0]), int(mid[1]), pool))

    windows: list[CropWindow] = []
    seen: set[tuple[int, int, int, int]] = set()
    rows, cols = shape
    for row, col, pool in anchors:
        row0 = max(0, row - radius)
        row1 = min(rows, row + radius + 1)
        col0 = max(0, col - radius)
        col1 = min(cols, col + radius + 1)
        key = (row0, row1, col0, col1)
        if key in seen:
            continue
        seen.add(key)
        windows.append(CropWindow(row0=row0, row1=row1, col0=col0, col1=col1, anchor_pool=pool))
        if len(windows) >= count:
            break
    return windows


def get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def fit_image(image: Image.Image, size: int, resample: int = LANCZOS) -> Image.Image:
    image = image.convert("RGB")
    scale = min(size / image.width, size / image.height)
    new_size = (max(1, int(round(image.width * scale))), max(1, int(round(image.height * scale))))
    image = image.resize(new_size, resample)
    canvas = Image.new("RGB", (size, size), "white")
    x = (size - image.width) // 2
    y = (size - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def mask_rgb(mask: np.ndarray, color: tuple[int, int, int]) -> Image.Image:
    image = np.full((*mask.shape, 3), 245, dtype=np.uint8)
    image[mask] = np.asarray(color, dtype=np.uint8)
    return Image.fromarray(image, mode="RGB")


def pool_rgb(masks: dict[str, np.ndarray], pool: str) -> Image.Image:
    return mask_rgb(masks[pool], POOL_COLORS[pool])


def sample_overlay(shape: tuple[int, int], samples: list[SampleCell]) -> Image.Image:
    image = np.full((*shape, 3), 245, dtype=np.uint8)
    for sample in samples:
        if 0 <= sample.row < shape[0] and 0 <= sample.col < shape[1]:
            image[sample.row, sample.col] = np.asarray(POOL_COLORS.get(sample.pool, POOL_COLORS["other"]), dtype=np.uint8)
    return Image.fromarray(image, mode="RGB")


def overlay_mask_on_image(image: Image.Image, mask: np.ndarray, color: tuple[int, int, int] = (46, 204, 113)) -> Image.Image:
    base = image.convert("RGB")
    mask_image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L").resize(base.size, NEAREST)
    color_layer = Image.new("RGB", base.size, color)
    return Image.composite(Image.blend(base, color_layer, 0.5), base, mask_image)


def overlay_grid_mask_on_image(
    image: Image.Image,
    mask: np.ndarray,
    color: tuple[int, int, int],
    alpha: float = 0.48,
) -> Image.Image:
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    rows, cols = mask.shape
    for row, col in np.argwhere(mask):
        x0 = int(round(col / cols * base.width))
        x1 = int(round((col + 1) / cols * base.width))
        y0 = int(round(row / rows * base.height))
        y1 = int(round((row + 1) / rows * base.height))
        draw.rectangle((x0, y0, max(x0 + 1, x1), max(y0 + 1, y1)), fill=(*color, int(255 * alpha)))
    return Image.alpha_composite(base, overlay).convert("RGB")


def overlay_linear_probe_on_image(
    image: Image.Image,
    window: CropWindow,
    predictions: dict[tuple[int, int], LinearProbeCell],
    threshold: float = 0.5,
) -> Image.Image:
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    rows = window.row1 - window.row0
    cols = window.col1 - window.col0
    for (row, col), pred in predictions.items():
        if not (window.row0 <= row < window.row1 and window.col0 <= col < window.col1):
            continue
        local_row = row - window.row0
        local_col = col - window.col0
        x0 = int(round(local_col / cols * base.width))
        x1 = int(round((local_col + 1) / cols * base.width))
        y0 = int(round(local_row / rows * base.height))
        y1 = int(round((local_row + 1) / rows * base.height))
        prob = max(0.0, min(1.0, pred.prob_fg))
        if pred.pred_fg:
            color = (46, 204, 113)
            alpha = int(80 + 130 * prob)
            draw.rectangle((x0, y0, max(x0 + 1, x1), max(y0 + 1, y1)), fill=(*color, alpha))
        else:
            # Scored background cells get a light outline; unscored cells remain transparent.
            alpha = int(80 + 90 * (threshold - min(prob, threshold)) / threshold)
            draw.rectangle((x0, y0, max(x0 + 1, x1), max(y0 + 1, y1)), outline=(231, 76, 60, alpha), width=2)
    return Image.alpha_composite(base, overlay).convert("RGB")


def draw_tissue_overlay_page(
    case: CaseInputs,
    page_title: str,
    source_image: Image.Image,
    gt: np.ndarray,
    qwen: np.ndarray,
    pools: dict[str, np.ndarray],
    window: CropWindow | None,
    linear_predictions: dict[tuple[int, int], LinearProbeCell],
    out_path: Path,
    panel_size: int,
) -> Image.Image:
    if window is None:
        linear_panel = source_image
    else:
        linear_panel = overlay_linear_probe_on_image(source_image, window, linear_predictions)
    panels = [
        ("selected crop tissue", source_image),
        ("easy negatives: qwen BG, GT BG", overlay_grid_mask_on_image(source_image, pools["easy_negative"], POOL_COLORS["easy_negative"])),
        ("hard negatives: qwen FG, GT BG", overlay_grid_mask_on_image(source_image, pools["hard_negative"], POOL_COLORS["hard_negative"])),
        ("true positives: qwen FG, GT FG", overlay_grid_mask_on_image(source_image, pools["true_positive"], POOL_COLORS["true_positive"])),
        ("tissue + GT overlay", overlay_grid_mask_on_image(source_image, gt, (46, 204, 113))),
        ("tissue + qwen2b zero-shot", overlay_grid_mask_on_image(source_image, qwen, (155, 89, 182))),
        ("tissue + scale500 linear probe", linear_panel),
    ]

    pad = 18
    label_h = 34
    title_h = 80
    cols = 4
    rows = 2
    width = cols * panel_size + (cols + 1) * pad
    height = title_h + rows * (panel_size + label_h) + (rows + 1) * pad
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 14), page_title, fill=(0, 0, 0), font=get_font(24, bold=True))
    sub = f"{case.case_id} | qwen mask: {case.qwen_stage}"
    draw.text((pad, 48), sub, fill=(70, 70, 70), font=get_font(15))
    for idx, (label, panel) in enumerate(panels):
        grid_row = idx // cols
        grid_col = idx % cols
        x = pad + grid_col * (panel_size + pad)
        y = title_h + pad + grid_row * (panel_size + label_h + pad)
        fitted = fit_image(panel, panel_size)
        canvas.paste(fitted, (x, y))
        draw.rectangle((x, y, x + panel_size - 1, y + panel_size - 1), outline=(180, 180, 180), width=1)
        draw.text((x, y + panel_size + 6), label, fill=(0, 0, 0), font=get_font(13, bold=True))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return canvas


def crop_thumbnail(thumbnail: Image.Image, window: CropWindow, mask_shape: tuple[int, int]) -> Image.Image:
    rows, cols = mask_shape
    x0 = int(round(window.col0 / cols * thumbnail.width))
    x1 = int(round(window.col1 / cols * thumbnail.width))
    y0 = int(round(window.row0 / rows * thumbnail.height))
    y1 = int(round(window.row1 / rows * thumbnail.height))
    return thumbnail.crop((x0, y0, max(x0 + 1, x1), max(y0 + 1, y1)))


def crop_level0_image(
    image_path: Path | None,
    window: CropWindow,
    mask_shape: tuple[int, int],
    wsi_dimensions: dict[str, Any],
) -> Image.Image | None:
    bbox = parse_bbox_from_crop_path(image_path)
    if image_path is None or bbox is None:
        return None
    x0, y0, x1, y1 = bbox
    wsi_width = int(wsi_dimensions.get("width", mask_shape[1] * CELL_SIZE_LEVEL0))
    wsi_height = int(wsi_dimensions.get("height", mask_shape[0] * CELL_SIZE_LEVEL0))
    wx0 = min(window.col0 * CELL_SIZE_LEVEL0, wsi_width)
    wx1 = min(window.col1 * CELL_SIZE_LEVEL0, wsi_width)
    wy0 = min(window.row0 * CELL_SIZE_LEVEL0, wsi_height)
    wy1 = min(window.row1 * CELL_SIZE_LEVEL0, wsi_height)
    if wx0 < x0 or wx1 > x1 or wy0 < y0 or wy1 > y1:
        return None
    image = Image.open(image_path).convert("RGB")
    px0 = int(round((wx0 - x0) / (x1 - x0) * image.width))
    px1 = int(round((wx1 - x0) / (x1 - x0) * image.width))
    py0 = int(round((wy0 - y0) / (y1 - y0) * image.height))
    py1 = int(round((wy1 - y0) / (y1 - y0) * image.height))
    return image.crop((px0, py0, max(px0 + 1, px1), max(py0 + 1, py1)))


def crop_wsi_window(
    case: CaseInputs,
    window: CropWindow,
    mask_shape: tuple[int, int],
    target_max_dim: int = 1024,
) -> Image.Image | None:
    if openslide is None:
        return None
    wsi_path = case.meta.get("wsi_path")
    if not wsi_path or not Path(wsi_path).exists():
        return None
    wsi_width = int(case.meta.get("wsi_dimensions", {}).get("width", mask_shape[1] * CELL_SIZE_LEVEL0))
    wsi_height = int(case.meta.get("wsi_dimensions", {}).get("height", mask_shape[0] * CELL_SIZE_LEVEL0))
    x0 = min(window.col0 * CELL_SIZE_LEVEL0, wsi_width)
    y0 = min(window.row0 * CELL_SIZE_LEVEL0, wsi_height)
    x1 = min(window.col1 * CELL_SIZE_LEVEL0, wsi_width)
    y1 = min(window.row1 * CELL_SIZE_LEVEL0, wsi_height)
    if x1 <= x0 or y1 <= y0:
        return None
    try:
        with openslide.OpenSlide(wsi_path) as slide:
            downsample = max((x1 - x0) / target_max_dim, (y1 - y0) / target_max_dim, 1.0)
            level = slide.get_best_level_for_downsample(downsample)
            level_downsample = float(slide.level_downsamples[level])
            size = (
                max(1, int(round((x1 - x0) / level_downsample))),
                max(1, int(round((y1 - y0) / level_downsample))),
            )
            return slide.read_region((x0, y0), level, size).convert("RGB")
    except Exception:
        return None


def draw_panel_page(
    case: CaseInputs,
    page_title: str,
    source_image: Image.Image,
    gt_overlay_image: Image.Image,
    qwen: np.ndarray,
    gt: np.ndarray,
    pools: dict[str, np.ndarray],
    samples: list[SampleCell],
    out_path: Path,
    panel_size: int,
) -> None:
    labels = [
        "source",
        "source + GT overlay",
        "qwen2b output",
        "ground truth",
        "easy negatives",
        "hard negatives",
        "true positives",
        "sampled cells",
    ]
    panels = [
        fit_image(source_image, panel_size),
        fit_image(gt_overlay_image, panel_size),
        fit_image(mask_rgb(qwen, (155, 89, 182)), panel_size, NEAREST),
        fit_image(mask_rgb(gt, (39, 174, 96)), panel_size, NEAREST),
        fit_image(pool_rgb(pools, "easy_negative"), panel_size, NEAREST),
        fit_image(pool_rgb(pools, "hard_negative"), panel_size, NEAREST),
        fit_image(pool_rgb(pools, "true_positive"), panel_size, NEAREST),
        fit_image(sample_overlay(qwen.shape, samples), panel_size, NEAREST),
    ]
    pad = 18
    label_h = 28
    title_h = 72
    cols = len(panels)
    width = cols * panel_size + (cols + 1) * pad
    height = title_h + panel_size + label_h + pad * 2
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 14), page_title, fill=(0, 0, 0), font=get_font(24, bold=True))
    sub = f"{case.case_id} | qwen mask: {case.qwen_stage} | grid: {qwen.shape[0]}x{qwen.shape[1]}"
    draw.text((pad, 46), sub, fill=(70, 70, 70), font=get_font(15))
    for idx, (label, panel) in enumerate(zip(labels, panels, strict=True)):
        x = pad + idx * (panel_size + pad)
        y = title_h
        canvas.paste(panel, (x, y))
        draw.rectangle((x, y, x + panel_size - 1, y + panel_size - 1), outline=(180, 180, 180), width=1)
        draw.text((x, y + panel_size + 6), label, fill=(0, 0, 0), font=get_font(14, bold=True))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def git_value(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unavailable"


def write_index(out_dir: Path, pages: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    links = "\n".join(
        f'<li><a href="{html.escape(Path(page["path"]).name)}">{html.escape(page["title"])}</a></li>' for page in pages
    )
    body = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Stress32 qwen2b sampling-pool review</title>
  <style>
    body {{ font-family: sans-serif; margin: 24px; color: #222; }}
    code {{ background: #f3f3f3; padding: 2px 4px; }}
    li {{ margin: 6px 0; }}
  </style>
</head>
<body>
  <h1>Stress32 qwen2b sampling-pool review</h1>
  <p>Created at {html.escape(summary["created_at"])}. Mode: <code>{html.escape(summary["sample_mode"])}</code>.</p>
  <p>Panels show tissue crop, easy negatives, hard negatives, true positives, GT overlay, qwen2b zero-shot overlay, and scale500-trained linear-probe overlay.</p>
  <p>PDF: <a href="{html.escape(Path(summary["pdf_path"]).name)}">{html.escape(Path(summary["pdf_path"]).name)}</a></p>
  <ul>
    {links}
  </ul>
</body>
</html>
"""
    (out_dir / "index.html").write_text(body)


def write_reproduction(out_dir: Path, args: argparse.Namespace, summary: dict[str, Any]) -> None:
    command = " ".join(
        [
            "python",
            "scripts/build_stress32_sampling_pool_review.py",
            "--output-root",
            str(args.output_root),
            "--gt-root",
            str(args.gt_root),
            "--qwen-root",
            str(args.qwen_root),
            "--run-id",
            args.run_id,
            "--case-limit",
            str(args.case_limit),
            "--crops-per-case",
            str(args.crops_per_case),
            "--crop-grid-radius",
            str(args.crop_grid_radius),
            "--seed",
            str(args.seed),
        ]
    )
    if args.manifest:
        command += f" --manifest {args.manifest}"
    if args.case_ids:
        command += f" --case-ids {args.case_ids}"
    if args.read_wsi_crops:
        command += " --read-wsi-crops"
    text = f"""Stress32 qwen2b hard-negative sampling-pool review visuals

Created: {summary["created_at"]}
Ticket: PER-250
Git commit: {summary["git_commit"]}
Git status short saved in summary.json under git_status_short.

Inputs:
- sample manifest: {summary["manifest_path"] or "absent; deterministic mask dry-run samples used"}
- GT root: {args.gt_root.resolve()}
- qwen2b root: {args.qwen_root.resolve()}
- qwen run ID: {args.run_id}
- mask grid: 512px level-0 cells

Command:
{command}

Outputs:
- index: {(out_dir / "index.html").resolve()}
- summary: {(out_dir / "summary.json").resolve()}
- PDF: {summary["pdf_path"]}
- PNG pages: {len(summary["pages"])}
"""
    (out_dir / "reproduction.txt").write_text(text)


def main() -> None:
    args = parse_args()
    manifest = args.manifest or args.output_root / "sample_manifest.csv"
    out_dir = args.output_root / "visuals"
    cases = discover_cases(args.gt_root, args.qwen_root, args.run_id)
    manifest_samples = parse_manifest(manifest) if manifest.exists() else []
    linear_by_case = parse_linear_probe_predictions(args.linear_probe_predictions)
    sample_mode = "sample_manifest" if manifest_samples else "mask_dry_run"

    by_case_samples: dict[str, list[SampleCell]] = defaultdict(list)
    for sample in manifest_samples:
        by_case_samples[sample.case_id].append(sample)

    selected = select_cases(cases, manifest_samples, args.case_ids, args.case_limit)
    pages: list[dict[str, Any]] = []
    pdf_pages: list[Image.Image] = []
    case_summaries: list[dict[str, Any]] = []

    for case in selected:
        gt = load_mask(case.gt_mask_path)
        qwen = load_mask(case.qwen_mask_path)
        if gt.shape != qwen.shape:
            raise ValueError(f"Mask shape mismatch for {case.case_id}: GT {gt.shape}, qwen {qwen.shape}")
        pools = pool_masks(qwen, gt)
        samples = list(by_case_samples.get(case.case_id, []))
        if not samples:
            samples = deterministic_cells(case.case_id, pools, args.dry_run_samples_per_pool, args.seed)
        windows = choose_windows(gt.shape, pools, samples, args.crops_per_case, args.crop_grid_radius)
        thumbnail = Image.open(case.thumbnail_path).convert("RGB")

        for idx, window in enumerate(windows, start=1):
            crop_samples = [
                SampleCell(
                    case_id=s.case_id,
                    row=s.row - window.row0,
                    col=s.col - window.col0,
                    pool=s.pool,
                    source=s.source,
                )
                for s in samples
                if window.row0 <= s.row < window.row1 and window.col0 <= s.col < window.col1
            ]
            crop_path = out_dir / f"{case.case_id}_crop{idx:02d}_{window.anchor_pool}.png"
            source_crop = (
                crop_wsi_window(case, window, gt.shape) if args.read_wsi_crops else None
            ) or crop_level0_image(
                case.crop_image_path,
                window,
                gt.shape,
                case.meta.get("wsi_dimensions", {}),
            ) or crop_thumbnail(thumbnail, window, gt.shape)
            gt_crop = gt[window.row0 : window.row1, window.col0 : window.col1]
            qwen_crop = qwen[window.row0 : window.row1, window.col0 : window.col1]
            pool_crops = {name: mask[window.row0 : window.row1, window.col0 : window.col1] for name, mask in pools.items()}
            page_image = draw_tissue_overlay_page(
                case=case,
                page_title=(
                    f"Crop {idx} ({window.anchor_pool} anchor): rows {window.row0}-{window.row1 - 1}, "
                    f"cols {window.col0}-{window.col1 - 1}"
                ),
                source_image=source_crop,
                qwen=qwen_crop,
                gt=gt_crop,
                pools=pool_crops,
                window=window,
                linear_predictions=linear_by_case.get(case.case_id, {}),
                out_path=crop_path,
                panel_size=args.max_panel_size,
            )
            pdf_pages.append(page_image)
            pages.append({"title": f"{case.case_id} crop {idx} ({window.anchor_pool})", "path": str(crop_path.resolve())})

        case_summaries.append(
            {
                "case_id": case.case_id,
                "qwen_stage": case.qwen_stage,
                "grid_shape": list(gt.shape),
                "easy_negative_cells": int(pools["easy_negative"].sum()),
                "hard_negative_cells": int(pools["hard_negative"].sum()),
                "true_positive_cells": int(pools["true_positive"].sum()),
                "sample_cells_shown": len(samples),
                "sample_source": samples[0].source if samples else "none",
                "thumbnail_path": str(case.thumbnail_path),
                "crop_image_path": str(case.crop_image_path) if case.crop_image_path else None,
                "gt_crop_overlay_path": str(case.gt_crop_overlay_path) if case.gt_crop_overlay_path else None,
                "wsi_gt_overlay_path": str(case.wsi_gt_overlay_path) if case.wsi_gt_overlay_path else None,
                "gt_mask_path": str(case.gt_mask_path),
                "qwen_mask_path": str(case.qwen_mask_path),
                "linear_probe_scored_cells": len(linear_by_case.get(case.case_id, {})),
            }
        )

    pdf_path = out_dir / args.pdf_name
    if pdf_pages:
        pdf_pages[0].save(pdf_path, save_all=True, append_images=pdf_pages[1:])

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "ticket": "PER-250",
        "git_commit": git_value(["rev-parse", "HEAD"]),
        "git_status_short": git_value(["status", "--short"]),
        "sample_mode": sample_mode,
        "manifest_path": str(manifest.resolve()) if manifest.exists() else None,
        "linear_probe_predictions": str(args.linear_probe_predictions.resolve()) if args.linear_probe_predictions.exists() else None,
        "output_dir": str(out_dir.resolve()),
        "pdf_path": str(pdf_path.resolve()),
        "case_count": len(selected),
        "page_count": len(pages),
        "cases": case_summaries,
        "pages": pages,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "summary.json", summary)
    write_index(out_dir, pages, summary)
    write_reproduction(out_dir, args, summary)
    print(f"Wrote {len(pages)} visual pages to {out_dir}")
    print(f"Index: {out_dir / 'index.html'}")
    print(f"Summary: {out_dir / 'summary.json'}")
    print(f"Reproduction: {out_dir / 'reproduction.txt'}")


if __name__ == "__main__":
    main()
