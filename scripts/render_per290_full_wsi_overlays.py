#!/usr/bin/env python3
"""Render dense full-WSI overlays for the PER-290 SV40 cleaned-mask probe."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
import textwrap
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from train_per276_sv40_cleaned_mask_probe import (  # noqa: E402
    BatchPatchReader,
    PatchRecord,
    bbox_dirs,
    has_stage5_error,
    load_json,
    load_source_cases,
    metric_summary,
)
from train_per_wsi_dinov3_fg_bg_probe import (  # noqa: E402
    DEFAULT_DINOV2_SMALL,
    DEFAULT_DINOV3_SMALL,
    FeatureExtractor,
    package_versions,
)


DEFAULT_SOURCE_RUN = REPO_ROOT / "runs/scale500_sv40_icl1_alex_100_vertex_v1"
DEFAULT_PROBE_RUN = REPO_ROOT / "runs/per290_sv40_cleaned_mask_probe_v1"


@dataclass(frozen=True)
class DenseCaseResult:
    case_id: str
    wsi_path: str
    records: list[PatchRecord]
    labels: np.ndarray
    probs: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-root", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--probe-run-root", type=Path, default=DEFAULT_PROBE_RUN)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--case-split", choices=["holdout", "train", "all"], default="holdout")
    parser.add_argument("--case-limit", type=int, default=None)
    parser.add_argument("--case-ids", default="", help="Comma-separated case IDs to render.")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--probe-threshold", type=float, default=0.50)
    parser.add_argument("--include-stage5-error-bboxes", action="store_true")
    parser.add_argument("--model-backend", choices=["transformers", "timm"], default="transformers")
    parser.add_argument("--model-name", default=DEFAULT_DINOV3_SMALL)
    parser.add_argument("--fallback-model-name", default=DEFAULT_DINOV2_SMALL)
    parser.add_argument("--allow-timm-fallback", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--input-size", type=int, default=None)
    parser.add_argument("--wsi-reader", choices=["cucim", "openslide"], default="openslide")
    parser.add_argument("--read-workers", type=int, default=4)
    parser.add_argument("--thumbnail-max-dim", type=int, default=2200)
    parser.add_argument("--pdf-max-cases", type=int, default=None)
    parser.add_argument("--skip-pdf", action="store_true")
    parser.add_argument("--ticket", default="PER-290")
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = args.probe_run_root / "visuals/full_wsi_dense_overlays_heldout"
    if args.model_path is None:
        default_name = "logreg_train_split.joblib" if args.case_split == "holdout" else "logreg_all_samples.joblib"
        args.model_path = args.probe_run_root / "models" / default_name
    return args


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def git_status_short() -> str:
    try:
        return subprocess.check_output(["git", "status", "--short"], cwd=REPO_ROOT, text=True)
    except Exception as exc:
        return f"git status failed: {exc}\n"


def split_case_ids(args: argparse.Namespace) -> list[str]:
    split_path = args.probe_run_root / "split_manifest.csv"
    if not split_path.exists():
        raise FileNotFoundError(f"Missing split manifest: {split_path}")
    wanted = {x.strip() for x in args.case_ids.split(",") if x.strip()}
    rows = read_csv(split_path)
    case_ids = []
    for row in rows:
        case_id = row["case_id"]
        if wanted and case_id not in wanted:
            continue
        if args.case_split != "all" and row.get("split") != args.case_split:
            continue
        case_ids.append(case_id)
    case_ids = sorted(dict.fromkeys(case_ids))
    if args.case_limit is not None:
        case_ids = case_ids[: args.case_limit]
    if not case_ids:
        raise ValueError(f"No cases selected for split={args.case_split}")
    return case_ids


def load_dense_records(args: argparse.Namespace, wanted_case_ids: set[str]) -> tuple[list[PatchRecord], list[dict[str, Any]], list[dict[str, Any]]]:
    source_args = argparse.Namespace(
        source_run_root=args.source_run_root,
        case_ids=",".join(sorted(wanted_case_ids)),
        case_limit=None,
    )
    cases = load_source_cases(source_args)
    records: list[PatchRecord] = []
    bbox_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    sample_index = 0
    for case in cases:
        for bbox_index, bbox in enumerate(bbox_dirs(case.run_dir), start=1):
            if has_stage5_error(bbox) and not args.include_stage5_error_bboxes:
                skipped.append(
                    {
                        "case_id": case.case_id,
                        "run_id": case.run_id,
                        "bbox_id": bbox.name,
                        "reason": "stage5_response_error",
                    }
                )
                continue
            patches_path = bbox / "stage6/patches.csv"
            mask_path = args.probe_run_root / "cleaned_masks" / case.case_id / bbox.name / "tissue_mask_cleaned.npy"
            if not patches_path.exists() or not mask_path.exists():
                skipped.append(
                    {
                        "case_id": case.case_id,
                        "run_id": case.run_id,
                        "bbox_id": bbox.name,
                        "reason": "missing_patches_or_cleaned_mask",
                        "patches_path": str(patches_path),
                        "mask_path": str(mask_path),
                    }
                )
                continue
            mask = np.load(mask_path).astype(bool)
            stage6_meta = load_json(bbox / "stage6/metadata.json")
            wsi_path = str(stage6_meta.get("wsi_path", ""))
            crop_path = str(bbox / "stage3/crop.png")
            bbox_total = bbox_fg = bbox_bg = 0
            with patches_path.open() as f:
                for patch in csv.DictReader(f):
                    rr = int(patch["row"])
                    cc = int(patch["col"])
                    if rr < 0 or cc < 0 or rr >= mask.shape[0] or cc >= mask.shape[1]:
                        continue
                    label = int(bool(mask[rr, cc]))
                    records.append(
                        PatchRecord(
                            sample_index=sample_index,
                            case_id=case.case_id,
                            run_id=case.run_id,
                            bbox_id=bbox.name,
                            bbox_index=bbox_index,
                            row=rr,
                            col=cc,
                            x=int(patch["wsi_x"]),
                            y=int(patch["wsi_y"]),
                            width=int(patch["patch_w"]),
                            height=int(patch["patch_h"]),
                            label_fg=label,
                            source_mask_fg=label,
                            wsi_path=wsi_path,
                            crop_path=crop_path,
                        )
                    )
                    sample_index += 1
                    bbox_total += 1
                    bbox_fg += label
                    bbox_bg += 1 - label
            bbox_rows.append(
                {
                    "case_id": case.case_id,
                    "run_id": case.run_id,
                    "bbox_id": bbox.name,
                    "bbox_index": bbox_index,
                    "patch_count": bbox_total,
                    "label_fg": bbox_fg,
                    "label_bg": bbox_bg,
                    "mask_path": str(mask_path),
                    "patches_path": str(patches_path),
                    "wsi_path": wsi_path,
                }
            )
    missing_cases = wanted_case_ids - {r.case_id for r in records}
    for case_id in sorted(missing_cases):
        skipped.append({"case_id": case_id, "reason": "no_dense_records"})
    return records, bbox_rows, skipped


def extract_case_features(
    args: argparse.Namespace,
    case_id: str,
    records: list[PatchRecord],
    extractor: FeatureExtractor,
) -> tuple[np.ndarray, list[PatchRecord], dict[str, Any]]:
    cache_path = args.output_dir / "features/dense" / f"{case_id}_features.npz"
    expected_indices = np.asarray([r.sample_index for r in records], dtype="int64")
    if cache_path.exists():
        try:
            with np.load(cache_path, allow_pickle=False) as data:
                if (
                    str(data["model_backend"]) == extractor.backend
                    and str(data["model_name"]) == extractor.model_name
                    and np.array_equal(data["sample_index"].astype("int64"), expected_indices)
                ):
                    return data["features"].astype("float32"), records, {
                        "case_id": case_id,
                        "patch_count": len(records),
                        "cache_reused": True,
                        "cache_path": str(cache_path),
                        "extract_seconds": 0.0,
                        "patches_per_second": 0.0,
                    }
        except Exception:
            pass
    wsi_paths = sorted({r.wsi_path for r in records if r.wsi_path})
    if len(wsi_paths) != 1:
        raise ValueError(f"Expected one WSI path for {case_id}, found {wsi_paths}")
    wsi_path = Path(wsi_paths[0])
    if not wsi_path.exists():
        raise FileNotFoundError(f"WSI path does not exist for {case_id}: {wsi_path}")
    started = time.perf_counter()
    feature_parts: list[np.ndarray] = []
    kept_records: list[PatchRecord] = []
    failures: list[dict[str, Any]] = []
    reader = BatchPatchReader(wsi_path, args.wsi_reader, args.read_workers)
    try:
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.read_workers)) as pool:
            for start in range(0, len(records), args.batch_size):
                batch_records = records[start : start + args.batch_size]
                future_to_index = {pool.submit(reader.read_patch, record): idx for idx, record in enumerate(batch_records)}
                image_by_index: dict[int, Image.Image] = {}
                for future in concurrent.futures.as_completed(future_to_index):
                    idx = future_to_index[future]
                    record = batch_records[idx]
                    try:
                        image_by_index[idx] = future.result()
                    except Exception as exc:
                        failures.append({"record_id": record.record_id, "error": f"{type(exc).__name__}: {exc}"})
                ordered_indices = [idx for idx in range(len(batch_records)) if idx in image_by_index]
                if ordered_indices:
                    images = [image_by_index[idx] for idx in ordered_indices]
                    feature_parts.append(extractor.extract_batch(images))
                    kept_records.extend(batch_records[idx] for idx in ordered_indices)
    finally:
        reader.close()
    elapsed = time.perf_counter() - started
    if not feature_parts:
        raise RuntimeError(f"No dense features extracted for {case_id}")
    features = np.concatenate(feature_parts, axis=0).astype("float32")
    if len(features) != len(kept_records):
        raise RuntimeError(f"Feature/record mismatch for {case_id}: features={len(features)} records={len(kept_records)}")
    if failures:
        write_csv(args.output_dir / "features/failures" / f"{case_id}_failures.csv", failures)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        features=features,
        sample_index=np.asarray([r.sample_index for r in kept_records], dtype="int64"),
        label_fg=np.asarray([r.label_fg for r in kept_records], dtype="int64"),
        case_id=np.asarray([r.case_id for r in kept_records]),
        bbox_id=np.asarray([r.bbox_id for r in kept_records]),
        row=np.asarray([r.row for r in kept_records], dtype="int64"),
        col=np.asarray([r.col for r in kept_records], dtype="int64"),
        x_level0=np.asarray([r.x for r in kept_records], dtype="int64"),
        y_level0=np.asarray([r.y for r in kept_records], dtype="int64"),
        width_level0=np.asarray([r.width for r in kept_records], dtype="int64"),
        height_level0=np.asarray([r.height for r in kept_records], dtype="int64"),
        model_backend=np.asarray(extractor.backend),
        model_name=np.asarray(extractor.model_name),
        wsi_reader=np.asarray(args.wsi_reader),
        read_workers=np.asarray(args.read_workers),
        created_at=np.asarray(datetime.now(timezone.utc).isoformat()),
    )
    return features, kept_records, {
        "case_id": case_id,
        "patch_count": int(len(kept_records)),
        "cache_reused": False,
        "cache_path": str(cache_path),
        "failure_count": len(failures),
        "extract_seconds": float(elapsed),
        "patches_per_second": float(len(kept_records) / elapsed) if elapsed > 0 else 0.0,
    }


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


FONT_TITLE = font(46, True)
FONT_H2 = font(28, True)
FONT_BODY = font(22)
FONT_SMALL = font(17)
PAGE_W = 2600
PAGE_H = 1750
MARGIN = 55
GREEN = (26, 185, 90, 96)
RED = (230, 55, 55, 86)
BBOX = (20, 95, 220, 220)


def open_wsi_thumbnail(wsi_path: str, max_dim: int) -> tuple[Image.Image, tuple[int, int]]:
    try:
        import openslide
    except Exception as exc:
        raise RuntimeError("openslide is required for full-WSI thumbnail rendering") from exc
    slide = openslide.OpenSlide(wsi_path)
    try:
        slide_w, slide_h = slide.dimensions
        scale = min(max_dim / max(slide_w, slide_h), 1.0)
        thumb_size = (max(1, int(slide_w * scale)), max(1, int(slide_h * scale)))
        thumb = slide.get_thumbnail(thumb_size).convert("RGB")
    finally:
        slide.close()
    return thumb, (slide_w, slide_h)


def fit_with_box(img: Image.Image, box: tuple[int, int]) -> tuple[Image.Image, tuple[int, int, int, int]]:
    out_w, out_h = box
    work = img.copy()
    work.thumbnail((out_w, out_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (out_w, out_h), "white")
    x = (out_w - work.width) // 2
    y = (out_h - work.height) // 2
    canvas.paste(work, (x, y))
    return canvas, (x, y, work.width, work.height)


def overlay_panel(
    thumb: Image.Image,
    slide_size: tuple[int, int],
    records: list[PatchRecord],
    probs: np.ndarray | None,
    threshold: float,
    size: tuple[int, int],
    mode: str,
) -> Image.Image:
    panel, (ox, oy, iw, ih) = fit_with_box(thumb, size)
    slide_w, slide_h = slide_size
    sx = iw / slide_w
    sy = ih / slide_h
    overlay = Image.new("RGBA", panel.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    prob_by_sample = None
    if probs is not None:
        prob_by_sample = {record.sample_index: float(probs[idx]) for idx, record in enumerate(records)}
    for record in records:
        if mode == "original":
            continue
        if mode == "label":
            color = GREEN if record.label_fg else RED
        else:
            p = prob_by_sample[record.sample_index] if prob_by_sample is not None else 0.0
            color = GREEN if p >= threshold else RED
        x0 = int(ox + record.x * sx)
        y0 = int(oy + record.y * sy)
        x1 = int(ox + (record.x + record.width) * sx)
        y1 = int(oy + (record.y + record.height) * sy)
        draw.rectangle([x0, y0, max(x0 + 1, x1), max(y0 + 1, y1)], fill=color)
    by_bbox: dict[str, list[PatchRecord]] = defaultdict(list)
    for record in records:
        by_bbox[record.bbox_id].append(record)
    for bbox_records in by_bbox.values():
        x0 = min(r.x for r in bbox_records)
        y0 = min(r.y for r in bbox_records)
        x1 = max(r.x + r.width for r in bbox_records)
        y1 = max(r.y + r.height for r in bbox_records)
        draw.rectangle(
            [int(ox + x0 * sx), int(oy + y0 * sy), int(ox + x1 * sx), int(oy + y1 * sy)],
            outline=BBOX,
            width=3,
        )
    return Image.alpha_composite(panel.convert("RGBA"), overlay).convert("RGB")


def render_pdf(args: argparse.Namespace, case_results: list[DenseCaseResult], case_metrics: list[dict[str, Any]]) -> Path:
    pages_dir = args.output_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    pages: list[Image.Image] = []

    summary_page = Image.new("RGB", (PAGE_W, PAGE_H), "white")
    draw = ImageDraw.Draw(summary_page)
    draw.text((MARGIN, 45), "PER-290 heldout full-WSI dense overlays", fill=(0, 0, 0), font=FONT_TITLE)
    y = 115
    lines = [
        f"Cases: {len(case_results)} from split={args.case_split}; model={args.model_path.name}; threshold={args.probe_threshold:.2f}",
        "Each patch cell inside selected PER-276 crop/bbox regions is colored red/green. Outside those regions is intentionally unscored.",
        f"Source run: {args.source_run_root}",
        f"Probe run: {args.probe_run_root}",
    ]
    for line in lines:
        for wrapped in textwrap.wrap(line, width=145):
            draw.text((MARGIN, y), wrapped, fill=(45, 45, 45), font=FONT_BODY)
            y += 32
    y += 25
    draw.text((MARGIN, y), "Dense heldout metrics vs cleaned Stage7 pseudo-labels", fill=(0, 0, 0), font=FONT_H2)
    y += 45
    for row in case_metrics[:36]:
        line = (
            f"{row['case_id']}: n={row['n']} fg={row['fg']} bg={row['bg']} "
            f"precision={row['precision_fg']:.3f} recall={row['recall_fg']:.3f} "
            f"f1={row['f1_fg']:.3f} bg_fpr={row['bg_false_positive_rate']:.3f}"
        )
        draw.text((MARGIN, y), line[:170], fill=(35, 35, 35), font=FONT_SMALL)
        y += 25
        if y > PAGE_H - 80:
            break
    pages.append(summary_page)

    panel_w = (PAGE_W - 2 * MARGIN - 50) // 3
    panel_h = 1130
    for page_idx, result in enumerate(case_results, start=1):
        thumb, slide_size = open_wsi_thumbnail(result.wsi_path, args.thumbnail_max_dim)
        page = Image.new("RGB", (PAGE_W, PAGE_H), "white")
        draw = ImageDraw.Draw(page)
        draw.text((MARGIN, 42), f"{result.case_id} | heldout dense full-WSI overlay", fill=(0, 0, 0), font=FONT_TITLE)
        metric = next(row for row in case_metrics if row["case_id"] == result.case_id)
        subtitle = (
            f"n={metric['n']} fg={metric['fg']} bg={metric['bg']} | "
            f"precision={metric['precision_fg']:.3f} recall={metric['recall_fg']:.3f} "
            f"f1={metric['f1_fg']:.3f} bg_fpr={metric['bg_false_positive_rate']:.3f}"
        )
        draw.text((MARGIN, 100), subtitle, fill=(55, 55, 55), font=FONT_BODY)
        headers = ["Original WSI thumbnail", "Cleaned Stage7 labels", "Linear-probe predictions"]
        x_positions = [MARGIN, MARGIN + panel_w + 25, MARGIN + 2 * (panel_w + 25)]
        panels = [
            overlay_panel(thumb, slide_size, result.records, None, args.probe_threshold, (panel_w, panel_h), "original"),
            overlay_panel(thumb, slide_size, result.records, None, args.probe_threshold, (panel_w, panel_h), "label"),
            overlay_panel(thumb, slide_size, result.records, result.probs, args.probe_threshold, (panel_w, panel_h), "prediction"),
        ]
        for x, header, panel in zip(x_positions, headers, panels):
            draw.text((x, 160), header, fill=(0, 0, 0), font=FONT_H2)
            page.paste(panel, (x, 205))
        y = 205 + panel_h + 32
        captions = [
            "Blue outlines show selected crop/bbox regions.",
            "Green = cleaned VLM Stage7 foreground; red = cleaned Stage7 background.",
            "Green/red = dense probe foreground/background predictions for every cell inside selected regions.",
        ]
        for x, caption in zip(x_positions, captions):
            for wrapped in textwrap.wrap(caption, width=48):
                draw.text((x, y), wrapped, fill=(60, 60, 60), font=FONT_BODY)
                y += 28
            y = 205 + panel_h + 32
        page_path = pages_dir / f"page_{page_idx:03d}_{result.case_id}.png"
        page.save(page_path)
        pages.append(page)

    pdf_path = args.output_dir / "per290_heldout_full_wsi_dense_overlays.pdf"
    pages[0].save(pdf_path, save_all=True, append_images=pages[1:], resolution=150.0)
    return pdf_path


def prediction_row(record: PatchRecord, prob: float, split: str, threshold: float) -> dict[str, Any]:
    return {
        "sample_index": record.sample_index,
        "record_id": record.record_id,
        "case_id": record.case_id,
        "split": split,
        "run_id": record.run_id,
        "bbox_id": record.bbox_id,
        "bbox_index": record.bbox_index,
        "row": record.row,
        "col": record.col,
        "x_level0": record.x,
        "y_level0": record.y,
        "width_level0": record.width,
        "height_level0": record.height,
        "label_fg_cleaned_stage7": record.label_fg,
        "prob_fg": float(prob),
        "pred_fg": int(prob >= threshold),
        "wsi_path": record.wsi_path,
    }


def write_reproduction(args: argparse.Namespace, summary: dict[str, Any]) -> None:
    command = [
        "python",
        "scripts/render_per290_full_wsi_overlays.py",
        "--source-run-root",
        str(args.source_run_root),
        "--probe-run-root",
        str(args.probe_run_root),
        "--output-dir",
        str(args.output_dir),
        "--case-split",
        str(args.case_split),
        "--model-path",
        str(args.model_path),
        "--model-name",
        str(args.model_name),
        "--batch-size",
        str(args.batch_size),
        "--wsi-reader",
        str(args.wsi_reader),
        "--read-workers",
        str(args.read_workers),
        "--thumbnail-max-dim",
        str(args.thumbnail_max_dim),
    ]
    if args.case_limit is not None:
        command.extend(["--case-limit", str(args.case_limit)])
    lines = [
        "PER-290 heldout full-WSI dense overlay packet",
        "",
        f"Created: {datetime.now(timezone.utc).isoformat()}",
        f"Ticket: {args.ticket}",
        f"Git commit: {git_commit()}",
        "",
        "Command:",
        "  " + " ".join(shlex.quote(part) for part in command),
        "",
        "Parent artifacts:",
        f"  source_run_root: {args.source_run_root.resolve()}",
        f"  probe_run_root: {args.probe_run_root.resolve()}",
        f"  model_path: {args.model_path.resolve()}",
        "",
        "Feature extraction:",
        f"  model_backend={args.model_backend}",
        f"  model_name={args.model_name}",
        f"  batch_size={args.batch_size}",
        f"  wsi_reader={args.wsi_reader}",
        f"  read_workers={args.read_workers}",
        "  HF_TOKEN is required for DINOv3 access if the model is not already cached; token value intentionally not recorded.",
        "",
        "Interpretation boundary:",
        "  Metrics are against cleaned PER-276 Stage7 pseudo-labels, not human GT.",
        "  Prediction overlays color every 512px patch cell inside selected crop/bbox regions. Outside those regions is unscored.",
        "",
        "Outputs:",
        f"  summary: {summary.get('summary_json')}",
        f"  metrics: {summary.get('case_metrics_csv')}",
        f"  predictions: {summary.get('dense_predictions_csv')}",
        f"  pdf: {summary.get('review_pdf')}",
        "",
        "DVC:",
        "  no .dvc directory",
        "",
        "Git status at render time:",
        git_status_short().rstrip() or "clean",
        "",
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "reproduction.txt").write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    case_ids = split_case_ids(args)
    wanted = set(case_ids)
    print(f"[cases] split={args.case_split} cases={len(case_ids)}", flush=True)
    dense_records, bbox_rows, skipped_rows = load_dense_records(args, wanted)
    if not dense_records:
        raise SystemExit("No dense records resolved")
    write_csv(args.output_dir / "dense_bbox_pool.csv", bbox_rows)
    write_csv(args.output_dir / "skipped_bboxes.csv", skipped_rows)
    print(f"[pool] records={len(dense_records)} bboxes={len(bbox_rows)} skipped={len(skipped_rows)}", flush=True)

    extractor_args = argparse.Namespace(
        model_backend=args.model_backend,
        model_name=args.model_name,
        fallback_model_name=args.fallback_model_name,
        allow_timm_fallback=args.allow_timm_fallback,
        device=args.device,
        batch_size=args.batch_size,
        input_size=args.input_size,
    )
    extractor = FeatureExtractor(extractor_args)
    model = joblib.load(args.model_path)

    by_case: dict[str, list[PatchRecord]] = defaultdict(list)
    for record in dense_records:
        by_case[record.case_id].append(record)
    case_results: list[DenseCaseResult] = []
    prediction_rows: list[dict[str, Any]] = []
    case_metrics: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    split_by_case = {row["case_id"]: row["split"] for row in read_csv(args.probe_run_root / "split_manifest.csv")}
    for case_id in case_ids:
        records = by_case.get(case_id, [])
        if not records:
            continue
        records.sort(key=lambda r: (r.bbox_index, r.row, r.col, r.x, r.y))
        print(f"[features] {case_id}: {len(records)} dense patches", flush=True)
        x_case, kept_records, feature_meta = extract_case_features(args, case_id, records, extractor)
        probs = model.predict_proba(x_case)[:, 1].astype("float32")
        labels = np.asarray([r.label_fg for r in kept_records], dtype="int64")
        metrics = {"case_id": case_id, "split": split_by_case.get(case_id, ""), **metric_summary(labels, probs, args.probe_threshold)}
        case_metrics.append(metrics)
        feature_rows.append(feature_meta)
        for record, prob in zip(kept_records, probs):
            prediction_rows.append(prediction_row(record, float(prob), split_by_case.get(case_id, ""), args.probe_threshold))
        case_results.append(DenseCaseResult(case_id=case_id, wsi_path=kept_records[0].wsi_path, records=kept_records, labels=labels, probs=probs))

    write_csv(args.output_dir / "dense_feature_summary.csv", feature_rows)
    write_csv(args.output_dir / "dense_patch_predictions.csv", prediction_rows)
    write_csv(args.output_dir / "dense_case_metrics.csv", case_metrics)
    review_pdf = ""
    rendered_results = case_results
    if args.pdf_max_cases is not None:
        rendered_results = case_results[: args.pdf_max_cases]
    if not args.skip_pdf:
        review_pdf = str(render_pdf(args, rendered_results, case_metrics))
    all_labels = np.asarray([row["label_fg_cleaned_stage7"] for row in prediction_rows], dtype="int64")
    all_probs = np.asarray([row["prob_fg"] for row in prediction_rows], dtype="float32")
    summary = {
        "ticket": args.ticket,
        "source_run_root": str(args.source_run_root.resolve()),
        "probe_run_root": str(args.probe_run_root.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "case_split": args.case_split,
        "case_count": len(case_results),
        "bbox_count": len(bbox_rows),
        "dense_patch_count": len(prediction_rows),
        "model_path": str(args.model_path.resolve()),
        "threshold": float(args.probe_threshold),
        "overall_metrics_vs_cleaned_stage7": metric_summary(all_labels, all_probs, args.probe_threshold),
        "feature_extractor": extractor.meta,
        "package_versions": package_versions(),
        "summary_json": str((args.output_dir / "summary.json").resolve()),
        "case_metrics_csv": str((args.output_dir / "dense_case_metrics.csv").resolve()),
        "dense_predictions_csv": str((args.output_dir / "dense_patch_predictions.csv").resolve()),
        "review_pdf": review_pdf,
    }
    write_json(args.output_dir / "summary.json", summary)
    write_reproduction(args, summary)
    print(json.dumps(summary, indent=2)[:4000], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
