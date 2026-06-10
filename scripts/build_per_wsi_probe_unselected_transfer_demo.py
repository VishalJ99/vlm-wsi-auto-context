#!/usr/bin/env python3
"""Apply a per-WSI DINOv3 FG/BG probe from selected crops to unselected detector crops."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import queue
import shlex
import subprocess
import sys
import textwrap
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import openslide
from PIL import Image, ImageDraw, ImageFont
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from train_per_wsi_dinov3_fg_bg_probe import FeatureExtractor  # noqa: E402

try:
    from cucim import CuImage
except Exception:  # pragma: no cover - optional runtime dependency
    CuImage = None  # type: ignore[assignment]


DEFAULT_PROBE_RUN_DIR = REPO_ROOT / "runs/per_wsi_dinov3_fg_bg_probe_baseline_v1_dinov3small"
DEFAULT_SELECTOR_RUN_ROOT = REPO_ROOT / "runs/auto_context_scale500_selector_all500_v1"
DEFAULT_SELECTOR_MANIFEST = (
    DEFAULT_SELECTOR_RUN_ROOT / "manifests/completed_cases_500_20260604_openrouter_review_current.csv"
)
DEFAULT_DETECTOR_CASE_DIR = (
    REPO_ROOT
    / "runs/detector_pipeline_scale500_v1/non_sv40/anon_02665c40_cc43_42f3_8ab1_fb9a1416e3e6"
)
DEFAULT_CASE_ID = "anon_02665c40_cc43_42f3_8ab1_fb9a1416e3e6"
DEFAULT_OUTPUT_DIR = (
    DEFAULT_PROBE_RUN_DIR
    / "visuals/per_wsi_probe_transfer_to_unselected_case023_allselected_train_v1"
)


@dataclass(frozen=True)
class CandidateInfo:
    candidate_order: int
    candidate_id: str
    bbox_level0: tuple[int, int, int, int]
    crop_path: Path
    metadata_path: Path
    selected_for_train: bool


@dataclass(frozen=True)
class PatchRecord:
    candidate_order: int
    candidate_id: str
    row: int
    col: int
    x: int
    y: int
    width: int
    height: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-run-dir", type=Path, default=DEFAULT_PROBE_RUN_DIR)
    parser.add_argument("--selector-manifest", type=Path, default=DEFAULT_SELECTOR_MANIFEST)
    parser.add_argument("--detector-case-dir", type=Path, default=DEFAULT_DETECTOR_CASE_DIR)
    parser.add_argument("--case-id", default=DEFAULT_CASE_ID)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ticket", default="PER-250")
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument(
        "--train-policy",
        choices=["all", "balanced-budget", "per-crop-balanced"],
        default="all",
        help=(
            "Use all selected pseudo-labeled patches, sample a global balanced per-class budget, "
            "or sample budget-per-class FG and BG patches from each selected crop."
        ),
    )
    parser.add_argument("--budget-per-class", type=int, default=100)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--model-backend", choices=["transformers", "timm"], default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--fallback-model-name", default="vit_small_patch14_dinov2")
    parser.add_argument("--allow-timm-fallback", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--input-size", type=int, default=None)
    parser.add_argument("--max-dim", type=int, default=1024)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
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


def get_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size)
    except Exception:
        return ImageFont.load_default()


def draw_wrapped_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    *,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    width_chars: int,
    line_spacing: int = 6,
) -> int:
    x, y = xy
    for line in textwrap.wrap(text, width=width_chars):
        draw.text((x, y), line, fill=fill, font=font)
        bbox = draw.textbbox((x, y), line, font=font)
        y += (bbox[3] - bbox[1]) + line_spacing
    return y


def safe_case_filename(case_id: str) -> str:
    return case_id.replace("/", "_")


def feature_path(run_dir: Path, case_id: str) -> Path:
    return run_dir / "features" / f"{safe_case_filename(case_id)}_features.npz"


def parse_box_ids(raw: str) -> list[int]:
    if not raw:
        return []
    try:
        value = ast.literal_eval(raw)
    except Exception:
        return []
    if isinstance(value, (list, tuple)):
        return [int(x) for x in value]
    return []


def selected_manifest_row(path: Path, case_id: str) -> dict[str, str]:
    for row in read_csv(path):
        if row.get("case_id") == case_id:
            return row
    raise ValueError(f"case_id not found in selector manifest: {case_id}")


def resolve_wsi_path(selected_rows: list[dict[str, str]], selector_row: dict[str, str], detector_json: dict[str, Any]) -> Path:
    candidates: list[str] = []
    # Prefer the source path over the local symlink; some symlinked SVS paths
    # have emitted TIFF-directory warnings while the source path reads cleanly.
    for key in ("source_wsi_path", "wsi_path"):
        for row in selected_rows:
            raw = row.get(key, "")
            if raw:
                candidates.append(raw)
        raw = selector_row.get(key, "")
        if raw:
            candidates.append(raw)
    raw = str(detector_json.get("wsi_path", ""))
    if raw:
        candidates.append(raw)
    for raw in candidates:
        path = Path(raw)
        if path.exists():
            return path
    raise FileNotFoundError("No readable WSI path found in selected manifest or detector detections.json")


def load_candidate_infos(detector_case_dir: Path, selected_ids: set[int]) -> list[CandidateInfo]:
    root = detector_case_dir / "intermediate_stage_artifacts/stage5_post_redetect_merge_and_crop/candidates"
    infos: list[CandidateInfo] = []
    for metadata_path in sorted(root.glob("*_stage4/metadata.json")):
        data = json.loads(metadata_path.read_text())
        candidate = data["candidate"]
        read_info = candidate["read_info"]
        order = int(candidate["candidate_order"])
        x0, y0, x1, y1 = [int(round(v)) for v in read_info["source_bbox_level0"]]
        infos.append(
            CandidateInfo(
                candidate_order=order,
                candidate_id=str(candidate["candidate_id"]),
                bbox_level0=(x0, y0, x1, y1),
                crop_path=Path(candidate["crop_path"]),
                metadata_path=metadata_path,
                selected_for_train=order in selected_ids,
            )
        )
    if not infos:
        raise FileNotFoundError(f"No stage5 candidate metadata found under {root}")
    return sorted(infos, key=lambda c: c.candidate_order)


def bbox_name_to_level0(name: str) -> tuple[int, int, int, int] | None:
    try:
        vals = [int(part) for part in name.split("_")]
    except Exception:
        return None
    if len(vals) != 4:
        return None
    return tuple(vals)  # type: ignore[return-value]


def select_train_indices(labels: np.ndarray, bbox_indices: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    all_idx = np.arange(len(labels), dtype="int64")
    if args.train_policy == "all":
        return all_idx
    fg = all_idx[labels == 1]
    bg = all_idx[labels == 0]
    budget = int(args.budget_per_class)
    rng = np.random.default_rng(int(args.sample_seed))
    if args.train_policy == "per-crop-balanced":
        chosen_parts: list[np.ndarray] = []
        for bbox_index in sorted(set(int(x) for x in bbox_indices.tolist())):
            crop_idx = all_idx[bbox_indices == bbox_index]
            crop_fg = crop_idx[labels[crop_idx] == 1]
            crop_bg = crop_idx[labels[crop_idx] == 0]
            if len(crop_fg) < budget or len(crop_bg) < budget:
                raise ValueError(
                    f"insufficient selected patches for bbox_index={bbox_index} budget={budget}: "
                    f"fg={len(crop_fg)} bg={len(crop_bg)}"
                )
            chosen_parts.extend(
                [
                    rng.choice(crop_fg, size=budget, replace=False),
                    rng.choice(crop_bg, size=budget, replace=False),
                ]
            )
        chosen = np.concatenate(chosen_parts)
        rng.shuffle(chosen)
        return chosen.astype("int64")

    if len(fg) < budget or len(bg) < budget:
        raise ValueError(f"insufficient selected patches for budget={budget}: fg={len(fg)} bg={len(bg)}")
    chosen = np.concatenate(
        [
            rng.choice(fg, size=budget, replace=False),
            rng.choice(bg, size=budget, replace=False),
        ]
    )
    rng.shuffle(chosen)
    return chosen.astype("int64")


def metric_summary(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y,
        pred,
        labels=[1],
        average="binary",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision_fg": float(precision),
        "recall_fg": float(recall),
        "f1_fg": float(f1),
    }


def build_patch_records(candidate: CandidateInfo, patch_size: int) -> list[PatchRecord]:
    x0, y0, x1, y1 = candidate.bbox_level0
    rows = int(math.ceil((y1 - y0) / patch_size))
    cols = int(math.ceil((x1 - x0) / patch_size))
    records: list[PatchRecord] = []
    for row in range(rows):
        y = y0 + row * patch_size
        height = max(1, min(patch_size, y1 - y))
        for col in range(cols):
            x = x0 + col * patch_size
            width = max(1, min(patch_size, x1 - x))
            records.append(
                PatchRecord(
                    candidate_order=candidate.candidate_order,
                    candidate_id=candidate.candidate_id,
                    row=row,
                    col=col,
                    x=x,
                    y=y,
                    width=width,
                    height=height,
                )
            )
    return records


class WsiPatchReader:
    def __init__(self, path: Path, backend: str, read_workers: int) -> None:
        self.backend = backend
        self.read_workers = int(read_workers)
        if backend == "cucim":
            if CuImage is None:
                raise RuntimeError("cuCIM is not available in this environment")
            self.slide = CuImage(str(path))
        elif backend == "openslide":
            self.slide = openslide.OpenSlide(str(path))
        else:
            raise ValueError(f"Unsupported WSI reader: {backend}")

    def read_patch(self, record: PatchRecord) -> Image.Image:
        if self.backend == "cucim":
            region = self.slide.read_region(
                location=(record.x, record.y),
                size=(record.width, record.height),
                level=0,
                num_workers=self.read_workers,
            )
            arr = np.asarray(region)
            if arr.ndim == 2:
                return Image.fromarray(arr).convert("RGB")
            return Image.fromarray(arr[:, :, :3]).convert("RGB")
        return self.slide.read_region((record.x, record.y), 0, (record.width, record.height)).convert("RGB")

    def close(self) -> None:
        close = getattr(self.slide, "close", None)
        if callable(close):
            close()


def read_patch(slide: Any, record: PatchRecord) -> Image.Image:
    if hasattr(slide, "read_patch"):
        return slide.read_patch(record)
    return slide.read_region((record.x, record.y), 0, (record.width, record.height)).convert("RGB")


def extract_unselected_features(
    slide: openslide.OpenSlide,
    records: list[PatchRecord],
    extractor: FeatureExtractor,
    cache_path: Path,
    resume: bool,
) -> tuple[np.ndarray, list[PatchRecord], dict[str, Any]]:
    if resume and cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as data:
            if str(data["model_backend"]) == extractor.backend and str(data["model_name"]) == extractor.model_name:
                loaded_records = [
                    PatchRecord(
                        candidate_order=int(order),
                        candidate_id=str(candidate_id),
                        row=int(row),
                        col=int(col),
                        x=int(x),
                        y=int(y),
                        width=int(width),
                        height=int(height),
                    )
                    for order, candidate_id, row, col, x, y, width, height in zip(
                        data["candidate_order"],
                        data["candidate_id"],
                        data["row"],
                        data["col"],
                        data["x_level0"],
                        data["y_level0"],
                        data["width_level0"],
                        data["height_level0"],
                    )
                ]
                meta = {
                    "cache_reused": True,
                    "cache_path": str(cache_path.resolve()),
                    "model_backend": extractor.backend,
                    "model_name": extractor.model_name,
                    "wsi_reader": str(data["wsi_reader"]) if "wsi_reader" in data.files else "missing",
                    "read_workers": int(data["read_workers"]) if "read_workers" in data.files else -1,
                    "pipeline_mode": str(data["pipeline_mode"]) if "pipeline_mode" in data.files else "missing",
                    "extract_seconds": 0.0,
                    "patches_per_second": 0.0,
                }
                return data["features"].astype("float32"), loaded_records, meta

    started = time.perf_counter()
    features: list[np.ndarray] = []

    def infer_images(images: list[Image.Image]) -> None:
        if not images:
            return
        features.extend(list(extractor.extract_batch(images)))

    pipeline_mode = str(getattr(extractor, "pipeline_mode", "serial"))
    prefetch_queue_batches = int(getattr(extractor, "prefetch_queue_batches", 4))
    if pipeline_mode == "prefetch":
        batches: queue.Queue[list[Image.Image] | Exception | None] = queue.Queue(
            maxsize=max(1, prefetch_queue_batches)
        )

        def producer() -> None:
            try:
                images: list[Image.Image] = []
                for record in records:
                    images.append(read_patch(slide, record))
                    if len(images) >= extractor.batch_size:
                        batches.put(images)
                        images = []
                if images:
                    batches.put(images)
            except Exception as exc:
                batches.put(exc)
            finally:
                batches.put(None)

        thread = threading.Thread(target=producer, name="unselected-feature-prefetch", daemon=True)
        thread.start()
        while True:
            item = batches.get()
            if item is None:
                break
            if isinstance(item, Exception):
                thread.join(timeout=5)
                raise item
            infer_images(item)
        thread.join()
    else:
        batch_images: list[Image.Image] = []
        for record in records:
            batch_images.append(read_patch(slide, record))
            if len(batch_images) >= extractor.batch_size:
                infer_images(batch_images)
                batch_images = []
        infer_images(batch_images)

    elapsed = time.perf_counter() - started
    feature_array = np.stack(features, axis=0).astype("float32")
    wsi_reader = getattr(slide, "backend", "openslide")
    read_workers = int(getattr(slide, "read_workers", 0))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        features=feature_array,
        candidate_order=np.asarray([r.candidate_order for r in records], dtype="int64"),
        candidate_id=np.asarray([r.candidate_id for r in records]),
        row=np.asarray([r.row for r in records], dtype="int64"),
        col=np.asarray([r.col for r in records], dtype="int64"),
        x_level0=np.asarray([r.x for r in records], dtype="int64"),
        y_level0=np.asarray([r.y for r in records], dtype="int64"),
        width_level0=np.asarray([r.width for r in records], dtype="int64"),
        height_level0=np.asarray([r.height for r in records], dtype="int64"),
        model_backend=np.asarray(extractor.backend),
        model_name=np.asarray(extractor.model_name),
        wsi_reader=np.asarray(wsi_reader),
        read_workers=np.asarray(read_workers),
        pipeline_mode=np.asarray(pipeline_mode),
        created_at=np.asarray(datetime.now(timezone.utc).isoformat()),
    )
    meta = {
        "cache_reused": False,
        "cache_path": str(cache_path.resolve()),
        "model_backend": extractor.backend,
        "model_name": extractor.model_name,
        "wsi_reader": wsi_reader,
        "read_workers": read_workers,
        "pipeline_mode": pipeline_mode,
        "extract_seconds": float(elapsed),
        "patches_per_second": float(len(records) / elapsed) if elapsed > 0 else 0.0,
    }
    return feature_array, records, meta


def read_bbox_thumbnail(
    slide: openslide.OpenSlide,
    bbox: tuple[int, int, int, int],
    max_dim: int,
) -> tuple[Image.Image, float]:
    x0, y0, x1, y1 = bbox
    width = x1 - x0
    height = y1 - y0
    downsample = max(width, height) / max_dim if max(width, height) > max_dim else 1.0
    level = int(slide.get_best_level_for_downsample(downsample))
    level_downsample = float(slide.level_downsamples[level])
    read_w = max(1, int(math.ceil(width / level_downsample)))
    read_h = max(1, int(math.ceil(height / level_downsample)))
    image = slide.read_region((x0, y0), level, (read_w, read_h)).convert("RGB")
    scale = min(max_dim / max(width, height), 1.0)
    out_w = max(1, int(round(width * scale)))
    out_h = max(1, int(round(height * scale)))
    if image.size != (out_w, out_h):
        image = image.resize((out_w, out_h), Image.Resampling.LANCZOS)
    return image, scale


def draw_prediction_grid_panel(
    slide: openslide.OpenSlide,
    candidate: CandidateInfo,
    records: list[PatchRecord],
    pred_by_index: dict[int, int],
    prob_by_index: dict[int, float],
    *,
    title: str,
    max_dim: int,
) -> Image.Image:
    base, scale = read_bbox_thumbnail(slide, candidate.bbox_level0, max_dim)
    rgba = base.convert("RGBA")
    overlay = Image.new("RGBA", rgba.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    x0, y0, _x1, _y1 = candidate.bbox_level0
    for local_idx, record in enumerate(records):
        pred = int(pred_by_index[local_idx])
        prob = float(prob_by_index[local_idx])
        if pred:
            alpha = int(45 + 95 * max(0.0, min(1.0, prob)))
            fill = (34, 197, 94, alpha)
        else:
            alpha = int(25 + 65 * max(0.0, min(1.0, 1.0 - prob)))
            fill = (239, 68, 68, alpha)
        rect = [
            int(round((record.x - x0) * scale)),
            int(round((record.y - y0) * scale)),
            int(round((record.x + record.width - x0) * scale)),
            int(round((record.y + record.height - y0) * scale)),
        ]
        draw.rectangle(rect, fill=fill, outline=(35, 35, 35, 150), width=2)
    combined = Image.alpha_composite(rgba, overlay).convert("RGB")
    top_pad = 62
    caption_pad = 54
    title_font = get_font(24, bold=True)
    small_font = get_font(16)
    measure = ImageDraw.Draw(Image.new("RGB", (1, 1), "white"))
    fg = sum(int(pred_by_index[i]) for i in range(len(records)))
    avg_prob = float(np.mean([prob_by_index[i] for i in range(len(records))])) if records else float("nan")
    stat_text = f"predicted FG {fg}/{len(records)} patches | mean p(FG)={avg_prob:.3f}"
    caption = "green=probe FG, red=probe BG; exact detector source bbox lattice"
    panel_w = max(
        combined.width,
        460,
        measure.textbbox((0, 0), title, font=title_font)[2] + 24,
        measure.textbbox((0, 0), stat_text, font=small_font)[2] + 24,
        measure.textbbox((0, 0), caption, font=small_font)[2] + 24,
    )
    panel = Image.new("RGB", (panel_w, combined.height + top_pad + caption_pad), "white")
    d = ImageDraw.Draw(panel)
    d.text((12, 12), title, fill=(0, 0, 0), font=title_font)
    d.text(
        (12, 40),
        stat_text,
        fill=(45, 45, 45),
        font=small_font,
    )
    panel.paste(combined, ((panel_w - combined.width) // 2, top_pad))
    d.text(
        (12, top_pad + combined.height + 14),
        caption,
        fill=(65, 65, 65),
        font=small_font,
    )
    return panel


def draw_selected_label_panel(
    slide: openslide.OpenSlide,
    rows: list[dict[str, str]],
    labels: np.ndarray,
    record_indices: np.ndarray,
    *,
    title: str,
    max_dim: int,
    sampled_by_record_index: dict[int, int] | None = None,
) -> Image.Image:
    x0 = min(int(r["x_level0"]) for r in rows)
    y0 = min(int(r["y_level0"]) for r in rows)
    x1 = max(int(r["x_level0"]) + int(r["width_level0"]) for r in rows)
    y1 = max(int(r["y_level0"]) + int(r["height_level0"]) for r in rows)
    bbox = (x0, y0, x1, y1)
    base, scale = read_bbox_thumbnail(slide, bbox, max_dim)
    label_by_record_index = {int(record_idx): int(label) for record_idx, label in zip(record_indices.tolist(), labels.tolist())}
    rgba = base.convert("RGBA")
    overlay = Image.new("RGBA", rgba.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for row in rows:
        local_idx = int(row["_local_record_index"])
        value = label_by_record_index[local_idx]
        fill = (34, 197, 94, 95) if value else (239, 68, 68, 45)
        px = int(row["x_level0"])
        py = int(row["y_level0"])
        pw = int(row["width_level0"])
        ph = int(row["height_level0"])
        rect = [
            int(round((px - x0) * scale)),
            int(round((py - y0) * scale)),
            int(round((px + pw - x0) * scale)),
            int(round((py + ph - y0) * scale)),
        ]
        outline = (35, 35, 35, 150)
        width = 2
        if sampled_by_record_index and local_idx in sampled_by_record_index:
            outline = (250, 204, 21, 255) if sampled_by_record_index[local_idx] == 1 else (56, 189, 248, 255)
            width = 5
        draw.rectangle(rect, fill=fill, outline=outline, width=width)
    combined = Image.alpha_composite(rgba, overlay).convert("RGB")
    top_pad = 54
    title_font = get_font(22, bold=True)
    measure = ImageDraw.Draw(Image.new("RGB", (1, 1), "white"))
    panel_w = max(combined.width, 360, measure.textbbox((0, 0), title, font=title_font)[2] + 24)
    panel = Image.new("RGB", (panel_w, combined.height + top_pad), "white")
    d = ImageDraw.Draw(panel)
    d.text((12, 12), title, fill=(0, 0, 0), font=title_font)
    panel.paste(combined, ((panel_w - combined.width) // 2, top_pad))
    return panel


def candidate_thumbnail_path(detector_case_dir: Path, candidate: CandidateInfo) -> Path:
    thumb = (
        detector_case_dir
        / "intermediate_stage_artifacts/stage7_comparative_thumbnail_filter/thumbnail_crops"
        / f"{candidate.candidate_order:02d}_candidate_{candidate.candidate_order:02d}.png"
    )
    if thumb.exists():
        return thumb
    return candidate.crop_path


def draw_test_crop_thumbnail_panel(
    detector_case_dir: Path,
    candidate: CandidateInfo,
    stats: dict[str, Any],
    *,
    max_dim: int,
) -> Image.Image:
    image_path = candidate_thumbnail_path(detector_case_dir, candidate)
    image = Image.open(image_path).convert("RGB")
    image = resize_to_fit(image, max_dim, max_dim)
    title = f"test detector ID {candidate.candidate_order}"
    stat_text = (
        f"thumbnail crop | predicted FG {stats.get('pred_fg', 'n/a')}/{stats.get('patch_count', 'n/a')} "
        f"| mean p(FG)={float(stats.get('mean_prob_fg', float('nan'))):.3f}"
        if stats
        else "thumbnail crop"
    )
    title_font = get_font(24, bold=True)
    small_font = get_font(16)
    measure = ImageDraw.Draw(Image.new("RGB", (1, 1), "white"))
    panel_w = max(
        image.width + 20,
        430,
        measure.textbbox((0, 0), title, font=title_font)[2] + 24,
        measure.textbbox((0, 0), stat_text, font=small_font)[2] + 24,
    )
    panel_h = image.height + 98
    panel = Image.new("RGB", (panel_w, panel_h), "white")
    d = ImageDraw.Draw(panel)
    d.text((12, 10), title, fill=(0, 0, 0), font=title_font)
    d.text((12, 40), stat_text, fill=(45, 45, 45), font=small_font)
    x = (panel_w - image.width) // 2
    y = 72
    panel.paste(image, (x, y))
    d.rectangle([x, y, x + image.width - 1, y + image.height - 1], outline=(220, 38, 38), width=5)
    return panel


def resize_to_width(image: Image.Image, max_width: int) -> Image.Image:
    if image.width <= max_width:
        return image
    scale = max_width / image.width
    return image.resize((max_width, max(1, int(round(image.height * scale)))), Image.Resampling.LANCZOS)


def resize_to_fit(image: Image.Image, max_width: int, max_height: int) -> Image.Image:
    scale = min(max_width / image.width, max_height / image.height, 1.0)
    if scale >= 1.0:
        return image
    return image.resize(
        (max(1, int(round(image.width * scale))), max(1, int(round(image.height * scale)))),
        Image.Resampling.LANCZOS,
    )


def make_contact_sheet(panels: list[Image.Image], *, cols: int, gap: int, bg: str = "white") -> Image.Image:
    if not panels:
        return Image.new("RGB", (800, 160), bg)
    col_w = max(panel.width for panel in panels)
    rows: list[list[Image.Image]] = [panels[i : i + cols] for i in range(0, len(panels), cols)]
    row_heights = [max(panel.height for panel in row) for row in rows]
    width = cols * col_w + (cols - 1) * gap
    height = sum(row_heights) + (len(rows) - 1) * gap
    sheet = Image.new("RGB", (width, height), bg)
    y = 0
    for row, row_h in zip(rows, row_heights):
        x = 0
        for panel in row:
            sheet.paste(panel, (x + (col_w - panel.width) // 2, y))
            x += col_w + gap
        y += row_h + gap
    return sheet


def draw_detector_overview(
    detector_case_dir: Path,
    detections_json: dict[str, Any],
    selected_ids: set[int],
    max_width: int,
) -> Image.Image:
    thumb_path = (
        detector_case_dir
        / "intermediate_stage_artifacts/stage1_thumbnail_detection/thumbnail.png"
    )
    if thumb_path.exists():
        base = Image.open(thumb_path).convert("RGB")
    else:
        base = Image.open(detector_case_dir / "final_detected_bboxes.png").convert("RGB")
    draw = ImageDraw.Draw(base)
    w, h = base.size
    for det in detections_json["detections"]:
        order = int(det["source_candidate_order"])
        y0, x0, y1, x1 = [float(v) for v in det["box_2d"]]
        color = (22, 163, 74) if order in selected_ids else (220, 38, 38)
        rect = [
            int(round(x0 / 1000.0 * w)),
            int(round(y0 / 1000.0 * h)),
            int(round(x1 / 1000.0 * w)),
            int(round(y1 / 1000.0 * h)),
        ]
        draw.rectangle(rect, outline=color, width=5)
        label = str(order)
        font = get_font(28, bold=True)
        tb = draw.textbbox((0, 0), label, font=font)
        lx = rect[0] + 5
        ly = rect[1] + 5
        draw.rectangle([lx, ly, lx + tb[2] - tb[0] + 10, ly + tb[3] - tb[1] + 8], fill="white", outline=color, width=3)
        draw.text((lx + 5, ly + 3), label, fill=color, font=font)
    return resize_to_width(base, max_width)


def make_page(
    title: str,
    subtitle: str,
    body: Image.Image,
    *,
    footer: str | None = None,
    page_width: int = 1700,
) -> Image.Image:
    body = resize_to_fit(body, page_width - 100, 2100)
    page_h = 170 + body.height + (70 if footer else 35)
    page = Image.new("RGB", (page_width, page_h), "white")
    d = ImageDraw.Draw(page)
    d.text((50, 34), title, fill=(0, 0, 0), font=get_font(36, bold=True))
    y = draw_wrapped_text(d, (50, 82), subtitle, font=get_font(21), fill=(45, 45, 45), width_chars=122)
    page.paste(body, ((page_width - body.width) // 2, max(142, y + 24)))
    if footer:
        draw_wrapped_text(
            d,
            (50, page.height - 54),
            footer,
            font=get_font(15),
            fill=(80, 80, 80),
            width_chars=150,
        )
    return page


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "git_status.txt").write_text(git_status_short())

    detector_json_path = args.detector_case_dir / "detections.json"
    detections_json = json.loads(detector_json_path.read_text())
    selector_row = selected_manifest_row(args.selector_manifest, args.case_id)
    selected_ids = set(parse_box_ids(selector_row.get("verifier_selected_box_ids", "")))
    if not selected_ids:
        raise ValueError(f"No verifier_selected_box_ids found for case {args.case_id}")
    candidates = load_candidate_infos(args.detector_case_dir, selected_ids)
    unselected_candidates = [candidate for candidate in candidates if not candidate.selected_for_train]
    selected_candidates = [candidate for candidate in candidates if candidate.selected_for_train]
    if not unselected_candidates:
        raise ValueError("No unselected detector candidates to score")

    selected_rows_all = read_csv(args.probe_run_dir / "manifests/selected_patch_manifest.csv")
    selected_rows = [dict(row) for row in selected_rows_all if row["case_id"] == args.case_id]
    if not selected_rows:
        raise ValueError(f"No selected patch rows for case_id={args.case_id}")
    for idx, row in enumerate(selected_rows):
        row["_local_record_index"] = str(idx)

    with np.load(feature_path(args.probe_run_dir, args.case_id), allow_pickle=False) as data:
        selected_features = data["features"].astype("float32")
        selected_labels = data["labels"].astype("int64")
        selected_bbox_indices = data["bbox_indices"].astype("int64")
        selected_record_indices = data["record_indices"].astype("int64")
        feature_backend = str(data["model_backend"])
        feature_model = str(data["model_name"])
    if args.model_backend is None:
        args.model_backend = feature_backend
    if args.model_name is None:
        args.model_name = feature_model

    train_idx = select_train_indices(selected_labels, selected_bbox_indices, args)
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, solver="liblinear", random_state=int(args.sample_seed)),
    )
    clf.fit(selected_features[train_idx], selected_labels[train_idx])
    train_prob = clf.predict_proba(selected_features[train_idx])[:, 1]
    train_pred = (train_prob >= 0.5).astype("int64")
    train_metrics = metric_summary(selected_labels[train_idx], train_pred)

    wsi_path = resolve_wsi_path(selected_rows, selector_row, detections_json)
    slide = openslide.OpenSlide(str(wsi_path))
    try:
        extractor = FeatureExtractor(args)
        unselected_records: list[PatchRecord] = []
        for candidate in unselected_candidates:
            unselected_records.extend(build_patch_records(candidate, int(args.patch_size)))
        unselected_features, unselected_records, feature_cache_meta = extract_unselected_features(
            slide,
            unselected_records,
            extractor,
            args.output_dir / "features/unselected_detector_candidates_features.npz",
            bool(args.resume),
        )
        unselected_prob = clf.predict_proba(unselected_features)[:, 1]
        unselected_pred = (unselected_prob >= 0.5).astype("int64")

        prediction_rows: list[dict[str, Any]] = []
        by_candidate_records: dict[int, list[tuple[int, PatchRecord]]] = defaultdict(list)
        for idx, record in enumerate(unselected_records):
            by_candidate_records[record.candidate_order].append((idx, record))
            prediction_rows.append(
                {
                    "case_id": args.case_id,
                    "candidate_order": record.candidate_order,
                    "candidate_id": record.candidate_id,
                    "row": record.row,
                    "col": record.col,
                    "x_level0": record.x,
                    "y_level0": record.y,
                    "width_level0": record.width,
                    "height_level0": record.height,
                    "prob_fg": float(unselected_prob[idx]),
                    "pred_fg": int(unselected_pred[idx]),
                }
            )
        write_csv(args.output_dir / "unselected_patch_predictions.csv", prediction_rows)

        candidate_rows: list[dict[str, Any]] = []
        candidate_stats: dict[int, dict[str, Any]] = {}
        for candidate in candidates:
            patches = [i for i, rec in enumerate(unselected_records) if rec.candidate_order == candidate.candidate_order]
            row: dict[str, Any] = {
                "case_id": args.case_id,
                "candidate_order": candidate.candidate_order,
                "candidate_id": candidate.candidate_id,
                "selected_for_train": candidate.selected_for_train,
                "bbox_level0": list(candidate.bbox_level0),
            }
            if patches:
                probs = unselected_prob[patches]
                preds = unselected_pred[patches]
                row.update(
                    {
                        "patch_count": int(len(patches)),
                        "pred_fg": int(preds.sum()),
                        "pred_bg": int(len(preds) - preds.sum()),
                        "pred_fg_fraction": float(preds.mean()),
                        "mean_prob_fg": float(probs.mean()),
                    }
                )
                candidate_stats[candidate.candidate_order] = row
            candidate_rows.append(row)
        write_csv(args.output_dir / "candidate_transfer_summary.csv", candidate_rows)

        coord_to_candidate = {candidate.bbox_level0: candidate.candidate_order for candidate in candidates}
        selected_by_bbox: dict[int, list[dict[str, str]]] = defaultdict(list)
        for row in selected_rows:
            selected_by_bbox[int(row["bbox_index"])].append(row)

        sampled_by_record_index = {
            int(selected_record_indices[feature_idx]): int(selected_labels[feature_idx])
            for feature_idx in train_idx.tolist()
        }
        selected_panels: list[Image.Image] = []
        for bbox_index in sorted(selected_by_bbox):
            rows = selected_by_bbox[bbox_index]
            coord = bbox_name_to_level0(rows[0]["bbox_name"])
            candidate_order = coord_to_candidate.get(coord or (-1, -1, -1, -1), -1)
            panel = draw_selected_label_panel(
                slide,
                rows,
                selected_labels,
                selected_record_indices,
                title=f"selected train bbox {bbox_index} | detector ID {candidate_order}",
                max_dim=520,
                sampled_by_record_index=sampled_by_record_index,
            )
            selected_panels.append(panel)
        selected_sheet = make_contact_sheet(selected_panels, cols=3, gap=24)

        test_thumbnail_panels: list[Image.Image] = []
        for candidate in unselected_candidates:
            panel = draw_test_crop_thumbnail_panel(
                args.detector_case_dir,
                candidate,
                candidate_stats.get(candidate.candidate_order, {}),
                max_dim=460,
            )
            panel_path = args.output_dir / f"unselected_candidate_{candidate.candidate_order:02d}_thumbnail.png"
            panel.save(panel_path)
            test_thumbnail_panels.append(panel)
        test_thumbnail_sheet = make_contact_sheet(test_thumbnail_panels, cols=3, gap=28)

        prediction_panels: list[Image.Image] = []
        page_images: list[Image.Image] = []
        for candidate in unselected_candidates:
            pairs = by_candidate_records[candidate.candidate_order]
            records = [record for _global_idx, record in pairs]
            pred_by_local = {local: int(unselected_pred[global_idx]) for local, (global_idx, _record) in enumerate(pairs)}
            prob_by_local = {local: float(unselected_prob[global_idx]) for local, (global_idx, _record) in enumerate(pairs)}
            panel = draw_prediction_grid_panel(
                slide,
                candidate,
                records,
                pred_by_local,
                prob_by_local,
                title=f"unselected detector ID {candidate.candidate_order}",
                max_dim=620,
            )
            panel_path = args.output_dir / f"unselected_candidate_{candidate.candidate_order:02d}_prediction.png"
            panel.save(panel_path)
            prediction_panels.append(panel)
        prediction_sheet = make_contact_sheet(prediction_panels, cols=2, gap=34)

        overview = draw_detector_overview(args.detector_case_dir, detections_json, selected_ids, max_width=1240)
        summary_text = (
            f"Verifier-selected training detector IDs: {sorted(selected_ids)}. "
            f"Unselected detector IDs scored by the trained probe: {[c.candidate_order for c in unselected_candidates]}. "
            f"Train policy: {args.train_policy}; train patches={len(train_idx)} "
            f"(FG={(selected_labels[train_idx] == 1).sum()}, BG={(selected_labels[train_idx] == 0).sum()}). "
            f"Apparent train FG F1={train_metrics['f1_fg']:.3f}."
        )
        overview_page = make_page(
            "Selected-to-Unselected DINOv3 Probe Transfer",
            f"{args.case_id}. {summary_text}",
            overview,
            footer="Overview colors: green boxes trained the probe; red boxes are detector candidates scored only by transfer.",
        )
        selected_page = make_page(
            "2. Training crops: verifier-selected detector bboxes",
            (
                "These Stage 7 pseudo-label grids are the source supervision. "
                "Yellow borders mark sampled FG train patches; cyan borders mark sampled BG train patches."
            ),
            selected_sheet,
            footer=(
                "Green patches are Stage 7 foreground labels; red patches are Stage 7 background labels. "
                "Sampled train patches are outlined in yellow for FG and cyan for BG."
            ),
        )
        test_thumbnail_page = make_page(
            "1. Test crops: unselected detector bbox thumbnails",
            "These are the detector-pipeline thumbnail-level views of the crops scored by transfer.",
            test_thumbnail_sheet,
            footer="Red borders indicate crops that were not used for training and are scored only by the fitted probe.",
        )
        transfer_page = make_page(
            "3. Transfer to unselected detector bboxes",
            "The fitted DINOv3-small linear probe is applied to the remaining detector candidates in the same WSI. These are predictions, not reviewed ground truth.",
            prediction_sheet,
            footer="Green patches are probe foreground predictions; red patches are probe background predictions.",
        )
        page_images.extend([overview_page, test_thumbnail_page, selected_page, transfer_page])

        for idx, page in enumerate(page_images, start=1):
            page.save(args.output_dir / f"page_{idx:02d}.png")

        pdf_path = args.output_dir / f"{args.case_id}_selected_to_unselected_probe_transfer.pdf"
        page_images[0].save(pdf_path, save_all=True, append_images=page_images[1:])
    finally:
        slide.close()

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "ticket": args.ticket,
        "git_commit": git_commit(),
        "case_id": args.case_id,
        "wsi_path": str(wsi_path),
        "probe_run_dir": str(args.probe_run_dir.resolve()),
        "selector_manifest": str(args.selector_manifest.resolve()),
        "detector_case_dir": str(args.detector_case_dir.resolve()),
        "selected_detector_ids_for_train": sorted(selected_ids),
        "unselected_detector_ids_scored": [candidate.candidate_order for candidate in unselected_candidates],
        "train_policy": args.train_policy,
        "budget_per_class": args.budget_per_class if args.train_policy in {"balanced-budget", "per-crop-balanced"} else None,
        "sample_seed": args.sample_seed,
        "train_patch_count": int(len(train_idx)),
        "train_fg": int((selected_labels[train_idx] == 1).sum()),
        "train_bg": int((selected_labels[train_idx] == 0).sum()),
        "train_selected_bbox_indices": sorted(set(int(x) for x in selected_bbox_indices[train_idx].tolist())),
        "train_patch_count_by_selected_bbox": {
            str(bbox_index): int((selected_bbox_indices[train_idx] == bbox_index).sum())
            for bbox_index in sorted(set(int(x) for x in selected_bbox_indices.tolist()))
        },
        "train_metrics_apparent": train_metrics,
        "feature_backend": feature_backend,
        "feature_model": feature_model,
        "unselected_feature_cache": feature_cache_meta,
        "candidate_summary_csv": str((args.output_dir / "candidate_transfer_summary.csv").resolve()),
        "patch_predictions_csv": str((args.output_dir / "unselected_patch_predictions.csv").resolve()),
        "pdf": str(pdf_path.resolve()),
        "preview_pages": [str((args.output_dir / f"page_{idx:02d}.png").resolve()) for idx in range(1, len(page_images) + 1)],
    }
    write_json(args.output_dir / "summary.json", summary)
    (args.output_dir / "reproduction.txt").write_text(
        "\n".join(
            [
                "PER-250 Per-WSI DINOv3 Selected-to-Unselected Transfer Demo",
                "==========================================================",
                "",
                f"Created: {summary['created_at']}",
                f"Ticket: {args.ticket}",
                f"Git commit: {summary['git_commit']}",
                f"Dirty git state captured in: {args.output_dir / 'git_status.txt'}",
                "",
                "Command:",
                " ".join(
                    shlex.quote(part)
                    for part in [
                        "python",
                        "scripts/build_per_wsi_probe_unselected_transfer_demo.py",
                        "--probe-run-dir",
                        str(args.probe_run_dir),
                        "--selector-manifest",
                        str(args.selector_manifest),
                        "--detector-case-dir",
                        str(args.detector_case_dir),
                        "--case-id",
                        args.case_id,
                        "--output-dir",
                        str(args.output_dir),
                        "--train-policy",
                        args.train_policy,
                        "--patch-size",
                        str(args.patch_size),
                        "--max-dim",
                        str(args.max_dim),
                        "--device",
                        str(args.device),
                        "--batch-size",
                        str(args.batch_size),
                    ]
                ),
                "",
                "Inputs:",
                f"- Probe run: {args.probe_run_dir.resolve()}",
                f"- Selected feature cache: {feature_path(args.probe_run_dir, args.case_id).resolve()}",
                f"- Selected patch manifest: {(args.probe_run_dir / 'manifests/selected_patch_manifest.csv').resolve()}",
                f"- Selector manifest: {args.selector_manifest.resolve()}",
                f"- Detector case dir: {args.detector_case_dir.resolve()}",
                f"- WSI path: {wsi_path}",
                "",
                "Split semantics:",
                f"- Train on verifier-selected detector candidate IDs: {sorted(selected_ids)}.",
                f"- Apply the fitted probe to unselected detector candidate IDs: {[c.candidate_order for c in unselected_candidates]}.",
                "- Supervision is the selected crops' Stage 7 pseudo-label grid only.",
                "- Unselected crops are predictions only; no precision/recall is reported for them here.",
                "",
                "Outputs:",
                f"- PDF: {pdf_path.resolve()}",
                f"- Candidate summary: {(args.output_dir / 'candidate_transfer_summary.csv').resolve()}",
                f"- Patch predictions: {(args.output_dir / 'unselected_patch_predictions.csv').resolve()}",
                f"- Feature cache: {(args.output_dir / 'features/unselected_detector_candidates_features.npz').resolve()}",
                "",
            ]
        )
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
