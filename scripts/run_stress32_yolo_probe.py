#!/usr/bin/env python3
"""Run YOLO conf=0.01 plus pooled DINOv3 FG/BG probe on stress-32 WSIs."""

from __future__ import annotations

import argparse
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
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from train_per_wsi_dinov3_fg_bg_probe import FeatureExtractor, WsiPatchReader, package_versions  # noqa: E402


DEFAULT_STRESS_DATASET_ROOT = (
    Path("/data2/vj724/wsi-agents")
    / "vlm_gt_seg_comparison_experiment/dataset_thumbnails_harder_jones_leica"
)
DEFAULT_STRESS_PIPELINE_ROOT = (
    Path("/data2/vj724/wsi-agents")
    / "new_vlm_gt_preds/leica_hard_jones_evg_qwen_8b_zero_shot"
)
DEFAULT_TRAIN_FEATURE_DIR = REPO_ROOT / "runs/scale500_selected_dinov3small_features_sample100_per_wsi_v1"
DEFAULT_YOLO_WEIGHTS = (
    REPO_ROOT
    / "runs/detector_distillation/yolo_scale500_per248_v1"
    / "yolo11n_img1024_e60_stainjitter/ultralytics/train/weights/best.pt"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "runs/stress32_yolo_dinov3_probe_v1"
DEFAULT_DINOV3_SMALL = "facebook/dinov3-vits16-pretrain-lvd1689m"
DEFAULT_DINOV2_SMALL = "vit_small_patch14_dinov2"


@dataclass(frozen=True)
class StressCase:
    case_id: str
    scanner: str
    stain: str
    wsi_path: Path
    thumbnail_path: Path
    gt_mask_path: Path
    wsi_width: int
    wsi_height: int
    thumbnail_width: int
    thumbnail_height: int
    case_dir: Path


@dataclass(frozen=True)
class Detection:
    case_id: str
    detection_id: int
    conf: float
    cls: int
    x0_thumb: float
    y0_thumb: float
    x1_thumb: float
    y1_thumb: float
    x0: int
    y0: int
    x1: int
    y1: int


@dataclass(frozen=True)
class PatchRecord:
    case_id: str
    detection_id: int
    detection_ids: tuple[int, ...]
    row: int
    col: int
    x: int
    y: int
    width: int
    height: int

    @property
    def record_id(self) -> str:
        dets = ".".join(str(x) for x in self.detection_ids)
        return f"{self.case_id}|det{dets}|r{self.row}c{self.col}|{self.x}_{self.y}_{self.width}_{self.height}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["manifest", "detect", "score", "all"], default="all")
    parser.add_argument("--stress-dataset-root", type=Path, default=DEFAULT_STRESS_DATASET_ROOT)
    parser.add_argument("--stress-pipeline-root", type=Path, default=DEFAULT_STRESS_PIPELINE_ROOT)
    parser.add_argument("--train-feature-dir", type=Path, default=DEFAULT_TRAIN_FEATURE_DIR)
    parser.add_argument("--yolo-weights", type=Path, default=DEFAULT_YOLO_WEIGHTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ticket", default="PER-250")
    parser.add_argument("--run-id", default="harder_jones_leica_manual")
    parser.add_argument("--scanner", default="leica")
    parser.add_argument("--stain", default="jones")
    parser.add_argument("--case-ids", default="", help="Comma-separated stress case IDs to process; default uses all seeded stress cases.")
    parser.add_argument("--case-limit", type=int, default=None)
    parser.add_argument("--conf", type=float, default=0.01)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--yolo-device", default="")
    parser.add_argument("--suppress-contained", action="store_true", default=True)
    parser.add_argument("--no-suppress-contained", dest="suppress_contained", action="store_false")
    parser.add_argument(
        "--containment-threshold",
        type=float,
        default=0.90,
        help="Suppress lower-confidence YOLO boxes whose area is this covered by an already-kept box.",
    )
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--probe-threshold", type=float, default=0.50)
    parser.add_argument("--model-backend", choices=["transformers", "timm"], default="transformers")
    parser.add_argument("--model-name", default=DEFAULT_DINOV3_SMALL)
    parser.add_argument("--fallback-model-name", default=DEFAULT_DINOV2_SMALL)
    parser.add_argument("--allow-timm-fallback", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--pipeline-mode", choices=["serial", "prefetch"], default="serial")
    parser.add_argument("--prefetch-queue-batches", type=int, default=4)
    parser.add_argument("--input-size", type=int, default=None)
    parser.add_argument("--wsi-reader", choices=["cucim", "openslide"], default="cucim")
    parser.add_argument("--read-workers", type=int, default=16)
    parser.add_argument("--sample-seed", type=int, default=250)
    parser.add_argument("--holdout-frac", type=float, default=0.20)
    parser.add_argument("--max-crop-panels-per-case", type=int, default=6)
    parser.add_argument("--max-overview-width", type=int, default=1450)
    parser.add_argument("--max-grid-dim", type=int, default=620)
    parser.add_argument("--max-patches-per-case", type=int, default=None)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    return parser.parse_args()


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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


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


def resize_to_fit(image: Image.Image, max_w: int, max_h: int) -> Image.Image:
    scale = min(max_w / image.width, max_h / image.height, 1.0)
    if scale >= 1.0:
        return image
    return image.resize((max(1, int(round(image.width * scale))), max(1, int(round(image.height * scale)))), Image.Resampling.LANCZOS)


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


def make_page(title: str, subtitle: str, body: Image.Image, footer: str = "", page_width: int = 1700) -> Image.Image:
    body = resize_to_fit(body, page_width - 100, 2200)
    page_h = 170 + body.height + (88 if footer else 34)
    page = Image.new("RGB", (page_width, page_h), "white")
    draw = ImageDraw.Draw(page)
    draw.text((50, 34), title, fill=(0, 0, 0), font=get_font(36, bold=True))
    y = draw_wrapped_text(draw, (50, 84), subtitle, font=get_font(21), fill=(45, 45, 45), width_chars=122)
    page.paste(body, ((page_width - body.width) // 2, max(144, y + 22)))
    if footer:
        draw_wrapped_text(draw, (50, page.height - 70), footer, font=get_font(16), fill=(80, 80, 80), width_chars=150)
    return page


def make_contact_sheet(panels: list[Image.Image], cols: int, gap: int) -> Image.Image:
    if not panels:
        return Image.new("RGB", (900, 180), "white")
    col_w = max(panel.width for panel in panels)
    rows = [panels[i : i + cols] for i in range(0, len(panels), cols)]
    row_heights = [max(panel.height for panel in row) for row in rows]
    out = Image.new("RGB", (cols * col_w + (cols - 1) * gap, sum(row_heights) + (len(rows) - 1) * gap), "white")
    y = 0
    for row, row_h in zip(rows, row_heights):
        x = 0
        for panel in row:
            out.paste(panel, (x + (col_w - panel.width) // 2, y))
            x += col_w + gap
        y += row_h + gap
    return out


def seeded_case_ids(pipeline_root: Path, run_id: str) -> list[str]:
    paths = sorted(pipeline_root.glob(f"anon_*/{run_id}/pipeline_metadata.json"))
    if not paths:
        raise FileNotFoundError(f"No stress pipeline metadata under {pipeline_root}/*/{run_id}")
    return [path.parts[-3] for path in paths]


def load_stress_cases(args: argparse.Namespace) -> list[StressCase]:
    cases: list[StressCase] = []
    requested = {part.strip() for part in str(args.case_ids).split(",") if part.strip()}
    for case_id in seeded_case_ids(args.stress_pipeline_root, args.run_id):
        if requested and case_id not in requested:
            continue
        case_dir = args.stress_dataset_root / args.scanner / args.stain / case_id
        meta_path = case_dir / "case_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing case_meta.json for {case_id}: {meta_path}")
        meta = json.loads(meta_path.read_text())
        wsi_path = Path(meta["wsi_path"])
        if not wsi_path.exists():
            raise FileNotFoundError(f"Missing WSI for {case_id}: {wsi_path}")
        thumb = case_dir / "thumbnail.png"
        mask = case_dir / "mask.npy"
        if not thumb.exists() or not mask.exists():
            raise FileNotFoundError(f"Missing thumbnail or GT mask for {case_id}: {case_dir}")
        cases.append(
            StressCase(
                case_id=case_id,
                scanner=str(meta.get("scanner", args.scanner)),
                stain=str(meta.get("stain", args.stain)),
                wsi_path=wsi_path,
                thumbnail_path=thumb,
                gt_mask_path=mask,
                wsi_width=int(meta["wsi_dimensions"]["width"]),
                wsi_height=int(meta["wsi_dimensions"]["height"]),
                thumbnail_width=int(meta["thumbnail_dimensions"]["width"]),
                thumbnail_height=int(meta["thumbnail_dimensions"]["height"]),
                case_dir=case_dir,
            )
        )
    cases.sort(key=lambda c: c.case_id)
    if args.case_limit is not None:
        cases = cases[: int(args.case_limit)]
    if not cases:
        raise ValueError("No stress cases selected")
    return cases


def stress_case_rows(cases: list[StressCase]) -> list[dict[str, Any]]:
    return [
        {
            "case_id": case.case_id,
            "scanner": case.scanner,
            "stain": case.stain,
            "wsi_path": str(case.wsi_path),
            "thumbnail_path": str(case.thumbnail_path),
            "gt_mask_path": str(case.gt_mask_path),
            "wsi_width": case.wsi_width,
            "wsi_height": case.wsi_height,
            "thumbnail_width": case.thumbnail_width,
            "thumbnail_height": case.thumbnail_height,
            "case_dir": str(case.case_dir),
        }
        for case in cases
    ]


def stage_manifest(args: argparse.Namespace) -> list[StressCase]:
    cases = load_stress_cases(args)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "stress32_manifest.csv", stress_case_rows(cases))
    write_json(
        out / "stress32_manifest_summary.json",
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "case_count": len(cases),
            "stress_dataset_root": str(args.stress_dataset_root.resolve()),
            "stress_pipeline_root": str(args.stress_pipeline_root.resolve()),
            "scanner": args.scanner,
            "stain": args.stain,
            "run_id": args.run_id,
            "case_ids": [case.case_id for case in cases],
        },
    )
    return cases


def run_yolo_detect(args: argparse.Namespace, cases: list[StressCase]) -> list[dict[str, Any]]:
    from ultralytics import YOLO

    model = YOLO(str(args.yolo_weights))
    rows: list[dict[str, Any]] = []
    for case in cases:
        result = model.predict(
            source=str(case.thumbnail_path),
            imgsz=int(args.imgsz),
            conf=float(args.conf),
            iou=float(args.iou),
            max_det=int(args.max_det),
            device=args.yolo_device or None,
            verbose=False,
        )[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or boxes.xyxy is None:
            continue
        xyxy = boxes.xyxy.detach().cpu().numpy()
        confs = boxes.conf.detach().cpu().numpy()
        classes = boxes.cls.detach().cpu().numpy() if boxes.cls is not None else np.zeros(len(xyxy))
        sx = case.wsi_width / case.thumbnail_width
        sy = case.wsi_height / case.thumbnail_height
        order = np.argsort(-confs)
        for det_idx, source_idx in enumerate(order.tolist(), start=1):
            x0t, y0t, x1t, y1t = [float(v) for v in xyxy[source_idx]]
            x0t = max(0.0, min(float(case.thumbnail_width), x0t))
            x1t = max(0.0, min(float(case.thumbnail_width), x1t))
            y0t = max(0.0, min(float(case.thumbnail_height), y0t))
            y1t = max(0.0, min(float(case.thumbnail_height), y1t))
            x0 = max(0, min(case.wsi_width - 1, int(math.floor(x0t * sx))))
            x1 = max(0, min(case.wsi_width, int(math.ceil(x1t * sx))))
            y0 = max(0, min(case.wsi_height - 1, int(math.floor(y0t * sy))))
            y1 = max(0, min(case.wsi_height, int(math.ceil(y1t * sy))))
            if x1 <= x0 or y1 <= y0:
                continue
            rows.append(
                {
                    "case_id": case.case_id,
                    "detection_id": det_idx,
                    "yolo_conf": float(confs[source_idx]),
                    "yolo_cls": int(classes[source_idx]),
                    "x0_thumb": x0t,
                    "y0_thumb": y0t,
                    "x1_thumb": x1t,
                    "y1_thumb": y1t,
                    "x0_level0": x0,
                    "y0_level0": y0,
                    "x1_level0": x1,
                    "y1_level0": y1,
                    "width_level0": x1 - x0,
                    "height_level0": y1 - y0,
                    "thumbnail_path": str(case.thumbnail_path),
                    "wsi_path": str(case.wsi_path),
                }
            )
    write_csv(args.output_dir / "yolo_detections.csv", rows)
    write_json(
        args.output_dir / "yolo_detection_summary.json",
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "case_count": len(cases),
            "detection_count": len(rows),
            "yolo_weights": str(args.yolo_weights.resolve()),
            "conf": args.conf,
            "iou": args.iou,
            "imgsz": args.imgsz,
            "max_det": args.max_det,
        },
    )
    return rows


def load_detections(path: Path) -> list[Detection]:
    detections: list[Detection] = []
    for row in read_csv(path):
        detections.append(
            Detection(
                case_id=row["case_id"],
                detection_id=int(row["detection_id"]),
                conf=float(row["yolo_conf"]),
                cls=int(row["yolo_cls"]),
                x0_thumb=float(row["x0_thumb"]),
                y0_thumb=float(row["y0_thumb"]),
                x1_thumb=float(row["x1_thumb"]),
                y1_thumb=float(row["y1_thumb"]),
                x0=int(row["x0_level0"]),
                y0=int(row["y0_level0"]),
                x1=int(row["x1_level0"]),
                y1=int(row["y1_level0"]),
            )
        )
    return detections


def detection_area(det: Detection) -> float:
    return float(max(0, det.x1 - det.x0) * max(0, det.y1 - det.y0))


def detection_intersection_area(a: Detection, b: Detection) -> float:
    x0 = max(a.x0, b.x0)
    y0 = max(a.y0, b.y0)
    x1 = min(a.x1, b.x1)
    y1 = min(a.y1, b.y1)
    return float(max(0, x1 - x0) * max(0, y1 - y0))


def suppress_contained_detections(
    detections: list[Detection],
    containment_threshold: float,
) -> tuple[list[Detection], list[dict[str, Any]]]:
    ordered = sorted(detections, key=lambda det: (-det.conf, det.detection_id))
    kept: list[Detection] = []
    suppressed: list[dict[str, Any]] = []
    for det in ordered:
        area = detection_area(det)
        suppressor: Detection | None = None
        coverage = 0.0
        if area > 0:
            for kept_det in kept:
                overlap = detection_intersection_area(det, kept_det)
                coverage = max(coverage, overlap / area)
                if overlap / area >= containment_threshold:
                    suppressor = kept_det
                    break
        if suppressor is None:
            kept.append(det)
        else:
            suppressed.append(
                {
                    "case_id": det.case_id,
                    "suppressed_detection_id": det.detection_id,
                    "suppressor_detection_id": suppressor.detection_id,
                    "suppressed_conf": det.conf,
                    "suppressor_conf": suppressor.conf,
                    "suppressed_area": area,
                    "covered_fraction": coverage,
                }
            )
    return sorted(kept, key=lambda det: det.detection_id), suppressed


def detection_rows(detections: list[Detection]) -> list[dict[str, Any]]:
    return [
        {
            "case_id": det.case_id,
            "detection_id": det.detection_id,
            "yolo_conf": det.conf,
            "yolo_cls": det.cls,
            "x0_thumb": det.x0_thumb,
            "y0_thumb": det.y0_thumb,
            "x1_thumb": det.x1_thumb,
            "y1_thumb": det.y1_thumb,
            "x0_level0": det.x0,
            "y0_level0": det.y0,
            "x1_level0": det.x1,
            "y1_level0": det.y1,
            "width_level0": det.x1 - det.x0,
            "height_level0": det.y1 - det.y0,
        }
        for det in detections
    ]


def build_patch_records(case: StressCase, detections: list[Detection], patch_size: int) -> list[PatchRecord]:
    cells: dict[tuple[int, int], set[int]] = defaultdict(set)
    for det in detections:
        row0 = max(0, int(math.floor(det.y0 / patch_size)))
        row1 = min(int(math.ceil(case.wsi_height / patch_size)), int(math.ceil(det.y1 / patch_size)))
        col0 = max(0, int(math.floor(det.x0 / patch_size)))
        col1 = min(int(math.ceil(case.wsi_width / patch_size)), int(math.ceil(det.x1 / patch_size)))
        for rr in range(row0, row1):
            y = rr * patch_size
            h = max(1, min(patch_size, case.wsi_height - y))
            for cc in range(col0, col1):
                x = cc * patch_size
                w = max(1, min(patch_size, case.wsi_width - x))
                if x < det.x1 and x + w > det.x0 and y < det.y1 and y + h > det.y0:
                    cells[(rr, cc)].add(det.detection_id)
    records: list[PatchRecord] = []
    for (rr, cc), det_ids in sorted(cells.items()):
        y = rr * patch_size
        x = cc * patch_size
        records.append(
            PatchRecord(
                case_id=case.case_id,
                detection_id=min(det_ids),
                detection_ids=tuple(sorted(det_ids)),
                row=rr,
                col=cc,
                x=x,
                y=y,
                width=max(1, min(patch_size, case.wsi_width - x)),
                height=max(1, min(patch_size, case.wsi_height - y)),
            )
        )
    return records


def load_training_features(feature_dir: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    backends: set[str] = set()
    models: set[str] = set()
    files = sorted((feature_dir / "features").glob("*_features.npz"))
    if not files:
        raise FileNotFoundError(f"No feature NPZ files under {feature_dir / 'features'}")
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            x_parts.append(data["features"].astype("float32"))
            y_parts.append(data["labels"].astype("int64"))
            backends.add(str(data["model_backend"]))
            models.add(str(data["model_name"]))
    if len(backends) != 1 or len(models) != 1:
        raise ValueError(f"Mixed feature backends/models: {backends} / {models}")
    x = np.concatenate(x_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    meta = {
        "feature_files": len(files),
        "patches": int(len(y)),
        "fg": int((y == 1).sum()),
        "bg": int((y == 0).sum()),
        "model_backend": next(iter(backends)),
        "model_name": next(iter(models)),
    }
    return x, y, meta


def fit_linear_probe(x: np.ndarray, y: np.ndarray, seed: int, holdout_frac: float) -> tuple[Any, dict[str, Any]]:
    indices = np.arange(len(y), dtype="int64")
    train_idx, test_idx = train_test_split(indices, test_size=holdout_frac, random_state=seed, stratify=y)
    val_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, solver="lbfgs", class_weight="balanced", random_state=seed),
    )
    val_model.fit(x[train_idx], y[train_idx])
    prob = val_model.predict_proba(x[test_idx])[:, 1]
    pred = (prob >= 0.5).astype("int64")
    precision, recall, f1, _ = precision_recall_fscore_support(y[test_idx], pred, labels=[1], average="binary", zero_division=0)
    metrics = {
        "model": "linear_logreg",
        "train_n": int(len(train_idx)),
        "test_n": int(len(test_idx)),
        "test_fg": int((y[test_idx] == 1).sum()),
        "test_bg": int((y[test_idx] == 0).sum()),
        "accuracy": float(accuracy_score(y[test_idx], pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y[test_idx], pred)),
        "precision_fg": float(precision),
        "recall_fg": float(recall),
        "f1_fg": float(f1),
        "roc_auc": safe_metric(roc_auc_score, y[test_idx], prob),
        "average_precision": safe_metric(average_precision_score, y[test_idx], prob),
    }
    final_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, solver="lbfgs", class_weight="balanced", random_state=seed),
    )
    final_model.fit(x, y)
    return final_model, metrics


def safe_metric(fn: Any, y: np.ndarray, prob: np.ndarray) -> float:
    try:
        return float(fn(y, prob))
    except Exception:
        return float("nan")


def extract_case_features(
    args: argparse.Namespace,
    case: StressCase,
    records: list[PatchRecord],
    extractor: FeatureExtractor,
) -> tuple[np.ndarray, list[PatchRecord], dict[str, Any]]:
    started = time.perf_counter()
    cache_path = args.output_dir / "features" / f"{case.case_id}_yolo_probe_features.npz"
    if args.resume and cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as data:
            if str(data["model_backend"]) == extractor.backend and str(data["model_name"]) == extractor.model_name:
                loaded = [
                    PatchRecord(
                        case_id=str(case_id),
                        detection_id=int(det_id),
                        detection_ids=tuple(int(x) for x in str(det_ids).split(";") if str(x)),
                        row=int(row),
                        col=int(col),
                        x=int(x),
                        y=int(y),
                        width=int(width),
                        height=int(height),
                    )
                    for case_id, det_id, row, col, x, y, width, height in zip(
                        data["case_id"],
                        data["detection_id"],
                        data["detection_ids"],
                        data["row"],
                        data["col"],
                        data["x_level0"],
                        data["y_level0"],
                        data["width_level0"],
                        data["height_level0"],
                    )
                ]
                requested_key = [(r.row, r.col, r.x, r.y, r.width, r.height, r.detection_ids) for r in records]
                loaded_key = [(r.row, r.col, r.x, r.y, r.width, r.height, r.detection_ids) for r in loaded]
                if requested_key == loaded_key:
                    return data["features"].astype("float32"), loaded, {
                        "cache_reused": True,
                        "cache_path": str(cache_path),
                        "wsi_reader": str(data["wsi_reader"]) if "wsi_reader" in data.files else "missing",
                        "read_workers": int(data["read_workers"]) if "read_workers" in data.files else -1,
                        "pipeline_mode": str(data["pipeline_mode"]) if "pipeline_mode" in data.files else "missing",
                        "extract_seconds": 0.0,
                        "patches_per_second": 0.0,
                    }
    if args.max_patches_per_case is not None and len(records) > args.max_patches_per_case:
        raise ValueError(
            f"{case.case_id} has {len(records)} patches inside YOLO boxes, above --max-patches-per-case={args.max_patches_per_case}"
        )
    features: list[np.ndarray] = []

    def infer_images(images: list[Image.Image]) -> None:
        if not images:
            return
        features.extend(list(extractor.extract_batch(images)))

    if args.pipeline_mode == "serial":
        reader = WsiPatchReader(case.wsi_path, args.wsi_reader, args.read_workers)
        images: list[Image.Image] = []
        try:
            for record in records:
                images.append(reader.read_patch(record))
                if len(images) >= extractor.batch_size:
                    infer_images(images)
                    images = []
            infer_images(images)
        finally:
            reader.close()
    else:
        batches: queue.Queue[list[Image.Image] | Exception | None] = queue.Queue(maxsize=max(1, int(args.prefetch_queue_batches)))

        def producer() -> None:
            reader = WsiPatchReader(case.wsi_path, args.wsi_reader, args.read_workers)
            try:
                images: list[Image.Image] = []
                for record in records:
                    images.append(reader.read_patch(record))
                    if len(images) >= extractor.batch_size:
                        batches.put(images)
                        images = []
                if images:
                    batches.put(images)
            except Exception as exc:
                batches.put(exc)
            finally:
                reader.close()
                batches.put(None)

        thread = threading.Thread(target=producer, name=f"prefetch-{case.case_id}", daemon=True)
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

    elapsed = time.perf_counter() - started
    feature_array = np.stack(features, axis=0).astype("float32") if features else np.zeros((0, 0), dtype="float32")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        features=feature_array,
        case_id=np.asarray([r.case_id for r in records]),
        detection_id=np.asarray([r.detection_id for r in records], dtype="int64"),
        detection_ids=np.asarray([";".join(str(x) for x in r.detection_ids) for r in records]),
        row=np.asarray([r.row for r in records], dtype="int64"),
        col=np.asarray([r.col for r in records], dtype="int64"),
        x_level0=np.asarray([r.x for r in records], dtype="int64"),
        y_level0=np.asarray([r.y for r in records], dtype="int64"),
        width_level0=np.asarray([r.width for r in records], dtype="int64"),
        height_level0=np.asarray([r.height for r in records], dtype="int64"),
        model_backend=np.asarray(extractor.backend),
        model_name=np.asarray(extractor.model_name),
        wsi_reader=np.asarray(args.wsi_reader),
        read_workers=np.asarray(int(args.read_workers)),
        pipeline_mode=np.asarray(args.pipeline_mode),
        created_at=np.asarray(datetime.now(timezone.utc).isoformat()),
    )
    return feature_array, records, {
        "cache_reused": False,
        "cache_path": str(cache_path),
        "wsi_reader": args.wsi_reader,
        "read_workers": int(args.read_workers),
        "pipeline_mode": args.pipeline_mode,
        "extract_seconds": float(elapsed),
        "patches_per_second": float(len(records) / elapsed) if elapsed > 0 else 0.0,
    }


def assign_gt(record: PatchRecord, gt: np.ndarray, patch_size: int) -> int:
    row = max(0, min(gt.shape[0] - 1, int(record.y // patch_size)))
    col = max(0, min(gt.shape[1] - 1, int(record.x // patch_size)))
    return int(bool(gt[row, col]))


def mask_metrics(gt: np.ndarray, pred: np.ndarray, prob: np.ndarray) -> dict[str, Any]:
    gt_b = gt.astype(bool)
    pred_b = pred.astype(bool)
    tp = int(np.logical_and(gt_b, pred_b).sum())
    fp = int(np.logical_and(~gt_b, pred_b).sum())
    fn = int(np.logical_and(gt_b, ~pred_b).sum())
    tn = int(np.logical_and(~gt_b, ~pred_b).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    dice = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0
    accuracy = (tp + tn) / max(1, gt.size)
    return {
        "grid_rows": int(gt.shape[0]),
        "grid_cols": int(gt.shape[1]),
        "grid_cells": int(gt.size),
        "gt_fg": int(gt_b.sum()),
        "pred_fg": int(pred_b.sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision_fg": float(precision),
        "recall_fg": float(recall),
        "f1_fg": float(f1),
        "dice_fg": float(dice),
        "accuracy": float(accuracy),
        "mean_pred_prob": float(prob[pred_b].mean()) if pred_b.any() else 0.0,
    }


def score_cases(args: argparse.Namespace, cases: list[StressCase], detections: list[Detection]) -> dict[str, Any]:
    x_train, y_train, train_meta = load_training_features(args.train_feature_dir)
    model, validation_metrics = fit_linear_probe(x_train, y_train, args.sample_seed, args.holdout_frac)
    write_csv(args.output_dir / "model_summary.csv", [{**train_meta, **validation_metrics}])

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

    raw_detections_by_case: dict[str, list[Detection]] = defaultdict(list)
    for det in detections:
        raw_detections_by_case[det.case_id].append(det)
    detections_by_case: dict[str, list[Detection]] = defaultdict(list)
    suppressed_rows: list[dict[str, Any]] = []
    for case in cases:
        case_raw = sorted(raw_detections_by_case.get(case.case_id, []), key=lambda d: d.detection_id)
        if args.suppress_contained:
            kept, suppressed = suppress_contained_detections(case_raw, args.containment_threshold)
            suppressed_rows.extend(suppressed)
        else:
            kept = case_raw
        detections_by_case[case.case_id] = kept
    filtered_detections = [det for case in cases for det in detections_by_case.get(case.case_id, [])]
    write_csv(args.output_dir / "yolo_detections_filtered.csv", detection_rows(filtered_detections))
    write_csv(args.output_dir / "yolo_detections_suppressed_contained.csv", suppressed_rows)

    patch_rows: list[dict[str, Any]] = []
    bbox_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    feature_meta_rows: list[dict[str, Any]] = []
    pred_payload: dict[str, tuple[list[Detection], list[PatchRecord], np.ndarray, np.ndarray, np.ndarray]] = {}

    for case in cases:
        case_dets = sorted(detections_by_case.get(case.case_id, []), key=lambda d: d.detection_id)
        records = build_patch_records(case, case_dets, args.patch_size)
        print(
            f"[score] {case.case_id}: detections={len(case_dets)} unique_patches={len(records)}",
            flush=True,
        )
        if not records:
            gt = np.load(case.gt_mask_path).astype(bool)
            case_rows.append(
                {
                    "case_id": case.case_id,
                    "detection_count": 0,
                    "patch_count": 0,
                    **mask_metrics(gt, np.zeros_like(gt, dtype=bool), np.zeros_like(gt, dtype="float32")),
                }
            )
            continue
        features, records, fmeta = extract_case_features(args, case, records, extractor)
        print(
            f"[score] {case.case_id}: features_shape={tuple(features.shape)} cache_reused={fmeta.get('cache_reused')}",
            flush=True,
        )
        feature_meta_rows.append({"case_id": case.case_id, "patch_count": len(records), **fmeta})
        prob = model.predict_proba(features)[:, 1]
        pred = (prob >= args.probe_threshold).astype("int64")
        gt = np.load(case.gt_mask_path).astype(bool)
        gt_labels = np.asarray([assign_gt(record, gt, args.patch_size) for record in records], dtype="int64")
        prob_grid = np.zeros(gt.shape, dtype="float32")
        pred_grid = np.zeros(gt.shape, dtype=bool)
        for record, is_fg, fg_prob in zip(records, pred, prob):
            rr = max(0, min(gt.shape[0] - 1, int(record.y // args.patch_size)))
            cc = max(0, min(gt.shape[1] - 1, int(record.x // args.patch_size)))
            prob_grid[rr, cc] = max(prob_grid[rr, cc], float(fg_prob))
            pred_grid[rr, cc] = bool(pred_grid[rr, cc] or int(is_fg) == 1)
        case_metric = mask_metrics(gt, pred_grid, prob_grid)
        case_rows.append(
            {
                "case_id": case.case_id,
                "scanner": case.scanner,
                "stain": case.stain,
                "detection_count": len(case_dets),
                "patch_count": len(records),
                **case_metric,
            }
        )
        print(
            f"[score] {case.case_id}: precision={case_metric['precision_fg']:.3f} "
            f"recall={case_metric['recall_fg']:.3f} f1={case_metric['f1_fg']:.3f} "
            f"dice={case_metric['dice_fg']:.3f}",
            flush=True,
        )
        pred_payload[case.case_id] = (case_dets, records, prob, pred, gt_labels)
        for record, fg_prob, is_fg, gt_fg in zip(records, prob, pred, gt_labels):
            patch_rows.append(
                {
                    "case_id": case.case_id,
                    "detection_id": record.detection_id,
                    "detection_ids": ";".join(str(x) for x in record.detection_ids),
                    "row": record.row,
                    "col": record.col,
                    "x_level0": record.x,
                    "y_level0": record.y,
                    "width_level0": record.width,
                    "height_level0": record.height,
                    "prob_fg": float(fg_prob),
                    "pred_fg": int(is_fg),
                    "gt_fg": int(gt_fg),
                }
            )
        by_det: dict[int, list[int]] = defaultdict(list)
        for idx, record in enumerate(records):
            for det_id in record.detection_ids:
                by_det[det_id].append(idx)
        for det in case_dets:
            idxs = by_det.get(det.detection_id, [])
            if not idxs:
                continue
            det_pred = pred[np.asarray(idxs, dtype="int64")]
            det_prob = prob[np.asarray(idxs, dtype="int64")]
            det_gt = gt_labels[np.asarray(idxs, dtype="int64")]
            bbox_rows.append(
                {
                    "case_id": case.case_id,
                    "detection_id": det.detection_id,
                    "yolo_conf": det.conf,
                    "patch_count": len(idxs),
                    "pred_fg": int(det_pred.sum()),
                    "gt_fg_in_box": int(det_gt.sum()),
                    "pred_fg_fraction": float(det_pred.mean()),
                    "mean_prob_fg": float(det_prob.mean()),
                    "x0_level0": det.x0,
                    "y0_level0": det.y0,
                    "x1_level0": det.x1,
                    "y1_level0": det.y1,
                }
            )

    write_csv(args.output_dir / "patch_predictions.csv", patch_rows)
    write_csv(args.output_dir / "bbox_summary.csv", bbox_rows)
    write_csv(args.output_dir / "case_metrics.csv", case_rows)
    write_csv(args.output_dir / "feature_cache_summary.csv", feature_meta_rows)
    pdf_path = args.output_dir / "stress32_yolo_conf001_dinov3_linear_probe_review.pdf"
    pages = build_pdf_pages(args, cases, pred_payload, bbox_rows, case_rows, validation_metrics, train_meta, extractor.meta)
    if pages:
        pages[0].save(pdf_path, save_all=True, append_images=pages[1:])
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "ticket": args.ticket,
        "git_commit": git_commit(),
        "git_status_path": str(args.output_dir / "git_status.txt"),
        "case_count": len(cases),
        "raw_detection_count": len(detections),
        "filtered_detection_count": len(filtered_detections),
        "suppressed_detection_count": len(suppressed_rows),
        "suppress_contained": bool(args.suppress_contained),
        "containment_threshold": float(args.containment_threshold),
        "pipeline_mode": args.pipeline_mode,
        "prefetch_queue_batches": int(args.prefetch_queue_batches),
        "patch_predictions": len(patch_rows),
        "train_meta": train_meta,
        "validation_metrics": validation_metrics,
        "extractor": extractor.meta,
        "package_versions": package_versions(),
        "pdf": str(pdf_path.resolve()),
        "case_metrics_csv": str((args.output_dir / "case_metrics.csv").resolve()),
        "bbox_summary_csv": str((args.output_dir / "bbox_summary.csv").resolve()),
        "patch_predictions_csv": str((args.output_dir / "patch_predictions.csv").resolve()),
    }
    write_json(args.output_dir / "summary.json", summary)
    return summary


def draw_metric_table(train_meta: dict[str, Any], validation: dict[str, Any], case_rows: list[dict[str, Any]]) -> Image.Image:
    width = 1200
    height = 620
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    y = 18
    draw.text((18, y), "Pooled DINOv3-small linear probe", fill=(0, 0, 0), font=get_font(28, bold=True))
    y += 48
    lines = [
        f"Training cache: {train_meta['feature_files']} WSIs, {train_meta['patches']} selected-crop patches "
        f"({train_meta['fg']} FG / {train_meta['bg']} BG).",
        f"Holdout validation: F1={validation['f1_fg']:.3f}, precision={validation['precision_fg']:.3f}, "
        f"recall={validation['recall_fg']:.3f}, AUROC={validation['roc_auc']:.3f}, AP={validation['average_precision']:.3f}.",
    ]
    for line in lines:
        y = draw_wrapped_text(draw, (18, y), line, font=get_font(22), fill=(35, 35, 35), width_chars=92)
        y += 14
    draw.text((18, y), "Stress-32 YOLO+probe WSI metrics", fill=(0, 0, 0), font=get_font(26, bold=True))
    y += 42
    if case_rows:
        vals = {key: np.asarray([float(row[key]) for row in case_rows], dtype="float64") for key in ["precision_fg", "recall_fg", "f1_fg", "dice_fg"]}
        for key, label in [
            ("precision_fg", "precision"),
            ("recall_fg", "recall"),
            ("f1_fg", "F1"),
            ("dice_fg", "Dice"),
        ]:
            draw.text((36, y), f"mean {label}: {vals[key].mean():.3f}  (median {np.median(vals[key]):.3f})", fill=(35, 35, 35), font=get_font(22))
            y += 34
    draw.text((18, height - 78), "Metrics are full WSI-grid metrics: outside YOLO boxes is treated as predicted background.", fill=(70, 70, 70), font=get_font(18))
    draw.text((18, height - 48), "Patch size is 512px at level 0; YOLO detector confidence is 0.01.", fill=(70, 70, 70), font=get_font(18))
    return image


def draw_case_overview(
    args: argparse.Namespace,
    case: StressCase,
    detections: list[Detection],
    records: list[PatchRecord],
    prob: np.ndarray,
    pred: np.ndarray,
    bbox_rows: list[dict[str, Any]],
    case_metric: dict[str, Any],
) -> Image.Image:
    base = Image.open(case.thumbnail_path).convert("RGB")
    rgba = base.convert("RGBA")
    overlay = Image.new("RGBA", rgba.size, (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)
    sx = base.width / case.wsi_width
    sy = base.height / case.wsi_height
    for record, fg_prob, is_fg in zip(records, prob, pred):
        if int(is_fg) != 1:
            continue
        rect = [
            int(round(record.x * sx)),
            int(round(record.y * sy)),
            int(round((record.x + record.width) * sx)),
            int(round((record.y + record.height) * sy)),
        ]
        alpha = int(45 + 115 * max(0.0, min(1.0, float(fg_prob))))
        odraw.rectangle(rect, fill=(34, 197, 94, alpha), outline=(22, 163, 74, 210), width=1)
    combined = Image.alpha_composite(rgba, overlay).convert("RGB")
    draw = ImageDraw.Draw(combined)
    stats_by_det = {int(row["detection_id"]): row for row in bbox_rows}
    for det in detections:
        row = stats_by_det.get(det.detection_id, {})
        rect = [
            int(round(det.x0 * sx)),
            int(round(det.y0 * sy)),
            int(round(det.x1 * sx)),
            int(round(det.y1 * sy)),
        ]
        color = (220, 38, 38)
        draw.rectangle(rect, outline=color, width=4)
        fg_frac = float(row.get("pred_fg_fraction", 0.0)) if row else 0.0
        label = f"{det.detection_id} c{det.conf:.2f} f{fg_frac:.2f}"
        font = get_font(21, bold=True)
        tb = draw.textbbox((0, 0), label, font=font)
        lx, ly = rect[0] + 4, rect[1] + 4
        draw.rectangle([lx, ly, lx + tb[2] - tb[0] + 8, ly + tb[3] - tb[1] + 6], fill="white", outline=color, width=2)
        draw.text((lx + 4, ly + 2), label, fill=color, font=font)
    footer_h = 94
    canvas = Image.new("RGB", (combined.width, combined.height + footer_h), "white")
    canvas.paste(combined, (0, 0))
    d = ImageDraw.Draw(canvas)
    footer = (
        f"{case.case_id} | detections={len(detections)} | patches={len(records)} | "
        f"precision={float(case_metric['precision_fg']):.3f}, recall={float(case_metric['recall_fg']):.3f}, "
        f"F1={float(case_metric['f1_fg']):.3f}, Dice={float(case_metric['dice_fg']):.3f}. "
        "Green squares are linear-probe FG predictions inside YOLO boxes only."
    )
    draw_wrapped_text(d, (12, combined.height + 12), footer, font=get_font(18), fill=(40, 40, 40), width_chars=125)
    return resize_to_fit(canvas, args.max_overview_width, 1000)


def draw_crop_panel(
    args: argparse.Namespace,
    case: StressCase,
    det: Detection,
    records: list[PatchRecord],
    prob: np.ndarray,
    pred: np.ndarray,
) -> Image.Image:
    base_full = Image.open(case.thumbnail_path).convert("RGB")
    sx = base_full.width / case.wsi_width
    sy = base_full.height / case.wsi_height
    crop_rect = [
        int(round(det.x0 * sx)),
        int(round(det.y0 * sy)),
        int(round(det.x1 * sx)),
        int(round(det.y1 * sy)),
    ]
    crop = base_full.crop(crop_rect).convert("RGBA")
    overlay = Image.new("RGBA", crop.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for record, fg_prob, is_fg in zip(records, prob, pred):
        fill = (34, 197, 94, int(55 + 110 * max(0.0, min(1.0, float(fg_prob))))) if int(is_fg) else (239, 68, 68, 55)
        rect = [
            int(round((record.x - det.x0) * sx)),
            int(round((record.y - det.y0) * sy)),
            int(round((record.x + record.width - det.x0) * sx)),
            int(round((record.y + record.height - det.y0) * sy)),
        ]
        draw.rectangle(rect, fill=fill, outline=(35, 35, 35, 150), width=1)
    image = Image.alpha_composite(crop, overlay).convert("RGB")
    image = resize_to_fit(image, args.max_grid_dim, args.max_grid_dim)
    panel_w = max(460, image.width + 24)
    panel_h = image.height + 90
    panel = Image.new("RGB", (panel_w, panel_h), "white")
    d = ImageDraw.Draw(panel)
    d.text((12, 8), f"det {det.detection_id} | conf={det.conf:.3f}", fill=(0, 0, 0), font=get_font(22, bold=True))
    d.text((12, 38), "green=probe FG, red=probe BG", fill=(60, 60, 60), font=get_font(16))
    panel.paste(image, ((panel_w - image.width) // 2, 76))
    return panel


def build_pdf_pages(
    args: argparse.Namespace,
    cases: list[StressCase],
    pred_payload: dict[str, tuple[list[Detection], list[PatchRecord], np.ndarray, np.ndarray, np.ndarray]],
    bbox_rows: list[dict[str, Any]],
    case_rows: list[dict[str, Any]],
    validation_metrics: dict[str, Any],
    train_meta: dict[str, Any],
    extractor_meta: dict[str, Any],
) -> list[Image.Image]:
    pages: list[Image.Image] = [
        make_page(
            "Stress-32 YOLO+Linear Probe",
            "Scale500 YOLO detector at conf=0.01, then pooled DINOv3-small linear FG/BG probe on 512px level-0 patches inside YOLO boxes.",
            draw_metric_table(train_meta, validation_metrics, case_rows),
            footer=f"Feature backend: {extractor_meta.get('backend', 'unknown')} / {extractor_meta.get('model_name', 'unknown')}; batch={args.batch_size}; WSI reader={args.wsi_reader}; read_workers={args.read_workers}.",
        )
    ]
    bbox_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in bbox_rows:
        bbox_by_case[str(row["case_id"])].append(row)
    metrics_by_case = {str(row["case_id"]): row for row in case_rows}
    for case in cases:
        payload = pred_payload.get(case.case_id)
        if payload is None:
            continue
        detections, records, prob, pred, _gt = payload
        overview = draw_case_overview(args, case, detections, records, prob, pred, bbox_by_case[case.case_id], metrics_by_case[case.case_id])
        pages.append(
            make_page(
                f"{case.case_id} | {case.stain} | thumbnail overview",
                "YOLO boxes are outlined and labeled with detector id, YOLO confidence, and linear-probe FG fraction; green squares are predicted foreground patches.",
                overview,
                footer="Patch squares are rendered only for patches inside YOLO-detected boxes.",
            )
        )
        rows = sorted(bbox_by_case[case.case_id], key=lambda row: float(row.get("pred_fg_fraction", 0.0)), reverse=True)
        keep_ids = {int(row["detection_id"]) for row in rows[: args.max_crop_panels_per_case]}
        panels: list[Image.Image] = []
        for det in detections:
            if det.detection_id not in keep_ids:
                continue
            idxs = [idx for idx, record in enumerate(records) if det.detection_id in record.detection_ids]
            if not idxs:
                continue
            panels.append(
                draw_crop_panel(
                    args,
                    case,
                    det,
                    [records[idx] for idx in idxs],
                    prob[np.asarray(idxs, dtype="int64")],
                    pred[np.asarray(idxs, dtype="int64")],
                )
            )
        if panels:
            pages.append(
                make_page(
                    f"{case.case_id} | crop-level probe overlays",
                    f"Top {len(panels)} YOLO boxes by predicted FG fraction, rendered as patch-level classification overlays.",
                    make_contact_sheet(panels, cols=2, gap=28),
                    footer="These crop panels are thumbnail-derived visualizations; CSV outputs preserve exact level-0 patch coordinates.",
                )
            )
    return pages


def write_reproduction(args: argparse.Namespace, summary: dict[str, Any] | None = None) -> None:
    detect_cmd = [
        "/data2/vj724/venvs/vlm-wsi-yolo-ultralytics-per241/bin/python",
        "scripts/run_stress32_yolo_probe.py",
        "--stage",
        "detect",
        "--output-dir",
        str(args.output_dir),
        "--conf",
        str(args.conf),
        "--iou",
        str(args.iou),
        "--imgsz",
        str(args.imgsz),
        "--max-det",
        str(args.max_det),
        "--yolo-weights",
        str(args.yolo_weights),
    ]
    if args.case_ids:
        detect_cmd.extend(["--case-ids", args.case_ids])
    if args.case_limit is not None:
        detect_cmd.extend(["--case-limit", str(args.case_limit)])
    score_cmd = [
        "/vol/biomedic3/vj724/.conda/envs/path-agent/bin/python",
        "scripts/run_stress32_yolo_probe.py",
        "--stage",
        "score",
        "--output-dir",
        str(args.output_dir),
        "--train-feature-dir",
        str(args.train_feature_dir),
        "--batch-size",
        str(args.batch_size),
        "--pipeline-mode",
        str(args.pipeline_mode),
        "--prefetch-queue-batches",
        str(args.prefetch_queue_batches),
        "--wsi-reader",
        str(args.wsi_reader),
        "--read-workers",
        str(args.read_workers),
        "--patch-size",
        str(args.patch_size),
        "--probe-threshold",
        str(args.probe_threshold),
        "--containment-threshold",
        str(args.containment_threshold),
        "--device",
        str(args.device),
    ]
    if not args.suppress_contained:
        score_cmd.append("--no-suppress-contained")
    if args.case_ids:
        score_cmd.extend(["--case-ids", args.case_ids])
    if args.case_limit is not None:
        score_cmd.extend(["--case-limit", str(args.case_limit)])
    lines = [
        "PER-250 Stress-32 YOLO plus DINOv3 linear probe",
        "================================================",
        "",
        f"Created: {datetime.now(timezone.utc).isoformat()}",
        f"Ticket: {args.ticket}",
        f"Git commit: {git_commit()}",
        f"Dirty git state captured in: {args.output_dir / 'git_status.txt'}",
        "",
        "Pipeline:",
        "- Stress-32 manifest is the 32 seeded/evaluable cases from the harder-Jones Leica manual pipeline bundle.",
        "- YOLO detector runs on the saved 1024px WSI thumbnails at confidence 0.01.",
        "- DINOv3-small linear probe scores only 512px level-0 patches inside YOLO-detected boxes.",
        f"- Contained-box suppression is {'on' if args.suppress_contained else 'off'} at threshold {args.containment_threshold}.",
        f"- Patch extraction pipeline mode: {args.pipeline_mode}; prefetch_queue_batches={args.prefetch_queue_batches}.",
        "- Full WSI-grid metrics treat all patches outside YOLO boxes as predicted background.",
        "",
        "Commands:",
        " ".join(shlex.quote(part) for part in detect_cmd),
        " ".join(shlex.quote(part) for part in score_cmd),
        "",
        "Inputs:",
        f"- Stress dataset root: {args.stress_dataset_root.resolve()}",
        f"- Stress pipeline root: {args.stress_pipeline_root.resolve()}",
        f"- YOLO weights: {args.yolo_weights.resolve()}",
        f"- Training feature cache: {args.train_feature_dir.resolve()}",
        "",
        "Outputs:",
        f"- Manifest: {(args.output_dir / 'stress32_manifest.csv').resolve()}",
        f"- YOLO detections: {(args.output_dir / 'yolo_detections.csv').resolve()}",
        f"- YOLO detections after contained-box filtering: {(args.output_dir / 'yolo_detections_filtered.csv').resolve()}",
        f"- Suppressed contained boxes: {(args.output_dir / 'yolo_detections_suppressed_contained.csv').resolve()}",
        f"- Case metrics: {(args.output_dir / 'case_metrics.csv').resolve()}",
        f"- Bbox summary: {(args.output_dir / 'bbox_summary.csv').resolve()}",
        f"- Patch predictions: {(args.output_dir / 'patch_predictions.csv').resolve()}",
        f"- PDF: {(args.output_dir / 'stress32_yolo_conf001_dinov3_linear_probe_review.pdf').resolve()}",
    ]
    if summary:
        lines.extend(["", "Summary JSON:", json.dumps(summary, indent=2, sort_keys=True)])
    (args.output_dir / "reproduction.txt").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "git_status.txt").write_text(git_status_short())
    cases: list[StressCase] = []
    if args.stage in {"manifest", "detect", "score", "all"}:
        cases = stage_manifest(args)
    if args.stage in {"detect", "all"}:
        run_yolo_detect(args, cases)
    summary: dict[str, Any] | None = None
    if args.stage in {"score", "all"}:
        detections_path = args.output_dir / "yolo_detections.csv"
        if not detections_path.exists():
            raise FileNotFoundError(f"Run --stage detect first; missing {detections_path}")
        summary = score_cases(args, cases, load_detections(detections_path))
    write_reproduction(args, summary)


if __name__ == "__main__":
    main()
