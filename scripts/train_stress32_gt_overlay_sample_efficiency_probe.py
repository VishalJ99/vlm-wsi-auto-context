#!/usr/bin/env python3
"""Stress-32 DINOv3 sample-efficiency probe from GT-overlay foreground pools."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import queue
import shlex
import subprocess
import sys
import textwrap
import threading
import time
import zlib
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import openslide
import torch
from PIL import Image, ImageDraw, ImageFont
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from cucim import CuImage
except Exception:  # pragma: no cover - optional runtime dependency
    CuImage = None  # type: ignore[assignment]


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GT_ROOT = (
    Path("/data2/vj724/wsi-agents")
    / "vlm_gt_seg_comparison_experiment/dataset_thumbnails_harder_jones_leica/leica/jones"
)
DEFAULT_QWEN_ROOT = (
    Path("/data2/vj724/wsi-agents")
    / "new_vlm_gt_preds/leica_hard_jones_evg_qwen_2b_zero_shot"
)
DEFAULT_SOURCE_PDF = (
    REPO_ROOT
    / "runs/stress32_qwen2b_hard_negative_probe_v1/visuals/stress32_qwen_stage1_bbox_pool_sampling_review.pdf"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "runs/stress32_gt_overlay_sample_efficiency_probe_v1"
DEFAULT_RUN_ID = "harder_jones_leica_manual"
DEFAULT_DINOV3_SMALL = "facebook/dinov3-vits16-pretrain-lvd1689m"
DEFAULT_DINOV2_SMALL = "vit_small_patch14_dinov2"
DEFAULT_HOLDOUT_CASE_IDS = (
    "anon_d2e4cb7a-e08a-4993-bc16-bbe34e7f6503,"
    "anon_27bbc335-2f23-42c8-92f3-76ec09fe6c15,"
    "anon_8468d97d-ea6e-4ba2-b620-61e8ad72cb0e,"
    "anon_69eb2434-188b-4fcb-8690-489d222b1079,"
    "anon_b1cd41a1-a834-4a4f-a5ad-25b27cee7068,"
    "anon_4239ac62-4106-45d8-95d6-4c774d5b0dfe"
)
LANCZOS = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
NEAREST = getattr(getattr(Image, "Resampling", Image), "NEAREST")


@dataclass(frozen=True)
class PatchRecord:
    case_id: str
    split: str
    bucket: str
    row: int
    col: int
    x: int
    y: int
    width: int
    height: int
    label_fg: int
    qwen_fg: int
    gt_fg: int

    @property
    def record_id(self) -> str:
        return f"{self.case_id}|{self.bucket}|r{self.row}c{self.col}|{self.x}_{self.y}"


@dataclass(frozen=True)
class CaseBundle:
    case_id: str
    case_dir: Path
    wsi_path: Path
    bbox_level0: tuple[int, int, int, int]
    bbox_thumbnail: tuple[int, int, int, int]
    mask_shape: tuple[int, int]
    records: list[PatchRecord]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticket", default="PER-269")
    parser.add_argument("--gt-root", type=Path, default=DEFAULT_GT_ROOT)
    parser.add_argument("--qwen-root", type=Path, default=DEFAULT_QWEN_ROOT)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--source-pdf", type=Path, default=DEFAULT_SOURCE_PDF)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--case-ids", default="")
    parser.add_argument("--case-limit", type=int, default=None)
    parser.add_argument("--holdout-case-ids", default=DEFAULT_HOLDOUT_CASE_IDS)
    parser.add_argument("--sample-sizes", default="10,20,50,100,200,300,500")
    parser.add_argument("--sample-seed", type=int, default=250)
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--probe-threshold", type=float, default=0.5)
    parser.add_argument("--model-backend", choices=["transformers", "timm"], default="transformers")
    parser.add_argument("--model-name", default=DEFAULT_DINOV3_SMALL)
    parser.add_argument("--fallback-model-name", default=DEFAULT_DINOV2_SMALL)
    parser.add_argument("--allow-timm-fallback", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--input-size", type=int, default=None)
    parser.add_argument("--wsi-reader", choices=["cucim", "openslide"], default="cucim")
    parser.add_argument("--read-workers", type=int, default=16)
    parser.add_argument("--prefetch-queue-batches", type=int, default=4)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--max-review-cases", type=int, default=6)
    parser.add_argument("--review-sample-sizes", default="10,100,500")
    parser.add_argument("--max-review-dim", type=int, default=900)
    return parser.parse_args()


def parse_int_list(text: str) -> list[int]:
    values = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError(f"No integer values parsed from {text!r}")
    return values


def parse_str_list(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


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
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def capture_command(command: list[str]) -> str:
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


def git_commit() -> str:
    return capture_command(["git", "rev-parse", "HEAD"])


def git_status_short() -> str:
    return capture_command(["git", "status", "--short"]) + "\n"


def dvc_status_text() -> str:
    return capture_command(["bash", "-lc", "command -v dvc >/dev/null 2>&1 && dvc status || echo 'dvc: command not found'"]) + "\n"


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {
        "python": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
        "torch": getattr(torch, "__version__", "unknown"),
        "openslide": getattr(openslide, "__version__", "unknown"),
        "numpy": getattr(np, "__version__", "unknown"),
    }
    for name in ("sklearn", "PIL", "transformers", "timm", "cucim", "matplotlib"):
        try:
            module = __import__(name)
            versions[name] = str(getattr(module, "__version__", "unknown"))
        except Exception as exc:
            versions[name] = f"unavailable: {type(exc).__name__}: {exc}"
    return versions


def qwen_mask_path(qwen_root: Path, run_id: str, case_id: str) -> Path:
    base = qwen_root / case_id / run_id
    for stage in ("stage7_new", "stage7"):
        path = base / stage / "mask.npy"
        if path.exists():
            return path
    raise FileNotFoundError(f"No qwen2b stage7_new/stage7 mask for {case_id} under {base}")


def qwen_stage1_bboxes_path(qwen_root: Path, run_id: str, case_id: str) -> Path:
    return qwen_root / case_id / run_id / "stage1/bboxes.json"


def load_qwen_stage1_bbox(case_dir: Path, qwen_root: Path, run_id: str) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int], Path]:
    case_id = case_dir.name
    path = qwen_stage1_bboxes_path(qwen_root, run_id, case_id)
    if not path.exists():
        raise FileNotFoundError(f"Missing qwen-stage1 bbox JSON: {path}")
    payload = read_json(path)
    region = payload["detected_regions"][0]
    return (
        tuple(int(x) for x in region["bbox_level0"]),
        tuple(int(x) for x in region["bbox_thumbnail"]),
        path,
    )


def cell_rect_level0(row: int, col: int, cell_size_level0: int) -> tuple[int, int, int, int]:
    x0 = int(col) * int(cell_size_level0)
    y0 = int(row) * int(cell_size_level0)
    return x0, y0, x0 + int(cell_size_level0), y0 + int(cell_size_level0)


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


def rect_intersection(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[int, int, int, int] | None:
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


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


def case_wsi_path(case_dir: Path) -> Path:
    meta = read_json(case_dir / "case_meta.json")
    path = Path(str(meta["wsi_path"]))
    if not path.exists():
        raise FileNotFoundError(f"WSI path from case_meta does not exist: {path}")
    return path


def bbox_crop_file(case_dir: Path, bbox_level0: tuple[int, int, int, int]) -> Path | None:
    path = case_dir / ("_".join(str(int(x)) for x in bbox_level0) + ".png")
    return path if path.exists() else None


def choose_read_level(level_downsamples: list[float], long_edge_level0: int, max_dim: int) -> int:
    desired_downsample = max(1.0, long_edge_level0 / float(max(1, max_dim)))
    level = 0
    for idx, downsample in enumerate(level_downsamples):
        if float(downsample) <= desired_downsample:
            level = idx
    return level


def read_bbox_preview(case: CaseBundle, max_dim: int) -> Image.Image:
    crop_file = bbox_crop_file(case.case_dir, case.bbox_level0)
    if crop_file is not None:
        image = Image.open(crop_file).convert("RGB")
    else:
        x0, y0, x1, y1 = case.bbox_level0
        width = max(1, x1 - x0)
        height = max(1, y1 - y0)
        slide = openslide.OpenSlide(str(case.wsi_path))
        try:
            level_downsamples = [float(x) for x in slide.level_downsamples]
            level = choose_read_level(level_downsamples, max(width, height), max_dim)
            downsample = float(level_downsamples[level]) if level_downsamples else 1.0
            read_size = (max(1, int(math.ceil(width / downsample))), max(1, int(math.ceil(height / downsample))))
            image = slide.read_region((x0, y0), level, read_size).convert("RGB")
        finally:
            slide.close()
    if max(image.size) > max_dim:
        scale = max_dim / float(max(image.size))
        image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), LANCZOS)
    return image


def build_case_bundle(args: argparse.Namespace, case_dir: Path, split: str) -> tuple[CaseBundle, dict[str, Any]]:
    case_id = case_dir.name
    bbox_level0, bbox_thumbnail, bbox_path = load_qwen_stage1_bbox(case_dir, args.qwen_root, args.run_id)
    gt = np.load(case_dir / "mask.npy").astype(bool)
    qwen_path = qwen_mask_path(args.qwen_root, args.run_id, case_id)
    qwen = np.load(qwen_path).astype(bool)
    if gt.shape != qwen.shape:
        raise ValueError(f"Mask shape mismatch for {case_id}: GT={gt.shape}, qwen={qwen.shape}")
    records: list[PatchRecord] = []
    inside_count = 0
    for row in range(gt.shape[0]):
        for col in range(gt.shape[1]):
            if not cell_center_in_bbox_level0(row, col, bbox_level0, args.patch_size):
                continue
            inside_count += 1
            gt_fg = bool(gt[row, col])
            qwen_fg = bool(qwen[row, col])
            if gt_fg:
                bucket = "foreground_gt"
                label_fg = 1
            elif qwen_fg:
                bucket = "hard_negative"
                label_fg = 0
            else:
                bucket = "easy_negative"
                label_fg = 0
            records.append(
                PatchRecord(
                    case_id=case_id,
                    split=split,
                    bucket=bucket,
                    row=int(row),
                    col=int(col),
                    x=int(col * args.patch_size),
                    y=int(row * args.patch_size),
                    width=int(args.patch_size),
                    height=int(args.patch_size),
                    label_fg=label_fg,
                    qwen_fg=1 if qwen_fg else 0,
                    gt_fg=1 if gt_fg else 0,
                )
            )
    counts = defaultdict(int)
    for record in records:
        counts[record.bucket] += 1
    qwen_tp = int((gt & qwen).sum())
    qwen_fn = int((gt & ~qwen).sum())
    qwen_fp = int((~gt & qwen).sum())
    census = {
        "case_id": case_id,
        "split": split,
        "wsi_path": str(case_wsi_path(case_dir)),
        "gt_mask_path": str(case_dir / "mask.npy"),
        "qwen_mask_path": str(qwen_path),
        "bbox_json_path": str(bbox_path),
        "bbox_level0": list(bbox_level0),
        "bbox_thumbnail": list(bbox_thumbnail),
        "grid_rows": int(gt.shape[0]),
        "grid_cols": int(gt.shape[1]),
        "inside_bbox_cells": int(inside_count),
        "foreground_gt": int(counts["foreground_gt"]),
        "hard_negative": int(counts["hard_negative"]),
        "easy_negative": int(counts["easy_negative"]),
        "qwen_tp_global": qwen_tp,
        "qwen_fn_global": qwen_fn,
        "qwen_fp_global": qwen_fp,
        "hard_negative_burden_inside_bbox": float(counts["hard_negative"] / max(1, counts["foreground_gt"] + counts["hard_negative"])),
    }
    bundle = CaseBundle(
        case_id=case_id,
        case_dir=case_dir,
        wsi_path=case_wsi_path(case_dir),
        bbox_level0=bbox_level0,
        bbox_thumbnail=bbox_thumbnail,
        mask_shape=(int(gt.shape[0]), int(gt.shape[1])),
        records=records,
    )
    return bundle, census


def load_case_bundles(args: argparse.Namespace) -> tuple[list[CaseBundle], list[dict[str, Any]]]:
    requested = parse_str_list(args.case_ids)
    case_dirs = [args.gt_root / case_id for case_id in requested] if requested else sorted(path for path in args.gt_root.iterdir() if path.is_dir())
    case_dirs = [
        path
        for path in case_dirs
        if (path / "mask.npy").exists() and (path / "case_meta.json").exists() and qwen_stage1_bboxes_path(args.qwen_root, args.run_id, path.name).exists()
    ]
    if args.case_limit is not None:
        case_dirs = case_dirs[: args.case_limit]
    holdout = set(parse_str_list(args.holdout_case_ids))
    present = {path.name for path in case_dirs}
    active_holdout = holdout & present
    if not active_holdout:
        raise ValueError("No holdout case IDs are present in the selected case set.")
    bundles: list[CaseBundle] = []
    census: list[dict[str, Any]] = []
    for case_dir in case_dirs:
        split = "heldout" if case_dir.name in active_holdout else "train"
        bundle, row = build_case_bundle(args, case_dir, split)
        bundles.append(bundle)
        census.append(row)
    if not any(bundle.records and bundle.records[0].split == "train" for bundle in bundles):
        raise ValueError("No training cases after holdout split.")
    if not any(bundle.records and bundle.records[0].split == "heldout" for bundle in bundles):
        raise ValueError("No heldout cases after holdout split.")
    return bundles, census


def patch_record_row(record: PatchRecord) -> dict[str, Any]:
    return {
        "record_id": record.record_id,
        "case_id": record.case_id,
        "split": record.split,
        "bucket": record.bucket,
        "label_fg": record.label_fg,
        "gt_fg": record.gt_fg,
        "qwen_fg": record.qwen_fg,
        "row": record.row,
        "col": record.col,
        "x_level0": record.x,
        "y_level0": record.y,
        "width_level0": record.width,
        "height_level0": record.height,
    }


def stable_order(records: list[PatchRecord], seed: int, case_id: str, bucket: str) -> list[PatchRecord]:
    if not records:
        return []
    order_seed = int(seed + zlib.crc32(f"{case_id}:{bucket}".encode("utf-8")))
    rng = np.random.default_rng(order_seed)
    indices = rng.permutation(len(records))
    return [records[int(idx)] for idx in indices]


def choose_bg_counts(hard_count: int, easy_count: int, target_total: int) -> tuple[int, int]:
    total = min(target_total, hard_count + easy_count)
    best: tuple[int, int, int, int] | None = None
    for hard_take in range(0, min(total, hard_count) + 1):
        easy_take = total - hard_take
        if easy_take < 0 or easy_take > easy_count:
            continue
        balance = abs(hard_take - easy_take)
        target_balance = abs(hard_take - target_total // 2)
        candidate = (balance, target_balance, -hard_take, hard_take)
        if best is None or candidate < best:
            best = candidate
    if best is None:
        return 0, 0
    hard_take = best[3]
    easy_take = total - hard_take
    return int(hard_take), int(easy_take)


def build_sample_manifests(
    args: argparse.Namespace,
    bundles: list[CaseBundle],
    sample_sizes: list[int],
) -> tuple[dict[int, list[PatchRecord]], list[dict[str, Any]], list[dict[str, Any]]]:
    train_by_case = {bundle.case_id: bundle for bundle in bundles if bundle.records and bundle.records[0].split == "train"}
    samples_by_size: dict[int, list[PatchRecord]] = {}
    sample_rows: list[dict[str, Any]] = []
    sample_count_rows: list[dict[str, Any]] = []
    orders: dict[tuple[str, str], list[PatchRecord]] = {}
    for case_id, bundle in train_by_case.items():
        by_bucket: dict[str, list[PatchRecord]] = defaultdict(list)
        for record in bundle.records:
            by_bucket[record.bucket].append(record)
        for bucket in ("foreground_gt", "hard_negative", "easy_negative"):
            orders[(case_id, bucket)] = stable_order(by_bucket[bucket], args.sample_seed, case_id, bucket)
    for sample_size in sample_sizes:
        rows: list[PatchRecord] = []
        for case_id in sorted(train_by_case):
            fg_order = orders[(case_id, "foreground_gt")]
            hard_order = orders[(case_id, "hard_negative")]
            easy_order = orders[(case_id, "easy_negative")]
            fg_take = min(sample_size, len(fg_order))
            hard_take, easy_take = choose_bg_counts(len(hard_order), len(easy_order), sample_size)
            chosen = fg_order[:fg_take] + hard_order[:hard_take] + easy_order[:easy_take]
            rows.extend(chosen)
            sample_count_rows.append(
                {
                    "sample_size_per_wsi": sample_size,
                    "case_id": case_id,
                    "requested_fg": sample_size,
                    "actual_fg": fg_take,
                    "requested_bg": sample_size,
                    "actual_bg": hard_take + easy_take,
                    "actual_hard_negative": hard_take,
                    "actual_easy_negative": easy_take,
                    "available_fg": len(fg_order),
                    "available_hard_negative": len(hard_order),
                    "available_easy_negative": len(easy_order),
                }
            )
            for idx, record in enumerate(chosen):
                row = patch_record_row(record)
                row.update(
                    {
                        "sample_size_per_wsi": sample_size,
                        "sample_order_in_case_size": idx,
                        "used_for_training": 1,
                    }
                )
                sample_rows.append(row)
        samples_by_size[sample_size] = rows
    return samples_by_size, sample_rows, sample_count_rows


class FeatureExtractor:
    def __init__(self, args: argparse.Namespace):
        self.requested_backend = args.model_backend
        self.requested_model = args.model_name
        self.fallback_model = args.fallback_model_name
        self.device = torch.device(args.device if args.device != "auto" else ("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.batch_size = int(args.batch_size)
        self.input_size_arg = args.input_size
        self.backend = args.model_backend
        self.model_name = args.model_name
        self.fallback_used = False
        self.load_error: str | None = None
        self.meta: dict[str, Any] = {}
        self._load(args)

    def _load(self, args: argparse.Namespace) -> None:
        started = time.monotonic()
        if args.model_backend == "transformers":
            try:
                self._load_transformers(args.model_name)
            except Exception as exc:
                self.load_error = f"{type(exc).__name__}: {exc}"
                if not args.allow_timm_fallback:
                    raise
                self.backend = "timm"
                self.model_name = args.fallback_model_name
                self.fallback_used = True
                self._load_timm(args.fallback_model_name)
        elif args.model_backend == "timm":
            self._load_timm(args.model_name)
        else:
            raise ValueError(args.model_backend)
        self.meta.update(
            {
                "requested_backend": self.requested_backend,
                "requested_model": self.requested_model,
                "backend": self.backend,
                "model_name": self.model_name,
                "fallback_used": self.fallback_used,
                "fallback_model": self.fallback_model if self.fallback_used else None,
                "load_error": self.load_error,
                "device": str(self.device),
                "batch_size": self.batch_size,
                "load_seconds": float(time.monotonic() - started),
            }
        )

    def _load_transformers(self, model_name: str) -> None:
        from transformers import AutoImageProcessor, AutoModel

        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.input_size = self.input_size_arg
        hidden = getattr(getattr(self.model, "config", None), "hidden_size", None)
        self.meta = {
            "feature_api": "transformers.AutoModel",
            "feature_dim_config": hidden,
            "preprocess": "AutoImageProcessor.from_pretrained",
            "input_size_override": self.input_size_arg,
        }

    def _load_timm(self, model_name: str) -> None:
        import timm

        self.model = timm.create_model(model_name, pretrained=True, num_classes=0).to(self.device).eval()
        cfg = timm.data.resolve_model_data_config(self.model)
        self.input_size = self.input_size_arg or int(cfg.get("input_size", (3, 224, 224))[-1])
        self.mean = torch.tensor(tuple(float(x) for x in cfg.get("mean", (0.485, 0.456, 0.406))), dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(tuple(float(x) for x in cfg.get("std", (0.229, 0.224, 0.225))), dtype=torch.float32).view(3, 1, 1)
        self.meta = {
            "feature_api": "timm.create_model(num_classes=0)",
            "input_size": self.input_size,
            "preprocess": "RGB resize bicubic, ImageNet/timm normalization",
        }

    @torch.inference_mode()
    def extract_batch(self, images: list[Image.Image]) -> np.ndarray:
        if self.backend == "transformers":
            inputs = self.processor(images=images, return_tensors="pt")
            inputs = {key: value.to(self.device, non_blocking=True) for key, value in inputs.items()}
            with torch.autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
                outputs = self.model(**inputs)
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                features = outputs.pooler_output
            elif hasattr(outputs, "last_hidden_state"):
                features = outputs.last_hidden_state[:, 0]
            else:
                first = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
                features = first[:, 0] if first.ndim == 3 else first
            return features.detach().float().cpu().numpy().astype("float32")
        tensors = []
        for image in images:
            image = image.resize((self.input_size, self.input_size), Image.Resampling.BICUBIC)
            arr = np.asarray(image).astype("float32") / 255.0
            tensor = torch.from_numpy(arr).permute(2, 0, 1)
            tensors.append((tensor - self.mean) / self.std)
        batch = torch.stack(tensors, dim=0).to(self.device, non_blocking=True)
        with torch.autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
            output = self.model(batch)
        if isinstance(output, (tuple, list)):
            output = output[0]
        if output.ndim > 2:
            output = output.mean(dim=tuple(range(2, output.ndim)))
        return output.detach().float().cpu().numpy().astype("float32")


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
                image = Image.fromarray(arr).convert("RGB")
            else:
                image = Image.fromarray(arr[:, :, :3]).convert("RGB")
        else:
            image = self.slide.read_region((record.x, record.y), 0, (record.width, record.height)).convert("RGB")
        if image.size != (record.width, record.height):
            padded = Image.new("RGB", (record.width, record.height), (255, 255, 255))
            padded.paste(image, (0, 0))
            image = padded
        if image.size[0] <= 0 or image.size[1] <= 0:
            raise ValueError(f"Empty patch read for {record.record_id}")
        return image

    def close(self) -> None:
        close = getattr(self.slide, "close", None)
        if callable(close):
            close()


def cache_meta_matches(path: Path, extractor: FeatureExtractor, records: list[PatchRecord]) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            return (
                str(data["model_backend"]) == extractor.backend
                and str(data["model_name"]) == extractor.model_name
                and np.array_equal(data["record_id"].astype(str), np.asarray([record.record_id for record in records], dtype=str))
            )
    except Exception:
        return False


def infer_images(extractor: FeatureExtractor, images: list[Image.Image], features: list[np.ndarray]) -> None:
    if images:
        features.extend(list(extractor.extract_batch(images)))


def extract_case_features(
    args: argparse.Namespace,
    case: CaseBundle,
    records: list[PatchRecord],
    extractor: FeatureExtractor,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    cache_path = args.output_dir / "features" / f"{case.case_id}_features.npz"
    if args.resume and cache_meta_matches(cache_path, extractor, records):
        with np.load(cache_path, allow_pickle=False) as data:
            return data["features"].astype("float32"), [], {
                "case_id": case.case_id,
                "cache_reused": True,
                "cache_path": str(cache_path),
                "patch_count": int(len(records)),
                "extract_seconds": 0.0,
            }

    started = time.perf_counter()
    features: list[np.ndarray] = []
    failures: list[dict[str, Any]] = []
    batches: queue.Queue[list[Image.Image] | Exception | None] = queue.Queue(maxsize=max(1, int(args.prefetch_queue_batches)))

    def producer() -> None:
        reader = WsiPatchReader(case.wsi_path, args.wsi_reader, args.read_workers)
        try:
            images: list[Image.Image] = []
            for record in records:
                try:
                    images.append(reader.read_patch(record))
                except Exception as exc:
                    failures.append({"record_id": record.record_id, "error": f"{type(exc).__name__}: {exc}"})
                    continue
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
        infer_images(extractor, item, features)
    thread.join()
    elapsed = time.perf_counter() - started
    if len(features) + len(failures) != len(records):
        raise RuntimeError(f"Feature/record count mismatch for {case.case_id}: features={len(features)}, failures={len(failures)}, records={len(records)}")
    if failures:
        raise RuntimeError(f"Patch read failures for {case.case_id}: first={failures[0]}")
    feature_array = np.stack(features, axis=0).astype("float32") if features else np.zeros((0, 0), dtype="float32")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        features=feature_array,
        record_id=np.asarray([record.record_id for record in records]),
        case_id=np.asarray([record.case_id for record in records]),
        split=np.asarray([record.split for record in records]),
        bucket=np.asarray([record.bucket for record in records]),
        label_fg=np.asarray([record.label_fg for record in records], dtype="int64"),
        gt_fg=np.asarray([record.gt_fg for record in records], dtype="int64"),
        qwen_fg=np.asarray([record.qwen_fg for record in records], dtype="int64"),
        row=np.asarray([record.row for record in records], dtype="int64"),
        col=np.asarray([record.col for record in records], dtype="int64"),
        x_level0=np.asarray([record.x for record in records], dtype="int64"),
        y_level0=np.asarray([record.y for record in records], dtype="int64"),
        width_level0=np.asarray([record.width for record in records], dtype="int64"),
        height_level0=np.asarray([record.height for record in records], dtype="int64"),
        model_backend=np.asarray(extractor.backend),
        model_name=np.asarray(extractor.model_name),
        wsi_reader=np.asarray(args.wsi_reader),
        read_workers=np.asarray(args.read_workers),
        created_at=np.asarray(datetime.now(timezone.utc).isoformat()),
    )
    return feature_array, [], {
        "case_id": case.case_id,
        "cache_reused": False,
        "cache_path": str(cache_path),
        "patch_count": int(len(records)),
        "extract_seconds": float(elapsed),
        "patches_per_second": float(len(records) / elapsed) if elapsed > 0 else 0.0,
    }


def build_feature_inputs(
    bundles: list[CaseBundle],
    samples_by_size: dict[int, list[PatchRecord]],
) -> dict[str, list[PatchRecord]]:
    required: dict[str, dict[str, PatchRecord]] = defaultdict(dict)
    for records in samples_by_size.values():
        for record in records:
            required[record.case_id][record.record_id] = record
    for bundle in bundles:
        if bundle.records and bundle.records[0].split == "heldout":
            for record in bundle.records:
                required[record.case_id][record.record_id] = record
    return {
        case_id: sorted(records.values(), key=lambda record: (record.row, record.col, record.bucket))
        for case_id, records in required.items()
    }


def fit_linear_probe(x: np.ndarray, y: np.ndarray, seed: int) -> Any:
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, solver="lbfgs", class_weight="balanced", random_state=seed),
    )
    model.fit(x, y)
    return model


def predict_prob(model: Any, x: np.ndarray) -> np.ndarray:
    return model.predict_proba(x)[:, 1]


def safe_metric(fn: Any, y: np.ndarray, prob: np.ndarray) -> float:
    try:
        return float(fn(y, prob))
    except Exception:
        return float("nan")


def metric_row(
    *,
    sample_size: int,
    scope: str,
    case_id: str,
    y: np.ndarray,
    prob: np.ndarray,
    buckets: np.ndarray,
    threshold: float,
    sample_counts: dict[str, Any],
) -> dict[str, Any]:
    pred = (prob >= threshold).astype("int64")
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    precision, recall, f1, _ = precision_recall_fscore_support(y, pred, labels=[1], average="binary", zero_division=0)
    hard_mask = buckets == "hard_negative"
    easy_mask = buckets == "easy_negative"

    def specificity(mask: np.ndarray) -> float:
        denom = int(mask.sum())
        if denom == 0:
            return float("nan")
        return float(((pred[mask] == 0) & (y[mask] == 0)).sum() / denom)

    def fp_rate(mask: np.ndarray) -> float:
        denom = int(mask.sum())
        if denom == 0:
            return float("nan")
        return float(((pred[mask] == 1) & (y[mask] == 0)).sum() / denom)

    return {
        "sample_size_per_wsi": sample_size,
        "scope": scope,
        "case_id": case_id,
        "n_eval": int(len(y)),
        "eval_fg": int((y == 1).sum()),
        "eval_bg": int((y == 0).sum()),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision_fg": float(precision),
        "recall_fg": float(recall),
        "f1_fg": float(f1),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "roc_auc": safe_metric(roc_auc_score, y, prob),
        "average_precision": safe_metric(average_precision_score, y, prob),
        "gt_background_false_positive_rate": float(fp / max(1, fp + tn)),
        "hard_negative_specificity": specificity(hard_mask),
        "hard_negative_false_positive_rate": fp_rate(hard_mask),
        "easy_negative_specificity": specificity(easy_mask),
        "easy_negative_false_positive_rate": fp_rate(easy_mask),
        **sample_counts,
    }


def get_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size)
    except Exception:
        return ImageFont.load_default()


def fit_image(image: Image.Image, width: int, height: int) -> Image.Image:
    image = image.convert("RGB")
    scale = min(width / image.width, height / image.height)
    resized = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), LANCZOS)
    canvas = Image.new("RGB", (width, height), "white")
    canvas.paste(resized, ((width - resized.width) // 2, (height - resized.height) // 2))
    return canvas


def draw_prediction_overlay(
    case: CaseBundle,
    base_crop: Image.Image,
    records: list[PatchRecord],
    prob: np.ndarray,
    threshold: float,
    title: str,
) -> Image.Image:
    crop = base_crop.convert("RGBA")
    overlay = Image.new("RGBA", crop.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    pred = (prob >= threshold).astype("int64")
    for record, is_fg in zip(records, pred):
        rect = local_cell_rect_level0(record.row, record.col, case.bbox_level0, crop.size, record.width)
        if rect is None:
            continue
        if record.label_fg == 1 and int(is_fg) == 1:
            fill = (46, 204, 113, 125)
        elif record.label_fg == 0 and int(is_fg) == 1:
            fill = (231, 76, 60, 150)
        elif record.label_fg == 1 and int(is_fg) == 0:
            fill = (52, 152, 219, 130)
        else:
            continue
        draw.rectangle(rect, fill=fill)
    out = Image.alpha_composite(crop, overlay).convert("RGB")
    draw_out = ImageDraw.Draw(out)
    draw_out.rectangle((0, 0, out.width - 1, out.height - 1), outline=(80, 80, 80), width=1)
    draw_out.text((8, 8), title, fill=(0, 0, 0), font=get_font(16, bold=True))
    return out


def draw_gt_overlay(case: CaseBundle, base_crop: Image.Image) -> Image.Image:
    crop = base_crop.convert("RGBA")
    overlay = Image.new("RGBA", crop.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for record in case.records:
        if record.label_fg != 1:
            continue
        rect = local_cell_rect_level0(record.row, record.col, case.bbox_level0, crop.size, record.width)
        if rect is not None:
            draw.rectangle(rect, fill=(46, 204, 113, 110))
    out = Image.alpha_composite(crop, overlay).convert("RGB")
    draw_out = ImageDraw.Draw(out)
    draw_out.text((8, 8), "GT foreground", fill=(0, 0, 0), font=get_font(16, bold=True))
    return out


def render_metric_plot(output_dir: Path, metrics: list[dict[str, Any]]) -> Path:
    import matplotlib.pyplot as plt

    overall = [row for row in metrics if row["scope"] == "overall"]
    xs = [int(row["sample_size_per_wsi"]) for row in overall]
    precision = [float(row["precision_fg"]) for row in overall]
    recall = [float(row["recall_fg"]) for row in overall]
    f1 = [float(row["f1_fg"]) for row in overall]
    bg_fpr = [float(row["gt_background_false_positive_rate"]) for row in overall]
    hard_spec = [float(row["hard_negative_specificity"]) for row in overall]
    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    ax1.plot(xs, precision, marker="o", label="FG precision")
    ax1.plot(xs, recall, marker="o", label="FG recall")
    ax1.plot(xs, f1, marker="o", label="FG F1")
    ax1.plot(xs, hard_spec, marker="o", label="hard-negative specificity")
    ax1.set_xscale("log")
    ax1.set_ylim(0.0, 1.02)
    ax1.set_xlabel("training samples per WSI per class")
    ax1.set_ylabel("score")
    ax1.grid(True, alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(xs, bg_fpr, marker="s", color="#d62728", linestyle="--", label="GT-bg FP rate")
    ax2.set_ylim(0.0, max(0.05, min(1.0, max(bg_fpr) * 1.2)))
    ax2.set_ylabel("false-positive rate")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best")
    fig.tight_layout()
    path = output_dir / "plots/sample_efficiency_metrics.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def draw_wrapped_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font: ImageFont.ImageFont, width_chars: int) -> int:
    x, y = xy
    for line in textwrap.wrap(text, width=width_chars):
        draw.text((x, y), line, fill=(35, 35, 35), font=font)
        bbox = draw.textbbox((x, y), line, font=font)
        y += (bbox[3] - bbox[1]) + 6
    return y


def render_review_pdf(
    args: argparse.Namespace,
    bundles: list[CaseBundle],
    metrics: list[dict[str, Any]],
    predictions_by_size_case: dict[tuple[int, str], tuple[list[PatchRecord], np.ndarray]],
    plot_path: Path,
) -> Path:
    pages_dir = args.output_dir / "review_pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    pages: list[Image.Image] = []
    page = Image.new("RGB", (1600, 1100), "white")
    draw = ImageDraw.Draw(page)
    draw.text((40, 32), "Stress-32 GT-overlay DINOv3 sample-efficiency probe", fill=(0, 0, 0), font=get_font(34, bold=True))
    y = 92
    y = draw_wrapped_text(
        draw,
        (40, y),
        "Foreground training pool is all GT-overlay foreground inside the selected qwen-stage1 bbox. Background is split into hard negatives (qwen foreground, GT background) and easy negatives (qwen background, GT background). Primary readout is false-positive reduction on held-out WSIs.",
        get_font(18),
        125,
    )
    plot = fit_image(Image.open(plot_path).convert("RGB"), 900, 560)
    page.paste(plot, (40, 180))
    overall = [row for row in metrics if row["scope"] == "overall"]
    table_x = 980
    table_y = 190
    draw.text((table_x, table_y - 36), "Held-out overall metrics", fill=(0, 0, 0), font=get_font(21, bold=True))
    header = "N     Prec   BG-FPR  HardSpec Recall  F1"
    draw.text((table_x, table_y), header, fill=(0, 0, 0), font=get_font(16, bold=True))
    y2 = table_y + 30
    for row in overall:
        text = (
            f"{int(row['sample_size_per_wsi']):<5} "
            f"{float(row['precision_fg']):.3f}  "
            f"{float(row['gt_background_false_positive_rate']):.3f}   "
            f"{float(row['hard_negative_specificity']):.3f}    "
            f"{float(row['recall_fg']):.3f}  "
            f"{float(row['f1_fg']):.3f}"
        )
        draw.text((table_x, y2), text, fill=(25, 25, 25), font=get_font(16))
        y2 += 28
    draw.text((40, 790), "Overlay colors: green=true positive, red=false positive, blue=false negative. Blank cells are true background.", fill=(50, 50, 50), font=get_font(18))
    pages.append(page)
    page.save(pages_dir / "page_001.png")

    review_sizes = [size for size in parse_int_list(args.review_sample_sizes) if any(row["sample_size_per_wsi"] == size for row in overall)]
    if not review_sizes:
        review_sizes = [int(overall[-1]["sample_size_per_wsi"])] if overall else []
    heldout_cases = [bundle for bundle in bundles if bundle.records and bundle.records[0].split == "heldout"][: args.max_review_cases]
    page_idx = 2
    for case in heldout_cases:
        base = read_bbox_preview(case, args.max_review_dim)
        panels = [("GT foreground", draw_gt_overlay(case, base))]
        for size in review_sizes[:3]:
            records, prob = predictions_by_size_case[(size, case.case_id)]
            panels.append((f"N={size}", draw_prediction_overlay(case, base, records, prob, args.probe_threshold, f"N={size}")))
        page = Image.new("RGB", (1600, 1100), "white")
        draw = ImageDraw.Draw(page)
        draw.text((36, 30), f"{case.case_id} | held-out bbox prediction overlays", fill=(0, 0, 0), font=get_font(28, bold=True))
        draw.text((36, 66), f"bbox_level0={list(case.bbox_level0)}", fill=(60, 60, 60), font=get_font(16))
        panel_w = 740
        panel_h = 430
        for idx, (_label, panel) in enumerate(panels[:4]):
            x = 36 + (idx % 2) * (panel_w + 42)
            y = 115 + (idx // 2) * (panel_h + 65)
            fitted = fit_image(panel, panel_w, panel_h)
            page.paste(fitted, (x, y))
            draw.rectangle((x, y, x + panel_w - 1, y + panel_h - 1), outline=(170, 170, 170), width=1)
        pages.append(page)
        page.save(pages_dir / f"page_{page_idx:03d}.png")
        page_idx += 1
    pdf_path = args.output_dir / "stress32_gt_overlay_sample_efficiency_probe_review.pdf"
    pages[0].save(pdf_path, save_all=True, append_images=pages[1:])
    return pdf_path


def write_reproduction(args: argparse.Namespace, summary: dict[str, Any]) -> Path:
    command = " ".join(shlex.quote(part) for part in [sys.executable, *sys.argv])
    path = args.output_dir / "reproduction.txt"
    content = f"""Generated timestamp: {datetime.now(timezone.utc).isoformat()}
Ticket: {args.ticket}
Source repository: {REPO_ROOT}
Working directory: {Path.cwd()}
Git commit: {git_commit()}
Dirty-worktree status at creation time:
{git_status_short()}
DVC status:
{dvc_status_text()}

Exact command:
{command}

Inputs:
- GT root: {args.gt_root.resolve()}
- qwen2b root: {args.qwen_root.resolve()}
- qwen2b run id: {args.run_id}
- source review PDF: {args.source_pdf.resolve() if args.source_pdf.exists() else args.source_pdf}
- selected bbox source: qwen-stage1 stage1/bboxes.json

Experiment:
- foreground pool: GT foreground inside selected qwen-stage1 bbox
- hard negative pool: qwen foreground and GT background inside selected qwen-stage1 bbox
- easy negative pool: qwen background and GT background inside selected qwen-stage1 bbox
- heldout case ids: {args.holdout_case_ids}
- sample sizes per training WSI: {args.sample_sizes}
- sample seed: {args.sample_seed}
- DINO model: {args.model_backend} / {args.model_name}
- wsi reader: {args.wsi_reader}; read workers: {args.read_workers}; batch size: {args.batch_size}

Outputs:
- output root: {args.output_dir.resolve()}
- review PDF: {summary.get("review_pdf")}
- metrics CSV: {summary.get("metrics_csv")}
- pool census CSV: {summary.get("pool_census_csv")}

Credentials:
- HF_TOKEN or HUGGING_FACE_HUB_TOKEN must be set for gated DINOv3 model access. Token values are intentionally not recorded.

Notes:
- Feature caches store DINO embeddings and patch metadata, not raw WSI pixels.
- Results are deterministic for the recorded sample seed except for GPU/library numerical variation.
"""
    path.write_text(content)
    return path


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_sizes = parse_int_list(args.sample_sizes)
    started = time.perf_counter()
    bundles, census_rows = load_case_bundles(args)
    write_csv(args.output_dir / "pool_census.csv", census_rows)
    split_rows = [
        {
            "case_id": row["case_id"],
            "split": row["split"],
            "foreground_gt": row["foreground_gt"],
            "hard_negative": row["hard_negative"],
            "easy_negative": row["easy_negative"],
            "hard_negative_burden_inside_bbox": row["hard_negative_burden_inside_bbox"],
        }
        for row in census_rows
    ]
    write_csv(args.output_dir / "split_manifest.csv", split_rows)
    all_candidate_rows = [patch_record_row(record) for bundle in bundles for record in bundle.records]
    write_csv(args.output_dir / "candidate_cells.csv", all_candidate_rows)
    samples_by_size, sample_rows, sample_count_rows = build_sample_manifests(args, bundles, sample_sizes)
    write_csv(args.output_dir / "sample_manifest_all_sizes.csv", sample_rows)
    write_csv(args.output_dir / "sample_count_summary.csv", sample_count_rows)
    sample_dir = args.output_dir / "sampled_manifests"
    for size, records in samples_by_size.items():
        write_csv(sample_dir / f"N{size:03d}.csv", [patch_record_row(record) for record in records])

    print(f"[setup] cases={len(bundles)} train={sum(1 for b in bundles if b.records and b.records[0].split == 'train')} heldout={sum(1 for b in bundles if b.records and b.records[0].split == 'heldout')}", flush=True)
    feature_inputs = build_feature_inputs(bundles, samples_by_size)
    feature_input_rows = [patch_record_row(record) for records in feature_inputs.values() for record in records]
    write_csv(args.output_dir / "feature_input_manifest.csv", feature_input_rows)
    extractor = FeatureExtractor(args)
    feature_by_record: dict[str, np.ndarray] = {}
    feature_rows: list[dict[str, Any]] = []
    bundle_by_case = {bundle.case_id: bundle for bundle in bundles}
    for case_id in sorted(feature_inputs):
        records = feature_inputs[case_id]
        print(f"[features] {case_id}: {len(records)} patches", flush=True)
        features, _failures, meta = extract_case_features(args, bundle_by_case[case_id], records, extractor)
        feature_rows.append(meta)
        for record, feature in zip(records, features):
            feature_by_record[record.record_id] = feature
    write_csv(args.output_dir / "feature_cache_summary.csv", feature_rows)

    metrics: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    predictions_by_size_case: dict[tuple[int, str], tuple[list[PatchRecord], np.ndarray]] = {}
    heldout_records_by_case = {
        bundle.case_id: sorted(bundle.records, key=lambda record: (record.row, record.col))
        for bundle in bundles
        if bundle.records and bundle.records[0].split == "heldout"
    }
    sample_count_by_size: dict[int, dict[str, Any]] = {}
    for size, records in samples_by_size.items():
        counts = defaultdict(int)
        for record in records:
            counts[record.bucket] += 1
        sample_count_by_size[size] = {
            "train_fg_samples": int(counts["foreground_gt"]),
            "train_bg_samples": int(counts["hard_negative"] + counts["easy_negative"]),
            "train_hard_negative_samples": int(counts["hard_negative"]),
            "train_easy_negative_samples": int(counts["easy_negative"]),
            "train_total_samples": int(len(records)),
        }
    for size in sample_sizes:
        records = samples_by_size[size]
        x_train = np.stack([feature_by_record[record.record_id] for record in records], axis=0).astype("float32")
        y_train = np.asarray([record.label_fg for record in records], dtype="int64")
        model = fit_linear_probe(x_train, y_train, args.sample_seed + size)
        all_y: list[int] = []
        all_prob: list[float] = []
        all_bucket: list[str] = []
        for case_id, eval_records in heldout_records_by_case.items():
            x_eval = np.stack([feature_by_record[record.record_id] for record in eval_records], axis=0).astype("float32")
            y_eval = np.asarray([record.label_fg for record in eval_records], dtype="int64")
            buckets = np.asarray([record.bucket for record in eval_records])
            prob = predict_prob(model, x_eval)
            predictions_by_size_case[(size, case_id)] = (eval_records, prob)
            row = metric_row(
                sample_size=size,
                scope="case",
                case_id=case_id,
                y=y_eval,
                prob=prob,
                buckets=buckets,
                threshold=args.probe_threshold,
                sample_counts=sample_count_by_size[size],
            )
            metrics.append(row)
            all_y.extend(int(x) for x in y_eval)
            all_prob.extend(float(x) for x in prob)
            all_bucket.extend(str(x) for x in buckets)
            pred = (prob >= args.probe_threshold).astype("int64")
            for record, p, pred_fg in zip(eval_records, prob, pred):
                pred_row = patch_record_row(record)
                pred_row.update(
                    {
                        "sample_size_per_wsi": size,
                        "prob_fg": float(p),
                        "pred_fg": int(pred_fg),
                    }
                )
                predictions.append(pred_row)
        metrics.append(
            metric_row(
                sample_size=size,
                scope="overall",
                case_id="ALL_HELDOUT",
                y=np.asarray(all_y, dtype="int64"),
                prob=np.asarray(all_prob, dtype="float32"),
                buckets=np.asarray(all_bucket),
                threshold=args.probe_threshold,
                sample_counts=sample_count_by_size[size],
            )
        )
    metrics.sort(key=lambda row: (int(row["sample_size_per_wsi"]), row["scope"], row["case_id"]))
    write_csv(args.output_dir / "metrics.csv", metrics)
    write_csv(args.output_dir / "heldout_patch_predictions.csv", predictions)
    plot_path = render_metric_plot(args.output_dir, metrics)
    pdf_path = render_review_pdf(args, bundles, metrics, predictions_by_size_case, plot_path)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "ticket": args.ticket,
        "output_dir": str(args.output_dir.resolve()),
        "source_pdf": str(args.source_pdf.resolve()) if args.source_pdf.exists() else str(args.source_pdf),
        "case_count": len(bundles),
        "train_case_count": sum(1 for bundle in bundles if bundle.records and bundle.records[0].split == "train"),
        "heldout_case_count": sum(1 for bundle in bundles if bundle.records and bundle.records[0].split == "heldout"),
        "holdout_case_ids": parse_str_list(args.holdout_case_ids),
        "sample_sizes": sample_sizes,
        "pool_census_csv": str((args.output_dir / "pool_census.csv").resolve()),
        "split_manifest_csv": str((args.output_dir / "split_manifest.csv").resolve()),
        "candidate_cells_csv": str((args.output_dir / "candidate_cells.csv").resolve()),
        "sample_manifest_csv": str((args.output_dir / "sample_manifest_all_sizes.csv").resolve()),
        "feature_input_manifest_csv": str((args.output_dir / "feature_input_manifest.csv").resolve()),
        "feature_cache_summary_csv": str((args.output_dir / "feature_cache_summary.csv").resolve()),
        "metrics_csv": str((args.output_dir / "metrics.csv").resolve()),
        "heldout_patch_predictions_csv": str((args.output_dir / "heldout_patch_predictions.csv").resolve()),
        "plot_path": str(plot_path.resolve()),
        "review_pdf": str(pdf_path.resolve()),
        "model": extractor.meta,
        "package_versions": package_versions(),
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    write_json(args.output_dir / "summary.json", summary)
    repro_path = write_reproduction(args, summary)
    summary["reproduction_txt"] = str(repro_path.resolve())
    write_json(args.output_dir / "summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    summary = run_experiment(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
