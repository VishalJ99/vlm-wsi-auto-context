#!/usr/bin/env python3
"""Train a stress-32 DINOv3 probe from qwen2b hard/easy negatives."""

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
import zlib
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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from build_per_wsi_probe_unselected_transfer_demo import (  # noqa: E402
    PatchRecord as ScalePatchRecord,
    draw_prediction_grid_panel,
    make_contact_sheet,
    make_page,
)
from train_per_wsi_dinov3_fg_bg_probe import (  # noqa: E402
    FeatureExtractor,
    WsiPatchReader,
    package_versions,
)
from train_pooled_dinov3_probe_transfer import (  # noqa: E402
    draw_detector_overview_with_stats,
    load_case_bundles,
)
from run_stress32_yolo_probe import (  # noqa: E402
    Detection,
    StressCase,
    assign_gt,
    build_patch_records,
    detection_rows,
    load_detections,
    load_stress_cases,
    mask_metrics,
    suppress_contained_detections,
)


DEFAULT_STRESS_DATASET_ROOT = (
    Path("/data2/vj724/wsi-agents")
    / "vlm_gt_seg_comparison_experiment/dataset_thumbnails_harder_jones_leica"
)
DEFAULT_QWEN2B_ROOT = (
    Path("/data2/vj724/wsi-agents")
    / "new_vlm_gt_preds/leica_hard_jones_evg_qwen_2b_zero_shot"
)
DEFAULT_STRESS_YOLO_SOURCE = REPO_ROOT / "runs/stress32_yolo_dinov3_probe_v1"
DEFAULT_SCALE500_FEATURE_RUN = REPO_ROOT / "runs/scale500_selected_dinov3small_features_sample100_per_wsi_v1"
DEFAULT_SCALE500_TRANSFER_FEATURES = (
    DEFAULT_SCALE500_FEATURE_RUN / "visuals/pooled_probe_transfer_20perstain_v1/features"
)
DEFAULT_SCALE500_TRANSFER_MANIFEST = (
    DEFAULT_SCALE500_FEATURE_RUN / "visuals/pooled_probe_transfer_20perstain_v1/transfer_case_manifest_20perstain.csv"
)
DEFAULT_SELECTOR_MANIFEST = (
    REPO_ROOT
    / "runs/auto_context_scale500_selector_all500_v1/manifests/completed_cases_500_20260604_openrouter_review_current.csv"
)
DEFAULT_DETECTOR_ROOT = REPO_ROOT / "runs/detector_pipeline_scale500_v1"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "runs/stress32_qwen2b_hard_negative_probe_v1"
DEFAULT_DINOV3_SMALL = "facebook/dinov3-vits16-pretrain-lvd1689m"
DEFAULT_DINOV2_SMALL = "vit_small_patch14_dinov2"


@dataclass(frozen=True)
class SampleRecord:
    case_id: str
    row: int
    col: int
    x: int
    y: int
    width: int
    height: int
    bucket: str
    label_fg: int
    gt_fg: int
    qwen2b_fg: int

    @property
    def record_id(self) -> str:
        return f"{self.case_id}|{self.bucket}|r{self.row}c{self.col}|{self.x}_{self.y}"


@dataclass(frozen=True)
class YoloPatchRecord:
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
        return f"{self.case_id}|det{dets}|r{self.row}c{self.col}|{self.x}_{self.y}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ticket", default="PER-250")
    parser.add_argument("--stress-dataset-root", type=Path, default=DEFAULT_STRESS_DATASET_ROOT)
    parser.add_argument("--qwen2b-root", type=Path, default=DEFAULT_QWEN2B_ROOT)
    parser.add_argument("--run-id", default="harder_jones_leica_manual")
    parser.add_argument("--scanner", default="leica")
    parser.add_argument("--stain", default="jones")
    parser.add_argument("--case-ids", default="")
    parser.add_argument("--case-limit", type=int, default=None)
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--sample-seed", type=int, default=250)
    parser.add_argument("--hard-negatives-per-wsi", type=int, default=100)
    parser.add_argument("--easy-negatives-per-wsi", type=int, default=100)
    parser.add_argument("--foreground-per-wsi", type=int, default=200)
    parser.add_argument("--model-backend", choices=["transformers", "timm"], default="transformers")
    parser.add_argument("--model-name", default=DEFAULT_DINOV3_SMALL)
    parser.add_argument("--fallback-model-name", default=DEFAULT_DINOV2_SMALL)
    parser.add_argument("--allow-timm-fallback", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--pipeline-mode", choices=["serial", "prefetch"], default="prefetch")
    parser.add_argument("--prefetch-queue-batches", type=int, default=4)
    parser.add_argument("--input-size", type=int, default=None)
    parser.add_argument("--wsi-reader", choices=["cucim", "openslide"], default="cucim")
    parser.add_argument("--read-workers", type=int, default=16)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--stress-yolo-source-dir", type=Path, default=DEFAULT_STRESS_YOLO_SOURCE)
    parser.add_argument("--suppress-contained", action="store_true", default=True)
    parser.add_argument("--no-suppress-contained", dest="suppress_contained", action="store_false")
    parser.add_argument("--containment-threshold", type=float, default=0.90)
    parser.add_argument("--probe-threshold", type=float, default=0.50)
    parser.add_argument("--scale500-feature-run", type=Path, default=DEFAULT_SCALE500_FEATURE_RUN)
    parser.add_argument("--scale500-transfer-feature-dir", type=Path, default=DEFAULT_SCALE500_TRANSFER_FEATURES)
    parser.add_argument("--scale500-transfer-manifest", type=Path, default=DEFAULT_SCALE500_TRANSFER_MANIFEST)
    parser.add_argument("--selector-manifest", type=Path, default=DEFAULT_SELECTOR_MANIFEST)
    parser.add_argument("--detector-root", type=Path, default=DEFAULT_DETECTOR_ROOT)
    parser.add_argument("--scale500-cases-per-stain", type=int, default=4)
    parser.add_argument("--max-scale500-crop-panels", type=int, default=4)
    parser.add_argument("--max-overview-width", type=int, default=760)
    parser.add_argument("--max-grid-dim", type=int, default=500)
    parser.add_argument("--skip-scale500-qual", action="store_true")
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


def dvc_status_text() -> str:
    try:
        return subprocess.check_output(["dvc", "status"], cwd=REPO_ROOT, text=True, stderr=subprocess.STDOUT)
    except Exception as exc:
        return f"dvc status unavailable: {type(exc).__name__}: {exc}\n"


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


def case_seed(base_seed: int, case_id: str) -> int:
    return int(base_seed + zlib.crc32(case_id.encode("utf-8")))


def qwen_mask_path(args: argparse.Namespace, case_id: str) -> Path:
    base = args.qwen2b_root / case_id / args.run_id
    preferred = base / "stage7_new/mask.npy"
    if preferred.exists():
        return preferred
    fallback = base / "stage7/mask.npy"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"No qwen2b stage7_new/stage7 mask for {case_id} under {base}")


def load_cases_for_qwen2b(args: argparse.Namespace) -> list[StressCase]:
    ns = argparse.Namespace(
        stress_dataset_root=args.stress_dataset_root,
        stress_pipeline_root=args.qwen2b_root,
        run_id=args.run_id,
        scanner=args.scanner,
        stain=args.stain,
        case_ids=args.case_ids,
        case_limit=args.case_limit,
    )
    return load_stress_cases(ns)


def qwen_stage6_patch_lookup(args: argparse.Namespace, case: StressCase, mask_shape: tuple[int, int]) -> dict[tuple[int, int], tuple[int, int, int, int]]:
    run_dir = args.qwen2b_root / case.case_id / args.run_id
    lookup: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    for csv_path in sorted(run_dir.glob("bboxes/*/stage6/patches.csv")):
        with csv_path.open() as f:
            for row in csv.DictReader(f):
                x = int(row["wsi_x"])
                y = int(row["wsi_y"])
                width = int(args.patch_size)
                height = int(args.patch_size)
                rr = int(y // args.patch_size)
                cc = int(x // args.patch_size)
                if 0 <= rr < mask_shape[0] and 0 <= cc < mask_shape[1]:
                    lookup.setdefault((rr, cc), (x, y, width, height))
    if not lookup:
        raise FileNotFoundError(f"No qwen2b stage6 patch coordinates for {case.case_id}: {run_dir}/bboxes/*/stage6/patches.csv")
    return lookup


def sample_cells(
    case: StressCase,
    gt: np.ndarray,
    qwen: np.ndarray,
    bucket: str,
    mask: np.ndarray,
    count: int,
    seed: int,
    patch_lookup: dict[tuple[int, int], tuple[int, int, int, int]],
) -> list[SampleRecord]:
    coords = np.argwhere(mask)
    if not len(coords) or count <= 0:
        return []
    rng = np.random.default_rng(seed)
    take = min(count, len(coords))
    idxs = rng.choice(np.arange(len(coords)), size=take, replace=False)
    rows: list[SampleRecord] = []
    for idx in sorted(int(i) for i in idxs):
        rr = int(coords[idx, 0])
        cc = int(coords[idx, 1])
        if (rr, cc) not in patch_lookup:
            continue
        x, y, width, height = patch_lookup[(rr, cc)]
        rows.append(
            SampleRecord(
                case_id=case.case_id,
                row=rr,
                col=cc,
                x=x,
                y=y,
                width=width,
                height=height,
                bucket=bucket,
                label_fg=1 if bool(gt[rr, cc]) else 0,
                gt_fg=1 if bool(gt[rr, cc]) else 0,
                qwen2b_fg=1 if bool(qwen[rr, cc]) else 0,
            )
        )
    return rows


def build_census_and_samples(args: argparse.Namespace, cases: list[StressCase]) -> tuple[list[dict[str, Any]], list[SampleRecord]]:
    census_rows: list[dict[str, Any]] = []
    samples: list[SampleRecord] = []
    for case in cases:
        gt = np.load(case.gt_mask_path).astype(bool)
        qwen_path = qwen_mask_path(args, case.case_id)
        qwen = np.load(qwen_path).astype(bool)
        if gt.shape != qwen.shape:
            raise ValueError(f"Mask shape mismatch for {case.case_id}: gt={gt.shape}, qwen={qwen.shape}")
        patch_lookup = qwen_stage6_patch_lookup(args, case, gt.shape)
        stage6_cells = np.zeros(gt.shape, dtype=bool)
        for rr, cc in patch_lookup:
            stage6_cells[rr, cc] = True
        tp = qwen & gt
        fp = qwen & ~gt
        tn = ~qwen & ~gt
        fn = ~qwen & gt
        sample_fp = fp & stage6_cells
        sample_tn = tn & stage6_cells
        sample_fg = gt & stage6_cells
        census_rows.append(
            {
                "case_id": case.case_id,
                "gt_mask_path": str(case.gt_mask_path),
                "qwen2b_mask_path": str(qwen_path),
                "qwen2b_stage6_patch_cells": int(stage6_cells.sum()),
                "grid_rows": int(gt.shape[0]),
                "grid_cols": int(gt.shape[1]),
                "grid_cells": int(gt.size),
                "gt_fg": int(gt.sum()),
                "qwen2b_fg": int(qwen.sum()),
                "true_positive": int(tp.sum()),
                "false_positive_hard_negative": int(fp.sum()),
                "true_negative_easy_negative": int(tn.sum()),
                "false_negative": int(fn.sum()),
                "sample_pool_gt_fg": int(sample_fg.sum()),
                "sample_pool_fp_hard_negative": int(sample_fp.sum()),
                "sample_pool_tn_easy_negative": int(sample_tn.sum()),
                "qwen2b_precision": float(tp.sum() / max(1, tp.sum() + fp.sum())),
                "qwen2b_recall": float(tp.sum() / max(1, tp.sum() + fn.sum())),
            }
        )
        seed = case_seed(args.sample_seed, case.case_id)
        samples.extend(sample_cells(case, gt, qwen, "hard_negative_fp", sample_fp, args.hard_negatives_per_wsi, seed + 11, patch_lookup))
        samples.extend(sample_cells(case, gt, qwen, "easy_negative_tn", sample_tn, args.easy_negatives_per_wsi, seed + 23, patch_lookup))
        samples.extend(sample_cells(case, gt, qwen, "foreground_gt", sample_fg, args.foreground_per_wsi, seed + 37, patch_lookup))
    return census_rows, sorted(samples, key=lambda r: (r.case_id, r.bucket, r.row, r.col))


def sample_row(record: SampleRecord, index: int) -> dict[str, Any]:
    return {
        "sample_index": index,
        "record_id": record.record_id,
        "case_id": record.case_id,
        "bucket": record.bucket,
        "label_fg": record.label_fg,
        "gt_fg": record.gt_fg,
        "qwen2b_fg": record.qwen2b_fg,
        "row": record.row,
        "col": record.col,
        "x_level0": record.x,
        "y_level0": record.y,
        "width_level0": record.width,
        "height_level0": record.height,
    }


def cache_meta_matches(path: Path, extractor: FeatureExtractor) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            return str(data["model_backend"]) == extractor.backend and str(data["model_name"]) == extractor.model_name
    except Exception:
        return False


def infer_images(extractor: FeatureExtractor, images: list[Image.Image], features: list[np.ndarray]) -> None:
    if images:
        features.extend(list(extractor.extract_batch(images)))


def extract_sample_features(
    args: argparse.Namespace,
    case: StressCase,
    records: list[SampleRecord],
    extractor: FeatureExtractor,
) -> tuple[np.ndarray, dict[str, Any]]:
    cache_path = args.output_dir / "features/sampled_training" / f"{case.case_id}_sampled_qwen2b_features.npz"
    if args.resume and cache_meta_matches(cache_path, extractor):
        with np.load(cache_path, allow_pickle=False) as data:
            core_fields = {"sample_index", "row", "col", "x_level0", "y_level0", "bucket"}
            core_matches = core_fields.issubset(set(data.files)) and (
                len(data["sample_index"]) == len(records)
                and np.array_equal(data["row"].astype("int64"), np.asarray([r.row for r in records], dtype="int64"))
                and np.array_equal(data["col"].astype("int64"), np.asarray([r.col for r in records], dtype="int64"))
                and np.array_equal(data["x_level0"].astype("int64"), np.asarray([r.x for r in records], dtype="int64"))
                and np.array_equal(data["y_level0"].astype("int64"), np.asarray([r.y for r in records], dtype="int64"))
                and np.array_equal(data["bucket"].astype(str), np.asarray([r.bucket for r in records], dtype=str))
            )
            if {"width_level0", "height_level0"}.issubset(set(data.files)):
                shape_matches = (
                    np.array_equal(data["width_level0"].astype("int64"), np.asarray([r.width for r in records], dtype="int64"))
                    and np.array_equal(data["height_level0"].astype("int64"), np.asarray([r.height for r in records], dtype="int64"))
                )
            else:
                shape_matches = True
            cache_matches_records = core_matches and shape_matches
            if cache_matches_records:
                return data["features"].astype("float32"), {
                    "case_id": case.case_id,
                    "cache_reused": True,
                    "cache_path": str(cache_path),
                    "patch_count": int(len(records)),
                    "extract_seconds": 0.0,
                }

    started = time.perf_counter()
    features: list[np.ndarray] = []
    if args.pipeline_mode == "serial":
        reader = WsiPatchReader(case.wsi_path, args.wsi_reader, args.read_workers)
        images: list[Image.Image] = []
        try:
            for record in records:
                images.append(reader.read_patch(record))
                if len(images) >= extractor.batch_size:
                    infer_images(extractor, images, features)
                    images = []
            infer_images(extractor, images, features)
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

        thread = threading.Thread(target=producer, name=f"sample-prefetch-{case.case_id}", daemon=True)
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
    feature_array = np.stack(features, axis=0).astype("float32") if features else np.zeros((0, 0), dtype="float32")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        features=feature_array,
        sample_index=np.asarray([i for i in range(len(records))], dtype="int64"),
        bucket=np.asarray([r.bucket for r in records]),
        label_fg=np.asarray([r.label_fg for r in records], dtype="int64"),
        row=np.asarray([r.row for r in records], dtype="int64"),
        col=np.asarray([r.col for r in records], dtype="int64"),
        x_level0=np.asarray([r.x for r in records], dtype="int64"),
        y_level0=np.asarray([r.y for r in records], dtype="int64"),
        width_level0=np.asarray([r.width for r in records], dtype="int64"),
        height_level0=np.asarray([r.height for r in records], dtype="int64"),
        model_backend=np.asarray(extractor.backend),
        model_name=np.asarray(extractor.model_name),
        wsi_reader=np.asarray(args.wsi_reader),
        read_workers=np.asarray(args.read_workers),
        pipeline_mode=np.asarray(args.pipeline_mode),
        created_at=np.asarray(datetime.now(timezone.utc).isoformat()),
    )
    return feature_array, {
        "case_id": case.case_id,
        "cache_reused": False,
        "cache_path": str(cache_path),
        "patch_count": int(len(records)),
        "extract_seconds": float(elapsed),
        "patches_per_second": float(len(records) / elapsed) if elapsed > 0 else 0.0,
    }


def extract_all_sample_features(
    args: argparse.Namespace,
    cases: list[StressCase],
    samples: list[SampleRecord],
    extractor: FeatureExtractor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    case_by_id = {case.case_id: case for case in cases}
    by_case: dict[str, list[SampleRecord]] = defaultdict(list)
    for record in samples:
        by_case[record.case_id].append(record)
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    case_parts: list[np.ndarray] = []
    meta_rows: list[dict[str, Any]] = []
    for case_id in sorted(by_case):
        records = by_case[case_id]
        print(f"[features] sampled {case_id}: {len(records)} patches", flush=True)
        x_case, meta = extract_sample_features(args, case_by_id[case_id], records, extractor)
        x_parts.append(x_case)
        y_parts.append(np.asarray([r.label_fg for r in records], dtype="int64"))
        case_parts.append(np.asarray([case_id] * len(records)))
        meta_rows.append(meta)
    return np.concatenate(x_parts), np.concatenate(y_parts), np.concatenate(case_parts), meta_rows


def fit_linear_probe(x: np.ndarray, y: np.ndarray, seed: int) -> Any:
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, solver="lbfgs", class_weight="balanced", random_state=seed),
    )
    model.fit(x, y)
    return model


def predict_prob(model: Any, x: np.ndarray) -> np.ndarray:
    return model.predict_proba(x)[:, 1]


def patch_metric_summary(y: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    pred = (prob >= threshold).astype("int64")
    precision, recall, f1, _ = precision_recall_fscore_support(y, pred, labels=[1], average="binary", zero_division=0)
    return {
        "n": int(len(y)),
        "fg": int((y == 1).sum()),
        "bg": int((y == 0).sum()),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision_fg": float(precision),
        "recall_fg": float(recall),
        "f1_fg": float(f1),
        "roc_auc": safe_metric(roc_auc_score, y, prob),
        "average_precision": safe_metric(average_precision_score, y, prob),
    }


def safe_metric(fn: Any, y: np.ndarray, prob: np.ndarray) -> float:
    try:
        return float(fn(y, prob))
    except Exception:
        return float("nan")


def yolo_record_from_arrays(data: Any, idx: int) -> YoloPatchRecord:
    det_ids = tuple(int(x) for x in str(data["detection_ids"][idx]).split(";") if str(x))
    return YoloPatchRecord(
        case_id=str(data["case_id"][idx]),
        detection_id=int(data["detection_id"][idx]),
        detection_ids=det_ids,
        row=int(data["row"][idx]),
        col=int(data["col"][idx]),
        x=int(data["x_level0"][idx]),
        y=int(data["y_level0"][idx]),
        width=int(data["width_level0"][idx]),
        height=int(data["height_level0"][idx]),
    )


def load_or_extract_yolo_features(
    args: argparse.Namespace,
    case: StressCase,
    detections: list[Detection],
    extractor: FeatureExtractor,
) -> tuple[np.ndarray, list[YoloPatchRecord], dict[str, Any]]:
    source_cache = args.stress_yolo_source_dir / "features" / f"{case.case_id}_yolo_probe_features.npz"
    cache_path = args.output_dir / "features/stress_yolo" / f"{case.case_id}_yolo_probe_features.npz"
    for path in (cache_path, source_cache):
        if args.resume and cache_meta_matches(path, extractor):
            with np.load(path, allow_pickle=False) as data:
                records = [yolo_record_from_arrays(data, idx) for idx in range(len(data["row"]))]
                return data["features"].astype("float32"), records, {
                    "case_id": case.case_id,
                    "cache_reused": True,
                    "cache_path": str(path),
                    "source_cache": str(source_cache),
                    "patch_count": int(len(records)),
                }

    raw_records = build_patch_records(case, detections, args.patch_size)
    records = [
        YoloPatchRecord(
            case_id=r.case_id,
            detection_id=r.detection_id,
            detection_ids=r.detection_ids,
            row=r.row,
            col=r.col,
            x=r.x,
            y=r.y,
            width=r.width,
            height=r.height,
        )
        for r in raw_records
    ]
    started = time.perf_counter()
    features: list[np.ndarray] = []
    reader = WsiPatchReader(case.wsi_path, args.wsi_reader, args.read_workers)
    try:
        images: list[Image.Image] = []
        for record in records:
            images.append(reader.read_patch(record))
            if len(images) >= extractor.batch_size:
                infer_images(extractor, images, features)
                images = []
        infer_images(extractor, images, features)
    finally:
        reader.close()
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
        "case_id": case.case_id,
        "cache_reused": False,
        "cache_path": str(cache_path),
        "patch_count": int(len(records)),
        "extract_seconds": float(elapsed),
        "patches_per_second": float(len(records) / elapsed) if elapsed > 0 else 0.0,
    }


def filtered_detections_by_case(args: argparse.Namespace, cases: list[StressCase]) -> tuple[dict[str, list[Detection]], list[dict[str, Any]]]:
    detections = load_detections(args.stress_yolo_source_dir / "yolo_detections.csv")
    raw_by_case: dict[str, list[Detection]] = defaultdict(list)
    for det in detections:
        raw_by_case[det.case_id].append(det)
    by_case: dict[str, list[Detection]] = {}
    suppressed_rows: list[dict[str, Any]] = []
    for case in cases:
        raw = sorted(raw_by_case.get(case.case_id, []), key=lambda d: d.detection_id)
        if args.suppress_contained:
            kept, suppressed = suppress_contained_detections(raw, args.containment_threshold)
            suppressed_rows.extend(suppressed)
        else:
            kept = raw
        by_case[case.case_id] = kept
    return by_case, suppressed_rows


def evaluate_yolo_case(
    args: argparse.Namespace,
    case: StressCase,
    records: list[YoloPatchRecord],
    prob: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray, np.ndarray]:
    gt = np.load(case.gt_mask_path).astype(bool)
    pred = (prob >= args.probe_threshold).astype("int64")
    prob_grid = np.zeros(gt.shape, dtype="float32")
    pred_grid = np.zeros(gt.shape, dtype=bool)
    patch_rows: list[dict[str, Any]] = []
    for record, fg_prob, is_fg in zip(records, prob, pred):
        rr = max(0, min(gt.shape[0] - 1, int(record.row)))
        cc = max(0, min(gt.shape[1] - 1, int(record.col)))
        prob_grid[rr, cc] = max(prob_grid[rr, cc], float(fg_prob))
        pred_grid[rr, cc] = bool(pred_grid[rr, cc] or int(is_fg) == 1)
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
                "gt_fg": assign_gt(record, gt, args.patch_size),
            }
        )
    case_row = {"case_id": case.case_id, "patch_count": int(len(records)), **mask_metrics(gt, pred_grid, prob_grid)}
    return case_row, patch_rows, prob_grid, pred_grid


def run_stress_yolo_evaluation(
    args: argparse.Namespace,
    cases: list[StressCase],
    x_sample: np.ndarray,
    y_sample: np.ndarray,
    case_ids_sample: np.ndarray,
    final_model: Any,
    extractor: FeatureExtractor,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, tuple[list[YoloPatchRecord], np.ndarray]]]:
    detections_by_case, suppressed_rows = filtered_detections_by_case(args, cases)
    write_csv(args.output_dir / "stress_yolo_detections_filtered.csv", [row for case in cases for row in detection_rows(detections_by_case[case.case_id])])
    write_csv(args.output_dir / "stress_yolo_detections_suppressed_contained.csv", suppressed_rows)
    yolo_features: dict[str, tuple[np.ndarray, list[YoloPatchRecord], dict[str, Any]]] = {}
    feature_rows: list[dict[str, Any]] = []
    for case in cases:
        print(f"[stress-yolo] loading features for {case.case_id}", flush=True)
        features, records, meta = load_or_extract_yolo_features(args, case, detections_by_case[case.case_id], extractor)
        yolo_features[case.case_id] = (features, records, meta)
        feature_rows.append(meta)

    loso_rows: list[dict[str, Any]] = []
    if len(cases) >= 2:
        for heldout in cases:
            train_idx = np.where(case_ids_sample != heldout.case_id)[0]
            model = fit_linear_probe(
                x_sample[train_idx],
                y_sample[train_idx],
                args.sample_seed + zlib.crc32(heldout.case_id.encode("utf-8")),
            )
            features, records, _meta = yolo_features[heldout.case_id]
            prob = predict_prob(model, features) if len(features) else np.asarray([], dtype="float32")
            case_row, _patch_rows, _prob_grid, _pred_grid = evaluate_yolo_case(args, heldout, records, prob)
            case_row.update({"fold": "loso", "heldout_case_id": heldout.case_id, "train_sample_count": int(len(train_idx))})
            loso_rows.append(case_row)

    final_case_rows: list[dict[str, Any]] = []
    final_patch_rows: list[dict[str, Any]] = []
    visual_payload: dict[str, tuple[list[YoloPatchRecord], np.ndarray]] = {}
    for case in cases:
        features, records, _meta = yolo_features[case.case_id]
        prob = predict_prob(final_model, features) if len(features) else np.asarray([], dtype="float32")
        case_row, patch_rows, _prob_grid, _pred_grid = evaluate_yolo_case(args, case, records, prob)
        case_row.update({"fold": "final_all_stress_train"})
        final_case_rows.append(case_row)
        final_patch_rows.extend(patch_rows)
        visual_payload[case.case_id] = (records, prob)
    return loso_rows, final_case_rows, final_patch_rows, feature_rows, visual_payload


def load_feature_cache(feature_dir: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
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
    return np.concatenate(x_parts), np.concatenate(y_parts), {
        "feature_files": len(files),
        "patches": int(sum(len(y) for y in y_parts)),
        "fg": int(sum(int((y == 1).sum()) for y in y_parts)),
        "bg": int(sum(int((y == 0).sum()) for y in y_parts)),
        "model_backend": ",".join(sorted(backends)),
        "model_name": ",".join(sorted(models)),
    }


def select_scale500_case_ids(args: argparse.Namespace) -> list[str]:
    rows = read_csv(args.scale500_transfer_manifest)
    by_stain: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_stain[row["stain"]].append(row)
    selected: list[str] = []
    selected_rows: list[dict[str, Any]] = []
    for stain in sorted(by_stain):
        for row in sorted(by_stain[stain], key=lambda r: int(r["selection_index_within_stain"]))[: args.scale500_cases_per_stain]:
            selected.append(row["case_id"])
            selected_rows.append(row)
    write_csv(args.output_dir / "scale500_qualitative_cases.csv", selected_rows)
    return selected


def load_scale500_unselected_features(args: argparse.Namespace, case_id: str) -> tuple[np.ndarray, list[ScalePatchRecord], dict[str, Any]]:
    path = args.scale500_transfer_feature_dir / f"{case_id}_unselected_detector_candidates_features.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing scale500 unselected feature cache for {case_id}: {path}")
    with np.load(path, allow_pickle=False) as data:
        records = [
            ScalePatchRecord(
                candidate_order=int(data["candidate_order"][idx]),
                candidate_id=str(data["candidate_id"][idx]),
                row=int(data["row"][idx]),
                col=int(data["col"][idx]),
                x=int(data["x_level0"][idx]),
                y=int(data["y_level0"][idx]),
                width=int(data["width_level0"][idx]),
                height=int(data["height_level0"][idx]),
            )
            for idx in range(len(data["row"]))
        ]
        meta = {
            "path": str(path),
            "model_backend": str(data["model_backend"]),
            "model_name": str(data["model_name"]),
            "patch_count": int(len(records)),
        }
        return data["features"].astype("float32"), records, meta


def candidate_stats(records: list[ScalePatchRecord], prob: np.ndarray, pred: np.ndarray) -> dict[int, dict[str, Any]]:
    by_candidate: dict[int, list[int]] = defaultdict(list)
    for idx, record in enumerate(records):
        by_candidate[record.candidate_order].append(idx)
    out: dict[int, dict[str, Any]] = {}
    for order, idxs in by_candidate.items():
        idx = np.asarray(idxs, dtype="int64")
        probs = prob[idx]
        preds = pred[idx]
        out[order] = {
            "candidate_order": order,
            "patch_count": int(len(idx)),
            "pred_fg": int(preds.sum()),
            "pred_fg_fraction": float(preds.mean()) if len(preds) else 0.0,
            "mean_prob_fg": float(probs.mean()) if len(probs) else 0.0,
        }
    return out


def stats_rows(case_id: str, stain: str, model_name: str, stats: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for order, row in sorted(stats.items()):
        rows.append({"case_id": case_id, "stain": stain, "model": model_name, **row})
    return rows


def build_comparison_table(stress_stats: dict[int, dict[str, Any]], scale_stats: dict[int, dict[str, Any]]) -> Image.Image:
    orders = sorted(set(stress_stats) | set(scale_stats))
    row_h = 34
    width = 760
    height = 60 + max(1, len(orders)) * row_h
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((10, 10), "Unselected candidate FG fractions", fill=(0, 0, 0), font=get_font(24, bold=True))
    y = 48
    for order in orders:
        s = stress_stats.get(order, {})
        c = scale_stats.get(order, {})
        text = (
            f"{order:>2}: stress={float(s.get('pred_fg_fraction', 0.0)):.2f} "
            f"(p={float(s.get('mean_prob_fg', 0.0)):.3f}) | "
            f"scale500={float(c.get('pred_fg_fraction', 0.0)):.2f} "
            f"(p={float(c.get('mean_prob_fg', 0.0)):.3f})"
        )
        draw.text((12, y), text, fill=(35, 35, 35), font=get_font(19))
        y += row_h
    return image


def build_scale500_qualitative_pages(
    args: argparse.Namespace,
    stress_model: Any,
    scale_model: Any,
) -> tuple[list[Image.Image], list[dict[str, Any]], list[dict[str, Any]]]:
    case_ids = select_scale500_case_ids(args)
    bundle_args = argparse.Namespace(
        probe_run_dir=args.scale500_feature_run,
        selector_manifest=args.selector_manifest,
        detector_root=args.detector_root,
        case_ids=",".join(case_ids),
        case_limit=None,
    )
    bundles = load_case_bundles(bundle_args)
    pages: list[Image.Image] = []
    candidate_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    for bundle in bundles:
        features, records, meta = load_scale500_unselected_features(args, bundle.case_id)
        feature_rows.append({"case_id": bundle.case_id, "stain": bundle.stain, **meta})
        stress_prob = predict_prob(stress_model, features)
        scale_prob = predict_prob(scale_model, features)
        stress_pred = (stress_prob >= args.probe_threshold).astype("int64")
        scale_pred = (scale_prob >= args.probe_threshold).astype("int64")
        stress_stats = candidate_stats(records, stress_prob, stress_pred)
        scale_stats = candidate_stats(records, scale_prob, scale_pred)
        candidate_rows.extend(stats_rows(bundle.case_id, bundle.stain, "stress32_qwen2b_hardneg_linear", stress_stats))
        candidate_rows.extend(stats_rows(bundle.case_id, bundle.stain, "scale500_selected_linear", scale_stats))
        left = draw_detector_overview_with_stats(
            bundle,
            "stress32_qwen2b_hardneg_linear",
            stress_stats,
            (records, stress_pred, stress_prob),
            args.max_overview_width,
        )
        right = draw_detector_overview_with_stats(
            bundle,
            "scale500_selected_linear",
            scale_stats,
            (records, scale_pred, scale_prob),
            args.max_overview_width,
        )
        table = build_comparison_table(stress_stats, scale_stats)
        pages.append(
            make_page(
                f"{bundle.case_id} | {bundle.stain} | scale500 qualitative transfer",
                "No manual GT for scale500: left is stress32 qwen2b-hard-negative probe, right is scale500-selected probe on the same unselected detector candidates.",
                make_contact_sheet([left, right, table], cols=1, gap=24),
                footer="Green squares are predicted foreground patches inside detector boxes only.",
            )
        )
        unselected = [candidate for candidate in bundle.candidates if not candidate.selected_for_train]
        top_orders = sorted(
            {c.candidate_order for c in unselected},
            key=lambda order: max(
                float(stress_stats.get(order, {}).get("pred_fg_fraction", 0.0)),
                float(scale_stats.get(order, {}).get("pred_fg_fraction", 0.0)),
            ),
            reverse=True,
        )[: args.max_scale500_crop_panels]
        panels: list[Image.Image] = []
        slide_path = bundle.selector_row.get("source_wsi_path") or bundle.selector_row.get("wsi_path")
        slide = openslide.OpenSlide(slide_path)
        try:
            for candidate in unselected:
                if candidate.candidate_order not in top_orders:
                    continue
                pairs = [(idx, record) for idx, record in enumerate(records) if record.candidate_order == candidate.candidate_order]
                if not pairs:
                    continue
                local_records = [record for _idx, record in pairs]
                for model_label, pred, prob in [
                    ("stress32 probe", stress_pred, stress_prob),
                    ("scale500 probe", scale_pred, scale_prob),
                ]:
                    panels.append(
                        draw_prediction_grid_panel(
                            slide,
                            candidate,
                            local_records,
                            {local: int(pred[global_idx]) for local, (global_idx, _record) in enumerate(pairs)},
                            {local: float(prob[global_idx]) for local, (global_idx, _record) in enumerate(pairs)},
                            title=f"{model_label} | detector ID {candidate.candidate_order}",
                            max_dim=args.max_grid_dim,
                        )
                    )
        finally:
            slide.close()
        if panels:
            pages.append(
                make_page(
                    f"{bundle.case_id} | {bundle.stain} | crop-level comparison",
                    "Patch-grid predictions on selected unselected detector crops, paired by model.",
                    make_contact_sheet(panels, cols=2, gap=28),
                    footer="Green=predicted foreground, red=predicted background.",
                )
            )
    return pages, candidate_rows, feature_rows


def draw_metric_summary_page(
    samples: list[SampleRecord],
    census_rows: list[dict[str, Any]],
    loso_rows: list[dict[str, Any]],
    final_rows: list[dict[str, Any]],
) -> Image.Image:
    width = 1250
    height = 760
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((20, 18), "Stress32 qwen2b Hard-Negative Linear Probe", fill=(0, 0, 0), font=get_font(30, bold=True))
    bucket_counts: dict[str, int] = defaultdict(int)
    for sample in samples:
        bucket_counts[sample.bucket] += 1
    qwen_fp = sum(int(r["false_positive_hard_negative"]) for r in census_rows)
    qwen_tn = sum(int(r["true_negative_easy_negative"]) for r in census_rows)
    qwen_tp = sum(int(r["true_positive"]) for r in census_rows)
    qwen_fn = sum(int(r["false_negative"]) for r in census_rows)
    y = 72
    lines = [
        f"Census cases={len(census_rows)} | qwen2b TP={qwen_tp:,} FP/hard-neg={qwen_fp:,} TN/easy-neg={qwen_tn:,} FN={qwen_fn:,}.",
        "Training samples: "
        + ", ".join(f"{key}={value:,}" for key, value in sorted(bucket_counts.items())),
        "Stress metrics below are full-grid metrics: outside YOLO boxes is predicted background.",
    ]
    for line in lines:
        y = draw_wrapped_text(draw, (24, y), line, font=get_font(21), fill=(35, 35, 35), width_chars=104)
        y += 14
    for title, rows in [("LOSO stress32 YOLO route", loso_rows), ("Final all-stress train YOLO route", final_rows)]:
        draw.text((24, y), title, fill=(0, 0, 0), font=get_font(25, bold=True))
        y += 40
        if rows:
            for key, label in [
                ("precision_fg", "precision"),
                ("recall_fg", "recall"),
                ("f1_fg", "F1"),
                ("dice_fg", "Dice"),
            ]:
                vals = np.asarray([float(row[key]) for row in rows], dtype="float64")
                draw.text(
                    (48, y),
                    f"mean {label}: {vals.mean():.3f}  median {np.median(vals):.3f}",
                    fill=(35, 35, 35),
                    font=get_font(21),
                )
                y += 31
        y += 18
    draw.text((24, height - 58), "Scale500 pages are qualitative only because scale500 has no manual GT.", fill=(80, 80, 80), font=get_font(18))
    return image


def write_reproduction(args: argparse.Namespace, summary: dict[str, Any]) -> None:
    cmd = [
        "python",
        "scripts/train_stress32_qwen2b_hard_negative_probe.py",
        "--output-dir",
        str(args.output_dir),
        "--qwen2b-root",
        str(args.qwen2b_root),
        "--stress-dataset-root",
        str(args.stress_dataset_root),
        "--stress-yolo-source-dir",
        str(args.stress_yolo_source_dir),
        "--scale500-feature-run",
        str(args.scale500_feature_run),
        "--scale500-transfer-feature-dir",
        str(args.scale500_transfer_feature_dir),
        "--batch-size",
        str(args.batch_size),
        "--wsi-reader",
        str(args.wsi_reader),
        "--read-workers",
        str(args.read_workers),
        "--pipeline-mode",
        str(args.pipeline_mode),
        "--sample-seed",
        str(args.sample_seed),
        "--hard-negatives-per-wsi",
        str(args.hard_negatives_per_wsi),
        "--easy-negatives-per-wsi",
        str(args.easy_negatives_per_wsi),
        "--foreground-per-wsi",
        str(args.foreground_per_wsi),
    ]
    if args.skip_scale500_qual:
        cmd.append("--skip-scale500-qual")
    lines = [
        "PER-250 Stress32 qwen2b hard-negative probe",
        "============================================",
        "",
        f"Created: {summary['created_at']}",
        f"Ticket: {args.ticket}",
        f"Git commit: {summary['git_commit']}",
        f"Working directory: {REPO_ROOT}",
        f"Dirty git state captured in: {args.output_dir / 'git_status.txt'}",
        f"DVC status captured in: {args.output_dir / 'dvc_status.txt'}",
        "",
        "Command:",
        " ".join(shlex.quote(part) for part in cmd),
        "",
        "Environment:",
        f"- Python executable: {sys.executable}",
        "- Expected env: path-agent with transformers, torch, sklearn, openslide, cuCIM.",
        "",
        "Inputs:",
        f"- Stress dataset root: {args.stress_dataset_root.resolve()}",
        f"- qwen2b zero-shot root: {args.qwen2b_root.resolve()}",
        f"- qwen2b mask choice: stage7_new/mask.npy, falling back to stage7/mask.npy.",
        f"- Existing stress YOLO/DINO feature source: {args.stress_yolo_source_dir.resolve()}",
        f"- Scale500 selected feature run: {args.scale500_feature_run.resolve()}",
        f"- Scale500 unselected transfer feature dir: {args.scale500_transfer_feature_dir.resolve()}",
        "",
        "Sampling:",
        f"- hard_negative_fp per WSI <= {args.hard_negatives_per_wsi}",
        f"- easy_negative_tn per WSI <= {args.easy_negatives_per_wsi}",
        f"- foreground_gt per WSI <= {args.foreground_per_wsi}",
        "- No crop-aware balancing; stress cases are treated as single-crop for reuse.",
        "",
        "Interpretation:",
        "- Stress32 has manual GT, so stress metrics are quantitative.",
        "- Scale500 has no manual GT here, so scale500 output is qualitative visual transfer only.",
        "",
        "Outputs:",
        f"- PDF: {summary['pdf']}",
        f"- patch_census.csv: {summary['patch_census_csv']}",
        f"- sample_manifest.csv: {summary['sample_manifest_csv']}",
        f"- loso_metrics.csv: {summary['loso_metrics_csv']}",
        f"- stress_yolo_patch_predictions.csv: {summary['stress_yolo_patch_predictions_csv']}",
        f"- scale500_qualitative_cases.csv: {summary['scale500_qualitative_cases_csv']}",
        "",
    ]
    (args.output_dir / "reproduction.txt").write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "git_status.txt").write_text(git_status_short())
    (args.output_dir / "dvc_status.txt").write_text(dvc_status_text())

    cases = load_cases_for_qwen2b(args)
    census_rows, samples = build_census_and_samples(args, cases)
    write_csv(args.output_dir / "patch_census.csv", census_rows)
    write_csv(args.output_dir / "sample_manifest.csv", [sample_row(record, idx) for idx, record in enumerate(samples)])

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
    x_sample, y_sample, sample_case_ids, sample_feature_rows = extract_all_sample_features(args, cases, samples, extractor)
    write_csv(args.output_dir / "sample_feature_cache_summary.csv", sample_feature_rows)
    stress_model = fit_linear_probe(x_sample, y_sample, args.sample_seed)
    sample_prob = predict_prob(stress_model, x_sample)
    sample_summary = patch_metric_summary(y_sample, sample_prob, args.probe_threshold)
    write_csv(args.output_dir / "training_sample_fit_metrics.csv", [sample_summary])

    loso_rows, final_case_rows, final_patch_rows, stress_feature_rows, _visual_payload = run_stress_yolo_evaluation(
        args,
        cases,
        x_sample,
        y_sample,
        sample_case_ids,
        stress_model,
        extractor,
    )
    write_csv(args.output_dir / "loso_metrics.csv", loso_rows)
    write_csv(args.output_dir / "stress_yolo_case_metrics.csv", final_case_rows)
    write_csv(args.output_dir / "stress_yolo_patch_predictions.csv", final_patch_rows)
    write_csv(args.output_dir / "stress_yolo_feature_cache_summary.csv", stress_feature_rows)

    pages = [
        make_page(
            "Stress32 qwen2b Hard-Negative Probe",
            "qwen2b-vs-GT sampling, LOSO stress32 YOLO-route metrics, and qualitative scale500 transfer comparison.",
            draw_metric_summary_page(samples, census_rows, loso_rows, final_case_rows),
            footer="Stress metrics use manual GT. Scale500 pages are qualitative because no manual scale500 GT is available.",
        )
    ]

    scale_train_meta: dict[str, Any] = {}
    scale500_candidate_rows: list[dict[str, Any]] = []
    scale500_feature_rows: list[dict[str, Any]] = []
    if not args.skip_scale500_qual:
        x_scale, y_scale, scale_train_meta = load_feature_cache(args.scale500_feature_run)
        scale500_model = fit_linear_probe(x_scale, y_scale, args.sample_seed + 10000)
        scale_pages, scale500_candidate_rows, scale500_feature_rows = build_scale500_qualitative_pages(args, stress_model, scale500_model)
        pages.extend(scale_pages)
    else:
        write_csv(args.output_dir / "scale500_qualitative_cases.csv", [])
    write_csv(args.output_dir / "scale500_qualitative_candidate_predictions.csv", scale500_candidate_rows)
    write_csv(args.output_dir / "scale500_qualitative_feature_cache_summary.csv", scale500_feature_rows)

    for idx, page in enumerate(pages, start=1):
        page.save(args.output_dir / f"page_{idx:03d}.png")
    pdf_path = args.output_dir / "stress32_qwen2b_hard_negative_probe_review.pdf"
    pages[0].save(pdf_path, save_all=True, append_images=pages[1:])

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": float(time.perf_counter() - started),
        "ticket": args.ticket,
        "git_commit": git_commit(),
        "output_dir": str(args.output_dir.resolve()),
        "case_count": len(cases),
        "sample_count": len(samples),
        "sample_fg": int((y_sample == 1).sum()),
        "sample_bg": int((y_sample == 0).sum()),
        "sample_fit_metrics": sample_summary,
        "qwen2b_root": str(args.qwen2b_root.resolve()),
        "stress_yolo_source_dir": str(args.stress_yolo_source_dir.resolve()),
        "scale500_feature_run": str(args.scale500_feature_run.resolve()),
        "scale500_train_meta": scale_train_meta,
        "feature_extractor": extractor.meta,
        "package_versions": package_versions(),
        "patch_census_csv": str((args.output_dir / "patch_census.csv").resolve()),
        "sample_manifest_csv": str((args.output_dir / "sample_manifest.csv").resolve()),
        "loso_metrics_csv": str((args.output_dir / "loso_metrics.csv").resolve()),
        "stress_yolo_patch_predictions_csv": str((args.output_dir / "stress_yolo_patch_predictions.csv").resolve()),
        "scale500_qualitative_cases_csv": str((args.output_dir / "scale500_qualitative_cases.csv").resolve()),
        "pdf": str(pdf_path.resolve()),
        "preview_pages": [str((args.output_dir / f"page_{idx:03d}.png").resolve()) for idx in range(1, len(pages) + 1)],
    }
    write_json(args.output_dir / "summary.json", summary)
    write_reproduction(args, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
