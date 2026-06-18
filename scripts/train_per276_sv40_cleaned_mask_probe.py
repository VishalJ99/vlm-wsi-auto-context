#!/usr/bin/env python3
"""Train a DINOv3 linear probe from cleaned PER-276 SV40 Stage7 masks."""

from __future__ import annotations

import argparse
import concurrent.futures
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
from typing import Any, Iterable

import joblib
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from postprocess_mask import count_components, fill_small_holes, remove_small_components  # noqa: E402
from train_per_wsi_dinov3_fg_bg_probe import (  # noqa: E402
    DEFAULT_DINOV2_SMALL,
    DEFAULT_DINOV3_SMALL,
    FeatureExtractor,
    WsiPatchReader,
    package_versions,
)


DEFAULT_SOURCE_RUN = REPO_ROOT / "runs/scale500_sv40_icl1_alex_100_vertex_v1"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "runs/per290_sv40_cleaned_mask_probe_v1"
STAGE5_ERROR_PATTERNS = ("Gemini API Error", "Response parse error", "Parse failed")


@dataclass(frozen=True)
class SourceCase:
    case_id: str
    run_id: str
    run_dir: Path


@dataclass(frozen=True)
class PatchRecord:
    sample_index: int
    case_id: str
    run_id: str
    bbox_id: str
    bbox_index: int
    row: int
    col: int
    x: int
    y: int
    width: int
    height: int
    label_fg: int
    source_mask_fg: int
    wsi_path: str
    crop_path: str

    @property
    def record_id(self) -> str:
        return f"{self.case_id}|{self.bbox_id}|r{self.row}c{self.col}|{self.x}_{self.y}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-root", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ticket", default="PER-290")
    parser.add_argument("--case-limit", type=int, default=None)
    parser.add_argument("--case-ids", default="", help="Comma-separated case IDs to include.")
    parser.add_argument("--sample-max-per-wsi", type=int, default=500)
    parser.add_argument("--sample-seed", type=int, default=290)
    parser.add_argument("--holdout-frac", type=float, default=0.20)
    parser.add_argument("--holdout-wsis", type=int, default=None)
    parser.add_argument("--probe-threshold", type=float, default=0.50)
    parser.add_argument("--remove-component-max-size", type=int, default=8)
    parser.add_argument("--fill-hole-max-size", type=int, default=8)
    parser.add_argument("--connectivity", type=int, choices=[4, 8], default=8)
    parser.add_argument("--include-stage5-error-bboxes", action="store_true")
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--model-backend", choices=["transformers", "timm"], default="transformers")
    parser.add_argument("--model-name", default=DEFAULT_DINOV3_SMALL)
    parser.add_argument("--fallback-model-name", default=DEFAULT_DINOV2_SMALL)
    parser.add_argument("--allow-timm-fallback", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--input-size", type=int, default=None)
    parser.add_argument("--wsi-reader", choices=["cucim", "openslide"], default="cucim")
    parser.add_argument("--read-workers", type=int, default=16)
    parser.add_argument("--read-queue-batches", type=int, default=4)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--pdf-max-cases", type=int, default=20)
    parser.add_argument("--pdf-max-bboxes-per-case", type=int, default=2)
    parser.add_argument("--skip-pdf", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


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


def case_dir_name(case_id: str) -> str:
    if case_id.startswith("anon_"):
        return "anon_" + case_id[len("anon_") :].replace("_", "-")
    return case_id.replace("_", "-")


def marker_run_id(path: Path) -> str:
    name = path.name
    if name.endswith(".complete.txt"):
        return name[: -len(".complete.txt")]
    return path.stem


def find_run_dir(source_root: Path, case_id: str, run_id: str) -> Path | None:
    direct = source_root / case_dir_name(case_id) / run_id
    if direct.exists():
        return direct
    matches = [p.parent for p in source_root.rglob(f"{run_id}/pipeline_status.json")]
    return matches[0] if matches else None


def load_source_cases(args: argparse.Namespace) -> list[SourceCase]:
    marker_root = args.source_run_root / "_array_task_cases"
    markers = sorted(marker_root.glob("*.complete.txt"))
    if not markers:
        raise FileNotFoundError(f"No complete markers found under {marker_root}")
    wanted = {x.strip() for x in args.case_ids.split(",") if x.strip()}
    cases: list[SourceCase] = []
    seen: set[str] = set()
    for marker in markers:
        run_id = marker_run_id(marker)
        case_id = marker.read_text().strip()
        if not case_id or (wanted and case_id not in wanted):
            continue
        if case_id in seen:
            continue
        run_dir = find_run_dir(args.source_run_root, case_id, run_id)
        if run_dir is None:
            continue
        cases.append(SourceCase(case_id=case_id, run_id=run_id, run_dir=run_dir))
        seen.add(case_id)
    cases.sort(key=lambda c: c.case_id)
    if args.case_limit is not None:
        cases = cases[: args.case_limit]
    return cases


def bbox_dirs(run_dir: Path) -> list[Path]:
    root = run_dir / "bboxes"
    if not root.exists():
        return []
    return sorted([p for p in root.iterdir() if p.is_dir()])


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def has_stage5_error(bbox_dir: Path) -> bool:
    response = bbox_dir / "stage5/intermediate/vlm_multiclass_ranking_response.txt"
    if not response.exists():
        return False
    text = response.read_text(errors="replace")
    return any(pattern in text for pattern in STAGE5_ERROR_PATTERNS)


def clean_mask(mask: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, dict[str, Any]]:
    source = mask.astype(bool)
    before_components = count_components(source, connectivity=args.connectivity)
    cleaned, removed_components, removed_pixels = remove_small_components(
        source,
        min_size=int(args.remove_component_max_size) + 1,
        connectivity=args.connectivity,
    )
    after_remove_components = count_components(cleaned, connectivity=args.connectivity)
    filled = fill_small_holes(
        cleaned,
        max_hole_size=int(args.fill_hole_max_size),
        connectivity=args.connectivity,
    )
    after_components = count_components(filled, connectivity=args.connectivity)
    added = (~cleaned) & filled
    removed = source & (~filled)
    stats = {
        "source_fg": int(source.sum()),
        "source_bg": int(source.size - source.sum()),
        "cleaned_fg": int(filled.sum()),
        "cleaned_bg": int(filled.size - filled.sum()),
        "source_components": int(before_components),
        "components_after_remove_small": int(after_remove_components),
        "cleaned_components": int(after_components),
        "removed_components_le_max_size": int(removed_components),
        "removed_pixels_small_components": int(removed_pixels),
        "filled_hole_pixels": int(added.sum()),
        "net_removed_pixels": int(removed.sum()),
        "delta_fg": int(filled.sum() - source.sum()),
        "connectivity": int(args.connectivity),
        "remove_component_max_size": int(args.remove_component_max_size),
        "fill_hole_max_size": int(args.fill_hole_max_size),
    }
    return filled.astype(bool), stats


def mask_png(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = (mask.astype(np.uint8) * 255)
    Image.fromarray(arr, mode="L").save(path)


def build_patch_pool(args: argparse.Namespace, cases: list[SourceCase]) -> tuple[list[PatchRecord], list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[PatchRecord] = []
    census_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    sample_index = 0
    for case in cases:
        case_source_fg = case_source_bg = 0
        case_cleaned_fg = case_cleaned_bg = 0
        for bbox_index, bbox in enumerate(bbox_dirs(case.run_dir), start=1):
            error_bbox = has_stage5_error(bbox)
            if error_bbox and not args.include_stage5_error_bboxes:
                skipped_rows.append(
                    {
                        "case_id": case.case_id,
                        "run_id": case.run_id,
                        "bbox_id": bbox.name,
                        "reason": "stage5_response_error",
                    }
                )
                continue
            mask_path = bbox / "stage7/tissue_mask_post.npy"
            patches_path = bbox / "stage6/patches.csv"
            if not mask_path.exists() or not patches_path.exists():
                skipped_rows.append(
                    {
                        "case_id": case.case_id,
                        "run_id": case.run_id,
                        "bbox_id": bbox.name,
                        "reason": "missing_mask_or_patches",
                    }
                )
                continue
            source_mask = np.load(mask_path).astype(bool)
            cleaned, stats = clean_mask(source_mask, args)
            clean_dir = args.output_dir / "cleaned_masks" / case.case_id / bbox.name
            npy_path = clean_dir / "tissue_mask_cleaned.npy"
            npy_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(npy_path, cleaned.astype(np.uint8))
            mask_png(cleaned, clean_dir / "tissue_mask_cleaned.png")

            stage6_meta = load_json(bbox / "stage6/metadata.json")
            wsi_path = str(stage6_meta.get("wsi_path", ""))
            crop_path = str(bbox / "stage3/crop.png")
            bbox_fg = bbox_bg = 0
            with patches_path.open() as f:
                for patch in csv.DictReader(f):
                    rr = int(patch["row"])
                    cc = int(patch["col"])
                    if rr < 0 or cc < 0 or rr >= cleaned.shape[0] or cc >= cleaned.shape[1]:
                        continue
                    label = 1 if bool(cleaned[rr, cc]) else 0
                    rows.append(
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
                            source_mask_fg=1 if bool(source_mask[rr, cc]) else 0,
                            wsi_path=wsi_path,
                            crop_path=crop_path,
                        )
                    )
                    sample_index += 1
                    if label:
                        bbox_fg += 1
                    else:
                        bbox_bg += 1
            case_source_fg += int(stats["source_fg"])
            case_source_bg += int(stats["source_bg"])
            case_cleaned_fg += bbox_fg
            case_cleaned_bg += bbox_bg
            census_rows.append(
                {
                    "case_id": case.case_id,
                    "run_id": case.run_id,
                    "bbox_id": bbox.name,
                    "bbox_index": bbox_index,
                    "stage5_error_bbox": bool(error_bbox),
                    "mask_shape_rows": int(source_mask.shape[0]),
                    "mask_shape_cols": int(source_mask.shape[1]),
                    "patch_rows_loaded": int(bbox_fg + bbox_bg),
                    "patch_cleaned_fg": int(bbox_fg),
                    "patch_cleaned_bg": int(bbox_bg),
                    "cleaned_mask_path": str(npy_path),
                    **stats,
                }
            )
        census_rows.append(
            {
                "case_id": case.case_id,
                "run_id": case.run_id,
                "bbox_id": "__case_total__",
                "bbox_index": -1,
                "patch_cleaned_fg": int(case_cleaned_fg),
                "patch_cleaned_bg": int(case_cleaned_bg),
                "source_fg": int(case_source_fg),
                "source_bg": int(case_source_bg),
            }
        )
    return rows, census_rows, skipped_rows


def sample_case_records(records: list[PatchRecord], cap: int, seed: int) -> tuple[list[PatchRecord], dict[str, Any]]:
    if cap <= 0:
        raise ValueError("--sample-max-per-wsi must be positive")
    fg = [r for r in records if r.label_fg == 1]
    bg = [r for r in records if r.label_fg == 0]
    rng = np.random.default_rng(seed)
    target_total = min(cap, len(records))
    target_fg = min(target_total // 2, len(fg))
    target_bg = min(target_total - target_fg, len(bg))
    remaining = target_total - target_fg - target_bg
    if remaining > 0:
        add_fg = min(remaining, len(fg) - target_fg)
        target_fg += add_fg
        remaining -= add_fg
    if remaining > 0:
        add_bg = min(remaining, len(bg) - target_bg)
        target_bg += add_bg
    chosen: list[PatchRecord] = []
    if target_fg:
        idx = rng.choice(len(fg), size=target_fg, replace=False)
        chosen.extend(fg[int(i)] for i in idx)
    if target_bg:
        idx = rng.choice(len(bg), size=target_bg, replace=False)
        chosen.extend(bg[int(i)] for i in idx)
    chosen.sort(key=lambda r: (r.bbox_index, r.row, r.col, r.x, r.y))
    return chosen, {
        "available_total": len(records),
        "available_fg": len(fg),
        "available_bg": len(bg),
        "target_total": target_total,
        "sampled_total": len(chosen),
        "sampled_fg": sum(1 for r in chosen if r.label_fg == 1),
        "sampled_bg": sum(1 for r in chosen if r.label_fg == 0),
        "sample_cap_per_wsi": cap,
    }


def sample_records(args: argparse.Namespace, pool: list[PatchRecord]) -> tuple[list[PatchRecord], list[dict[str, Any]]]:
    by_case: dict[str, list[PatchRecord]] = defaultdict(list)
    for record in pool:
        by_case[record.case_id].append(record)
    sampled: list[PatchRecord] = []
    audit_rows: list[dict[str, Any]] = []
    for case_id in sorted(by_case):
        seed = int(args.sample_seed) + zlib.crc32(case_id.encode("utf-8"))
        chosen, audit = sample_case_records(by_case[case_id], args.sample_max_per_wsi, seed)
        sampled.extend(chosen)
        audit_rows.append({"case_id": case_id, **audit})
    sampled = [
        PatchRecord(sample_index=i, **{k: v for k, v in record.__dict__.items() if k != "sample_index"})
        for i, record in enumerate(sampled)
    ]
    return sampled, audit_rows


def patch_row(record: PatchRecord) -> dict[str, Any]:
    return {
        "sample_index": record.sample_index,
        "record_id": record.record_id,
        "case_id": record.case_id,
        "run_id": record.run_id,
        "bbox_id": record.bbox_id,
        "bbox_index": record.bbox_index,
        "row": record.row,
        "col": record.col,
        "x_level0": record.x,
        "y_level0": record.y,
        "width_level0": record.width,
        "height_level0": record.height,
        "label_fg": record.label_fg,
        "source_mask_fg": record.source_mask_fg,
        "wsi_path": record.wsi_path,
        "crop_path": record.crop_path,
    }


def cache_meta_matches(path: Path, records: list[PatchRecord], extractor: FeatureExtractor) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            return (
                str(data["model_backend"]) == extractor.backend
                and str(data["model_name"]) == extractor.model_name
                and len(data["sample_index"]) == len(records)
                and np.array_equal(data["sample_index"].astype("int64"), np.asarray([r.sample_index for r in records], dtype="int64"))
                and np.array_equal(data["x_level0"].astype("int64"), np.asarray([r.x for r in records], dtype="int64"))
                and np.array_equal(data["y_level0"].astype("int64"), np.asarray([r.y for r in records], dtype="int64"))
                and np.array_equal(data["label_fg"].astype("int64"), np.asarray([r.label_fg for r in records], dtype="int64"))
            )
    except Exception:
        return False


class BatchPatchReader:
    def __init__(self, wsi_path: Path, backend: str, read_workers: int) -> None:
        self.wsi_path = wsi_path
        self.backend = backend
        self.read_workers = max(1, int(read_workers))
        self.local = threading.local()
        self._readers: list[WsiPatchReader] = []
        self._lock = threading.Lock()

    def _reader(self) -> WsiPatchReader:
        reader = getattr(self.local, "reader", None)
        if reader is None:
            reader = WsiPatchReader(self.wsi_path, self.backend, 1)
            self.local.reader = reader
            with self._lock:
                self._readers.append(reader)
        return reader

    def read_patch(self, record: PatchRecord) -> Image.Image:
        return self._reader().read_patch(record)

    def close(self) -> None:
        with self._lock:
            readers = list(self._readers)
            self._readers = []
        for reader in readers:
            try:
                reader.close()
            except Exception:
                pass


def infer_images(extractor: FeatureExtractor, images: list[Image.Image], parts: list[np.ndarray]) -> None:
    if images:
        parts.append(extractor.extract_batch(images))


def extract_case_features(
    args: argparse.Namespace,
    case_id: str,
    records: list[PatchRecord],
    extractor: FeatureExtractor,
) -> tuple[np.ndarray, list[PatchRecord], dict[str, Any]]:
    safe_case = case_id.replace("/", "_")
    cache_path = args.output_dir / "features/sampled" / f"{safe_case}_features.npz"
    if args.resume and cache_meta_matches(cache_path, records, extractor):
        with np.load(cache_path, allow_pickle=False) as data:
            return data["features"].astype("float32"), records, {
                "case_id": case_id,
                "patch_count": len(records),
                "cache_reused": True,
                "cache_path": str(cache_path),
                "extract_seconds": 0.0,
                "patches_per_second": 0.0,
            }
    if not records:
        raise ValueError(f"No sampled records for {case_id}")
    wsi_paths = sorted({r.wsi_path for r in records if r.wsi_path})
    if len(wsi_paths) != 1:
        raise ValueError(f"Expected one WSI path for {case_id}, found {wsi_paths}")
    wsi_path = Path(wsi_paths[0])
    if not wsi_path.exists():
        raise FileNotFoundError(f"WSI path does not exist for {case_id}: {wsi_path}")

    started = time.perf_counter()
    feature_parts: list[np.ndarray] = []
    failures: list[dict[str, Any]] = []
    reader = BatchPatchReader(wsi_path, args.wsi_reader, args.read_workers)
    try:
        if args.read_workers > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.read_workers) as pool:
                for start in range(0, len(records), args.batch_size):
                    batch_records = records[start : start + args.batch_size]
                    future_to_record = {pool.submit(reader.read_patch, record): record for record in batch_records}
                    image_by_index: dict[int, Image.Image] = {}
                    record_to_index = {id(record): idx for idx, record in enumerate(batch_records)}
                    for future in concurrent.futures.as_completed(future_to_record):
                        record = future_to_record[future]
                        try:
                            image_by_index[record_to_index[id(record)]] = future.result()
                        except Exception as exc:
                            failures.append({"record_id": record.record_id, "error": f"{type(exc).__name__}: {exc}"})
                    images = [image_by_index[i] for i in range(len(batch_records)) if i in image_by_index]
                    infer_images(extractor, images, feature_parts)
        else:
            for start in range(0, len(records), args.batch_size):
                batch_records = records[start : start + args.batch_size]
                images: list[Image.Image] = []
                for record in batch_records:
                    try:
                        images.append(reader.read_patch(record))
                    except Exception as exc:
                        failures.append({"record_id": record.record_id, "error": f"{type(exc).__name__}: {exc}"})
                infer_images(extractor, images, feature_parts)
    finally:
        reader.close()
    elapsed = time.perf_counter() - started
    if not feature_parts:
        raise RuntimeError(f"No features extracted for {case_id}")
    features = np.concatenate(feature_parts, axis=0).astype("float32")
    if len(features) != len(records) - len(failures):
        raise RuntimeError(f"Feature/record mismatch for {case_id}: features={len(features)} records={len(records)} failures={len(failures)}")
    if failures:
        write_csv(args.output_dir / "features/failures" / f"{safe_case}_failures.csv", failures)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    kept_records = records if not failures else [r for r in records if r.record_id not in {f["record_id"] for f in failures}]
    np.savez_compressed(
        cache_path,
        features=features,
        sample_index=np.asarray([r.sample_index for r in kept_records], dtype="int64"),
        case_id=np.asarray([r.case_id for r in kept_records]),
        bbox_id=np.asarray([r.bbox_id for r in kept_records]),
        row=np.asarray([r.row for r in kept_records], dtype="int64"),
        col=np.asarray([r.col for r in kept_records], dtype="int64"),
        x_level0=np.asarray([r.x for r in kept_records], dtype="int64"),
        y_level0=np.asarray([r.y for r in kept_records], dtype="int64"),
        width_level0=np.asarray([r.width for r in kept_records], dtype="int64"),
        height_level0=np.asarray([r.height for r in kept_records], dtype="int64"),
        label_fg=np.asarray([r.label_fg for r in kept_records], dtype="int64"),
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


def extract_all_features(
    args: argparse.Namespace,
    records: list[PatchRecord],
    extractor: FeatureExtractor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    by_case: dict[str, list[PatchRecord]] = defaultdict(list)
    for record in records:
        by_case[record.case_id].append(record)
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    sample_indices: list[np.ndarray] = []
    case_ids: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    for case_id in sorted(by_case):
        case_records = by_case[case_id]
        print(f"[features] {case_id}: {len(case_records)} sampled patches", flush=True)
        x_case, kept_records, meta = extract_case_features(args, case_id, case_records, extractor)
        kept = x_case.shape[0]
        features.append(x_case)
        labels.append(np.asarray([r.label_fg for r in kept_records], dtype="int64"))
        sample_indices.append(np.asarray([r.sample_index for r in kept_records], dtype="int64"))
        case_ids.append(np.asarray([case_id] * kept))
        rows.append(meta)
    return (
        np.concatenate(features, axis=0),
        np.concatenate(labels, axis=0),
        np.concatenate(sample_indices, axis=0),
        np.concatenate(case_ids, axis=0),
        rows,
    )


def choose_holdout(case_ids: np.ndarray, y: np.ndarray, args: argparse.Namespace) -> tuple[set[str], list[dict[str, Any]]]:
    unique_cases = sorted(set(str(x) for x in case_ids))
    if len(unique_cases) < 2:
        raise ValueError("Need at least two WSIs for heldout evaluation")
    holdout_n = args.holdout_wsis
    if holdout_n is None:
        holdout_n = max(1, int(round(len(unique_cases) * args.holdout_frac)))
    holdout_n = min(max(1, int(holdout_n)), len(unique_cases) - 1)
    case_stats = []
    for case_id in unique_cases:
        idx = case_ids == case_id
        total = int(idx.sum())
        fg = int(y[idx].sum())
        case_stats.append((case_id, total, fg, fg / total if total else 0.0))
    for attempt in range(1000):
        rng = np.random.default_rng(int(args.sample_seed) + 1009 * attempt)
        order = rng.permutation(len(unique_cases))
        chosen = {unique_cases[int(i)] for i in order[:holdout_n]}
        train = np.asarray([str(c) not in chosen for c in case_ids])
        test = ~train
        if len(set(y[train].tolist())) == 2 and len(set(y[test].tolist())) == 2:
            rows = [
                {
                    "case_id": case_id,
                    "split": "holdout" if case_id in chosen else "train",
                    "sampled_total": total,
                    "sampled_fg": fg,
                    "sampled_bg": total - fg,
                    "fg_fraction": frac,
                }
                for case_id, total, fg, frac in case_stats
            ]
            return chosen, rows
    raise RuntimeError("Could not find a WSI holdout split with both classes in train and holdout")


def fit_probe(x: np.ndarray, y: np.ndarray, seed: int) -> Any:
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, solver="lbfgs", class_weight="balanced", random_state=seed),
    )
    model.fit(x, y)
    return model


def safe_metric(fn: Any, y: np.ndarray, prob: np.ndarray) -> float:
    try:
        return float(fn(y, prob))
    except Exception:
        return float("nan")


def metric_summary(y: np.ndarray, prob: np.ndarray, threshold: float, prefix: str = "") -> dict[str, Any]:
    pred = (prob >= threshold).astype("int64")
    precision, recall, f1, _ = precision_recall_fscore_support(y, pred, labels=[1], average="binary", zero_division=0)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = [int(x) for x in cm.ravel()]
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    base = {
        "n": int(len(y)),
        "fg": int((y == 1).sum()),
        "bg": int((y == 0).sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision_fg": float(precision),
        "recall_fg": float(recall),
        "f1_fg": float(f1),
        "bg_false_positive_rate": float(fpr),
        "bg_specificity": float(specificity),
        "roc_auc": safe_metric(roc_auc_score, y, prob),
        "average_precision": safe_metric(average_precision_score, y, prob),
        "threshold": float(threshold),
    }
    if prefix:
        return {f"{prefix}{k}": v for k, v in base.items()}
    return base


def train_and_evaluate(
    args: argparse.Namespace,
    x: np.ndarray,
    y: np.ndarray,
    case_ids: np.ndarray,
) -> tuple[Any, Any, list[dict[str, Any]], list[dict[str, Any]], np.ndarray]:
    holdout, split_rows = choose_holdout(case_ids, y, args)
    train_idx = np.asarray([str(c) not in holdout for c in case_ids])
    test_idx = ~train_idx
    split_model = fit_probe(x[train_idx], y[train_idx], args.sample_seed)
    final_model = fit_probe(x, y, args.sample_seed + 10000)
    train_prob = split_model.predict_proba(x[train_idx])[:, 1]
    test_prob = split_model.predict_proba(x[test_idx])[:, 1]
    all_prob = split_model.predict_proba(x)[:, 1]
    metric_rows = [
        {"split": "train", **metric_summary(y[train_idx], train_prob, args.probe_threshold)},
        {"split": "heldout_wsi", **metric_summary(y[test_idx], test_prob, args.probe_threshold)},
    ]
    for case_id in sorted(holdout):
        idx = case_ids == case_id
        metric_rows.append(
            {"split": "heldout_case", "case_id": case_id, **metric_summary(y[idx], all_prob[idx], args.probe_threshold)}
        )
    return split_model, final_model, metric_rows, split_rows, all_prob


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


FONT_TITLE = font(42, True)
FONT_H2 = font(28, True)
FONT_BODY = font(22)
FONT_SMALL = font(18)
PAGE_W, PAGE_H = 1800, 2300
MARGIN = 55


def open_image(path: Path) -> Image.Image | None:
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def fit_with_box(img: Image.Image, box: tuple[int, int]) -> tuple[Image.Image, tuple[int, int, int, int]]:
    out_w, out_h = box
    work = img.copy()
    work.thumbnail((out_w, out_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (out_w, out_h), "white")
    x = (out_w - work.width) // 2
    y = (out_h - work.height) // 2
    canvas.paste(work, (x, y))
    return canvas, (x, y, work.width, work.height)


def draw_grid_overlay(
    crop: Image.Image,
    mask: np.ndarray,
    sampled: list[PatchRecord],
    prob_by_sample: dict[int, float],
    mode: str,
    size: tuple[int, int],
) -> Image.Image:
    panel, (ox, oy, iw, ih) = fit_with_box(crop, size)
    overlay = Image.new("RGBA", panel.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    rows, cols = mask.shape
    cell_w = iw / max(1, cols)
    cell_h = ih / max(1, rows)
    if mode == "label":
        for rr in range(rows):
            for cc in range(cols):
                if bool(mask[rr, cc]):
                    box = [
                        int(ox + cc * cell_w),
                        int(oy + rr * cell_h),
                        int(ox + (cc + 1) * cell_w),
                        int(oy + (rr + 1) * cell_h),
                    ]
                    draw.rectangle(box, fill=(30, 190, 80, 90))
    else:
        for record in sampled:
            p = prob_by_sample.get(record.sample_index)
            if p is None:
                continue
            color = (30, 190, 80, 110) if p >= 0.5 else (230, 60, 60, 90)
            box = [
                int(ox + record.col * cell_w),
                int(oy + record.row * cell_h),
                int(ox + (record.col + 1) * cell_w),
                int(oy + (record.row + 1) * cell_h),
            ]
            draw.rectangle(box, fill=color, outline=(40, 40, 40, 120), width=1)
    return Image.alpha_composite(panel.convert("RGBA"), overlay).convert("RGB")


def render_review_pdf(
    args: argparse.Namespace,
    sampled: list[PatchRecord],
    split_rows: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
    sample_indices: np.ndarray,
    prob: np.ndarray,
) -> Path:
    out_dir = args.output_dir / "visuals"
    pages_dir = out_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    prob_by_sample = {int(sample_index): float(prob[idx]) for idx, sample_index in enumerate(sample_indices)}
    split_by_case = {row["case_id"]: row["split"] for row in split_rows}
    heldout_cases = [row["case_id"] for row in split_rows if row["split"] == "holdout"][: args.pdf_max_cases]
    by_case_bbox: dict[tuple[str, str], list[PatchRecord]] = defaultdict(list)
    for record in sampled:
        by_case_bbox[(record.case_id, record.bbox_id)].append(record)

    pages: list[Image.Image] = []
    summary = Image.new("RGB", (PAGE_W, PAGE_H), "white")
    draw = ImageDraw.Draw(summary)
    draw.text((MARGIN, 45), "PER-290 SV40 cleaned-mask DINOv3 linear probe", fill=(0, 0, 0), font=FONT_TITLE)
    y = 120
    lines = [
        f"Source: {args.source_run_root}",
        f"Morphology: remove FG components <= {args.remove_component_max_size}, fill BG holes <= {args.fill_hole_max_size}, connectivity={args.connectivity}",
        f"Sampling: up to {args.sample_max_per_wsi} patches per WSI; WSI-level holdout split",
        f"Feature model: {args.model_name}; reader={args.wsi_reader}; read_workers={args.read_workers}; batch={args.batch_size}",
    ]
    for line in lines:
        draw.text((MARGIN, y), line[:130], fill=(45, 45, 45), font=FONT_BODY)
        y += 34
    y += 20
    for row in metric_rows[:3]:
        draw.text((MARGIN, y), f"{row.get('split', '')}", fill=(0, 0, 0), font=FONT_H2)
        y += 36
        items = [
            f"n={row.get('n')} fg={row.get('fg')} bg={row.get('bg')}",
            f"precision={row.get('precision_fg'):.3f} recall={row.get('recall_fg'):.3f} f1={row.get('f1_fg'):.3f}",
            f"balanced_accuracy={row.get('balanced_accuracy'):.3f} bg_fpr={row.get('bg_false_positive_rate'):.3f} auc={row.get('roc_auc'):.3f}",
        ]
        for item in items:
            draw.text((MARGIN + 20, y), item, fill=(35, 35, 35), font=FONT_BODY)
            y += 30
        y += 16
    pages.append(summary)

    page_index = 0
    for case_id in heldout_cases:
        bbox_ids = sorted({bbox for (cid, bbox) in by_case_bbox if cid == case_id})[: args.pdf_max_bboxes_per_case]
        for bbox_id in bbox_ids:
            records = by_case_bbox[(case_id, bbox_id)]
            crop_path = Path(records[0].crop_path)
            mask_path = args.output_dir / "cleaned_masks" / case_id / bbox_id / "tissue_mask_cleaned.npy"
            crop = open_image(crop_path)
            if crop is None or not mask_path.exists():
                continue
            mask = np.load(mask_path).astype(bool)
            page = Image.new("RGB", (PAGE_W, PAGE_H), "white")
            draw = ImageDraw.Draw(page)
            draw.text((MARGIN, 40), f"{case_id} | {bbox_id}", fill=(0, 0, 0), font=FONT_TITLE)
            draw.text((MARGIN, 92), f"split={split_by_case.get(case_id)} | green label panel = cleaned VLM FG | prediction panel = sampled probe FG/BG", fill=(55, 55, 55), font=FONT_BODY)
            panel_w = (PAGE_W - 2 * MARGIN - 35) // 2
            panel_h = 1260
            label_panel = draw_grid_overlay(crop, mask, records, prob_by_sample, "label", (panel_w, panel_h))
            pred_panel = draw_grid_overlay(crop, mask, records, prob_by_sample, "prediction", (panel_w, panel_h))
            x0 = MARGIN
            x1 = MARGIN + panel_w + 35
            y0 = 165
            draw.text((x0, y0 - 36), "Cleaned Stage7 labels", fill=(0, 0, 0), font=FONT_H2)
            draw.text((x1, y0 - 36), "Linear-probe predictions", fill=(0, 0, 0), font=FONT_H2)
            page.paste(label_panel, (x0, y0))
            page.paste(pred_panel, (x1, y0))
            ys = y0 + panel_h + 35
            local_records = [
                r
                for r in sampled
                if r.case_id == case_id and r.bbox_id == bbox_id and r.sample_index in prob_by_sample
            ]
            if local_records:
                y_true = np.asarray([record.label_fg for record in local_records], dtype="int64")
                y_prob = np.asarray([prob_by_sample[record.sample_index] for record in local_records], dtype="float32")
                m = metric_summary(y_true, y_prob, args.probe_threshold)
                text = (
                    f"sampled cells in bbox: n={m['n']} fg={m['fg']} bg={m['bg']} | "
                    f"precision={m['precision_fg']:.3f} recall={m['recall_fg']:.3f} "
                    f"f1={m['f1_fg']:.3f} bg_fpr={m['bg_false_positive_rate']:.3f}"
                )
                for line in textwrap.wrap(text, width=120):
                    draw.text((MARGIN, ys), line, fill=(35, 35, 35), font=FONT_BODY)
                    ys += 30
            page_index += 1
            page_path = pages_dir / f"page_{page_index:03d}_{case_id}_{bbox_id}.png"
            page.save(page_path)
            pages.append(page)
    pdf_path = out_dir / "per290_sv40_cleaned_mask_probe_review.pdf"
    pages[0].save(pdf_path, save_all=True, append_images=pages[1:], resolution=150.0)
    return pdf_path


def write_reproduction(args: argparse.Namespace, summary: dict[str, Any]) -> None:
    command = [
        "python",
        "scripts/train_per276_sv40_cleaned_mask_probe.py",
        "--source-run-root",
        str(args.source_run_root),
        "--output-dir",
        str(args.output_dir),
        "--sample-max-per-wsi",
        str(args.sample_max_per_wsi),
        "--remove-component-max-size",
        str(args.remove_component_max_size),
        "--fill-hole-max-size",
        str(args.fill_hole_max_size),
        "--connectivity",
        str(args.connectivity),
        "--model-name",
        str(args.model_name),
        "--batch-size",
        str(args.batch_size),
        "--wsi-reader",
        str(args.wsi_reader),
        "--read-workers",
        str(args.read_workers),
    ]
    if args.case_limit is not None:
        command.extend(["--case-limit", str(args.case_limit)])
    if args.allow_timm_fallback:
        command.append("--allow-timm-fallback")
    lines = [
        "PER-290 SV40 cleaned-mask DINOv3 linear probe",
        "",
        f"Created: {datetime.now(timezone.utc).isoformat()}",
        f"Ticket: {args.ticket}",
        f"Git commit: {git_commit()}",
        "",
        "Command:",
        "  " + " ".join(shlex.quote(part) for part in command),
        "",
        "Source run:",
        f"  {args.source_run_root.resolve()}",
        "",
        "Morphology:",
        f"  Remove foreground connected components with size <= {args.remove_component_max_size}.",
        f"  Fill enclosed background holes with size <= {args.fill_hole_max_size}.",
        f"  Connectivity: {args.connectivity}-way.",
        "  Implementation uses postprocess_mask.remove_small_components with min_size=max_size+1 and fill_small_holes.",
        "",
        "Feature extraction:",
        f"  model_backend={args.model_backend}",
        f"  model_name={args.model_name}",
        f"  batch_size={args.batch_size}",
        f"  wsi_reader={args.wsi_reader}",
        f"  read_workers={args.read_workers}",
        "  HF_TOKEN is required for DINOv3 access if the model is not already cached; token value intentionally not recorded.",
        "",
        "DVC:",
        "  no .dvc directory",
        "",
        "Outputs:",
        f"  summary: {summary.get('summary_json')}",
        f"  metrics: {summary.get('metrics_csv')}",
        f"  review_pdf: {summary.get('review_pdf')}",
        "",
        "Git status at run time:",
        git_status_short().rstrip() or "clean",
        "",
    ]
    (args.output_dir / "reproduction.txt").write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[source] {args.source_run_root}", flush=True)
    cases = load_source_cases(args)
    if not cases:
        raise SystemExit("No source cases resolved")
    print(f"[source] cases={len(cases)}", flush=True)

    pool, census_rows, skipped_rows = build_patch_pool(args, cases)
    write_csv(args.output_dir / "patch_census.csv", census_rows)
    write_csv(args.output_dir / "skipped_bboxes.csv", skipped_rows)
    print(f"[pool] records={len(pool)} skipped_bboxes={len(skipped_rows)}", flush=True)
    sampled, audit_rows = sample_records(args, pool)
    write_csv(args.output_dir / "sample_manifest.csv", [patch_row(record) for record in sampled])
    write_csv(args.output_dir / "sample_audit.csv", audit_rows)
    print(f"[sample] records={len(sampled)}", flush=True)

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
    x, y, sample_indices, case_ids, feature_rows = extract_all_features(args, sampled, extractor)
    write_csv(args.output_dir / "feature_cache_summary.csv", feature_rows)
    np.savez_compressed(
        args.output_dir / "features/sampled_features_all.npz",
        features=x,
        labels=y,
        sample_index=sample_indices,
        case_id=case_ids,
        model_backend=np.asarray(extractor.backend),
        model_name=np.asarray(extractor.model_name),
        created_at=np.asarray(datetime.now(timezone.utc).isoformat()),
    )

    split_model, final_model, metric_rows, split_rows, prob = train_and_evaluate(args, x, y, case_ids)
    write_csv(args.output_dir / "split_manifest.csv", split_rows)
    write_csv(args.output_dir / "metrics.csv", metric_rows)
    model_dir = args.output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(split_model, model_dir / "logreg_train_split.joblib")
    joblib.dump(final_model, model_dir / "logreg_all_samples.joblib")
    prediction_rows = []
    record_by_sample = {record.sample_index: record for record in sampled}
    for idx, sample_index in enumerate(sample_indices):
        record = record_by_sample[int(sample_index)]
        p = float(prob[idx])
        prediction_rows.append(
            {
                **patch_row(record),
                "split": "holdout" if any(row["case_id"] == record.case_id and row["split"] == "holdout" for row in split_rows) else "train",
                "prob_fg": p,
                "pred_fg": int(p >= args.probe_threshold),
            }
        )
    write_csv(args.output_dir / "sample_predictions.csv", prediction_rows)

    review_pdf = ""
    if not args.skip_pdf:
        review_pdf = str(render_review_pdf(args, sampled, split_rows, metric_rows, sample_indices, prob))
    summary = {
        "ticket": args.ticket,
        "source_run_root": str(args.source_run_root.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "source_cases": len(cases),
        "pool_records": len(pool),
        "sample_records": len(sampled),
        "sample_fg": int(y.sum()),
        "sample_bg": int(len(y) - y.sum()),
        "morphology": {
            "remove_foreground_components_size_le": int(args.remove_component_max_size),
            "fill_background_holes_size_le": int(args.fill_hole_max_size),
            "connectivity": int(args.connectivity),
        },
        "feature_extractor": extractor.meta,
        "package_versions": package_versions(),
        "metrics": metric_rows,
        "summary_json": str((args.output_dir / "summary.json").resolve()),
        "metrics_csv": str((args.output_dir / "metrics.csv").resolve()),
        "review_pdf": review_pdf,
    }
    write_json(args.output_dir / "summary.json", summary)
    write_reproduction(args, summary)
    print(json.dumps(summary, indent=2)[:4000], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
