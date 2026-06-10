#!/usr/bin/env python3
"""Benchmark SV40 DINOv3 feature extraction paths on fixed smoke-run patches."""

from __future__ import annotations

import argparse
import csv
import json
import math
import queue
import shlex
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

try:
    import torch
    import torch.nn.functional as F
except Exception as exc:  # pragma: no cover - import-time environment failure.
    raise RuntimeError("PyTorch is required for the DINOv3 throughput benchmark") from exc

try:
    from cucim import CuImage
except Exception:  # pragma: no cover - depends on host environment.
    CuImage = None  # type: ignore[assignment]


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from compare_scale500_stress32_probe_transfer import load_stress_probe, predict_prob  # noqa: E402


DEFAULT_SMOKE_RUN_DIR = REPO_ROOT / "runs/stress32_n500_sv40_augmented_probe_v1_smoke"
DEFAULT_STRESS_RUN_DIR = REPO_ROOT / "runs/stress32_gt_overlay_sample_efficiency_probe_v1"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "runs/sv40_dinov3_throughput_probe_v1"
DEFAULT_CASE_IDS = [
    "anon_0f916c21_02b3_4b7f_a16e_260abfc2a664",
    "anon_e60142ee_b5fd_44da_b63e_daa0a506e472",
]
ALL_VARIANTS = [
    "baseline_current",
    "readpool_pil_hf",
    "readpool_tensor_preprocess",
    "gpu_probe_no_feature_cache",
]


@dataclass(frozen=True)
class PatchRecord:
    case_id: str
    candidate_order: int
    candidate_id: str
    row: int
    col: int
    x: int
    y: int
    width: int
    height: int

    @property
    def record_id(self) -> str:
        return f"{self.case_id}:{self.candidate_order}:{self.row}:{self.col}:{self.x}:{self.y}"


@dataclass
class CaseData:
    case_id: str
    task: str
    stain: str
    wsi_path: Path
    records: list[PatchRecord]
    candidate_bboxes: dict[int, tuple[int, int, int, int]]
    recorded_smoke_pps: float | None
    recorded_smoke_seconds: float | None


@dataclass
class TimedBatch:
    features_cpu: np.ndarray | None
    features_gpu: torch.Tensor | None
    probs: np.ndarray | None
    timings: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-run-dir", type=Path, default=DEFAULT_SMOKE_RUN_DIR)
    parser.add_argument("--stress-run-dir", type=Path, default=DEFAULT_STRESS_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--case-ids", default=",".join(DEFAULT_CASE_IDS))
    parser.add_argument("--variants", default=",".join(ALL_VARIANTS))
    parser.add_argument("--ticket", default="PER-272")
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--read-workers", type=int, default=16)
    parser.add_argument("--readpool-workers", type=int, default=16)
    parser.add_argument("--prefetch-queue-batches", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--probe-threshold", type=float, default=0.5)
    parser.add_argument("--max-patches-per-case", type=int, default=None)
    parser.add_argument("--write-feature-caches", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def git_output(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"git {' '.join(args)} failed: {exc}"


def parse_list(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def parse_bbox(text: str) -> tuple[int, int, int, int]:
    value = json.loads(text)
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"Invalid bbox: {text}")
    return tuple(int(round(float(v))) for v in value)  # type: ignore[return-value]


def sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def add_timings(target: dict[str, float], source: dict[str, float]) -> None:
    for key, value in source.items():
        target[key] = target.get(key, 0.0) + float(value)


def chunked(items: list[PatchRecord], size: int) -> Iterable[list[PatchRecord]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def pad_rgb_array(arr: np.ndarray, side: int = 512) -> np.ndarray:
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    else:
        arr = arr[:, :, :3]
    arr = np.asarray(arr, dtype=np.uint8)
    if arr.shape[0] == side and arr.shape[1] == side:
        return arr
    canvas = np.full((side, side, 3), 255, dtype=np.uint8)
    h = min(side, arr.shape[0])
    w = min(side, arr.shape[1])
    canvas[:h, :w, :] = arr[:h, :w, :]
    return canvas


class SerialCuCimReader:
    def __init__(self, wsi_path: Path, read_workers: int) -> None:
        if CuImage is None:
            raise RuntimeError("cuCIM is not available")
        self.slide = CuImage(str(wsi_path))
        self.read_workers = int(read_workers)

    def close(self) -> None:
        close = getattr(self.slide, "close", None)
        if callable(close):
            close()

    def read_pil(self, record: PatchRecord) -> tuple[Image.Image, dict[str, float]]:
        t0 = time.perf_counter()
        region = self.slide.read_region(
            location=(record.x, record.y),
            size=(record.width, record.height),
            level=0,
            num_workers=self.read_workers,
        )
        t1 = time.perf_counter()
        arr = np.asarray(region)
        t2 = time.perf_counter()
        image = Image.fromarray(pad_rgb_array(arr)).convert("RGB")
        t3 = time.perf_counter()
        return image, {
            "read_region_seconds_sum": t1 - t0,
            "array_convert_seconds_sum": t2 - t1,
            "pil_convert_pad_seconds_sum": t3 - t2,
        }


_THREAD_LOCAL = threading.local()


def _thread_slide(wsi_path: Path) -> Any:
    slide = getattr(_THREAD_LOCAL, "slide", None)
    slide_path = getattr(_THREAD_LOCAL, "slide_path", None)
    if slide is None or slide_path != str(wsi_path):
        if CuImage is None:
            raise RuntimeError("cuCIM is not available")
        slide = CuImage(str(wsi_path))
        _THREAD_LOCAL.slide = slide
        _THREAD_LOCAL.slide_path = str(wsi_path)
    return slide


def read_pool_record(args: tuple[Path, PatchRecord, str]) -> tuple[Any, dict[str, float]]:
    wsi_path, record, output_kind = args
    slide = _thread_slide(wsi_path)
    t0 = time.perf_counter()
    region = slide.read_region(location=(record.x, record.y), size=(record.width, record.height), level=0)
    t1 = time.perf_counter()
    arr = np.asarray(region)
    t2 = time.perf_counter()
    padded = pad_rgb_array(arr)
    t3 = time.perf_counter()
    if output_kind == "pil":
        payload = Image.fromarray(padded).convert("RGB")
        t4 = time.perf_counter()
        timings = {
            "read_region_seconds_sum": t1 - t0,
            "array_convert_seconds_sum": t2 - t1,
            "array_pad_seconds_sum": t3 - t2,
            "pil_convert_pad_seconds_sum": t4 - t3,
        }
    else:
        payload = padded
        timings = {
            "read_region_seconds_sum": t1 - t0,
            "array_convert_seconds_sum": t2 - t1,
            "array_pad_seconds_sum": t3 - t2,
        }
    return payload, timings


class TimedDinoExtractor:
    def __init__(self, model_name: str, device_arg: str, batch_size: int) -> None:
        from transformers import AutoImageProcessor, AutoModel

        self.model_name = model_name
        self.device = torch.device(device_arg if device_arg != "auto" else ("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.batch_size = int(batch_size)
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.height, self.width = self._processor_size()
        self.mean = torch.tensor(getattr(self.processor, "image_mean", (0.485, 0.456, 0.406)), dtype=torch.float32)
        self.std = torch.tensor(getattr(self.processor, "image_std", (0.229, 0.224, 0.225)), dtype=torch.float32)
        self.mean = self.mean.view(1, 3, 1, 1).to(self.device)
        self.std = self.std.view(1, 3, 1, 1).to(self.device)

    def _processor_size(self) -> tuple[int, int]:
        size = getattr(self.processor, "size", None) or {}
        if isinstance(size, dict):
            height = size.get("height") or size.get("shortest_edge") or size.get("longest_edge") or 224
            width = size.get("width") or size.get("shortest_edge") or size.get("longest_edge") or height
            return int(height), int(width)
        return 224, 224

    def _feature_from_outputs(self, outputs: Any) -> torch.Tensor:
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            return outputs.pooler_output
        if hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state[:, 0]
        first = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        return first[:, 0] if first.ndim == 3 else first

    @torch.inference_mode()
    def extract_hf(self, images: list[Image.Image]) -> TimedBatch:
        timings: dict[str, float] = {}
        t0 = time.perf_counter()
        inputs = self.processor(images=images, return_tensors="pt")
        t1 = time.perf_counter()
        sync_if_needed(self.device)
        inputs = {key: value.to(self.device, non_blocking=True) for key, value in inputs.items()}
        sync_if_needed(self.device)
        t2 = time.perf_counter()
        with torch.autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
            outputs = self.model(**inputs)
        sync_if_needed(self.device)
        t3 = time.perf_counter()
        features = self._feature_from_outputs(outputs)
        cpu = features.detach().float().cpu().numpy().astype("float32")
        t4 = time.perf_counter()
        timings.update(
            {
                "hf_preprocess_seconds": t1 - t0,
                "input_copy_seconds": t2 - t1,
                "gpu_forward_seconds": t3 - t2,
                "feature_cpu_copy_seconds": t4 - t3,
            }
        )
        return TimedBatch(features_cpu=cpu, features_gpu=None, probs=None, timings=timings)

    @torch.inference_mode()
    def extract_tensor(
        self,
        arrays: list[np.ndarray],
        *,
        return_cpu_features: bool,
        gpu_scorer: "GpuProbeScorer | None" = None,
    ) -> TimedBatch:
        timings: dict[str, float] = {}
        t0 = time.perf_counter()
        stacked = np.stack(arrays, axis=0)
        tensor = torch.from_numpy(stacked).permute(0, 3, 1, 2)
        t1 = time.perf_counter()
        sync_if_needed(self.device)
        tensor = tensor.to(self.device, non_blocking=True)
        sync_if_needed(self.device)
        t2 = time.perf_counter()
        tensor = tensor.float().div_(255.0)
        tensor = F.interpolate(tensor, size=(self.height, self.width), mode="bicubic", align_corners=False)
        tensor = (tensor - self.mean) / self.std
        sync_if_needed(self.device)
        t3 = time.perf_counter()
        with torch.autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
            outputs = self.model(pixel_values=tensor)
        sync_if_needed(self.device)
        t4 = time.perf_counter()
        features = self._feature_from_outputs(outputs).float()
        probs = None
        if gpu_scorer is not None:
            probs_tensor = gpu_scorer.predict_proba(features)
            sync_if_needed(self.device)
            t5 = time.perf_counter()
            probs = probs_tensor.detach().float().cpu().numpy().astype("float32")
            t6 = time.perf_counter()
            features_cpu = None
            timings["gpu_probe_seconds"] = t5 - t4
            timings["prob_cpu_copy_seconds"] = t6 - t5
            copy_end = t6
        elif return_cpu_features:
            features_cpu = features.detach().float().cpu().numpy().astype("float32")
            t5 = time.perf_counter()
            timings["feature_cpu_copy_seconds"] = t5 - t4
            copy_end = t5
        else:
            features_cpu = None
            copy_end = t4
        timings.update(
            {
                "cpu_stack_seconds": t1 - t0,
                "input_copy_seconds": t2 - t1,
                "tensor_preprocess_seconds": t3 - t2,
                "gpu_forward_seconds": t4 - t3,
                "batch_total_post_read_seconds": copy_end - t0,
            }
        )
        return TimedBatch(features_cpu=features_cpu, features_gpu=None, probs=probs, timings=timings)


class GpuProbeScorer:
    def __init__(self, probe_model: Any, device: torch.device) -> None:
        scaler = None
        logistic = None
        for _name, step in getattr(probe_model, "steps", []):
            if hasattr(step, "mean_") and hasattr(step, "scale_"):
                scaler = step
            if hasattr(step, "coef_") and hasattr(step, "intercept_"):
                logistic = step
        if scaler is None or logistic is None:
            raise ValueError("Expected sklearn Pipeline with StandardScaler and LogisticRegression")
        classes = list(getattr(logistic, "classes_", [0, 1]))
        if 1 not in classes:
            raise ValueError(f"Positive class 1 missing from logistic classes: {classes}")
        mean = np.asarray(scaler.mean_, dtype="float32")
        scale = np.asarray(scaler.scale_, dtype="float32")
        coef = np.asarray(logistic.coef_, dtype="float32")
        intercept = np.asarray(logistic.intercept_, dtype="float32")
        if coef.shape[0] != 1:
            raise ValueError(f"Expected binary logistic coef shape (1,d), got {coef.shape}")
        # sklearn's binary coef is the log-odds for classes_[1]. These probes are trained on 0/1 labels.
        sign = 1.0 if classes[-1] == 1 else -1.0
        self.mean = torch.from_numpy(mean).to(device)
        self.scale = torch.from_numpy(scale).to(device)
        self.coef = torch.from_numpy(coef[0] * sign).to(device)
        self.intercept = torch.tensor(float(intercept[0] * sign), dtype=torch.float32, device=device)

    def predict_proba(self, features: torch.Tensor) -> torch.Tensor:
        z = (features - self.mean) / self.scale
        logits = z.matmul(self.coef) + self.intercept
        return torch.sigmoid(logits)


def load_cases(args: argparse.Namespace) -> list[CaseData]:
    case_filter = set(parse_list(args.case_ids))
    rows = read_csv(args.smoke_run_dir / "sv40_candidate_crops.csv")
    rows_by_case: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if not case_filter or row["case_id"] in case_filter:
            rows_by_case[row["case_id"]].append(row)

    smoke_summary = json.loads((args.smoke_run_dir / "summary.json").read_text())
    meta_by_case = smoke_summary.get("feature_meta_by_case", {})
    cases: list[CaseData] = []
    for case_id in sorted(rows_by_case):
        feature_path = args.smoke_run_dir / "features" / f"{case_id}_sv40_review_candidates_features.npz"
        if not feature_path.exists():
            raise FileNotFoundError(feature_path)
        with np.load(feature_path, allow_pickle=False) as data:
            records = [
                PatchRecord(
                    case_id=case_id,
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
        if args.max_patches_per_case is not None:
            records = records[: int(args.max_patches_per_case)]
        candidate_bboxes = {int(row["candidate_order"]): parse_bbox(row["bbox_level0"]) for row in rows_by_case[case_id]}
        first = rows_by_case[case_id][0]
        meta = meta_by_case.get(case_id, {})
        cases.append(
            CaseData(
                case_id=case_id,
                task=first.get("task", ""),
                stain=first.get("stain", ""),
                wsi_path=Path(first["resolved_wsi_path"]),
                records=records,
                candidate_bboxes=candidate_bboxes,
                recorded_smoke_pps=float(meta["patches_per_second"]) if "patches_per_second" in meta else None,
                recorded_smoke_seconds=float(meta["extract_seconds"]) if "extract_seconds" in meta else None,
            )
        )
    return cases


def validate_records_inside_bboxes(case: CaseData) -> tuple[bool, int]:
    outside = 0
    for record in case.records:
        bbox = case.candidate_bboxes.get(record.candidate_order)
        if bbox is None:
            outside += 1
            continue
        x0, y0, x1, y1 = bbox
        if not (x0 <= record.x and y0 <= record.y and record.x + record.width <= x1 and record.y + record.height <= y1):
            outside += 1
    return outside == 0, outside


def run_baseline_current(case: CaseData, extractor: TimedDinoExtractor, args: argparse.Namespace) -> tuple[np.ndarray, dict[str, float]]:
    batches: queue.Queue[tuple[list[Image.Image], dict[str, float]] | Exception | None] = queue.Queue(
        maxsize=max(1, int(args.prefetch_queue_batches))
    )
    read_timing_totals: dict[str, float] = defaultdict(float)
    producer_wall = 0.0

    def producer() -> None:
        nonlocal producer_wall
        reader = SerialCuCimReader(case.wsi_path, int(args.read_workers))
        started = time.perf_counter()
        try:
            images: list[Image.Image] = []
            batch_timings: dict[str, float] = defaultdict(float)
            for record in case.records:
                image, timings = reader.read_pil(record)
                images.append(image)
                add_timings(batch_timings, timings)
                add_timings(read_timing_totals, timings)
                if len(images) >= extractor.batch_size:
                    batches.put((images, dict(batch_timings)))
                    images = []
                    batch_timings = defaultdict(float)
            if images:
                batches.put((images, dict(batch_timings)))
        except Exception as exc:
            batches.put(exc)
        finally:
            producer_wall = time.perf_counter() - started
            reader.close()
            batches.put(None)

    features: list[np.ndarray] = []
    timing_totals: dict[str, float] = defaultdict(float)
    wall_start = time.perf_counter()
    thread = threading.Thread(target=producer, name=f"baseline-current-{case.case_id}", daemon=True)
    thread.start()
    while True:
        item = batches.get()
        if item is None:
            break
        if isinstance(item, Exception):
            thread.join(timeout=5)
            raise item
        images, batch_read_timings = item
        batch = extractor.extract_hf(images)
        features.append(batch.features_cpu)  # type: ignore[arg-type]
        add_timings(timing_totals, batch_read_timings)
        add_timings(timing_totals, batch.timings)
    thread.join()
    wall_seconds = time.perf_counter() - wall_start
    timing_totals["variant_wall_seconds"] = wall_seconds
    timing_totals["producer_wall_seconds"] = producer_wall
    timing_totals["read_wall_seconds"] = producer_wall
    timing_totals["patches_per_second"] = len(case.records) / wall_seconds if wall_seconds > 0 else 0.0
    return np.concatenate(features, axis=0), dict(timing_totals)


def run_readpool_pil_hf(case: CaseData, extractor: TimedDinoExtractor, args: argparse.Namespace) -> tuple[np.ndarray, dict[str, float]]:
    from concurrent.futures import ThreadPoolExecutor

    features: list[np.ndarray] = []
    timing_totals: dict[str, float] = defaultdict(float)
    wall_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=int(args.readpool_workers), thread_name_prefix="cucim-read") as pool:
        for records in chunked(case.records, extractor.batch_size):
            read_start = time.perf_counter()
            results = list(pool.map(read_pool_record, [(case.wsi_path, record, "pil") for record in records]))
            read_wall = time.perf_counter() - read_start
            timing_totals["read_wall_seconds"] += read_wall
            images = [payload for payload, _timings in results]
            for _payload, timings in results:
                add_timings(timing_totals, timings)
            batch = extractor.extract_hf(images)
            features.append(batch.features_cpu)  # type: ignore[arg-type]
            add_timings(timing_totals, batch.timings)
    wall_seconds = time.perf_counter() - wall_start
    timing_totals["variant_wall_seconds"] = wall_seconds
    timing_totals["patches_per_second"] = len(case.records) / wall_seconds if wall_seconds > 0 else 0.0
    return np.concatenate(features, axis=0), dict(timing_totals)


def run_readpool_tensor(
    case: CaseData,
    extractor: TimedDinoExtractor,
    args: argparse.Namespace,
    *,
    gpu_scorer: GpuProbeScorer | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, float]]:
    from concurrent.futures import ThreadPoolExecutor

    features: list[np.ndarray] = []
    probs: list[np.ndarray] = []
    timing_totals: dict[str, float] = defaultdict(float)
    wall_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=int(args.readpool_workers), thread_name_prefix="cucim-read") as pool:
        for records in chunked(case.records, extractor.batch_size):
            read_start = time.perf_counter()
            results = list(pool.map(read_pool_record, [(case.wsi_path, record, "array") for record in records]))
            read_wall = time.perf_counter() - read_start
            timing_totals["read_wall_seconds"] += read_wall
            arrays = [payload for payload, _timings in results]
            for _payload, timings in results:
                add_timings(timing_totals, timings)
            batch = extractor.extract_tensor(arrays, return_cpu_features=gpu_scorer is None, gpu_scorer=gpu_scorer)
            if batch.features_cpu is not None:
                features.append(batch.features_cpu)
            if batch.probs is not None:
                probs.append(batch.probs)
            add_timings(timing_totals, batch.timings)
    wall_seconds = time.perf_counter() - wall_start
    timing_totals["variant_wall_seconds"] = wall_seconds
    timing_totals["patches_per_second"] = len(case.records) / wall_seconds if wall_seconds > 0 else 0.0
    feature_array = np.concatenate(features, axis=0) if features else None
    prob_array = np.concatenate(probs, axis=0) if probs else None
    return feature_array, prob_array, dict(timing_totals)


def cosine_summary(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    if a.shape != b.shape:
        return {"feature_cosine_mean": math.nan, "feature_cosine_min": math.nan}
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    denom = np.where(denom == 0, np.nan, denom)
    cos = np.sum(a * b, axis=1) / denom
    return {
        "feature_cosine_mean": float(np.nanmean(cos)),
        "feature_cosine_min": float(np.nanmin(cos)),
    }


def comparison_summary(
    baseline_features: np.ndarray,
    baseline_probs: np.ndarray,
    baseline_pred: np.ndarray,
    features: np.ndarray | None,
    probs: np.ndarray,
    pred: np.ndarray,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "prob_abs_delta_mean": float(np.mean(np.abs(probs - baseline_probs))),
        "prob_abs_delta_max": float(np.max(np.abs(probs - baseline_probs))),
        "prediction_agreement": float(np.mean(pred == baseline_pred)),
        "prediction_disagreement_count": int(np.sum(pred != baseline_pred)),
    }
    if features is not None:
        out.update(cosine_summary(baseline_features, features))
    return out


def maybe_write_feature_cache(
    args: argparse.Namespace,
    case: CaseData,
    variant: str,
    features: np.ndarray | None,
    probs: np.ndarray,
    pred: np.ndarray,
) -> float:
    if not args.write_feature_caches:
        return 0.0
    t0 = time.perf_counter()
    path = args.output_dir / "features" / f"{case.case_id}_{variant}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "prob_fg": probs.astype("float32"),
        "pred_fg": pred.astype("int64"),
        "record_id": np.asarray([record.record_id for record in case.records]),
        "candidate_order": np.asarray([record.candidate_order for record in case.records], dtype="int64"),
        "x_level0": np.asarray([record.x for record in case.records], dtype="int64"),
        "y_level0": np.asarray([record.y for record in case.records], dtype="int64"),
    }
    if features is not None:
        payload["features"] = features.astype("float32")
    np.savez_compressed(path, **payload)
    return time.perf_counter() - t0


def make_row(
    args: argparse.Namespace,
    case: CaseData,
    variant: str,
    timings: dict[str, float],
    probs: np.ndarray,
    pred: np.ndarray,
    inside_ok: bool,
    outside_count: int,
    comparison: dict[str, Any] | None,
    cache_write_seconds: float,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "ticket": args.ticket,
        "case_id": case.case_id,
        "task": case.task,
        "stain": case.stain,
        "variant": variant,
        "patch_count": len(case.records),
        "wsi_path": str(case.wsi_path),
        "inside_candidate_bboxes": inside_ok,
        "outside_candidate_bbox_count": outside_count,
        "recorded_smoke_patches_per_second": case.recorded_smoke_pps if case.recorded_smoke_pps is not None else "",
        "recorded_smoke_extract_seconds": case.recorded_smoke_seconds if case.recorded_smoke_seconds is not None else "",
        "pred_fg": int(pred.sum()),
        "pred_bg": int(len(pred) - pred.sum()),
        "pred_fg_fraction": float(pred.mean()) if len(pred) else 0.0,
        "mean_prob_fg": float(probs.mean()) if len(probs) else 0.0,
        "cache_write_seconds": cache_write_seconds,
        "cache_written": bool(args.write_feature_caches),
    }
    row.update({key: float(value) for key, value in timings.items()})
    if comparison:
        row.update(comparison)
    return row


def write_reproduction(args: argparse.Namespace, cases: list[CaseData], variants: list[str]) -> Path:
    path = args.output_dir / "reproduction.txt"
    lines = [
        f"Generated timestamp: {datetime.now(timezone.utc).isoformat()}",
        f"Ticket: {args.ticket}",
        f"Repository: {REPO_ROOT}",
        f"Git commit: {git_output(['rev-parse', 'HEAD'])}",
        "Git status --short:",
        git_output(["status", "--short"]) or "clean",
        "",
        "Command:",
        " ".join(shlex.quote(part) for part in [sys.executable, *sys.argv]),
        "",
        f"Parent smoke run: {args.smoke_run_dir.resolve()}",
        f"Parent stress probe run: {args.stress_run_dir.resolve()}",
        f"Output root: {args.output_dir.resolve()}",
        f"Cases: {', '.join(case.case_id for case in cases)}",
        f"Variants: {', '.join(variants)}",
        f"Batch size: {args.batch_size}",
        f"Baseline serial cuCIM num_workers per read: {args.read_workers}",
        f"Read-pool workers: {args.readpool_workers}",
        "Storage note: smoke WSI paths are resolved from sv40_candidate_crops.csv; current smoke paths are /vol-backed.",
        "HF token requirement: DINOv3 model access may require HF_TOKEN, but the token value is intentionally not printed.",
        "DVC: no DVC metadata was found in this repo root during this run; runs/ outputs are git-ignored.",
        "",
        "Outputs:",
        "- throughput_summary.csv",
        "- summary.json",
        "- timings/*.json",
        "- reproduction.txt",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


def main() -> None:
    args = parse_args()
    variants = parse_list(args.variants)
    unknown = sorted(set(variants) - set(ALL_VARIANTS))
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}; choices={ALL_VARIANTS}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "git_status.txt").write_text(git_output(["status", "--short"]) + "\n")

    cases = load_cases(args)
    if not cases:
        raise ValueError("No cases selected")

    probe_args = argparse.Namespace(stress_run_dir=args.stress_run_dir, sample_seed=args.sample_seed)
    probe = load_stress_probe(probe_args, int(args.sample_size))
    extractor = TimedDinoExtractor(probe.feature_model, args.device, int(args.batch_size))
    gpu_scorer = GpuProbeScorer(probe.model, extractor.device)

    all_rows: list[dict[str, Any]] = []
    per_case_summary: dict[str, Any] = {}
    for case in cases:
        inside_ok, outside_count = validate_records_inside_bboxes(case)
        if not inside_ok:
            raise ValueError(f"{case.case_id} has {outside_count} records outside candidate bboxes")
        baseline_features: np.ndarray | None = None
        baseline_probs: np.ndarray | None = None
        baseline_pred: np.ndarray | None = None
        per_case_summary[case.case_id] = {
            "patch_count": len(case.records),
            "wsi_path": str(case.wsi_path),
            "recorded_smoke_patches_per_second": case.recorded_smoke_pps,
            "variants": {},
        }
        for variant in variants:
            started = time.perf_counter()
            if variant == "baseline_current":
                features, timings = run_baseline_current(case, extractor, args)
                probs = predict_prob(probe.model, features).astype("float32")
            elif variant == "readpool_pil_hf":
                features, timings = run_readpool_pil_hf(case, extractor, args)
                probs = predict_prob(probe.model, features).astype("float32")
            elif variant == "readpool_tensor_preprocess":
                features, _gpu_probs, timings = run_readpool_tensor(case, extractor, args, gpu_scorer=None)
                if features is None:
                    raise RuntimeError("readpool_tensor_preprocess did not return features")
                probs = predict_prob(probe.model, features).astype("float32")
            elif variant == "gpu_probe_no_feature_cache":
                features, probs, timings = run_readpool_tensor(case, extractor, args, gpu_scorer=gpu_scorer)
                if probs is None:
                    raise RuntimeError("gpu_probe_no_feature_cache did not return probabilities")
            else:  # pragma: no cover - guarded above.
                raise ValueError(variant)
            timings["full_variant_seconds_with_probe"] = time.perf_counter() - started
            pred = (probs >= float(args.probe_threshold)).astype("int64")

            comparison = None
            if variant == "baseline_current":
                if features is None:
                    raise RuntimeError("baseline_current must return features")
                baseline_features = features
                baseline_probs = probs
                baseline_pred = pred
                comparison = {
                    "prob_abs_delta_mean": 0.0,
                    "prob_abs_delta_max": 0.0,
                    "prediction_agreement": 1.0,
                    "prediction_disagreement_count": 0,
                    "feature_cosine_mean": 1.0,
                    "feature_cosine_min": 1.0,
                }
            elif baseline_features is not None and baseline_probs is not None and baseline_pred is not None:
                comparison = comparison_summary(baseline_features, baseline_probs, baseline_pred, features, probs, pred)

            cache_write_seconds = maybe_write_feature_cache(args, case, variant, features, probs, pred)
            row = make_row(args, case, variant, timings, probs, pred, inside_ok, outside_count, comparison, cache_write_seconds)
            if variant != "baseline_current":
                baseline_pps = per_case_summary[case.case_id]["variants"].get("baseline_current", {}).get("patches_per_second")
                if baseline_pps:
                    row["speedup_vs_baseline_current"] = float(row["patches_per_second"]) / float(baseline_pps)
            if case.recorded_smoke_pps:
                row["speedup_vs_recorded_smoke"] = float(row["patches_per_second"]) / float(case.recorded_smoke_pps)

            all_rows.append(row)
            per_case_summary[case.case_id]["variants"][variant] = row
            timing_path = args.output_dir / "timings" / f"{case.case_id}_{variant}.json"
            write_json(timing_path, row)

    write_csv(args.output_dir / "throughput_summary.csv", all_rows)
    write_reproduction(args, cases, variants)
    aggregate: dict[str, Any] = {
        "ticket": args.ticket,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(args.output_dir.resolve()),
        "smoke_run_dir": str(args.smoke_run_dir.resolve()),
        "stress_run_dir": str(args.stress_run_dir.resolve()),
        "variants": variants,
        "case_count": len(cases),
        "patch_count_total": int(sum(len(case.records) for case in cases)),
        "probe": {
            "name": probe.name,
            "feature_backend": probe.feature_backend,
            "feature_model": probe.feature_model,
            "train_count": probe.train_count,
            "train_fg": probe.train_fg,
            "train_bg": probe.train_bg,
        },
        "extractor": {
            "device": str(extractor.device),
            "batch_size": extractor.batch_size,
            "processor_size": [extractor.height, extractor.width],
        },
        "variant_mean_patches_per_second": {
            variant: float(np.mean([row["patches_per_second"] for row in all_rows if row["variant"] == variant]))
            for variant in variants
        },
        "cases": per_case_summary,
    }
    write_json(args.output_dir / "summary.json", aggregate)
    print(json.dumps({
        "output_dir": str(args.output_dir.resolve()),
        "summary": str((args.output_dir / "summary.json").resolve()),
        "throughput_summary": str((args.output_dir / "throughput_summary.csv").resolve()),
        "variant_mean_patches_per_second": aggregate["variant_mean_patches_per_second"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
