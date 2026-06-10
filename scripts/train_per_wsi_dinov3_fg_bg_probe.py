#!/usr/bin/env python3
"""Per-WSI frozen-DINO linear probes for selected-crop foreground/background patches."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import openslide
import torch
from PIL import Image
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

try:
    from cucim import CuImage
except Exception:  # pragma: no cover - optional runtime dependency
    CuImage = None  # type: ignore[assignment]


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ROOT = REPO_ROOT / "runs/auto_context_scale500_selector_all500_v1"
DEFAULT_MANIFEST = DEFAULT_RUN_ROOT / "manifests/completed_cases_500_20260604_openrouter_review_current.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "runs/per_wsi_dinov3_fg_bg_probe_baseline_v1"
DEFAULT_DINOV3_SMALL = "facebook/dinov3-vits16-pretrain-lvd1689m"
DEFAULT_DINOV2_SMALL = "vit_small_patch14_dinov2"


@dataclass(frozen=True)
class CaseInfo:
    task: int
    case_id: str
    stain: str
    wsi_path: str
    source_wsi_path: str
    run_dir: Path
    bbox_count: int
    fg_count: int
    bg_count: int
    max_leave_one_crop_budget: int
    verifier_selected_box_ids: str


@dataclass(frozen=True)
class PatchRecord:
    case_id: str
    task: int
    stain: str
    wsi_path: str
    source_wsi_path: str
    run_dir: Path
    bbox_name: str
    bbox_index: int
    patch_id: str
    row: int
    col: int
    x: int
    y: int
    width: int
    height: int
    label_fg: int

    @property
    def record_id(self) -> str:
        return f"{self.case_id}|bbox{self.bbox_index:02d}:{self.bbox_name}|{self.patch_id}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ticket", default="PER-250")
    parser.add_argument("--num-wsis", type=int, default=10)
    parser.add_argument("--case-seed", type=int, default=250)
    parser.add_argument("--sample-seeds", default="0,1,2")
    parser.add_argument("--budgets", default="5,10,20,50,100")
    parser.add_argument("--min-bboxes", type=int, default=2)
    parser.add_argument("--model-backend", choices=["transformers", "timm"], default="transformers")
    parser.add_argument("--model-name", default=DEFAULT_DINOV3_SMALL)
    parser.add_argument("--fallback-model-name", default=DEFAULT_DINOV2_SMALL)
    parser.add_argument(
        "--allow-timm-fallback",
        action="store_true",
        help="If the requested transformers model cannot be loaded, run the same protocol with timm fallback.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--input-size", type=int, default=None)
    parser.add_argument("--wsi-reader", choices=["openslide", "cucim"], default="openslide")
    parser.add_argument(
        "--read-workers",
        type=int,
        default=16,
        help="Number of cuCIM read workers per patch read; ignored when --wsi-reader openslide.",
    )
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--max-patches-per-case", type=int, default=None)
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


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {
        "python": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
        "torch": getattr(torch, "__version__", "unknown"),
        "openslide": getattr(openslide, "__version__", "unknown"),
        "numpy": getattr(np, "__version__", "unknown"),
    }
    for name in ("sklearn", "PIL", "transformers", "timm", "cucim"):
        try:
            mod = __import__(name)
            versions[name] = str(getattr(mod, "__version__", "unknown"))
        except Exception as exc:
            versions[name] = f"unavailable: {type(exc).__name__}: {exc}"
    return versions


def parse_int_list(text: str) -> list[int]:
    values = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError(f"No integer values parsed from {text!r}")
    return values


def case_dir_name(case_id: str) -> str:
    if case_id.startswith("anon_"):
        return "anon_" + case_id[len("anon_") :].replace("_", "-")
    return case_id.replace("_", "-")


def resolve_run_dir(run_root: Path, row: dict[str, str]) -> Path | None:
    task = int(row["task"])
    case_dir = run_root / case_dir_name(row["case_id"])
    matches = sorted(case_dir.glob(f"*task{task:03d}"))
    if not matches:
        return None
    if len(matches) > 1:
        exact = [path for path in matches if path.is_dir()]
        return exact[0] if exact else matches[0]
    return matches[0]


def read_completed_manifest(path: Path, run_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open() as f:
        for row in csv.DictReader(f):
            run_dir = resolve_run_dir(run_root, row)
            if run_dir is None:
                continue
            row = dict(row)
            row["run_dir"] = str(run_dir)
            rows.append(row)
    return rows


def bbox_dirs(run_dir: Path) -> list[Path]:
    root = run_dir / "bboxes"
    if not root.exists():
        return []
    return sorted([path for path in root.iterdir() if path.is_dir()])


def load_mask_counts(run_dir: Path) -> tuple[int, int, int, dict[str, tuple[int, int]]]:
    fg_total = 0
    bg_total = 0
    per_bbox: dict[str, tuple[int, int]] = {}
    for bbox in bbox_dirs(run_dir):
        mask_path = bbox / "stage7/tissue_mask_post.npy"
        if not mask_path.exists():
            continue
        mask = np.load(mask_path)
        fg = int(mask.astype(bool).sum())
        bg = int(mask.size - fg)
        per_bbox[bbox.name] = (fg, bg)
        fg_total += fg
        bg_total += bg
    return len(per_bbox), fg_total, bg_total, per_bbox


def leave_one_crop_capacity(per_bbox: dict[str, tuple[int, int]]) -> int:
    if len(per_bbox) < 2:
        return 0
    fg_total = sum(v[0] for v in per_bbox.values())
    bg_total = sum(v[1] for v in per_bbox.values())
    best = 0
    for fg, bg in per_bbox.values():
        best = max(best, min(fg_total - fg, bg_total - bg))
    return int(best)


def build_case_inventory(rows: list[dict[str, str]], min_bboxes: int) -> list[CaseInfo]:
    cases: list[CaseInfo] = []
    for row in rows:
        run_dir = Path(row["run_dir"])
        bbox_count, fg_count, bg_count, per_bbox = load_mask_counts(run_dir)
        if bbox_count < min_bboxes:
            continue
        cases.append(
            CaseInfo(
                task=int(row["task"]),
                case_id=row["case_id"],
                stain=row["stain"],
                wsi_path=row["wsi_path"],
                source_wsi_path=row.get("source_wsi_path", ""),
                run_dir=run_dir,
                bbox_count=bbox_count,
                fg_count=fg_count,
                bg_count=bg_count,
                max_leave_one_crop_budget=leave_one_crop_capacity(per_bbox),
                verifier_selected_box_ids=row.get("verifier_selected_box_ids", ""),
            )
        )
    return cases


def select_cases(cases: list[CaseInfo], num_wsis: int, max_budget: int, seed: int) -> list[CaseInfo]:
    eligible = [case for case in cases if case.max_leave_one_crop_budget >= max_budget]
    if len(eligible) < num_wsis:
        eligible = [case for case in cases if case.max_leave_one_crop_budget > 0]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(eligible))
    selected = [eligible[int(i)] for i in order[:num_wsis]]
    selected.sort(key=lambda c: (c.stain, c.task, c.case_id))
    return selected


def patch_dict(record: PatchRecord) -> dict[str, Any]:
    return {
        "record_id": record.record_id,
        "case_id": record.case_id,
        "task": record.task,
        "stain": record.stain,
        "wsi_path": record.wsi_path,
        "source_wsi_path": record.source_wsi_path,
        "run_dir": str(record.run_dir),
        "bbox_name": record.bbox_name,
        "bbox_index": record.bbox_index,
        "patch_id": record.patch_id,
        "row": record.row,
        "col": record.col,
        "x_level0": record.x,
        "y_level0": record.y,
        "width_level0": record.width,
        "height_level0": record.height,
        "label_fg": record.label_fg,
    }


def load_case_patches(case: CaseInfo) -> list[PatchRecord]:
    records: list[PatchRecord] = []
    for bbox_index, bbox in enumerate(bbox_dirs(case.run_dir), start=1):
        mask_path = bbox / "stage7/tissue_mask_post.npy"
        csv_path = bbox / "stage6/patches.csv"
        if not mask_path.exists() or not csv_path.exists():
            continue
        mask = np.load(mask_path).astype(bool)
        with csv_path.open() as f:
            for row in csv.DictReader(f):
                rr = int(row["row"])
                cc = int(row["col"])
                if rr < 0 or cc < 0 or rr >= mask.shape[0] or cc >= mask.shape[1]:
                    continue
                records.append(
                    PatchRecord(
                        case_id=case.case_id,
                        task=case.task,
                        stain=case.stain,
                        wsi_path=case.wsi_path,
                        source_wsi_path=case.source_wsi_path,
                        run_dir=case.run_dir,
                        bbox_name=bbox.name,
                        bbox_index=bbox_index,
                        patch_id=row["patch_id"],
                        row=rr,
                        col=cc,
                        x=int(row["wsi_x"]),
                        y=int(row["wsi_y"]),
                        width=int(row["patch_w"]),
                        height=int(row["patch_h"]),
                        label_fg=1 if bool(mask[rr, cc]) else 0,
                    )
                )
    records.sort(key=lambda r: (r.bbox_index, r.row, r.col, r.patch_id))
    return records


def resolve_wsi_path(record: PatchRecord) -> Path:
    # Prefer source_wsi_path: several scale-500 symlinked WSI paths emit
    # OpenSlide TIFF-directory warnings while the source path reads cleanly.
    for raw in (record.source_wsi_path, record.wsi_path):
        if raw:
            path = Path(raw)
            if path.exists():
                return path
    raise FileNotFoundError(f"No readable WSI path for {record.case_id}: {record.wsi_path} / {record.source_wsi_path}")


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
                "load_seconds": time.monotonic() - started,
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
            "timm_version": getattr(timm, "__version__", "unknown"),
            "input_size": self.input_size,
            "mean": [float(x) for x in self.mean.flatten()],
            "std": [float(x) for x in self.std.flatten()],
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


def cache_meta_matches(path: Path, extractor: FeatureExtractor) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            return str(data["model_backend"]) == extractor.backend and str(data["model_name"]) == extractor.model_name
    except Exception:
        return False


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
        if image.size[0] <= 0 or image.size[1] <= 0:
            raise ValueError(f"Empty patch read for {record.record_id}")
        if image.size != (record.width, record.height):
            padded = Image.new("RGB", (record.width, record.height), (255, 255, 255))
            padded.paste(image, (0, 0))
            image = padded
        return image

    def close(self) -> None:
        close = getattr(self.slide, "close", None)
        if callable(close):
            close()


def read_patch(reader: WsiPatchReader, record: PatchRecord) -> Image.Image:
    image = reader.read_patch(record)
    if image.size[0] <= 0 or image.size[1] <= 0:
        raise ValueError(f"Empty patch read for {record.record_id}")
    return image


def extract_case_features(
    case: CaseInfo,
    records: list[PatchRecord],
    extractor: FeatureExtractor,
    output_dir: Path,
    resume: bool,
    *,
    wsi_reader: str = "openslide",
    read_workers: int = 16,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    safe_case = case.case_id.replace("/", "_")
    feature_path = output_dir / "features" / f"{safe_case}_features.npz"
    if resume and cache_meta_matches(feature_path, extractor):
        with np.load(feature_path, allow_pickle=False) as data:
            return (
                data["features"],
                data["labels"].astype("int64"),
                data["bbox_indices"].astype("int64"),
                data["record_indices"].astype("int64"),
                [],
            )

    features: list[np.ndarray] = []
    labels: list[int] = []
    bbox_indices: list[int] = []
    record_indices: list[int] = []
    failures: list[dict[str, Any]] = []
    slide_path = resolve_wsi_path(records[0])
    batch_images: list[Image.Image] = []
    batch_meta: list[tuple[int, PatchRecord]] = []

    def flush() -> None:
        nonlocal batch_images, batch_meta
        if not batch_images:
            return
        batch_features = extractor.extract_batch(batch_images)
        for feat, (idx, rec) in zip(batch_features, batch_meta):
            features.append(feat)
            labels.append(rec.label_fg)
            bbox_indices.append(rec.bbox_index)
            record_indices.append(idx)
        batch_images = []
        batch_meta = []

    slide = WsiPatchReader(slide_path, wsi_reader, read_workers)
    try:
        for idx, record in enumerate(records):
            try:
                batch_images.append(read_patch(slide, record))
                batch_meta.append((idx, record))
                if len(batch_images) >= extractor.batch_size:
                    flush()
            except Exception as exc:
                failures.append({"record_id": record.record_id, "error": f"{type(exc).__name__}: {exc}"})
        flush()
    finally:
        slide.close()

    if not features:
        raise RuntimeError(f"No features extracted for {case.case_id}")

    feature_array = np.stack(features, axis=0).astype("float32")
    label_array = np.asarray(labels, dtype="int64")
    bbox_array = np.asarray(bbox_indices, dtype="int64")
    record_array = np.asarray(record_indices, dtype="int64")
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        feature_path,
        features=feature_array,
        labels=label_array,
        bbox_indices=bbox_array,
        record_indices=record_array,
        model_backend=np.asarray(extractor.backend),
        model_name=np.asarray(extractor.model_name),
        wsi_reader=np.asarray(wsi_reader),
        read_workers=np.asarray(read_workers),
        created_at=np.asarray(datetime.now(timezone.utc).isoformat()),
    )
    return feature_array, label_array, bbox_array, record_array, failures


def sample_train_indices(labels: np.ndarray, pool: np.ndarray, budget: int, seed: int) -> np.ndarray | None:
    fg = pool[labels[pool] == 1]
    bg = pool[labels[pool] == 0]
    if len(fg) < budget or len(bg) < budget:
        return None
    rng = np.random.default_rng(seed)
    chosen_fg = rng.choice(fg, size=budget, replace=False)
    chosen_bg = rng.choice(bg, size=budget, replace=False)
    chosen = np.concatenate([chosen_fg, chosen_bg])
    rng.shuffle(chosen)
    return chosen


def metric_value(fn: Any, *args: Any) -> float:
    try:
        value = fn(*args)
        return float(value)
    except Exception:
        return float("nan")


def train_and_eval(
    case: CaseInfo,
    features: np.ndarray,
    labels: np.ndarray,
    bbox_indices: np.ndarray,
    budgets: list[int],
    sample_seeds: list[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for heldout_bbox in sorted(set(int(x) for x in bbox_indices.tolist())):
        train_pool = np.where(bbox_indices != heldout_bbox)[0]
        test_idx = np.where(bbox_indices == heldout_bbox)[0]
        train_fg = int((labels[train_pool] == 1).sum())
        train_bg = int((labels[train_pool] == 0).sum())
        test_fg = int((labels[test_idx] == 1).sum())
        test_bg = int((labels[test_idx] == 0).sum())
        for budget in budgets:
            for sample_seed in sample_seeds:
                row: dict[str, Any] = {
                    "case_id": case.case_id,
                    "task": case.task,
                    "stain": case.stain,
                    "heldout_bbox_index": heldout_bbox,
                    "budget_per_class": budget,
                    "sample_seed": sample_seed,
                    "train_pool_fg": train_fg,
                    "train_pool_bg": train_bg,
                    "test_fg": test_fg,
                    "test_bg": test_bg,
                    "test_n": int(len(test_idx)),
                    "skipped": False,
                    "skip_reason": "",
                }
                chosen = sample_train_indices(labels, train_pool, budget, seed=sample_seed + 100000 * case.task + heldout_bbox)
                if chosen is None:
                    row["skipped"] = True
                    row["skip_reason"] = "insufficient_train_fg_or_bg"
                    rows.append(row)
                    continue
                if len(set(labels[chosen].tolist())) < 2:
                    row["skipped"] = True
                    row["skip_reason"] = "single_class_train"
                    rows.append(row)
                    continue
                clf = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(max_iter=2000, solver="liblinear", random_state=sample_seed),
                )
                clf.fit(features[chosen], labels[chosen])
                prob = clf.predict_proba(features[test_idx])[:, 1]
                pred = (prob >= 0.5).astype("int64")
                y = labels[test_idx]
                precision, recall, f1, _support = precision_recall_fscore_support(
                    y,
                    pred,
                    labels=[1],
                    average="binary",
                    zero_division=0,
                )
                cm = confusion_matrix(y, pred, labels=[0, 1])
                row.update(
                    {
                        "train_n": int(len(chosen)),
                        "train_fg": int((labels[chosen] == 1).sum()),
                        "train_bg": int((labels[chosen] == 0).sum()),
                        "accuracy": float(accuracy_score(y, pred)),
                        "balanced_accuracy": metric_value(balanced_accuracy_score, y, pred),
                        "precision_fg": float(precision),
                        "recall_fg": float(recall),
                        "f1_fg": float(f1),
                        "roc_auc": metric_value(roc_auc_score, y, prob) if len(set(y.tolist())) > 1 else float("nan"),
                        "average_precision": metric_value(average_precision_score, y, prob)
                        if len(set(y.tolist())) > 1
                        else float("nan"),
                        "tn_bg_as_bg": int(cm[0, 0]),
                        "fp_bg_as_fg": int(cm[0, 1]),
                        "fn_fg_as_bg": int(cm[1, 0]),
                        "tp_fg_as_fg": int(cm[1, 1]),
                    }
                )
                rows.append(row)
    return rows


def finite_mean(values: list[float]) -> float:
    finite = [v for v in values if isinstance(v, (float, int)) and math.isfinite(float(v))]
    return float(np.mean(finite)) if finite else float("nan")


def finite_std(values: list[float]) -> float:
    finite = [v for v in values if isinstance(v, (float, int)) and math.isfinite(float(v))]
    return float(np.std(finite, ddof=1)) if len(finite) > 1 else float("nan")


def aggregate_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if not row.get("skipped"):
            groups[int(row["budget_per_class"])].append(row)
    metric_names = ["accuracy", "balanced_accuracy", "precision_fg", "recall_fg", "f1_fg", "roc_auc", "average_precision"]
    out: list[dict[str, Any]] = []
    for budget, items in sorted(groups.items()):
        agg: dict[str, Any] = {
            "budget_per_class": budget,
            "evaluations": len(items),
            "cases": len(set(row["case_id"] for row in items)),
            "heldout_folds": len(set((row["case_id"], row["heldout_bbox_index"]) for row in items)),
            "sample_seeds": len(set(row["sample_seed"] for row in items)),
        }
        for name in metric_names:
            values = [float(row[name]) for row in items if name in row and row[name] != ""]
            agg[f"{name}_mean"] = finite_mean(values)
            agg[f"{name}_std"] = finite_std(values)
        out.append(agg)
    return out


def aggregate_case_budget(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if not row.get("skipped"):
            groups[(str(row["case_id"]), int(row["budget_per_class"]))].append(row)
    out: list[dict[str, Any]] = []
    for (case_id, budget), items in sorted(groups.items()):
        first = items[0]
        out.append(
            {
                "case_id": case_id,
                "task": first["task"],
                "stain": first["stain"],
                "budget_per_class": budget,
                "evaluations": len(items),
                "heldout_folds": len(set(row["heldout_bbox_index"] for row in items)),
                "balanced_accuracy_mean": finite_mean([float(row["balanced_accuracy"]) for row in items]),
                "f1_fg_mean": finite_mean([float(row["f1_fg"]) for row in items]),
                "roc_auc_mean": finite_mean([float(row["roc_auc"]) for row in items]),
                "average_precision_mean": finite_mean([float(row["average_precision"]) for row in items]),
            }
        )
    return out


def make_plot(aggregate_rows: list[dict[str, Any]], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    if not aggregate_rows:
        return
    budgets = [int(row["budget_per_class"]) for row in aggregate_rows]
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for metric, label in [
        ("balanced_accuracy", "balanced accuracy"),
        ("f1_fg", "foreground F1"),
        ("roc_auc", "AUROC"),
        ("average_precision", "AUPRC"),
    ]:
        y = [float(row.get(f"{metric}_mean", float("nan"))) for row in aggregate_rows]
        yerr = [float(row.get(f"{metric}_std", float("nan"))) for row in aggregate_rows]
        ax.errorbar(budgets, y, yerr=yerr, marker="o", capsize=3, label=label)
    ax.set_xscale("log")
    ax.set_xticks(budgets)
    ax.set_xticklabels([str(x) for x in budgets])
    ax.set_ylim(0.0, 1.03)
    ax.set_xlabel("training patches per class")
    ax.set_ylabel("held-out crop performance")
    ax.set_title("Per-WSI frozen-DINO FG/BG probe sample efficiency")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def write_reproduction(output_dir: Path, args: argparse.Namespace, summary: dict[str, Any]) -> None:
    lines = [
        "Per-WSI Frozen-DINO FG/BG Probe Baseline",
        "========================================",
        "",
        f"Created: {summary['created_at']}",
        f"Ticket: {args.ticket}",
        f"Git commit: {summary['git_commit']}",
        f"Dirty git state captured in: {output_dir / 'git_status.txt'}",
        "",
        "Command:",
        " ".join(shlex.quote(part) for part in summary["command"]),
        "",
        "Inputs:",
        f"- Completed-case manifest: {args.manifest.resolve()}",
        f"- Auto-context run root: {args.run_root.resolve()}",
        "- WSI image patches are read on demand from manifest WSI links using Stage 6 patch coordinates.",
        f"- WSI reader: {args.wsi_reader}; read_workers={args.read_workers}.",
        "- Labels are the Stage 7 postprocessed tissue mask over each selected crop patch grid.",
        "",
        "Split policy:",
        "- One independent probe is fit per WSI, budget, held-out selected crop, and sampling seed.",
        "- Train pool is all selected crops in the same WSI except the held-out crop.",
        "- Test set is every patch in the held-out crop.",
        f"- Budgets are per class: budget 100 means 100 foreground plus 100 background training patches.",
        "",
        "Backbone:",
        f"- Requested backend/model: {args.model_backend} / {args.model_name}",
        f"- Actual backend/model: {summary['feature_extractor']['backend']} / {summary['feature_extractor']['model_name']}",
        f"- Fallback used: {summary['feature_extractor']['fallback_used']}",
        "",
        "Outputs:",
        f"- Summary: {output_dir / 'summary.json'}",
        f"- Selected WSIs: {output_dir / 'manifests/selected_wsis.csv'}",
        f"- Patch manifest: {output_dir / 'manifests/selected_patch_manifest.csv'}",
        f"- Per-fold metrics: {output_dir / 'metrics/per_fold_metrics.csv'}",
        f"- Aggregate metrics: {output_dir / 'metrics/aggregate_metrics.csv'}",
        f"- Case-budget metrics: {output_dir / 'metrics/case_budget_metrics.csv'}",
        f"- Plot PNG/PDF: {output_dir / 'visuals/sample_efficiency.png'}",
        f"- Feature cache: {output_dir / 'features/'}",
    ]
    (output_dir / "reproduction.txt").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    command = sys.argv[:]
    budgets = parse_int_list(args.budgets)
    sample_seeds = parse_int_list(args.sample_seeds)
    max_budget = max(budgets)
    created_at = datetime.now(timezone.utc).isoformat()
    (args.output_dir / "git_status.txt").write_text(git_status_short())

    manifest_rows = read_completed_manifest(args.manifest, args.run_root)
    case_inventory = build_case_inventory(manifest_rows, min_bboxes=args.min_bboxes)
    selected_cases = select_cases(case_inventory, args.num_wsis, max_budget=max_budget, seed=args.case_seed)
    if len(selected_cases) < args.num_wsis:
        raise RuntimeError(f"Only selected {len(selected_cases)} eligible WSIs; requested {args.num_wsis}")

    write_csv(
        args.output_dir / "manifests/eligible_wsis.csv",
        [
            {
                "task": case.task,
                "case_id": case.case_id,
                "stain": case.stain,
                "wsi_path": case.wsi_path,
                "source_wsi_path": case.source_wsi_path,
                "run_dir": str(case.run_dir),
                "bbox_count": case.bbox_count,
                "fg_count": case.fg_count,
                "bg_count": case.bg_count,
                "max_leave_one_crop_budget": case.max_leave_one_crop_budget,
                "verifier_selected_box_ids": case.verifier_selected_box_ids,
            }
            for case in case_inventory
        ],
    )
    write_csv(
        args.output_dir / "manifests/selected_wsis.csv",
        [
            {
                "task": case.task,
                "case_id": case.case_id,
                "stain": case.stain,
                "wsi_path": case.wsi_path,
                "source_wsi_path": case.source_wsi_path,
                "run_dir": str(case.run_dir),
                "bbox_count": case.bbox_count,
                "fg_count": case.fg_count,
                "bg_count": case.bg_count,
                "max_leave_one_crop_budget": case.max_leave_one_crop_budget,
                "verifier_selected_box_ids": case.verifier_selected_box_ids,
            }
            for case in selected_cases
        ],
    )

    extractor = FeatureExtractor(args)
    all_patch_rows: list[dict[str, Any]] = []
    all_metric_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    case_summaries: list[dict[str, Any]] = []

    for idx, case in enumerate(selected_cases, start=1):
        started = time.monotonic()
        records = load_case_patches(case)
        if args.max_patches_per_case is not None and len(records) > args.max_patches_per_case:
            records = records[: args.max_patches_per_case]
        all_patch_rows.extend(patch_dict(record) for record in records)
        print(
            f"[{idx}/{len(selected_cases)}] {case.case_id} {case.stain}: "
            f"{len(records)} patches across {case.bbox_count} bboxes",
            flush=True,
        )
        features, labels, bbox_indices, record_indices, case_failures = extract_case_features(
            case,
            records,
            extractor,
            args.output_dir,
            resume=args.resume,
            wsi_reader=args.wsi_reader,
            read_workers=args.read_workers,
        )
        failures.extend({"case_id": case.case_id, **failure} for failure in case_failures)
        metric_rows = train_and_eval(case, features, labels, bbox_indices, budgets, sample_seeds)
        all_metric_rows.extend(metric_rows)
        case_summaries.append(
            {
                "case_id": case.case_id,
                "task": case.task,
                "stain": case.stain,
                "patches_loaded": len(records),
                "features_extracted": int(features.shape[0]),
                "feature_dim": int(features.shape[1]),
                "fg_features": int(labels.sum()),
                "bg_features": int(len(labels) - labels.sum()),
                "bbox_count": len(set(bbox_indices.tolist())),
                "feature_failures": len(case_failures),
                "metric_rows": len(metric_rows),
                "metric_rows_skipped": sum(1 for row in metric_rows if row.get("skipped")),
                "seconds": time.monotonic() - started,
            }
        )

    aggregate_rows = aggregate_metrics(all_metric_rows)
    case_budget_rows = aggregate_case_budget(all_metric_rows)
    write_csv(args.output_dir / "manifests/selected_patch_manifest.csv", all_patch_rows)
    write_csv(args.output_dir / "metrics/per_fold_metrics.csv", all_metric_rows)
    write_csv(args.output_dir / "metrics/aggregate_metrics.csv", aggregate_rows)
    write_csv(args.output_dir / "metrics/case_budget_metrics.csv", case_budget_rows)
    write_json(args.output_dir / "failures.json", failures)
    make_plot(aggregate_rows, args.output_dir / "visuals/sample_efficiency.png")

    summary = {
        "created_at": created_at,
        "ticket": args.ticket,
        "git_commit": git_commit(),
        "command": command,
        "package_versions": package_versions(),
        "input": {
            "run_root": str(args.run_root.resolve()),
            "manifest": str(args.manifest.resolve()),
            "manifest_rows_resolved": len(manifest_rows),
            "eligible_case_count": len(case_inventory),
        },
        "selection": {
            "num_wsis_requested": args.num_wsis,
            "num_wsis_selected": len(selected_cases),
            "case_seed": args.case_seed,
            "min_bboxes": args.min_bboxes,
            "budgets_per_class": budgets,
            "sample_seeds": sample_seeds,
        },
        "label_policy": {
            "positive_class": "foreground",
            "negative_class": "background",
            "source": "stage7/tissue_mask_post.npy aligned to stage6/patches.csv row/col",
        },
        "patch_reader": {
            "wsi_reader": args.wsi_reader,
            "read_workers": args.read_workers,
        },
        "split_policy": {
            "unit": "selected bbox crop within WSI",
            "train": "all selected crops except held-out crop",
            "test": "held-out selected crop",
            "budget_unit": "patches per class",
        },
        "feature_extractor": extractor.meta,
        "case_summaries": case_summaries,
        "metrics": {
            "per_fold_rows": len(all_metric_rows),
            "per_fold_rows_skipped": sum(1 for row in all_metric_rows if row.get("skipped")),
            "aggregate": aggregate_rows,
        },
        "failures_count": len(failures),
        "outputs": {
            "selected_wsis_csv": str((args.output_dir / "manifests/selected_wsis.csv").resolve()),
            "patch_manifest_csv": str((args.output_dir / "manifests/selected_patch_manifest.csv").resolve()),
            "per_fold_metrics_csv": str((args.output_dir / "metrics/per_fold_metrics.csv").resolve()),
            "aggregate_metrics_csv": str((args.output_dir / "metrics/aggregate_metrics.csv").resolve()),
            "case_budget_metrics_csv": str((args.output_dir / "metrics/case_budget_metrics.csv").resolve()),
            "sample_efficiency_png": str((args.output_dir / "visuals/sample_efficiency.png").resolve()),
            "sample_efficiency_pdf": str((args.output_dir / "visuals/sample_efficiency.pdf").resolve()),
            "reproduction_txt": str((args.output_dir / "reproduction.txt").resolve()),
        },
    }
    write_json(args.output_dir / "summary.json", summary)
    write_reproduction(args.output_dir, args, summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
