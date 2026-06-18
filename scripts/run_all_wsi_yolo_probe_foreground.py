#!/usr/bin/env python3
"""Run YOLO ROI proposal plus DINOv3 linear-probe foreground scoring for all WSIs."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import shlex
import subprocess
import sys
import threading
import time
import traceback
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zipfile import ZipFile
import xml.etree.ElementTree as ET

import joblib
import numpy as np
import openslide
from PIL import Image, ImageDraw
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from train_per_wsi_dinov3_fg_bg_probe import FeatureExtractor, WsiPatchReader, package_versions  # noqa: E402


DEFAULT_RUN_DIR = REPO_ROOT / "runs/all_wsi_yolo_probe_foreground_v1"
DEFAULT_ALL_SVS = Path("/vol/biomedic3/histopatho/win_share/all_svs_fpaths.csv")
DEFAULT_WORKBOOK = Path("/vol/biomedic3/histopatho/win_share/anon_master_combined_v5_with_reports_report_fixed (2).xlsx")
DEFAULT_ALEX_WSI_ROOT = Path("/anvme/workspace/b180dc29-histopatho")
DEFAULT_SOURCE_WSI_ROOT = Path("/vol/biomedic3/histopatho/win_share")
DEFAULT_YOLO_WEIGHTS = (
    REPO_ROOT
    / "runs/detector_distillation/yolo_scale500_per248_v1"
    / "yolo11n_img1024_e60_stainjitter/ultralytics/train/weights/best.pt"
)
DEFAULT_STRESS_RUN_DIR = REPO_ROOT / "runs/stress32_gt_overlay_sample_efficiency_probe_v1"
DEFAULT_SV40_MODEL = REPO_ROOT / "runs/per290_sv40_cleaned_mask_probe_v1/models/logreg_all_samples.joblib"
DEFAULT_DINOV3_SMALL = "facebook/dinov3-vits16-pretrain-lvd1689m"
DEFAULT_DINOV2_SMALL = "vit_small_patch14_dinov2"
XLSX_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


@dataclass(frozen=True)
class Detection:
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


class ThreadLocalPatchReader:
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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        seen: set[str] = set()
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def git_status_short() -> str:
    try:
        return subprocess.check_output(["git", "status", "--short"], cwd=REPO_ROOT, text=True)
    except Exception as exc:
        return f"git status failed: {type(exc).__name__}: {exc}\n"


def col_to_idx(ref: str) -> int:
    letters = "".join(ch for ch in ref if ch.isalpha())
    n = 0
    for ch in letters:
        n = n * 26 + ord(ch.upper()) - 64
    return n


def xlsx_cell_value(cell: ET.Element, shared_strings: list[str]) -> str:
    cell_type = cell.attrib.get("t")
    value = cell.find("m:v", XLSX_NS)
    if value is None:
        inline = cell.find("m:is/m:t", XLSX_NS)
        return inline.text if inline is not None and inline.text is not None else ""
    text = value.text or ""
    if cell_type == "s":
        return shared_strings[int(text)]
    return text


def load_workbook_metadata(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with ZipFile(path) as zf:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for item in root.findall("m:si", XLSX_NS):
                parts = [
                    node.text or ""
                    for node in item.iter("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t")
                ]
                shared_strings.append("".join(parts))
        sheet = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))
        header: dict[int, str] = {}
        current: dict[str, str] = {}
        rows: dict[str, dict[str, str]] = {}
        for row in sheet.findall(".//m:sheetData/m:row", XLSX_NS):
            row_number = int(row.attrib["r"])
            cells = {
                col_to_idx(cell.attrib["r"]): xlsx_cell_value(cell, shared_strings).strip()
                for cell in row.findall("m:c", XLSX_NS)
            }
            if row_number == 1:
                header = cells
                continue
            record = {header[idx]: value for idx, value in cells.items() if idx in header}
            for key in ("Anon_Patient_ID", "Anon_Path_ID"):
                if record.get(key):
                    current[key] = record[key]
                elif current.get(key):
                    record[key] = current[key]
            slide_name = Path(record.get("location_college") or record.get("Anon_Slide_ID") or "").name
            if slide_name:
                rows[slide_name] = record
        return rows


def normalize_stain(stain: str) -> str:
    return " ".join(stain.strip().upper().replace("_", " ").replace("-", " ").split())


def route_for_stain(stain_raw: str) -> tuple[str, str, str]:
    stain = normalize_stain(stain_raw)
    if not stain:
        return "stress32_N500_logreg", "UNKNOWN", "missing_stain_default_stress"
    histology_exact = {"H&E", "?H&E?", "HE", "HE5", "PAS", "JONES", "EVG", "CR", "CONGO RED", "TOL BLUE", "TOLBLUE"}
    histology_terms = ("H&E", "JONES", "PAS", "EVG", "CONGO", "TOL BLUE", "TOLBLUE")
    if stain in histology_exact or any(term in stain for term in histology_terms):
        return "stress32_N500_logreg", stain, "histochemical_or_requested_stress_probe"
    return "sv40_cleaned_logreg", stain, "immuno_or_other_sv40_probe"


def relative_to_source(path: Path, source_root: Path) -> str:
    try:
        return str(path.relative_to(source_root))
    except ValueError:
        marker = "win_share/"
        text = str(path)
        if marker in text:
            return text.split(marker, 1)[1]
        return path.name


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir
    manifest_path = args.manifest_csv or output_dir / "manifests/all_wsi_probe_manifest.csv"
    metadata = load_workbook_metadata(args.workbook)
    source_paths = [line.strip() for line in args.all_svs.read_text().splitlines() if line.strip()]
    rows: list[dict[str, Any]] = []
    for index, raw_path in enumerate(source_paths):
        source_path = Path(raw_path)
        slide_name = source_path.name
        meta = metadata.get(slide_name, {})
        stain_raw = str(meta.get("Stain_from_wsi", ""))
        route_model, stain_norm, route_reason = route_for_stain(stain_raw)
        relative_path = relative_to_source(source_path, args.source_wsi_root)
        rows.append(
            {
                "index": index,
                "case_id": source_path.stem,
                "slide_name": slide_name,
                "source_wsi_path": str(source_path),
                "relative_path": relative_path,
                "alex_wsi_path": str(args.alex_wsi_root / relative_path),
                "metadata_match": bool(meta),
                "anon_patient_id": meta.get("Anon_Patient_ID", ""),
                "anon_path_id": meta.get("Anon_Path_ID", ""),
                "stain_raw": stain_raw,
                "stain_norm": stain_norm,
                "route_model": route_model,
                "route_reason": route_reason,
            }
        )
    write_csv(manifest_path, rows)
    counts = {
        "created_at": utc_now(),
        "ticket": args.ticket,
        "manifest_csv": str(manifest_path.resolve()),
        "row_count": len(rows),
        "metadata_matches": sum(1 for row in rows if row["metadata_match"]),
        "missing_stain": sum(1 for row in rows if row["route_reason"] == "missing_stain_default_stress"),
        "route_counts": dict(Counter(str(row["route_model"]) for row in rows)),
        "stain_counts": dict(Counter(str(row["stain_raw"] or "UNKNOWN") for row in rows).most_common()),
        "source_all_svs": str(args.all_svs),
        "workbook": str(args.workbook),
        "alex_wsi_root": str(args.alex_wsi_root),
    }
    write_json(output_dir / "manifests/manifest_summary.json", counts)
    write_reproduction(args, output_dir, extra=counts)
    return counts


def fit_linear_probe(features: np.ndarray, labels: np.ndarray, seed: int) -> Any:
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, solver="lbfgs", class_weight="balanced", random_state=seed),
    )
    model.fit(features, labels)
    return model


def prepare_stress_probe(args: argparse.Namespace) -> dict[str, Any]:
    model_dir = args.output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"stress32_N{args.stress_sample_size:03d}_logreg.joblib"
    summary_path = model_dir / f"stress32_N{args.stress_sample_size:03d}_summary.json"
    if model_path.exists() and summary_path.exists() and not args.force:
        return json.loads(summary_path.read_text())

    manifest_path = args.stress_run_dir / "sampled_manifests" / f"N{args.stress_sample_size:03d}.csv"
    rows = [row for row in read_csv(manifest_path) if row.get("split") == "train"]
    by_case: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_case[row["case_id"]].append(row)

    feature_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    backends: set[str] = set()
    models: set[str] = set()
    bucket_counts: Counter[str] = Counter()
    per_case_counts: dict[str, int] = {}
    missing: list[str] = []
    for case_id, case_rows in sorted(by_case.items()):
        feature_path = args.stress_run_dir / "features" / f"{case_id}_features.npz"
        with np.load(feature_path, allow_pickle=False) as data:
            record_ids = [str(x) for x in data["record_id"]]
            index_by_id = {record_id: idx for idx, record_id in enumerate(record_ids)}
            idxs: list[int] = []
            for row in case_rows:
                idx = index_by_id.get(row["record_id"])
                if idx is None:
                    missing.append(row["record_id"])
                else:
                    idxs.append(idx)
                    bucket_counts.update([row["bucket"]])
            if idxs:
                arr_idx = np.asarray(idxs, dtype="int64")
                feature_parts.append(data["features"][arr_idx].astype("float32"))
                label_parts.append(data["label_fg"][arr_idx].astype("int64"))
                per_case_counts[case_id] = len(arr_idx)
            backends.add(str(data["model_backend"]))
            models.add(str(data["model_name"]))
    if missing:
        raise RuntimeError(f"{len(missing)} sampled records are missing cached stress features; first={missing[0]}")
    if len(backends) != 1 or len(models) != 1:
        raise RuntimeError(f"Mixed stress feature models: backends={sorted(backends)} models={sorted(models)}")
    x = np.concatenate(feature_parts, axis=0)
    y = np.concatenate(label_parts, axis=0)
    seed = int(args.stress_sample_seed) + int(args.stress_sample_size)
    model = fit_linear_probe(x, y, seed)
    joblib.dump(model, model_path)
    summary = {
        "created_at": utc_now(),
        "model_path": str(model_path.resolve()),
        "source": "stress32_gt_overlay",
        "source_run_dir": str(args.stress_run_dir.resolve()),
        "sample_manifest": str(manifest_path.resolve()),
        "sample_size_per_wsi": int(args.stress_sample_size),
        "sample_seed": int(args.stress_sample_seed),
        "fit_seed": seed,
        "train_count": int(len(y)),
        "train_fg": int((y == 1).sum()),
        "train_bg": int((y == 0).sum()),
        "case_count": len(per_case_counts),
        "min_patches_per_case": min(per_case_counts.values()),
        "max_patches_per_case": max(per_case_counts.values()),
        "bucket_counts": dict(bucket_counts),
        "feature_backend": next(iter(backends)),
        "feature_model": next(iter(models)),
    }
    write_json(summary_path, summary)
    return summary


def prepare_probes(args: argparse.Namespace) -> dict[str, Any]:
    stress_summary = prepare_stress_probe(args)
    sv40_model = Path(args.sv40_model)
    if not sv40_model.exists():
        raise FileNotFoundError(f"Missing SV40 model: {sv40_model}")
    sv40_loaded = joblib.load(sv40_model)
    if not hasattr(sv40_loaded, "predict_proba"):
        raise RuntimeError(f"SV40 model does not expose predict_proba: {sv40_model}")
    summary = {
        "created_at": utc_now(),
        "ticket": args.ticket,
        "stress_probe": stress_summary,
        "sv40_probe": {
            "model_path": str(sv40_model.resolve()),
            "source": "PER-290 cleaned SV40 Stage7 masks",
        },
    }
    write_json(args.output_dir / "models/probe_summary.json", summary)
    write_reproduction(args, args.output_dir, extra=summary)
    return summary


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


def run_yolo(model: Any, thumbnail_path: Path, slide_size: tuple[int, int], args: argparse.Namespace) -> list[Detection]:
    result = model.predict(
        source=str(thumbnail_path),
        imgsz=int(args.imgsz),
        conf=float(args.conf),
        iou=float(args.iou),
        max_det=int(args.max_det),
        device=args.yolo_device or None,
        verbose=False,
    )[0]
    image = Image.open(thumbnail_path)
    thumb_w, thumb_h = image.size
    slide_w, slide_h = slide_size
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.xyxy is None:
        return []
    xyxy = boxes.xyxy.detach().cpu().numpy()
    confs = boxes.conf.detach().cpu().numpy()
    classes = boxes.cls.detach().cpu().numpy() if boxes.cls is not None else np.zeros(len(xyxy))
    sx = slide_w / max(1, thumb_w)
    sy = slide_h / max(1, thumb_h)
    order = np.argsort(-confs)
    detections: list[Detection] = []
    for det_idx, source_idx in enumerate(order.tolist(), start=1):
        x0t, y0t, x1t, y1t = [float(v) for v in xyxy[source_idx]]
        x0t = max(0.0, min(float(thumb_w), x0t))
        x1t = max(0.0, min(float(thumb_w), x1t))
        y0t = max(0.0, min(float(thumb_h), y0t))
        y1t = max(0.0, min(float(thumb_h), y1t))
        x0 = max(0, min(slide_w - 1, int(math.floor(x0t * sx))))
        x1 = max(0, min(slide_w, int(math.ceil(x1t * sx))))
        y0 = max(0, min(slide_h - 1, int(math.floor(y0t * sy))))
        y1 = max(0, min(slide_h, int(math.ceil(y1t * sy))))
        if x1 <= x0 or y1 <= y0:
            continue
        detections.append(
            Detection(
                detection_id=det_idx,
                conf=float(confs[source_idx]),
                cls=int(classes[source_idx]),
                x0_thumb=x0t,
                y0_thumb=y0t,
                x1_thumb=x1t,
                y1_thumb=y1t,
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
            )
        )
    return detections


def build_patch_records(case_id: str, slide_size: tuple[int, int], detections: list[Detection], patch_size: int) -> list[PatchRecord]:
    slide_w, slide_h = slide_size
    cells: dict[tuple[int, int], set[int]] = defaultdict(set)
    for det in detections:
        row0 = max(0, int(math.floor(det.y0 / patch_size)))
        row1 = min(int(math.ceil(slide_h / patch_size)), int(math.ceil(det.y1 / patch_size)))
        col0 = max(0, int(math.floor(det.x0 / patch_size)))
        col1 = min(int(math.ceil(slide_w / patch_size)), int(math.ceil(det.x1 / patch_size)))
        for row in range(row0, row1):
            y = row * patch_size
            h = max(1, min(patch_size, slide_h - y))
            for col in range(col0, col1):
                x = col * patch_size
                w = max(1, min(patch_size, slide_w - x))
                if x < det.x1 and x + w > det.x0 and y < det.y1 and y + h > det.y0:
                    cells[(row, col)].add(det.detection_id)
    records: list[PatchRecord] = []
    for (row, col), det_ids in sorted(cells.items()):
        x = col * patch_size
        y = row * patch_size
        records.append(
            PatchRecord(
                case_id=case_id,
                detection_id=min(det_ids),
                detection_ids=tuple(sorted(det_ids)),
                row=row,
                col=col,
                x=x,
                y=y,
                width=max(1, min(patch_size, slide_w - x)),
                height=max(1, min(patch_size, slide_h - y)),
            )
        )
    return records


def extract_features(args: argparse.Namespace, wsi_path: Path, records: list[PatchRecord], extractor: FeatureExtractor) -> np.ndarray:
    if not records:
        return np.zeros((0, 0), dtype="float32")
    feature_parts: list[np.ndarray] = []
    reader = ThreadLocalPatchReader(wsi_path, args.wsi_reader, args.read_workers)
    try:
        if int(args.read_workers) > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=int(args.read_workers)) as pool:
                for start in range(0, len(records), int(args.batch_size)):
                    batch_records = records[start : start + int(args.batch_size)]
                    future_to_idx = {
                        pool.submit(reader.read_patch, record): idx
                        for idx, record in enumerate(batch_records)
                    }
                    images_by_idx: dict[int, Image.Image] = {}
                    for future in concurrent.futures.as_completed(future_to_idx):
                        images_by_idx[future_to_idx[future]] = future.result()
                    images = [images_by_idx[idx] for idx in range(len(batch_records))]
                    feature_parts.append(extractor.extract_batch(images))
        else:
            for start in range(0, len(records), int(args.batch_size)):
                batch_records = records[start : start + int(args.batch_size)]
                images = [reader.read_patch(record) for record in batch_records]
                feature_parts.append(extractor.extract_batch(images))
    finally:
        reader.close()
    return np.concatenate(feature_parts, axis=0).astype("float32")


def foreground_components(records: list[PatchRecord], prob: np.ndarray, pred: np.ndarray) -> list[dict[str, Any]]:
    fg_indices = [idx for idx, is_fg in enumerate(pred.tolist()) if int(is_fg) == 1]
    by_cell = {(records[idx].row, records[idx].col): idx for idx in fg_indices}
    seen: set[tuple[int, int]] = set()
    regions: list[dict[str, Any]] = []
    for cell in sorted(by_cell):
        if cell in seen:
            continue
        queue: deque[tuple[int, int]] = deque([cell])
        seen.add(cell)
        idxs: list[int] = []
        while queue:
            cur = queue.popleft()
            idxs.append(by_cell[cur])
            row, col = cur
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nxt = (row + dr, col + dc)
                    if nxt in by_cell and nxt not in seen:
                        seen.add(nxt)
                        queue.append(nxt)
        xs0 = [records[idx].x for idx in idxs]
        ys0 = [records[idx].y for idx in idxs]
        xs1 = [records[idx].x + records[idx].width for idx in idxs]
        ys1 = [records[idx].y + records[idx].height for idx in idxs]
        det_ids = sorted({det for idx in idxs for det in records[idx].detection_ids})
        probs = prob[np.asarray(idxs, dtype="int64")]
        regions.append(
            {
                "region_id": len(regions) + 1,
                "patch_count": len(idxs),
                "x0_level0": min(xs0),
                "y0_level0": min(ys0),
                "x1_level0": max(xs1),
                "y1_level0": max(ys1),
                "width_level0": max(xs1) - min(xs0),
                "height_level0": max(ys1) - min(ys0),
                "mean_prob_fg": float(probs.mean()) if len(probs) else 0.0,
                "max_prob_fg": float(probs.max()) if len(probs) else 0.0,
                "detection_ids": ";".join(str(x) for x in det_ids),
            }
        )
    return regions


def draw_overlay(
    thumbnail_path: Path,
    slide_size: tuple[int, int],
    detections: list[Detection],
    records: list[PatchRecord],
    pred: np.ndarray,
    out_path: Path,
) -> None:
    image = Image.open(thumbnail_path).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    slide_w, slide_h = slide_size
    sx = image.width / max(1, slide_w)
    sy = image.height / max(1, slide_h)
    for det in detections:
        rect = (det.x0 * sx, det.y0 * sy, det.x1 * sx, det.y1 * sy)
        draw.rectangle(rect, outline=(255, 190, 0, 230), width=3)
    for record, is_fg in zip(records, pred.tolist()):
        if int(is_fg) != 1:
            continue
        rect = (record.x * sx, record.y * sy, (record.x + record.width) * sx, (record.y + record.height) * sy)
        draw.rectangle(rect, fill=(46, 204, 113, 85), outline=(39, 174, 96, 170), width=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def model_paths(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "stress32_N500_logreg": args.output_dir / f"models/stress32_N{args.stress_sample_size:03d}_logreg.joblib",
        "sv40_cleaned_logreg": Path(args.sv40_model),
    }


def load_models(args: argparse.Namespace) -> dict[str, Any]:
    paths = model_paths(args)
    return {name: joblib.load(path) for name, path in paths.items()}


def process_case(
    args: argparse.Namespace,
    row: dict[str, str],
    *,
    yolo_model: Any,
    models: dict[str, Any],
    extractor: FeatureExtractor,
) -> dict[str, Any]:
    case_id = row["case_id"]
    case_dir = args.output_dir / "slides" / case_id
    complete_path = case_dir / "complete.json"
    if complete_path.exists() and args.resume and not args.overwrite:
        payload = json.loads(complete_path.read_text())
        return {"case_id": case_id, "status": "skipped_complete", **payload}

    started = time.perf_counter()
    case_dir.mkdir(parents=True, exist_ok=True)
    try:
        wsi_path = Path(row.get("alex_wsi_path") or row.get("source_wsi_path") or "")
        if not wsi_path.exists():
            source = Path(row.get("source_wsi_path", ""))
            if source.exists():
                wsi_path = source
            else:
                raise FileNotFoundError(f"Missing WSI path: alex={wsi_path} source={source}")
        slide = openslide.OpenSlide(str(wsi_path))
        try:
            slide_w, slide_h = [int(v) for v in slide.dimensions]
            thumbnail = slide.get_thumbnail((int(args.imgsz), int(args.imgsz))).convert("RGB")
            # OpenSlide can attach very large metadata chunks to PIL images; OpenCV
            # inside Ultralytics may reject those when saved as PNG. Re-materialize
            # pixels only and use JPEG for detector input.
            thumbnail = Image.fromarray(np.asarray(thumbnail), mode="RGB")
        finally:
            slide.close()
        thumbnail_path = case_dir / "thumbnail.jpg"
        thumbnail.save(thumbnail_path, quality=92)

        raw_detections = run_yolo(yolo_model, thumbnail_path, (slide_w, slide_h), args)
        filtered_detections, suppressed = suppress_contained_detections(raw_detections, float(args.containment_threshold))
        records = build_patch_records(case_id, (slide_w, slide_h), filtered_detections, int(args.patch_size))
        if args.max_patches_per_wsi is not None and len(records) > int(args.max_patches_per_wsi):
            raise RuntimeError(f"{case_id} has {len(records)} patches, above max {args.max_patches_per_wsi}")

        route_model = row["route_model"]
        model = models[route_model]
        features = extract_features(args, wsi_path, records, extractor) if records else np.zeros((0, 0), dtype="float32")
        prob = model.predict_proba(features)[:, 1] if len(records) else np.zeros((0,), dtype="float32")
        pred = (prob >= float(args.probe_threshold)).astype("int64")
        regions = foreground_components(records, prob, pred)

        patch_rows = []
        for record, fg_prob, is_fg in zip(records, prob.tolist(), pred.tolist()):
            patch_rows.append(
                {
                    "record_id": record.record_id,
                    "case_id": case_id,
                    "route_model": route_model,
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
                }
            )
        write_csv(case_dir / "detections_raw.csv", detection_rows(raw_detections))
        write_csv(case_dir / "detections_filtered.csv", detection_rows(filtered_detections))
        write_csv(case_dir / "detections_suppressed_contained.csv", suppressed)
        write_csv(case_dir / "patch_predictions.csv", patch_rows)
        write_csv(case_dir / "foreground_regions.csv", regions)
        if args.write_overlays:
            draw_overlay(thumbnail_path, (slide_w, slide_h), filtered_detections, records, pred, case_dir / "foreground_overlay_thumb.png")

        summary = {
            "created_at": utc_now(),
            "status": "complete",
            "case_id": case_id,
            "slide_name": row.get("slide_name", ""),
            "stain_raw": row.get("stain_raw", ""),
            "stain_norm": row.get("stain_norm", ""),
            "route_model": route_model,
            "route_reason": row.get("route_reason", ""),
            "wsi_path": str(wsi_path),
            "slide_width": slide_w,
            "slide_height": slide_h,
            "thumbnail_path": str(thumbnail_path),
            "raw_detection_count": len(raw_detections),
            "filtered_detection_count": len(filtered_detections),
            "suppressed_detection_count": len(suppressed),
            "patch_count": len(records),
            "pred_fg_patch_count": int(pred.sum()),
            "foreground_region_count": len(regions),
            "mean_prob_fg": float(prob.mean()) if len(prob) else 0.0,
            "max_prob_fg": float(prob.max()) if len(prob) else 0.0,
            "elapsed_seconds": float(time.perf_counter() - started),
        }
        write_json(case_dir / "summary.json", summary)
        write_json(complete_path, summary)
        return summary
    except Exception as exc:
        payload = {
            "created_at": utc_now(),
            "status": "error",
            "case_id": case_id,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "elapsed_seconds": float(time.perf_counter() - started),
        }
        write_json(case_dir / "error.json", payload)
        if args.fail_fast:
            raise
        return payload


def select_shard_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    rows = read_csv(args.manifest_csv)
    if args.case_ids:
        wanted = {part.strip() for part in args.case_ids.split(",") if part.strip()}
        rows = [row for row in rows if row["case_id"] in wanted or row.get("slide_name") in wanted]
    if args.shard_index is not None:
        rows = [row for row in rows if int(row["index"]) % int(args.num_shards) == int(args.shard_index)]
    if args.start_index is not None:
        rows = [row for row in rows if int(row["index"]) >= int(args.start_index)]
    if args.end_index is not None:
        rows = [row for row in rows if int(row["index"]) < int(args.end_index)]
    if args.case_limit is not None:
        rows = rows[: int(args.case_limit)]
    return rows


def run_shard(args: argparse.Namespace) -> dict[str, Any]:
    from ultralytics import YOLO

    rows = select_shard_rows(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError("No manifest rows selected for this shard")
    models = load_models(args)
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
    yolo_model = YOLO(str(args.yolo_weights))

    status_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for offset, row in enumerate(rows, start=1):
        print(f"[case {offset}/{len(rows)}] index={row['index']} case={row['case_id']} route={row['route_model']}", flush=True)
        result = process_case(args, row, yolo_model=yolo_model, models=models, extractor=extractor)
        status_rows.append(result)
        print(
            f"[case {offset}/{len(rows)}] {row['case_id']} status={result.get('status')} "
            f"dets={result.get('filtered_detection_count')} patches={result.get('patch_count')} "
            f"regions={result.get('foreground_region_count')} elapsed={float(result.get('elapsed_seconds', 0.0)):.1f}s",
            flush=True,
        )
    shard_name = "manual"
    if args.shard_index is not None:
        shard_name = f"shard_{int(args.shard_index):04d}_of_{int(args.num_shards):04d}"
    shard_dir = args.output_dir / "shards"
    write_csv(shard_dir / f"{shard_name}_status.csv", status_rows)
    summary = {
        "created_at": utc_now(),
        "status": "complete",
        "shard_name": shard_name,
        "selected_rows": len(rows),
        "complete": sum(1 for row in status_rows if row.get("status") == "complete"),
        "skipped_complete": sum(1 for row in status_rows if row.get("status") == "skipped_complete"),
        "errors": sum(1 for row in status_rows if row.get("status") == "error"),
        "elapsed_seconds": float(time.perf_counter() - started),
        "extractor": extractor.meta,
        "package_versions": package_versions(),
    }
    write_json(shard_dir / f"{shard_name}_summary.json", summary)
    write_reproduction(args, args.output_dir, extra=summary)
    return summary


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    manifest_rows = read_csv(args.manifest_csv)
    status_rows: list[dict[str, Any]] = []
    for row in manifest_rows:
        case_id = row["case_id"]
        case_dir = args.output_dir / "slides" / case_id
        complete = case_dir / "complete.json"
        error = case_dir / "error.json"
        payload: dict[str, Any]
        if complete.exists():
            payload = json.loads(complete.read_text())
        elif error.exists():
            payload = json.loads(error.read_text())
        else:
            payload = {"case_id": case_id, "status": "pending"}
        status_rows.append({**row, **payload})
    write_csv(args.output_dir / "status/all_wsi_status.csv", status_rows)
    summary = {
        "created_at": utc_now(),
        "manifest_rows": len(manifest_rows),
        "status_counts": dict(Counter(str(row.get("status", "unknown")) for row in status_rows)),
        "route_counts": dict(Counter(str(row.get("route_model", "unknown")) for row in status_rows)),
        "completed_patch_count": int(sum(int(float(row.get("patch_count", 0) or 0)) for row in status_rows if row.get("status") == "complete")),
        "completed_region_count": int(sum(int(float(row.get("foreground_region_count", 0) or 0)) for row in status_rows if row.get("status") == "complete")),
    }
    write_json(args.output_dir / "status/summary.json", summary)
    return summary


def write_reproduction(args: argparse.Namespace, output_dir: Path, *, extra: dict[str, Any] | None = None) -> None:
    lines = [
        "All-WSI YOLO plus DINOv3 linear-probe foreground extraction",
        "",
        f"Updated: {utc_now()}",
        f"Ticket: {args.ticket}",
        f"Repository: {REPO_ROOT}",
        f"Git commit: {git_commit()}",
        "Git status --short at creation/update:",
        git_status_short() or "clean\n",
        "",
        "Current command:",
        shlex.join([sys.executable, *sys.argv]),
        "",
        "Core settings:",
        f"- YOLO weights: {args.yolo_weights}",
        f"- YOLO conf: {args.conf}",
        f"- YOLO iou: {args.iou}",
        f"- containment threshold: {args.containment_threshold}",
        f"- patch size: {args.patch_size}",
        f"- probe threshold: {args.probe_threshold}",
        f"- WSI reader: {args.wsi_reader}",
        f"- read workers: {args.read_workers}",
        f"- DINO model: {args.model_backend} / {args.model_name}",
        f"- batch size: {args.batch_size}",
        "",
        "Probe routing:",
        "- H&E, PAS, JONES, EVG, Congo Red/CR, and Tol Blue use stress32_N500_logreg.",
        "- Immuno/other known stains use the PER-290 SV40 cleaned-mask probe.",
        "- Missing stain metadata defaults to stress32_N500_logreg and is flagged in the manifest.",
        "",
        "Expected outputs:",
        f"- manifest: {args.manifest_csv}",
        f"- slide outputs: {output_dir / 'slides'}",
        f"- shard summaries: {output_dir / 'shards'}",
        f"- aggregate status: {output_dir / 'status'}",
        "",
        "Credentials:",
        "- HF_TOKEN or HUGGING_FACE_HUB_TOKEN may be required for gated DINOv3 access; token values are intentionally not recorded.",
    ]
    if extra is not None:
        lines.extend(["", "Last summary:", json.dumps(extra, indent=2, sort_keys=True)])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "reproduction.txt").write_text("\n".join(lines) + "\n")


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--manifest-csv", type=Path, default=None)
    parser.add_argument("--ticket", default="PER-290")
    parser.add_argument("--yolo-weights", type=Path, default=DEFAULT_YOLO_WEIGHTS)
    parser.add_argument("--stress-run-dir", type=Path, default=DEFAULT_STRESS_RUN_DIR)
    parser.add_argument("--stress-sample-size", type=int, default=500)
    parser.add_argument("--stress-sample-seed", type=int, default=270)
    parser.add_argument("--sv40-model", type=Path, default=DEFAULT_SV40_MODEL)
    parser.add_argument("--conf", type=float, default=0.01)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--yolo-device", default="")
    parser.add_argument("--containment-threshold", type=float, default=0.90)
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--probe-threshold", type=float, default=0.50)
    parser.add_argument("--model-backend", choices=["transformers", "timm"], default="transformers")
    parser.add_argument("--model-name", default=DEFAULT_DINOV3_SMALL)
    parser.add_argument("--fallback-model-name", default=DEFAULT_DINOV2_SMALL)
    parser.add_argument("--allow-timm-fallback", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--input-size", type=int, default=None)
    parser.add_argument("--wsi-reader", choices=["openslide", "cucim"], default="openslide")
    parser.add_argument("--read-workers", type=int, default=4)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_manifest = sub.add_parser("build-manifest")
    add_common(p_manifest)
    p_manifest.add_argument("--all-svs", type=Path, default=DEFAULT_ALL_SVS)
    p_manifest.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK)
    p_manifest.add_argument("--source-wsi-root", type=Path, default=DEFAULT_SOURCE_WSI_ROOT)
    p_manifest.add_argument("--alex-wsi-root", type=Path, default=DEFAULT_ALEX_WSI_ROOT)

    p_prepare = sub.add_parser("prepare-probes")
    add_common(p_prepare)
    p_prepare.add_argument("--force", action="store_true")

    p_run = sub.add_parser("run-shard")
    add_common(p_run)
    p_run.add_argument("--shard-index", type=int, default=None)
    p_run.add_argument("--num-shards", type=int, default=1)
    p_run.add_argument("--start-index", type=int, default=None)
    p_run.add_argument("--end-index", type=int, default=None)
    p_run.add_argument("--case-ids", default="")
    p_run.add_argument("--case-limit", type=int, default=None)
    p_run.add_argument("--max-patches-per-wsi", type=int, default=None)
    p_run.add_argument("--resume", action="store_true", default=True)
    p_run.add_argument("--no-resume", dest="resume", action="store_false")
    p_run.add_argument("--overwrite", action="store_true")
    p_run.add_argument("--write-overlays", action="store_true")
    p_run.add_argument("--fail-fast", action="store_true")

    p_sum = sub.add_parser("summarize")
    add_common(p_sum)
    return parser


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.manifest_csv is None:
        args.manifest_csv = args.output_dir / "manifests/all_wsi_probe_manifest.csv"
    return args


def main() -> None:
    args = normalize_args(create_parser().parse_args())
    if args.command == "build-manifest":
        result = build_manifest(args)
    elif args.command == "prepare-probes":
        result = prepare_probes(args)
    elif args.command == "run-shard":
        result = run_shard(args)
    elif args.command == "summarize":
        result = summarize(args)
    else:
        raise ValueError(args.command)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
