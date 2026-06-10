#!/usr/bin/env python3
"""Train pooled DINOv3 FG/BG probes and score unselected scale-500 detector crops."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import subprocess
import sys
import textwrap
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
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from build_per_wsi_probe_unselected_transfer_demo import (  # noqa: E402
    CandidateInfo,
    PatchRecord,
    build_patch_records,
    candidate_thumbnail_path,
    draw_prediction_grid_panel,
    extract_unselected_features,
    load_candidate_infos,
    parse_box_ids,
    read_csv,
    resize_to_fit,
    resolve_wsi_path,
    WsiPatchReader,
)
from train_per_wsi_dinov3_fg_bg_probe import FeatureExtractor, package_versions  # noqa: E402


DEFAULT_PROBE_RUN_DIR = REPO_ROOT / "runs/per_wsi_dinov3_fg_bg_probe_baseline_v1_dinov3small"
DEFAULT_SELECTOR_RUN_ROOT = REPO_ROOT / "runs/auto_context_scale500_selector_all500_v1"
DEFAULT_SELECTOR_MANIFEST = (
    DEFAULT_SELECTOR_RUN_ROOT / "manifests/completed_cases_500_20260604_openrouter_review_current.csv"
)
DEFAULT_DETECTOR_ROOT = REPO_ROOT / "runs/detector_pipeline_scale500_v1"
DEFAULT_OUTPUT_DIR = (
    DEFAULT_PROBE_RUN_DIR / "visuals/per_wsi_probe_pooled_scale500_transfer_v1"
)


@dataclass(frozen=True)
class CaseBundle:
    case_id: str
    task: int
    stain: str
    selector_row: dict[str, str]
    detector_case_dir: Path
    selected_ids: set[int]
    selected_features: np.ndarray
    selected_labels: np.ndarray
    feature_backend: str
    feature_model: str
    candidates: list[CandidateInfo]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-run-dir", type=Path, default=DEFAULT_PROBE_RUN_DIR)
    parser.add_argument("--selector-manifest", type=Path, default=DEFAULT_SELECTOR_MANIFEST)
    parser.add_argument("--detector-root", type=Path, default=DEFAULT_DETECTOR_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ticket", default="PER-250")
    parser.add_argument("--case-limit", type=int, default=None)
    parser.add_argument("--case-ids", default="", help="Comma-separated case IDs; default uses all cached feature files.")
    parser.add_argument(
        "--transfer-case-ids",
        default="",
        help="Comma-separated case IDs whose unselected detector crops should be scored/rendered; default scores all training cases.",
    )
    parser.add_argument("--transfer-case-limit", type=int, default=None)
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--model-backend", choices=["transformers", "timm"], default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--fallback-model-name", default="vit_small_patch14_dinov2")
    parser.add_argument("--allow-timm-fallback", action="store_true")
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
    parser.add_argument("--max-overview-width", type=int, default=1320)
    parser.add_argument("--max-grid-dim", type=int, default=620)
    parser.add_argument("--summary-top-k", type=int, default=10)
    parser.add_argument("--validation-mode", choices=["holdout", "loso"], default="holdout")
    parser.add_argument("--holdout-frac", type=float, default=0.2)
    parser.add_argument("--mlp-max-iter", type=int, default=45)
    parser.add_argument("--sample-seed", type=int, default=0)
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


def make_page(
    title: str,
    subtitle: str,
    body: Image.Image,
    *,
    footer: str | None = None,
    page_width: int = 1700,
    max_body_h: int = 2100,
) -> Image.Image:
    body = resize_to_fit(body, page_width - 100, max_body_h)
    page_h = 170 + body.height + (78 if footer else 38)
    page = Image.new("RGB", (page_width, page_h), "white")
    draw = ImageDraw.Draw(page)
    draw.text((50, 34), title, fill=(0, 0, 0), font=get_font(36, bold=True))
    y = draw_wrapped_text(draw, (50, 82), subtitle, font=get_font(21), fill=(45, 45, 45), width_chars=122)
    page.paste(body, ((page_width - body.width) // 2, max(142, y + 24)))
    if footer:
        draw_wrapped_text(draw, (50, page.height - 62), footer, font=get_font(15), fill=(80, 80, 80), width_chars=150)
    return page


def make_contact_sheet(panels: list[Image.Image], *, cols: int, gap: int, bg: str = "white") -> Image.Image:
    if not panels:
        return Image.new("RGB", (900, 180), bg)
    col_w = max(panel.width for panel in panels)
    rows = [panels[i : i + cols] for i in range(0, len(panels), cols)]
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


def selected_manifest_by_case(path: Path) -> dict[str, dict[str, str]]:
    return {row["case_id"]: row for row in read_csv(path)}


def feature_case_id(path: Path) -> str:
    return path.name.removesuffix("_features.npz")


def detector_case_dir(detector_root: Path, case_id: str) -> Path:
    matches: list[Path] = []
    for subdir in ("non_sv40", "sv40_skip_odd"):
        path = detector_root / subdir / case_id
        if path.exists():
            matches.append(path)
    if not matches:
        raise FileNotFoundError(f"No detector case dir found for {case_id} under {detector_root}")
    return matches[0]


def final_detector_orders(case_dir: Path) -> set[int]:
    detector_json = json.loads((case_dir / "detections.json").read_text())
    return {int(det["source_candidate_order"]) for det in detector_json.get("detections", [])}


def load_case_bundles(args: argparse.Namespace) -> list[CaseBundle]:
    selector_rows = selected_manifest_by_case(args.selector_manifest)
    requested = {part.strip() for part in args.case_ids.split(",") if part.strip()}
    feature_paths = sorted(args.probe_run_dir.glob("features/*_features.npz"))
    bundles: list[CaseBundle] = []
    for path in feature_paths:
        case_id = feature_case_id(path)
        if requested and case_id not in requested:
            continue
        selector_row = selector_rows.get(case_id)
        if selector_row is None:
            continue
        selected_ids = set(parse_box_ids(selector_row.get("verifier_selected_box_ids", "")))
        if not selected_ids:
            continue
        with np.load(path, allow_pickle=False) as data:
            features = data["features"].astype("float32")
            labels = data["labels"].astype("int64")
            backend = str(data["model_backend"])
            model = str(data["model_name"])
        case_dir = detector_case_dir(args.detector_root, case_id)
        detector_orders = final_detector_orders(case_dir)
        candidates = [
            candidate
            for candidate in load_candidate_infos(case_dir, selected_ids)
            if candidate.candidate_order in detector_orders
        ]
        bundles.append(
            CaseBundle(
                case_id=case_id,
                task=int(selector_row["task"]),
                stain=selector_row["stain"],
                selector_row=selector_row,
                detector_case_dir=case_dir,
                selected_ids=selected_ids,
                selected_features=features,
                selected_labels=labels,
                feature_backend=backend,
                feature_model=model,
                candidates=candidates,
            )
        )
    bundles.sort(key=lambda item: (item.stain, item.task, item.case_id))
    if args.case_limit is not None:
        bundles = bundles[: int(args.case_limit)]
    if not bundles:
        raise ValueError("No cached selected feature cases found for pooled training")
    return bundles


def select_transfer_bundles(args: argparse.Namespace, bundles: list[CaseBundle]) -> list[CaseBundle]:
    requested = {part.strip() for part in args.transfer_case_ids.split(",") if part.strip()}
    selected = [bundle for bundle in bundles if not requested or bundle.case_id in requested]
    selected.sort(key=lambda item: (item.stain, item.task, item.case_id))
    if args.transfer_case_limit is not None:
        selected = selected[: int(args.transfer_case_limit)]
    if not selected:
        raise ValueError(f"No transfer cases selected from {len(bundles)} training bundles")
    return selected


def class_weight_vector(y: np.ndarray) -> np.ndarray:
    counts = np.bincount(y.astype("int64"), minlength=2).astype("float64")
    weights = np.ones_like(y, dtype="float64")
    total = float(len(y))
    for cls in (0, 1):
        if counts[cls] > 0:
            weights[y == cls] = total / (2.0 * counts[cls])
    return weights


def make_model(name: str, seed: int, mlp_max_iter: int) -> Any:
    if name == "linear_logreg":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                solver="lbfgs",
                class_weight="balanced",
                random_state=seed,
            ),
        )
    if name == "mlp_1x64":
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(64,),
                activation="relu",
                solver="adam",
                alpha=1e-4,
                batch_size=256,
                learning_rate_init=1e-3,
                max_iter=mlp_max_iter,
                early_stopping=True,
                n_iter_no_change=5,
                validation_fraction=0.12,
                random_state=seed,
            ),
        )
    if name == "mlp_2x64":
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(64, 64),
                activation="relu",
                solver="adam",
                alpha=1e-4,
                batch_size=256,
                learning_rate_init=1e-3,
                max_iter=mlp_max_iter,
                early_stopping=True,
                n_iter_no_change=5,
                validation_fraction=0.12,
                random_state=seed,
            ),
        )
    raise ValueError(name)


def fit_model(model: Any, x: np.ndarray, y: np.ndarray) -> Any:
    kwargs: dict[str, Any] = {}
    if hasattr(model, "steps") and model.steps[-1][0] == "mlpclassifier":
        kwargs["mlpclassifier__sample_weight"] = class_weight_vector(y)
    model.fit(x, y, **kwargs)
    return model


def predict_prob(model: Any, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    raise TypeError(f"Model lacks predict_proba: {type(model)}")


def metric_summary(y: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    pred = (prob >= 0.5).astype("int64")
    precision, recall, f1, _ = precision_recall_fscore_support(
        y,
        pred,
        labels=[1],
        average="binary",
        zero_division=0,
    )
    return {
        "n": float(len(y)),
        "fg": float((y == 1).sum()),
        "bg": float((y == 0).sum()),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision_fg": float(precision),
        "recall_fg": float(recall),
        "f1_fg": float(f1),
        "mean_prob_fg": float(prob.mean()) if len(prob) else float("nan"),
        "roc_auc": safe_metric(roc_auc_score, y, prob),
        "average_precision": safe_metric(average_precision_score, y, prob),
    }


def safe_metric(fn: Any, y: np.ndarray, prob: np.ndarray) -> float:
    try:
        return float(fn(y, prob))
    except Exception:
        return float("nan")


def pool_selected_features(bundles: list[CaseBundle]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    case_indices: list[np.ndarray] = []
    for idx, bundle in enumerate(bundles):
        features.append(bundle.selected_features)
        labels.append(bundle.selected_labels)
        case_indices.append(np.full(len(bundle.selected_labels), idx, dtype="int64"))
    return np.concatenate(features), np.concatenate(labels), np.concatenate(case_indices)


def leave_one_case_out_metrics(
    bundles: list[CaseBundle],
    x: np.ndarray,
    y: np.ndarray,
    case_indices: np.ndarray,
    model_names: list[str],
    seed: int,
    mlp_max_iter: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fold_rows: list[dict[str, Any]] = []
    for model_name in model_names:
        for case_idx, bundle in enumerate(bundles):
            train_idx = np.where(case_indices != case_idx)[0]
            test_idx = np.where(case_indices == case_idx)[0]
            model = make_model(model_name, seed + case_idx, mlp_max_iter)
            fit_model(model, x[train_idx], y[train_idx])
            prob = predict_prob(model, x[test_idx])
            row: dict[str, Any] = {
                "model": model_name,
                "heldout_case_id": bundle.case_id,
                "heldout_task": bundle.task,
                "heldout_stain": bundle.stain,
                "train_n": int(len(train_idx)),
                "test_n": int(len(test_idx)),
            }
            row.update(metric_summary(y[test_idx], prob))
            fold_rows.append(row)

    summary_rows: list[dict[str, Any]] = []
    metric_keys = [
        "accuracy",
        "balanced_accuracy",
        "precision_fg",
        "recall_fg",
        "f1_fg",
        "roc_auc",
        "average_precision",
    ]
    for model_name in model_names:
        rows = [row for row in fold_rows if row["model"] == model_name]
        summary: dict[str, Any] = {"model": model_name, "folds": len(rows)}
        for key in metric_keys:
            values = np.asarray([float(row[key]) for row in rows if not math.isnan(float(row[key]))], dtype="float64")
            summary[f"mean_{key}"] = float(values.mean()) if len(values) else float("nan")
            summary[f"std_{key}"] = float(values.std(ddof=0)) if len(values) else float("nan")
        summary_rows.append(summary)
    return fold_rows, summary_rows


def case_holdout_metrics(
    bundles: list[CaseBundle],
    x: np.ndarray,
    y: np.ndarray,
    case_indices: np.ndarray,
    model_names: list[str],
    seed: int,
    holdout_frac: float,
    mlp_max_iter: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    case_count = len(bundles)
    holdout_count = max(1, int(math.ceil(case_count * holdout_frac)))
    holdout_case_indices = set(int(idx) for idx in rng.permutation(case_count)[:holdout_count])
    train_idx = np.asarray([idx for idx in range(len(y)) if int(case_indices[idx]) not in holdout_case_indices], dtype="int64")
    test_idx = np.asarray([idx for idx in range(len(y)) if int(case_indices[idx]) in holdout_case_indices], dtype="int64")
    heldout_cases = [bundles[idx] for idx in sorted(holdout_case_indices)]
    fold_rows: list[dict[str, Any]] = []
    for model_name in model_names:
        model = make_model(model_name, seed + 1000, mlp_max_iter)
        fit_model(model, x[train_idx], y[train_idx])
        prob = predict_prob(model, x[test_idx])
        row: dict[str, Any] = {
            "model": model_name,
            "heldout_case_id": ",".join(bundle.case_id for bundle in heldout_cases),
            "heldout_task": ",".join(str(bundle.task) for bundle in heldout_cases),
            "heldout_stain": ",".join(bundle.stain for bundle in heldout_cases),
            "train_n": int(len(train_idx)),
            "test_n": int(len(test_idx)),
            "validation_mode": "case_holdout",
            "holdout_case_count": len(heldout_cases),
        }
        row.update(metric_summary(y[test_idx], prob))
        fold_rows.append(row)

    summary_rows: list[dict[str, Any]] = []
    for row in fold_rows:
        summary: dict[str, Any] = {
            "model": row["model"],
            "folds": 1,
            "validation_mode": "case_holdout",
            "heldout_case_id": row["heldout_case_id"],
        }
        for key in [
            "accuracy",
            "balanced_accuracy",
            "precision_fg",
            "recall_fg",
            "f1_fg",
            "roc_auc",
            "average_precision",
        ]:
            summary[f"mean_{key}"] = float(row[key])
            summary[f"std_{key}"] = 0.0
        summary_rows.append(summary)
    return fold_rows, summary_rows


def best_model_name(summary_rows: list[dict[str, Any]]) -> str:
    ordered = sorted(
        summary_rows,
        key=lambda row: (
            float(row.get("mean_f1_fg", float("nan"))),
            float(row.get("mean_balanced_accuracy", float("nan"))),
        ),
        reverse=True,
    )
    return str(ordered[0]["model"])


def read_detector_json(case_dir: Path) -> dict[str, Any]:
    return json.loads((case_dir / "detections.json").read_text())


def extract_case_unselected(
    args: argparse.Namespace,
    bundle: CaseBundle,
    extractor: FeatureExtractor,
) -> tuple[np.ndarray, list[PatchRecord], dict[str, Any]]:
    selected_rows_all = read_csv(args.probe_run_dir / "manifests/selected_patch_manifest.csv")
    selected_rows = [dict(row) for row in selected_rows_all if row["case_id"] == bundle.case_id]
    if not selected_rows:
        raise ValueError(f"No selected patch rows for {bundle.case_id}")
    detector_json = read_detector_json(bundle.detector_case_dir)
    wsi_path = resolve_wsi_path(selected_rows, bundle.selector_row, detector_json)
    if args.wsi_reader == "cucim":
        slide = WsiPatchReader(wsi_path, args.wsi_reader, args.read_workers)
    else:
        slide = openslide.OpenSlide(str(wsi_path))
    try:
        records: list[PatchRecord] = []
        for candidate in bundle.candidates:
            if not candidate.selected_for_train:
                records.extend(build_patch_records(candidate, int(args.patch_size)))
        if not records:
            return np.zeros((0, bundle.selected_features.shape[1]), dtype="float32"), [], {
                "cache_reused": False,
                "cache_path": "",
                "wsi_path": str(wsi_path),
            }
        cache_path = args.output_dir / "features" / f"{bundle.case_id}_unselected_detector_candidates_features.npz"
        features, records, meta = extract_unselected_features(slide, records, extractor, cache_path, bool(args.resume))
        allowed_orders = {candidate.candidate_order for candidate in bundle.candidates}
        if records:
            keep_idx = [idx for idx, record in enumerate(records) if record.candidate_order in allowed_orders]
            if len(keep_idx) != len(records):
                dropped = len(records) - len(keep_idx)
                features = features[np.asarray(keep_idx, dtype="int64")]
                records = [records[idx] for idx in keep_idx]
                meta["filtered_to_final_detector_orders"] = True
                meta["dropped_nonfinal_patch_records"] = int(dropped)
                meta["final_detector_orders"] = sorted(allowed_orders)
        meta["wsi_path"] = str(wsi_path)
        return features, records, meta
    finally:
        slide.close()


def draw_detector_overview_with_stats(
    bundle: CaseBundle,
    chosen_model: str,
    stats_by_candidate: dict[int, dict[str, Any]],
    patch_prediction: tuple[list[PatchRecord], np.ndarray, np.ndarray] | None,
    max_width: int,
) -> Image.Image:
    thumb_path = bundle.detector_case_dir / "intermediate_stage_artifacts/stage1_thumbnail_detection/thumbnail.png"
    if thumb_path.exists():
        base = Image.open(thumb_path).convert("RGB")
    else:
        base = Image.open(bundle.detector_case_dir / "final_detected_bboxes.png").convert("RGB")
    w, h = base.size
    detector_json = read_detector_json(bundle.detector_case_dir)
    visible_orders = {int(det["source_candidate_order"]) for det in detector_json.get("detections", [])}
    if patch_prediction is not None:
        records, pred, prob = patch_prediction
        slide_path = bundle.selector_row.get("source_wsi_path") or bundle.selector_row.get("wsi_path") or detector_json.get("wsi_path")
        try:
            slide = openslide.OpenSlide(str(slide_path))
            slide_w, slide_h = slide.dimensions
            slide.close()
        except Exception:
            slide_w = slide_h = 0
        if slide_w > 0 and slide_h > 0:
            rgba = base.convert("RGBA")
            overlay = Image.new("RGBA", rgba.size, (0, 0, 0, 0))
            odraw = ImageDraw.Draw(overlay)
            for record, is_fg, fg_prob in zip(records, pred, prob):
                if record.candidate_order not in visible_orders:
                    continue
                if int(is_fg) != 1:
                    continue
                alpha = int(45 + 95 * max(0.0, min(1.0, float(fg_prob))))
                rect = [
                    int(round(record.x / slide_w * w)),
                    int(round(record.y / slide_h * h)),
                    int(round((record.x + record.width) / slide_w * w)),
                    int(round((record.y + record.height) / slide_h * h)),
                ]
                odraw.rectangle(rect, fill=(34, 197, 94, alpha), outline=(22, 163, 74, 210), width=1)
            base = Image.alpha_composite(rgba, overlay).convert("RGB")

    draw = ImageDraw.Draw(base)
    for det in detector_json["detections"]:
        order = int(det["source_candidate_order"])
        y0, x0, y1, x1 = [float(v) for v in det["box_2d"]]
        selected = order in bundle.selected_ids
        color = (22, 163, 74) if selected else (220, 38, 38)
        rect = [
            int(round(x0 / 1000.0 * w)),
            int(round(y0 / 1000.0 * h)),
            int(round(x1 / 1000.0 * w)),
            int(round(y1 / 1000.0 * h)),
        ]
        draw.rectangle(rect, outline=color, width=5)
        label = str(order)
        if not selected and order in stats_by_candidate:
            label = f"{order} {float(stats_by_candidate[order]['pred_fg_fraction']):.2f}"
        font = get_font(24, bold=True)
        tb = draw.textbbox((0, 0), label, font=font)
        lx = rect[0] + 5
        ly = rect[1] + 5
        draw.rectangle(
            [lx, ly, lx + tb[2] - tb[0] + 10, ly + tb[3] - tb[1] + 8],
            fill="white",
            outline=color,
            width=3,
        )
        draw.text((lx + 5, ly + 3), label, fill=color, font=font)

    footer_h = 78
    canvas = Image.new("RGB", (base.width, base.height + footer_h), "white")
    canvas.paste(base, (0, 0))
    d = ImageDraw.Draw(canvas)
    text = (
        f"{bundle.case_id} | {bundle.stain} task{bundle.task:03d} | "
        f"green=train selected IDs {sorted(bundle.selected_ids)}; red=test candidates; "
        f"green squares={chosen_model} predicted FG patches in unselected candidates; "
        "red labels show candidate ID and predicted FG fraction."
    )
    draw_wrapped_text(d, (12, base.height + 12), text, font=get_font(18), fill=(40, 40, 40), width_chars=130)
    return resize_to_fit(canvas, max_width, 980)


def draw_case_table(case_rows: list[dict[str, Any]], chosen_model: str) -> Image.Image:
    rows = [row for row in case_rows if row["model"] == chosen_model]
    rows.sort(key=lambda row: int(row["candidate_order"]))
    font = get_font(19)
    bold = get_font(21, bold=True)
    line_h = 31
    width = 760
    height = 54 + max(1, len(rows)) * line_h
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((8, 8), f"{chosen_model} test-candidate predictions", fill=(0, 0, 0), font=bold)
    y = 46
    if not rows:
        draw.text((8, y), "No unselected candidates.", fill=(50, 50, 50), font=font)
        return image
    for row in rows:
        text = (
            f"{int(row['candidate_order']):>2}: "
            f"FG {int(row['pred_fg'])}/{int(row['patch_count'])} "
            f"({float(row['pred_fg_fraction']):.2f}), mean p={float(row['mean_prob_fg']):.3f}"
        )
        draw.text((8, y), text, fill=(35, 35, 35), font=font)
        y += line_h
    return image


def draw_metric_table(summary_rows: list[dict[str, Any]], chosen_model: str) -> Image.Image:
    headers = ["model", "F1", "bal acc", "precision", "recall", "AUROC", "AP"]
    rows = sorted(summary_rows, key=lambda row: str(row["model"]))
    col_w = [250, 125, 150, 150, 125, 125, 125]
    row_h = 42
    width = sum(col_w) + 28
    height = 76 + row_h * (len(rows) + 1)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    mode = str(summary_rows[0].get("validation_mode", "loso")) if summary_rows else "validation"
    title = "Case-heldout selected-crop validation" if mode == "case_holdout" else "Leave-one-WSI-out selected-crop validation"
    draw.text((12, 10), title, fill=(0, 0, 0), font=get_font(26, bold=True))
    x = 14
    y = 58
    for header, cw in zip(headers, col_w):
        draw.text((x, y), header, fill=(0, 0, 0), font=get_font(18, bold=True))
        x += cw
    y += row_h
    for row in rows:
        bg = (232, 245, 233) if row["model"] == chosen_model else (255, 255, 255)
        draw.rectangle([8, y - 6, width - 8, y + row_h - 7], fill=bg)
        values = [
            str(row["model"]),
            f"{float(row['mean_f1_fg']):.3f}",
            f"{float(row['mean_balanced_accuracy']):.3f}",
            f"{float(row['mean_precision_fg']):.3f}",
            f"{float(row['mean_recall_fg']):.3f}",
            f"{float(row['mean_roc_auc']):.3f}",
            f"{float(row['mean_average_precision']):.3f}",
        ]
        x = 14
        for value, cw in zip(values, col_w):
            draw.text((x, y), value, fill=(35, 35, 35), font=get_font(18))
            x += cw
        y += row_h
    draw.text((12, height - 30), f"Selected overlay model: {chosen_model}", fill=(35, 35, 35), font=get_font(17))
    return image


def draw_test_thumbnail_panel(bundle: CaseBundle, candidate: CandidateInfo, stats: dict[str, Any], chosen_model: str) -> Image.Image:
    image_path = candidate_thumbnail_path(bundle.detector_case_dir, candidate)
    image = Image.open(image_path).convert("RGB")
    image = resize_to_fit(image, 430, 430)
    title = f"test detector ID {candidate.candidate_order}"
    stat = (
        f"{chosen_model}: FG {int(stats['pred_fg'])}/{int(stats['patch_count'])} "
        f"({float(stats['pred_fg_fraction']):.2f}), mean p={float(stats['mean_prob_fg']):.3f}"
        if stats
        else f"{chosen_model}: no patches"
    )
    panel_w = max(470, image.width + 24)
    panel_h = image.height + 104
    panel = Image.new("RGB", (panel_w, panel_h), "white")
    draw = ImageDraw.Draw(panel)
    draw.text((12, 8), title, fill=(0, 0, 0), font=get_font(23, bold=True))
    draw.text((12, 39), stat, fill=(45, 45, 45), font=get_font(15))
    x = (panel_w - image.width) // 2
    y = 74
    panel.paste(image, (x, y))
    draw.rectangle([x, y, x + image.width - 1, y + image.height - 1], outline=(220, 38, 38), width=5)
    return panel


def build_visual_pages(
    args: argparse.Namespace,
    transfer_bundles: list[CaseBundle],
    chosen_model: str,
    summary_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    patch_predictions: dict[tuple[str, str], tuple[list[PatchRecord], np.ndarray, np.ndarray]],
) -> list[Image.Image]:
    pages: list[Image.Image] = []
    metric_table = draw_metric_table(summary_rows, chosen_model)
    validation_mode = str(summary_rows[0].get("validation_mode", args.validation_mode)) if summary_rows else args.validation_mode
    validation_text = (
        "Model comparison uses a case-level holdout split over selected-crop pseudo-labels."
        if validation_mode == "case_holdout"
        else "Model comparison uses leave-one-WSI-out selected-crop pseudo-label validation."
    )
    pages.append(
        make_page(
            "Pooled Scale-500 DINOv3 Probe Transfer",
            (
                "Frozen DINOv3-small features pooled across selected scale-500 crops. "
                f"{validation_text}"
            ),
            metric_table,
            footer="All models are then refit on all selected pseudo-labeled patches before scoring unselected detector candidates.",
        )
    )

    rows_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        rows_by_case[str(row["case_id"])].append(row)

    for bundle in transfer_bundles:
        chosen_stats = {
            int(row["candidate_order"]): row
            for row in rows_by_case[bundle.case_id]
            if row["model"] == chosen_model
        }
        overview = draw_detector_overview_with_stats(
            bundle,
            chosen_model,
            chosen_stats,
            patch_predictions.get((bundle.case_id, chosen_model)),
            args.max_overview_width,
        )
        table = draw_case_table(rows_by_case[bundle.case_id], chosen_model)
        body = make_contact_sheet([overview, table], cols=1, gap=22)
        pages.append(
            make_page(
                f"{bundle.case_id} | {bundle.stain} | thumbnail overview",
                "Green detector bboxes were verifier-selected and contributed training patches; red bboxes are unselected test crops scored by transfer.",
                body,
                footer="This is the thumbnail-level overview requested: detector bboxes overlaid on the detector thumbnail, not separate crop thumbnails.",
                max_body_h=2300,
            )
        )

    for bundle in transfer_bundles:
        chosen_stats = {
            int(row["candidate_order"]): row
            for row in rows_by_case[bundle.case_id]
            if row["model"] == chosen_model
        }
        unselected_candidates = [candidate for candidate in bundle.candidates if not candidate.selected_for_train]
        thumb_panels = [
            draw_test_thumbnail_panel(bundle, candidate, chosen_stats.get(candidate.candidate_order, {}), chosen_model)
            for candidate in unselected_candidates
        ]
        if thumb_panels:
            pages.append(
                make_page(
                    f"{bundle.case_id} | {bundle.stain} | crop thumbnails",
                    "Crop-level unselected detector bbox thumbnails with per-crop transfer prediction summaries.",
                    make_contact_sheet(thumb_panels, cols=3, gap=28),
                    footer="Red borders mark crops that Stage 7/redundancy selection did not choose for training.",
                )
            )

        key = (bundle.case_id, chosen_model)
        if key not in patch_predictions:
            continue
        records, pred, prob = patch_predictions[key]
        pred_panels: list[Image.Image] = []
        slide_path = bundle.selector_row.get("source_wsi_path") or bundle.selector_row.get("wsi_path")
        slide = openslide.OpenSlide(slide_path)
        try:
            for candidate in unselected_candidates:
                pairs = [(idx, record) for idx, record in enumerate(records) if record.candidate_order == candidate.candidate_order]
                if not pairs:
                    continue
                local_records = [record for _idx, record in pairs]
                pred_by_local = {local: int(pred[global_idx]) for local, (global_idx, _record) in enumerate(pairs)}
                prob_by_local = {local: float(prob[global_idx]) for local, (global_idx, _record) in enumerate(pairs)}
                panel = draw_prediction_grid_panel(
                    slide,
                    candidate,
                    local_records,
                    pred_by_local,
                    prob_by_local,
                    title=f"{chosen_model} patch grid | test detector ID {candidate.candidate_order}",
                    max_dim=args.max_grid_dim,
                )
                pred_panels.append(panel)
        finally:
            slide.close()
        if pred_panels:
            pages.append(
                make_page(
                    f"{bundle.case_id} | {bundle.stain} | crop-level prediction overlays",
                    "Patch-level transfer prediction overlays on each unselected detector bbox crop.",
                    make_contact_sheet(pred_panels, cols=2, gap=34),
                    footer="Green patches are probe foreground predictions; red patches are probe background predictions.",
                    max_body_h=2300,
                )
            )
    return pages


def write_reproduction(args: argparse.Namespace, summary: dict[str, Any]) -> None:
    command = [
        "python",
        "scripts/train_pooled_dinov3_probe_transfer.py",
        "--probe-run-dir",
        str(args.probe_run_dir),
        "--selector-manifest",
        str(args.selector_manifest),
        "--detector-root",
        str(args.detector_root),
        "--output-dir",
        str(args.output_dir),
        "--patch-size",
        str(args.patch_size),
        "--device",
        str(args.device),
        "--batch-size",
        str(args.batch_size),
        "--wsi-reader",
        str(args.wsi_reader),
        "--read-workers",
        str(args.read_workers),
        "--validation-mode",
        str(args.validation_mode),
        "--holdout-frac",
        str(args.holdout_frac),
        "--mlp-max-iter",
        str(args.mlp_max_iter),
        "--sample-seed",
        str(args.sample_seed),
    ]
    if args.case_limit is not None:
        command.extend(["--case-limit", str(args.case_limit)])
    if args.case_ids:
        command.extend(["--case-ids", args.case_ids])
    if args.transfer_case_ids:
        command.extend(["--transfer-case-ids", args.transfer_case_ids])
    if args.transfer_case_limit is not None:
        command.extend(["--transfer-case-limit", str(args.transfer_case_limit)])
    lines = [
        "PER-250 Pooled DINOv3 Scale-500 Probe Transfer",
        "================================================",
        "",
        f"Created: {summary['created_at']}",
        f"Ticket: {args.ticket}",
        f"Git commit: {summary['git_commit']}",
        f"Dirty git state captured in: {args.output_dir / 'git_status.txt'}",
        "",
        "Command:",
        " ".join(shlex.quote(part) for part in command),
        "",
        "Environment:",
        f"- Python executable: {sys.executable}",
        "- Conda env expected: path-agent",
        "",
        "Inputs:",
        f"- Selected feature run: {args.probe_run_dir.resolve()}",
        f"- Selected patch manifest: {(args.probe_run_dir / 'manifests/selected_patch_manifest.csv').resolve()}",
        f"- Selector manifest: {args.selector_manifest.resolve()}",
        f"- Detector root: {args.detector_root.resolve()}",
        f"- WSI reader for unselected transfer patch features: {args.wsi_reader}; read_workers={args.read_workers}.",
        "",
        "Split semantics:",
        "- Training/evaluation labels are Stage 7 pseudo-labels from verifier-selected scale-500 detector crops.",
        f"- Validation mode: {args.validation_mode}; holdout_frac={args.holdout_frac}.",
        "- Final transfer predictions are applied to unselected detector candidates in each WSI; these have no ground-truth review labels here.",
        "",
        "Outputs:",
        f"- PDF: {summary['pdf']}",
        f"- Model summary CSV: {summary['model_summary_csv']}",
        f"- Fold metrics CSV: {summary['fold_metrics_csv']}",
        f"- Candidate transfer CSV: {summary['candidate_transfer_summary_csv']}",
        f"- Patch predictions CSV: {summary['patch_predictions_csv']}",
        "",
        "Non-determinism:",
        f"- sample_seed={args.sample_seed}; sklearn MLP uses early stopping and may vary across sklearn/runtime versions.",
        "",
    ]
    (args.output_dir / "reproduction.txt").write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "git_status.txt").write_text(git_status_short())

    bundles = load_case_bundles(args)
    transfer_bundles = select_transfer_bundles(args, bundles)
    backends = {bundle.feature_backend for bundle in bundles}
    models = {bundle.feature_model for bundle in bundles}
    if len(backends) != 1 or len(models) != 1:
        raise ValueError(f"Mixed selected feature models are not supported: backends={backends}, models={models}")
    feature_backend = next(iter(backends))
    feature_model = next(iter(models))
    if args.model_backend is None:
        args.model_backend = feature_backend
    if args.model_name is None:
        args.model_name = feature_model

    x, y, case_indices = pool_selected_features(bundles)
    model_names = ["linear_logreg", "mlp_1x64", "mlp_2x64"]
    if args.validation_mode == "loso":
        fold_rows, model_summary_rows = leave_one_case_out_metrics(
            bundles,
            x,
            y,
            case_indices,
            model_names,
            int(args.sample_seed),
            int(args.mlp_max_iter),
        )
    else:
        fold_rows, model_summary_rows = case_holdout_metrics(
            bundles,
            x,
            y,
            case_indices,
            model_names,
            int(args.sample_seed),
            float(args.holdout_frac),
            int(args.mlp_max_iter),
        )
    chosen_model = best_model_name(model_summary_rows)

    final_models: dict[str, Any] = {}
    for model_name in model_names:
        model = make_model(model_name, int(args.sample_seed) + 5000, int(args.mlp_max_iter))
        final_models[model_name] = fit_model(model, x, y)

    extractor = FeatureExtractor(args)
    candidate_rows: list[dict[str, Any]] = []
    patch_rows: list[dict[str, Any]] = []
    feature_cache_meta: dict[str, Any] = {}
    patch_predictions_for_visual: dict[tuple[str, str], tuple[list[PatchRecord], np.ndarray, np.ndarray]] = {}
    for bundle in transfer_bundles:
        features, records, meta = extract_case_unselected(args, bundle, extractor)
        feature_cache_meta[bundle.case_id] = meta
        if not len(records):
            continue
        for model_name, model in final_models.items():
            prob = predict_prob(model, features)
            pred = (prob >= 0.5).astype("int64")
            if model_name == chosen_model:
                patch_predictions_for_visual[(bundle.case_id, model_name)] = (records, pred, prob)
            by_candidate: dict[int, list[int]] = defaultdict(list)
            for idx, record in enumerate(records):
                by_candidate[record.candidate_order].append(idx)
                patch_rows.append(
                    {
                        "case_id": bundle.case_id,
                        "task": bundle.task,
                        "stain": bundle.stain,
                        "model": model_name,
                        "candidate_order": record.candidate_order,
                        "candidate_id": record.candidate_id,
                        "row": record.row,
                        "col": record.col,
                        "x_level0": record.x,
                        "y_level0": record.y,
                        "width_level0": record.width,
                        "height_level0": record.height,
                        "prob_fg": float(prob[idx]),
                        "pred_fg": int(pred[idx]),
                    }
                )
            for candidate in bundle.candidates:
                if candidate.selected_for_train:
                    continue
                idxs = by_candidate.get(candidate.candidate_order, [])
                if not idxs:
                    continue
                probs = prob[idxs]
                preds = pred[idxs]
                candidate_rows.append(
                    {
                        "case_id": bundle.case_id,
                        "task": bundle.task,
                        "stain": bundle.stain,
                        "model": model_name,
                        "candidate_order": candidate.candidate_order,
                        "candidate_id": candidate.candidate_id,
                        "selected_for_train": False,
                        "bbox_level0": list(candidate.bbox_level0),
                        "patch_count": int(len(idxs)),
                        "pred_fg": int(preds.sum()),
                        "pred_bg": int(len(preds) - preds.sum()),
                        "pred_fg_fraction": float(preds.mean()),
                        "mean_prob_fg": float(probs.mean()),
                    }
                )

    write_csv(args.output_dir / "fold_metrics.csv", fold_rows)
    write_csv(args.output_dir / "model_summary.csv", model_summary_rows)
    write_csv(args.output_dir / "candidate_transfer_summary.csv", candidate_rows)
    write_csv(args.output_dir / "unselected_patch_predictions.csv", patch_rows)

    pages = build_visual_pages(
        args,
        transfer_bundles,
        chosen_model,
        model_summary_rows,
        candidate_rows,
        patch_predictions_for_visual,
    )
    for idx, page in enumerate(pages, start=1):
        page.save(args.output_dir / f"page_{idx:02d}.png")
    pdf_path = args.output_dir / "pooled_scale500_dinov3_probe_transfer.pdf"
    pages[0].save(pdf_path, save_all=True, append_images=pages[1:])

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "ticket": args.ticket,
        "git_commit": git_commit(),
        "probe_run_dir": str(args.probe_run_dir.resolve()),
        "selector_manifest": str(args.selector_manifest.resolve()),
        "detector_root": str(args.detector_root.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "case_count": len(bundles),
        "transfer_case_count": len(transfer_bundles),
        "transfer_case_ids": [bundle.case_id for bundle in transfer_bundles],
        "cases": [
            {
                "case_id": bundle.case_id,
                "task": bundle.task,
                "stain": bundle.stain,
                "selected_detector_ids": sorted(bundle.selected_ids),
                "candidate_count": len(bundle.candidates),
                "unselected_candidate_count": sum(not c.selected_for_train for c in bundle.candidates),
                "selected_patch_count": int(len(bundle.selected_labels)),
                "selected_fg": int((bundle.selected_labels == 1).sum()),
                "selected_bg": int((bundle.selected_labels == 0).sum()),
            }
            for bundle in bundles
        ],
        "pooled_selected_patch_count": int(len(y)),
        "pooled_selected_fg": int((y == 1).sum()),
        "pooled_selected_bg": int((y == 0).sum()),
        "feature_backend": feature_backend,
        "feature_model": feature_model,
        "feature_extractor": extractor.meta,
        "patch_reader": {
            "wsi_reader": args.wsi_reader,
            "read_workers": args.read_workers,
        },
        "package_versions": package_versions(),
        "model_names": model_names,
        "validation_mode": args.validation_mode,
        "holdout_frac": args.holdout_frac if args.validation_mode == "holdout" else None,
        "mlp_max_iter": args.mlp_max_iter,
        "chosen_overlay_model": chosen_model,
        "model_summary_csv": str((args.output_dir / "model_summary.csv").resolve()),
        "fold_metrics_csv": str((args.output_dir / "fold_metrics.csv").resolve()),
        "candidate_transfer_summary_csv": str((args.output_dir / "candidate_transfer_summary.csv").resolve()),
        "patch_predictions_csv": str((args.output_dir / "unselected_patch_predictions.csv").resolve()),
        "feature_cache_meta": feature_cache_meta,
        "pdf": str(pdf_path.resolve()),
        "preview_pages": [str((args.output_dir / f"page_{idx:02d}.png").resolve()) for idx in range(1, len(pages) + 1)],
    }
    write_json(args.output_dir / "summary.json", summary)
    write_reproduction(args, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
