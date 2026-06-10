#!/usr/bin/env python3
"""Prepare and run the PER-271 SV40-augmented stress-probe comparison.

The first mode is intentionally a pause point: it renders SV40 candidate crops
so approved crops can be selected before any SV40 labels are added to training.
"""

from __future__ import annotations

import argparse
import json
import math
import shlex
import subprocess
import sys
import textwrap
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import openslide
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from build_per_wsi_probe_unselected_transfer_demo import (  # noqa: E402
    CandidateInfo,
    PatchRecord,
    WsiPatchReader,
    build_patch_records,
    draw_prediction_grid_panel,
    draw_wrapped_text,
    extract_unselected_features,
    get_font,
    load_candidate_infos,
    read_bbox_thumbnail,
    read_csv,
    resize_to_fit,
    resolve_wsi_path,
    write_csv,
)
from compare_scale500_stress32_probe_transfer import (  # noqa: E402
    ProbeSpec,
    assert_feature_compatible,
    load_scale500_probe,
    load_stress_probe,
    probe_summary_rows,
    predict_prob,
)
from train_per_wsi_dinov3_fg_bg_probe import FeatureExtractor, package_versions  # noqa: E402
from train_pooled_dinov3_probe_transfer import (  # noqa: E402
    CaseBundle,
    detector_case_dir,
    final_detector_orders,
    load_case_bundles,
)


DEFAULT_STRESS_RUN_DIR = REPO_ROOT / "runs/stress32_gt_overlay_sample_efficiency_probe_v1"
DEFAULT_SCALE500_FEATURE_DIR = REPO_ROOT / "runs/scale500_selected_dinov3small_features_sample100_per_wsi_v1"
DEFAULT_SELECTOR_MANIFEST = (
    REPO_ROOT
    / "runs/auto_context_scale500_selector_all500_v1/manifests/completed_cases_500_20260604_openrouter_review_current.csv"
)
DEFAULT_DETECTOR_ROOT = REPO_ROOT / "runs/detector_pipeline_scale500_v1"
DEFAULT_PER270_RUN_DIR = REPO_ROOT / "runs/scale500_stress32_probe_transfer_compare_v1"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "runs/stress32_n500_sv40_augmented_probe_v1"


SOURCE_PRIORITY = {
    "true_red": 0,
    "stage5_unselected_backfill": 1,
    "selected_fallback": 2,
}

SOURCE_COLORS = {
    "true_red": (220, 38, 38),
    "stage5_unselected_backfill": (234, 88, 12),
    "selected_fallback": (22, 163, 74),
}


@dataclass(frozen=True)
class ReviewCandidate:
    bundle: CaseBundle
    candidate: CandidateInfo
    candidate_source: str


@dataclass(frozen=True)
class SelectedVlmItem:
    bundle: CaseBundle
    candidate: CandidateInfo
    bbox_id: str
    run_dir: Path
    stage7_overlay_path: Path
    reviewer_overlay_path: Path
    reviewer_qc: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=["prepare-sv40-review", "compare-vlm-linear-selected", "train-augmented-probe", "compare"],
        help="PER-271 phase to run.",
    )
    parser.add_argument("--stress-run-dir", type=Path, default=DEFAULT_STRESS_RUN_DIR)
    parser.add_argument("--scale500-feature-dir", type=Path, default=DEFAULT_SCALE500_FEATURE_DIR)
    parser.add_argument("--selector-manifest", type=Path, default=DEFAULT_SELECTOR_MANIFEST)
    parser.add_argument("--detector-root", type=Path, default=DEFAULT_DETECTOR_ROOT)
    parser.add_argument("--per270-run-dir", type=Path, default=DEFAULT_PER270_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ticket", default="PER-271")
    parser.add_argument("--sv40-wsi-count", type=int, default=32)
    parser.add_argument("--max-crops-per-wsi", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--probe-threshold", type=float, default=0.5)
    parser.add_argument("--reviewer-precision-threshold", type=float, default=0.85)
    parser.add_argument("--reviewer-recall-threshold", type=float, default=0.85)
    parser.add_argument("--model-backend", choices=["transformers", "timm"], default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--fallback-model-name", default="vit_small_patch14_dinov2")
    parser.add_argument("--allow-timm-fallback", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--input-size", type=int, default=None)
    parser.add_argument("--wsi-reader", choices=["cucim", "openslide"], default="cucim")
    parser.add_argument("--read-workers", type=int, default=16)
    parser.add_argument("--pipeline-mode", choices=["serial", "prefetch"], default="prefetch")
    parser.add_argument("--prefetch-queue-batches", type=int, default=4)
    parser.add_argument("--max-overview-width", type=int, default=1320)
    parser.add_argument("--max-crop-dim", type=int, default=520)
    parser.add_argument("--sample-seed", type=int, default=271)
    parser.add_argument("--case-limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--selected-sv40-crops", type=Path, default=None)
    parser.add_argument(
        "--vlm-stage7-manifest",
        type=Path,
        default=None,
        help="Filtered all500 manifest for selected-crop VLM/Stage7 comparison.",
    )
    parser.add_argument(
        "--selector-run-root",
        type=Path,
        default=REPO_ROOT / "runs/auto_context_scale500_selector_all500_v1",
        help="Selector-seeded all500 auto-context run root containing VLM Stage7 outputs.",
    )
    parser.add_argument(
        "--reviewer-batch-name",
        default="per250_scale500_all500_openrouter_gemini3flash_high_calibration_review_v1",
    )
    parser.add_argument("--linear-probe-sample-sizes", default="50,500")
    parser.add_argument("--max-grid-dim", type=int, default=520)
    return parser.parse_args()


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


def dvc_status_text() -> str:
    if not (REPO_ROOT / ".dvc").exists():
        return "no .dvc directory"
    try:
        return subprocess.check_output(["dvc", "status"], cwd=REPO_ROOT, text=True, stderr=subprocess.STDOUT)
    except Exception as exc:
        return f"dvc status failed: {type(exc).__name__}: {exc}"


def make_case_bundle_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        probe_run_dir=args.scale500_feature_dir,
        selector_manifest=args.selector_manifest,
        detector_root=args.detector_root,
        case_limit=None,
        case_ids="",
    )


def load_per270_sv40_holdout(per270_run_dir: Path) -> set[str]:
    path = per270_run_dir / "case_selection_manifest.csv"
    if not path.exists():
        return set()
    return {row["case_id"] for row in read_csv(path) if row.get("stain") == "SV40"}


def candidate_sort_key(candidate: CandidateInfo) -> tuple[int, str]:
    return int(candidate.candidate_order), str(candidate.candidate_id)


def source_candidates(bundle: CaseBundle) -> tuple[str, list[CandidateInfo]]:
    final_orders = final_detector_orders(bundle.detector_case_dir)
    all_candidates = load_candidate_infos(bundle.detector_case_dir, bundle.selected_ids)
    true_red = [
        candidate
        for candidate in bundle.candidates
        if candidate.candidate_order in final_orders and not candidate.selected_for_train
    ]
    if true_red:
        return "true_red", sorted(true_red, key=candidate_sort_key)

    stage5_unselected = [
        candidate
        for candidate in all_candidates
        if not candidate.selected_for_train and candidate.candidate_order not in final_orders
    ]
    if stage5_unselected:
        return "stage5_unselected_backfill", sorted(stage5_unselected, key=candidate_sort_key)

    selected_fallback = [candidate for candidate in all_candidates if candidate.selected_for_train]
    if selected_fallback:
        return "selected_fallback", sorted(selected_fallback, key=candidate_sort_key)
    return "selected_fallback", []


def select_sv40_review_candidates(args: argparse.Namespace, bundles: list[CaseBundle]) -> list[ReviewCandidate]:
    holdout = load_per270_sv40_holdout(args.per270_run_dir)
    grouped: list[tuple[int, int, int, str, CaseBundle, str, list[CandidateInfo]]] = []
    for bundle in bundles:
        if bundle.stain != "SV40" or bundle.case_id in holdout:
            continue
        source, candidates = source_candidates(bundle)
        if not candidates:
            continue
        selection_idx = int(bundle.selector_row.get("selection_index_within_stain", bundle.task))
        grouped.append(
            (
                SOURCE_PRIORITY[source],
                selection_idx,
                int(bundle.task),
                bundle.case_id,
                bundle,
                source,
                candidates,
            )
        )

    review_candidates: list[ReviewCandidate] = []
    selected_cases: set[str] = set()
    for _priority, _selection_idx, _task, _case_id, bundle, source, candidates in sorted(grouped):
        if bundle.case_id in selected_cases:
            continue
        selected_cases.add(bundle.case_id)
        for candidate in candidates[: int(args.max_crops_per_wsi)]:
            review_candidates.append(ReviewCandidate(bundle=bundle, candidate=candidate, candidate_source=source))
        if len(selected_cases) >= int(args.sv40_wsi_count):
            break

    if len(selected_cases) < int(args.sv40_wsi_count):
        raise ValueError(
            f"Need {args.sv40_wsi_count} strict-holdout SV40 WSIs, found {len(selected_cases)}. "
            f"PER-270 SV40 holdout size={len(holdout)}"
        )
    return review_candidates


def load_selected_rows_by_case(scale500_feature_dir: Path) -> dict[str, list[dict[str, str]]]:
    path = scale500_feature_dir / "manifests/selected_patch_manifest.csv"
    by_case: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(path):
        by_case[row["case_id"]].append(row)
    return by_case


def open_slide_for_bundle(
    args: argparse.Namespace,
    bundle: CaseBundle,
    selected_rows_by_case: dict[str, list[dict[str, str]]],
) -> tuple[Any, Path]:
    detector_json = json.loads((bundle.detector_case_dir / "detections.json").read_text())
    wsi_path = resolve_wsi_path(selected_rows_by_case[bundle.case_id], bundle.selector_row, detector_json)
    if args.wsi_reader == "cucim":
        return WsiPatchReader(wsi_path, args.wsi_reader, args.read_workers), wsi_path
    return openslide.OpenSlide(str(wsi_path)), wsi_path


def open_openslide_for_bundle(
    bundle: CaseBundle,
    selected_rows_by_case: dict[str, list[dict[str, str]]],
) -> tuple[openslide.OpenSlide, Path]:
    detector_json = json.loads((bundle.detector_case_dir / "detections.json").read_text())
    wsi_path = resolve_wsi_path(selected_rows_by_case[bundle.case_id], bundle.selector_row, detector_json)
    return openslide.OpenSlide(str(wsi_path)), wsi_path


def feature_cache_path(args: argparse.Namespace, case_id: str) -> Path:
    return args.output_dir / "features" / f"{case_id}_sv40_review_candidates_features.npz"


def score_review_candidates(
    args: argparse.Namespace,
    review_candidates: list[ReviewCandidate],
    probe: ProbeSpec,
    extractor: FeatureExtractor,
) -> tuple[list[dict[str, Any]], dict[str, tuple[list[PatchRecord], np.ndarray, np.ndarray, dict[str, Any]]]]:
    selected_rows_by_case = load_selected_rows_by_case(args.scale500_feature_dir)
    by_case: dict[str, list[ReviewCandidate]] = defaultdict(list)
    for item in review_candidates:
        by_case[item.bundle.case_id].append(item)

    rows: list[dict[str, Any]] = []
    case_predictions: dict[str, tuple[list[PatchRecord], np.ndarray, np.ndarray, dict[str, Any]]] = {}
    for case_id, items in sorted(by_case.items(), key=lambda kv: (kv[1][0].bundle.task, kv[0])):
        bundle = items[0].bundle
        slide, wsi_path = open_slide_for_bundle(args, bundle, selected_rows_by_case)
        try:
            records: list[PatchRecord] = []
            for item in items:
                records.extend(build_patch_records(item.candidate, int(args.patch_size)))
            features, loaded_records, meta = extract_unselected_features(
                slide,
                records,
                extractor,
                feature_cache_path(args, case_id),
                bool(args.resume),
            )
            if len(loaded_records) != len(records):
                # Cache reuse may preserve the original record order; use loaded records
                # as the source of truth for prediction summarisation.
                records = loaded_records
            prob = predict_prob(probe.model, features)
            pred = (prob >= float(args.probe_threshold)).astype("int64")
            case_predictions[case_id] = (records, pred, prob, meta)
        finally:
            slide.close()

        candidate_by_order = {item.candidate.candidate_order: item for item in items}
        indexes_by_order: dict[int, list[int]] = defaultdict(list)
        for idx, record in enumerate(records):
            indexes_by_order[int(record.candidate_order)].append(idx)
        for order in sorted(indexes_by_order):
            item = candidate_by_order.get(order)
            if item is None:
                continue
            idxs = indexes_by_order[order]
            cand_prob = prob[np.asarray(idxs, dtype="int64")]
            cand_pred = pred[np.asarray(idxs, dtype="int64")]
            x0, y0, x1, y1 = item.candidate.bbox_level0
            rows.append(
                {
                    "case_id": case_id,
                    "task": bundle.task,
                    "stain": bundle.stain,
                    "selection_index_within_stain": bundle.selector_row.get("selection_index_within_stain", ""),
                    "source_wsi_path": bundle.selector_row.get("source_wsi_path", ""),
                    "resolved_wsi_path": str(wsi_path),
                    "candidate_order": item.candidate.candidate_order,
                    "candidate_id": item.candidate.candidate_id,
                    "candidate_source": item.candidate_source,
                    "selected_for_train_in_scale500": item.candidate.selected_for_train,
                    "bbox_level0": [x0, y0, x1, y1],
                    "patch_count": len(idxs),
                    "preview_model": probe.name,
                    "preview_pred_fg": int(cand_pred.sum()),
                    "preview_pred_bg": int((cand_pred == 0).sum()),
                    "preview_fg_fraction": float(cand_pred.mean()) if len(cand_pred) else float("nan"),
                    "preview_mean_prob_fg": float(cand_prob.mean()) if len(cand_prob) else float("nan"),
                    "stage7_mask_path": "",
                    "reviewer_status": "not_run_preview_only",
                    "reviewer_precision": "",
                    "reviewer_recall": "",
                    "reviewer_pass": "",
                }
            )
    return rows, case_predictions


def resize_to_width(image: Image.Image, max_width: int) -> Image.Image:
    if image.width <= max_width:
        return image
    scale = max_width / image.width
    return image.resize((max_width, max(1, int(round(image.height * scale)))), Image.Resampling.LANCZOS)


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


def make_page(
    title: str,
    subtitle: str,
    body: Image.Image,
    *,
    footer: str | None = None,
    page_width: int = 1800,
    max_body_h: int = 2450,
) -> Image.Image:
    body = resize_to_fit(body, page_width - 100, max_body_h)
    page_h = 178 + body.height + (78 if footer else 38)
    page = Image.new("RGB", (page_width, page_h), "white")
    draw = ImageDraw.Draw(page)
    draw.text((50, 34), title, fill=(0, 0, 0), font=get_font(35, bold=True))
    y = draw_wrapped_text(draw, (50, 82), subtitle, font=get_font(20), fill=(45, 45, 45), width_chars=130)
    page.paste(body, ((page_width - body.width) // 2, max(148, y + 24)))
    if footer:
        draw_wrapped_text(draw, (50, page.height - 62), footer, font=get_font(15), fill=(80, 80, 80), width_chars=155)
    return page


def draw_summary_page_body(
    args: argparse.Namespace,
    probe: ProbeSpec,
    candidate_rows: list[dict[str, Any]],
    holdout_cases: set[str],
) -> Image.Image:
    case_ids = sorted({str(row["case_id"]) for row in candidate_rows})
    source_counts = Counter(str(row["candidate_source"]) for row in candidate_rows)
    lines = [
        "PER-271 SV40 crop review packet",
        "",
        f"Strict-holdout PER-270 SV40 eval cases excluded: {len(holdout_cases)}",
        f"SV40 WSIs selected for review: {len(case_ids)}",
        f"Candidate crop rows: {len(candidate_rows)}",
        "Candidate source counts: " + ", ".join(f"{k}={source_counts[k]}" for k in SOURCE_PRIORITY),
        "",
        "Foreground preview:",
        f"{probe.name}: train={probe.train_count} (FG={probe.train_fg}, BG={probe.train_bg}); feature={probe.feature_model}",
        "",
        "Important boundary:",
        "This packet is for crop approval. Green/red patch grids are stress_N500 preview predictions.",
        "Training labels are not taken from this preview. Approved crops still need the Stage7 + calibration reviewer route.",
        f"Reviewer training gate will use precision >= {args.reviewer_precision_threshold:.2f} and recall >= {args.reviewer_recall_threshold:.2f}.",
    ]
    font = get_font(23)
    bold = get_font(30, bold=True)
    line_h = 36
    width = 1500
    height = 34 + len(lines) * line_h
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    y = 16
    for idx, line in enumerate(lines):
        draw.text((18, y), line, fill=(0, 0, 0), font=bold if idx == 0 else font)
        y += line_h
    return image


def draw_case_overview(
    bundle: CaseBundle,
    review_items: list[ReviewCandidate],
    selected_rows_by_case: dict[str, list[dict[str, str]]],
    *,
    max_width: int,
) -> Image.Image:
    slide, _wsi_path = open_openslide_for_bundle(bundle, selected_rows_by_case)
    try:
        slide_w, slide_h = slide.dimensions
    finally:
        slide.close()

    thumb_path = bundle.detector_case_dir / "intermediate_stage_artifacts/stage1_thumbnail_detection/thumbnail.png"
    if thumb_path.exists():
        base = Image.open(thumb_path).convert("RGB")
    else:
        base = Image.open(bundle.detector_case_dir / "final_detected_bboxes.png").convert("RGB")
    w, h = base.size
    draw = ImageDraw.Draw(base)
    font = get_font(24, bold=True)

    all_candidates = load_candidate_infos(bundle.detector_case_dir, bundle.selected_ids)
    review_by_order = {item.candidate.candidate_order: item for item in review_items}
    for candidate in all_candidates:
        x0, y0, x1, y1 = candidate.bbox_level0
        if candidate.candidate_order in review_by_order:
            source = review_by_order[candidate.candidate_order].candidate_source
            color = SOURCE_COLORS[source]
            width = 6
            label = f"{candidate.candidate_order} {source.replace('_', ' ')[:11]}"
        elif candidate.selected_for_train:
            color = (22, 163, 74)
            width = 3
            label = str(candidate.candidate_order)
        else:
            color = (185, 28, 28)
            width = 2
            label = str(candidate.candidate_order)
        rect = [
            int(round(x0 / slide_w * w)),
            int(round(y0 / slide_h * h)),
            int(round(x1 / slide_w * w)),
            int(round(y1 / slide_h * h)),
        ]
        draw.rectangle(rect, outline=color, width=width)
        if candidate.candidate_order in review_by_order:
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
    return resize_to_width(base, max_width)


def draw_raw_crop_panel(
    slide: openslide.OpenSlide,
    item: ReviewCandidate,
    row: dict[str, Any] | None,
    *,
    max_dim: int,
) -> Image.Image:
    crop, _scale = read_bbox_thumbnail(slide, item.candidate.bbox_level0, max_dim)
    color = SOURCE_COLORS[item.candidate_source]
    title = f"ID {item.candidate.candidate_order} | {item.candidate_source}"
    if row:
        stat = (
            f"preview FG {row['preview_pred_fg']}/{row['patch_count']} "
            f"({float(row['preview_fg_fraction']):.2f}); mean p={float(row['preview_mean_prob_fg']):.3f}"
        )
    else:
        stat = "preview not available"
    title_font = get_font(22, bold=True)
    small_font = get_font(15)
    panel_w = max(crop.width + 20, 480)
    panel_h = crop.height + 96
    panel = Image.new("RGB", (panel_w, panel_h), "white")
    draw = ImageDraw.Draw(panel)
    draw.text((12, 10), title, fill=color, font=title_font)
    draw.text((12, 40), stat, fill=(45, 45, 45), font=small_font)
    x = (panel_w - crop.width) // 2
    y = 70
    panel.paste(crop, (x, y))
    draw.rectangle([x, y, x + crop.width - 1, y + crop.height - 1], outline=color, width=5)
    return panel


def build_review_pdf_pages(
    args: argparse.Namespace,
    probe: ProbeSpec,
    review_candidates: list[ReviewCandidate],
    candidate_rows: list[dict[str, Any]],
    case_predictions: dict[str, tuple[list[PatchRecord], np.ndarray, np.ndarray, dict[str, Any]]],
) -> list[Image.Image]:
    holdout = load_per270_sv40_holdout(args.per270_run_dir)
    pages = [
        make_page(
            "PER-271 SV40 Review Packet",
            "Strict-holdout SV40 crop approval gate before SV40 augmentation training.",
            draw_summary_page_body(args, probe, candidate_rows, holdout),
            footer="No SV40 training has been run from this packet.",
        )
    ]
    selected_rows_by_case = load_selected_rows_by_case(args.scale500_feature_dir)
    rows_by_case_order = {
        (str(row["case_id"]), int(row["candidate_order"])): row
        for row in candidate_rows
    }
    items_by_case: dict[str, list[ReviewCandidate]] = defaultdict(list)
    for item in review_candidates:
        items_by_case[item.bundle.case_id].append(item)

    for case_id, items in sorted(items_by_case.items(), key=lambda kv: (kv[1][0].bundle.task, kv[0])):
        bundle = items[0].bundle
        overview = draw_case_overview(
            bundle,
            items,
            selected_rows_by_case,
            max_width=args.max_overview_width,
        )
        pages.append(
            make_page(
                f"{case_id} | SV40 task{bundle.task:03d} | detector overview",
                "Green boxes are scale500 verifier-selected context; red/orange/green thick boxes are PER-271 candidate crops by source.",
                overview,
                footer="Source priority is true_red, then Stage5-unselected backfill, then selected fallback.",
                max_body_h=2350,
            )
        )

        slide, _wsi_path = open_openslide_for_bundle(bundle, selected_rows_by_case)
        try:
            records, pred, prob, _meta = case_predictions[case_id]
            pairs_by_order: dict[int, list[tuple[int, PatchRecord]]] = defaultdict(list)
            for idx, record in enumerate(records):
                pairs_by_order[int(record.candidate_order)].append((idx, record))

            candidate_panels: list[Image.Image] = []
            for item in items:
                row = rows_by_case_order.get((case_id, item.candidate.candidate_order))
                raw_panel = draw_raw_crop_panel(slide, item, row, max_dim=args.max_crop_dim)
                pairs = pairs_by_order.get(item.candidate.candidate_order, [])
                grid_panel: Image.Image
                if pairs:
                    local_records = [record for _idx, record in pairs]
                    pred_by_local = {local: int(pred[global_idx]) for local, (global_idx, _record) in enumerate(pairs)}
                    prob_by_local = {local: float(prob[global_idx]) for local, (global_idx, _record) in enumerate(pairs)}
                    grid_panel = draw_prediction_grid_panel(
                        slide,
                        item.candidate,
                        local_records,
                        pred_by_local,
                        prob_by_local,
                        title=f"{probe.name} preview | ID {item.candidate.candidate_order}",
                        max_dim=args.max_crop_dim,
                    )
                else:
                    grid_panel = Image.new("RGB", (520, 180), "white")
                    ImageDraw.Draw(grid_panel).text((12, 12), "No patch records", fill=(0, 0, 0), font=get_font(20))
                candidate_panels.append(make_contact_sheet([raw_panel, grid_panel], cols=2, gap=18))
        finally:
            slide.close()

        for chunk_idx in range(0, len(candidate_panels), 3):
            chunk = candidate_panels[chunk_idx : chunk_idx + 3]
            pages.append(
                make_page(
                    f"{case_id} | SV40 task{bundle.task:03d} | candidate crops",
                    "Left is the raw crop; right is the stress_N500 foreground preview. Use selected_sv40_crops.csv to approve crops.",
                    make_contact_sheet(chunk, cols=1, gap=24),
                    footer="Preview predictions are for review only; Stage7/reviewer labels are still required before training.",
                    max_body_h=2600,
                )
            )
    return pages


def candidate_template_rows(candidate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in candidate_rows:
        rows.append(
            {
                "approved_for_training": "",
                "case_id": row["case_id"],
                "task": row["task"],
                "stain": row["stain"],
                "candidate_order": row["candidate_order"],
                "candidate_id": row["candidate_id"],
                "candidate_source": row["candidate_source"],
                "selected_for_train_in_scale500": row["selected_for_train_in_scale500"],
                "bbox_level0": row["bbox_level0"],
                "preview_model": row["preview_model"],
                "preview_fg_fraction": row["preview_fg_fraction"],
                "preview_mean_prob_fg": row["preview_mean_prob_fg"],
                "reviewer_status": row["reviewer_status"],
                "reviewer_precision": row["reviewer_precision"],
                "reviewer_recall": row["reviewer_recall"],
                "notes": "",
            }
        )
    return rows


def parse_int_list(text: str) -> list[int]:
    values = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError(f"No integer values parsed from {text!r}")
    return values


def case_to_dash(case_id: str) -> str:
    if case_id.startswith("anon_"):
        return "anon_" + case_id[len("anon_") :].replace("_", "-")
    return case_id.replace("_", "-")


def bbox_id_from_level0(bbox: tuple[int, int, int, int]) -> str:
    return "_".join(str(int(value)) for value in bbox)


def bbox_id_to_level0(bbox_id: str) -> tuple[int, int, int, int] | None:
    try:
        values = [int(part) for part in str(bbox_id).split("_")]
    except ValueError:
        return None
    if len(values) != 4:
        return None
    return tuple(values)  # type: ignore[return-value]


def default_vlm_stage7_manifest(args: argparse.Namespace) -> Path:
    if args.vlm_stage7_manifest is not None:
        return args.vlm_stage7_manifest
    return args.output_dir / "vlm_stage7_selected_review" / "sv40_32_all500_selected_stage7_manifest.csv"


def reviewer_run_id(row: dict[str, str]) -> str:
    return f"per250_scale500_all500_verifier_qwen16_icl0_array_v1_task{int(row['task']):03d}_stage7_l0_review_1024"


def auto_context_run_id(row: dict[str, str]) -> str:
    return f"per250_scale500_all500_verifier_qwen16_icl0_array_v1_task{int(row['task']):03d}"


def load_vlm_reviewer_qc(args: argparse.Namespace) -> dict[tuple[str, str, str], dict[str, Any]]:
    results_path = args.selector_run_root / "reviewer" / args.reviewer_batch_name / "results.jsonl"
    rows: dict[tuple[str, str, str], dict[str, Any]] = {}
    if not results_path.exists():
        return rows
    with results_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            case_id = str(row.get("case_id") or "")
            run_id = str(row.get("run_id") or "")
            bbox_id = str(row.get("bbox_id") or "")
            if case_id and run_id and bbox_id:
                rows[(case_id, run_id, bbox_id)] = row.get("qc") if isinstance(row.get("qc"), dict) else {}
    return rows


def load_selected_vlm_items(args: argparse.Namespace, bundles: list[CaseBundle]) -> list[SelectedVlmItem]:
    manifest_path = default_vlm_stage7_manifest(args)
    manifest_rows = read_csv(manifest_path)
    bundle_by_case = {bundle.case_id: bundle for bundle in bundles}
    reviewer_qc = load_vlm_reviewer_qc(args)
    items: list[SelectedVlmItem] = []
    for row in manifest_rows:
        case_id = row["case_id"]
        bundle = bundle_by_case.get(case_id)
        if bundle is None:
            continue
        run_id = auto_context_run_id(row)
        review_run_id = reviewer_run_id(row)
        run_dir = args.selector_run_root / case_to_dash(case_id) / run_id
        stage1_path = run_dir / "stage1" / "bboxes.json"
        selected_bbox_ids: list[str] = []
        if stage1_path.exists():
            stage1 = json.loads(stage1_path.read_text())
            for region in stage1.get("detected_regions", []):
                bbox = region.get("bbox_level0")
                if isinstance(bbox, list) and len(bbox) == 4:
                    selected_bbox_ids.append(bbox_id_from_level0(tuple(int(value) for value in bbox)))
        if not selected_bbox_ids:
            selected_bbox_ids = [
                path.name
                for path in sorted((run_dir / "bboxes").glob("*"))
                if path.is_dir() and bbox_id_to_level0(path.name) is not None
            ]
        detector_candidates = load_candidate_infos(bundle.detector_case_dir, bundle.selected_ids)
        candidates_by_bbox = {bbox_id_from_level0(candidate.bbox_level0): candidate for candidate in detector_candidates}
        selected_candidates = [
            candidates_by_bbox[bbox_id]
            for bbox_id in selected_bbox_ids
            if bbox_id in candidates_by_bbox
        ]
        for candidate in sorted(selected_candidates, key=candidate_sort_key):
            bbox_id = bbox_id_from_level0(candidate.bbox_level0)
            stage7_overlay = run_dir / "bboxes" / bbox_id / "stage7" / "postprocess_after.png"
            reviewer_overlay = (
                args.selector_run_root
                / "reviewer_inputs"
                / case_id
                / review_run_id
                / "bboxes"
                / bbox_id
                / "stage3"
                / "overlay.png"
            )
            if not stage7_overlay.exists() and not reviewer_overlay.exists():
                continue
            items.append(
                SelectedVlmItem(
                    bundle=bundle,
                    candidate=candidate,
                    bbox_id=bbox_id,
                    run_dir=run_dir,
                    stage7_overlay_path=stage7_overlay,
                    reviewer_overlay_path=reviewer_overlay,
                    reviewer_qc=reviewer_qc.get((case_id, review_run_id, bbox_id), {}),
                )
            )
    if not items:
        raise ValueError(f"No selected VLM Stage7 items found from {manifest_path}")
    return items


def selected_vlm_feature_cache_path(args: argparse.Namespace, case_id: str) -> Path:
    return args.output_dir / "vlm_stage7_selected_review" / "linear_compare" / "features" / f"{case_id}_selected_vlm_features.npz"


def score_selected_vlm_items(
    args: argparse.Namespace,
    items: list[SelectedVlmItem],
    probes: list[ProbeSpec],
    extractor: FeatureExtractor,
) -> tuple[
    list[dict[str, Any]],
    dict[tuple[str, str], tuple[list[PatchRecord], np.ndarray, np.ndarray]],
    dict[str, Any],
]:
    selected_rows_by_case = load_selected_rows_by_case(args.scale500_feature_dir)
    by_case: dict[str, list[SelectedVlmItem]] = defaultdict(list)
    for item in items:
        by_case[item.bundle.case_id].append(item)

    candidate_rows: list[dict[str, Any]] = []
    predictions: dict[tuple[str, str], tuple[list[PatchRecord], np.ndarray, np.ndarray]] = {}
    feature_meta: dict[str, Any] = {}
    for case_id, case_items in sorted(by_case.items(), key=lambda kv: (kv[1][0].bundle.task, kv[0])):
        bundle = case_items[0].bundle
        slide, _wsi_path = open_slide_for_bundle(args, bundle, selected_rows_by_case)
        try:
            records: list[PatchRecord] = []
            for item in case_items:
                records.extend(build_patch_records(item.candidate, int(args.patch_size)))
            cache_path = selected_vlm_feature_cache_path(args, case_id)
            if bool(args.resume) and cache_path.exists():
                try:
                    with np.load(cache_path, allow_pickle=False) as cached:
                        cached_order_array = cached["candidate_order"]
                        cached_count = int(cached_order_array.shape[0])
                        cached_orders = {int(value) for value in cached_order_array.tolist()}
                    requested_orders = {int(record.candidate_order) for record in records}
                    if cached_count != len(records) or cached_orders != requested_orders:
                        cache_path.unlink()
                except Exception:
                    cache_path.unlink(missing_ok=True)
            features, loaded_records, meta = extract_unselected_features(
                slide,
                records,
                extractor,
                cache_path,
                bool(args.resume),
            )
            if len(loaded_records) != len(records):
                records = loaded_records
            feature_meta[case_id] = meta
        finally:
            slide.close()

        item_by_order = {item.candidate.candidate_order: item for item in case_items}
        indexes_by_order: dict[int, list[int]] = defaultdict(list)
        for idx, record in enumerate(records):
            indexes_by_order[int(record.candidate_order)].append(idx)

        for probe in probes:
            prob = predict_prob(probe.model, features)
            pred = (prob >= float(args.probe_threshold)).astype("int64")
            predictions[(case_id, probe.name)] = (records, pred, prob)
            for order in sorted(indexes_by_order):
                item = item_by_order.get(order)
                if item is None:
                    continue
                idxs = indexes_by_order[order]
                arr_idx = np.asarray(idxs, dtype="int64")
                cand_prob = prob[arr_idx]
                cand_pred = pred[arr_idx]
                qc = item.reviewer_qc or {}
                precision = qc.get("precision")
                recall = qc.get("recall")
                precision_pass = precision is not None and float(precision) >= float(args.reviewer_precision_threshold)
                recall_pass = recall is not None and float(recall) >= float(args.reviewer_recall_threshold)
                candidate_rows.append(
                    {
                        "case_id": case_id,
                        "task": bundle.task,
                        "stain": bundle.stain,
                        "selection_index_within_stain": bundle.selector_row.get("selection_index_within_stain", ""),
                        "candidate_order": item.candidate.candidate_order,
                        "candidate_id": item.candidate.candidate_id,
                        "bbox_id": item.bbox_id,
                        "bbox_level0": list(item.candidate.bbox_level0),
                        "model": probe.name,
                        "patch_count": int(len(idxs)),
                        "pred_fg": int(cand_pred.sum()),
                        "pred_bg": int((cand_pred == 0).sum()),
                        "pred_fg_fraction": float(cand_pred.mean()) if len(cand_pred) else float("nan"),
                        "mean_prob_fg": float(cand_prob.mean()) if len(cand_prob) else float("nan"),
                        "vlm_stage7_overlay_path": str(item.stage7_overlay_path.resolve()) if item.stage7_overlay_path.exists() else "",
                        "vlm_reviewer_overlay_path": str(item.reviewer_overlay_path.resolve()) if item.reviewer_overlay_path.exists() else "",
                        "vlm_reviewer_precision": precision if precision is not None else "",
                        "vlm_reviewer_recall": recall if recall is not None else "",
                        "vlm_reviewer_pass_at_085": bool(precision_pass and recall_pass) if precision is not None and recall is not None else "",
                    }
                )
    return candidate_rows, predictions, feature_meta


def draw_labeled_image_panel(path: Path, title: str, subtitle: str, *, max_dim: int) -> Image.Image:
    if path.exists():
        image = Image.open(path).convert("RGB")
        image = resize_to_fit(image, max_dim, max_dim)
    else:
        image = Image.new("RGB", (max_dim, max(160, max_dim // 2)), "white")
        ImageDraw.Draw(image).text((16, 16), "missing", fill=(180, 0, 0), font=get_font(24, bold=True))
    title_font = get_font(22, bold=True)
    small_font = get_font(15)
    panel_w = max(image.width, 430)
    panel_h = image.height + 86
    panel = Image.new("RGB", (panel_w, panel_h), "white")
    draw = ImageDraw.Draw(panel)
    draw.text((10, 8), title, fill=(0, 0, 0), font=title_font)
    draw_wrapped_text(draw, (10, 36), subtitle, font=small_font, fill=(55, 55, 55), width_chars=max(28, panel_w // 10), line_spacing=2)
    panel.paste(image, ((panel_w - image.width) // 2, 78))
    draw.rectangle(
        [(panel_w - image.width) // 2, 78, (panel_w + image.width) // 2 - 1, 78 + image.height - 1],
        outline=(210, 210, 210),
        width=2,
    )
    return panel


def format_qc(qc: dict[str, Any], args: argparse.Namespace) -> str:
    if not qc:
        return "reviewer QC missing"
    precision = qc.get("precision")
    recall = qc.get("recall")
    if precision is None or recall is None:
        return "reviewer QC unavailable"
    passed = float(precision) >= float(args.reviewer_precision_threshold) and float(recall) >= float(args.reviewer_recall_threshold)
    return f"precision={float(precision):.2f}, recall={float(recall):.2f}, {'pass' if passed else 'fail'} at >=0.85"


def prediction_panel_for_item(
    slide: openslide.OpenSlide,
    item: SelectedVlmItem,
    probe: ProbeSpec,
    predictions: dict[tuple[str, str], tuple[list[PatchRecord], np.ndarray, np.ndarray]],
    args: argparse.Namespace,
) -> Image.Image:
    key = (item.bundle.case_id, probe.name)
    if key not in predictions:
        missing = Image.new("RGB", (430, 180), "white")
        ImageDraw.Draw(missing).text((14, 14), "missing predictions", fill=(180, 0, 0), font=get_font(22, bold=True))
        return missing
    records, pred, prob = predictions[key]
    pairs = [(idx, record) for idx, record in enumerate(records) if record.candidate_order == item.candidate.candidate_order]
    if not pairs:
        missing = Image.new("RGB", (430, 180), "white")
        ImageDraw.Draw(missing).text((14, 14), "missing crop records", fill=(180, 0, 0), font=get_font(22, bold=True))
        return missing
    local_records = [record for _idx, record in pairs]
    pred_by_local = {local: int(pred[global_idx]) for local, (global_idx, _record) in enumerate(pairs)}
    prob_by_local = {local: float(prob[global_idx]) for local, (global_idx, _record) in enumerate(pairs)}
    return draw_prediction_grid_panel(
        slide,
        item.candidate,
        local_records,
        pred_by_local,
        prob_by_local,
        title=f"{probe.name} | ID {item.candidate.candidate_order}",
        max_dim=args.max_grid_dim,
    )


def draw_vlm_linear_summary(
    args: argparse.Namespace,
    probes: list[ProbeSpec],
    items: list[SelectedVlmItem],
    candidate_rows: list[dict[str, Any]],
) -> Image.Image:
    case_count = len({item.bundle.case_id for item in items})
    qc_pass = 0
    qc_available = 0
    for item in items:
        qc = item.reviewer_qc or {}
        if qc.get("precision") is not None and qc.get("recall") is not None:
            qc_available += 1
            if float(qc["precision"]) >= float(args.reviewer_precision_threshold) and float(qc["recall"]) >= float(args.reviewer_recall_threshold):
                qc_pass += 1
    lines = [
        "PER-271 SV40 selected-crop VLM vs linear probe comparison",
        "",
        f"SV40 WSIs: {case_count}",
        f"Selected VLM bbox crops: {len(items)}",
        f"Reviewer QC available: {qc_available}; pass at >=0.85 precision and recall: {qc_pass}",
        "",
        "Models:",
    ]
    for probe in probes:
        lines.append(
            f"{probe.name}: source={probe.source}; train={probe.train_count} "
            f"(FG={probe.train_fg}, BG={probe.train_bg}); feature={probe.feature_model}"
        )
    lines.extend(
        [
            "",
            "Interpretation boundary:",
            "VLM Stage7 is the selected-crop foreground route used for labels/reviewer QC.",
            "Linear probes are shown as visual comparators on the same selected bbox lattice.",
            "This packet does not train the SV40-augmented probe.",
        ]
    )
    font = get_font(22)
    bold = get_font(29, bold=True)
    width = 1520
    line_h = 34
    image = Image.new("RGB", (width, 34 + line_h * len(lines)), "white")
    draw = ImageDraw.Draw(image)
    y = 14
    for idx, line in enumerate(lines):
        draw.text((16, y), line, fill=(0, 0, 0), font=bold if idx == 0 else font)
        y += line_h
    return image


def build_vlm_linear_pdf_pages(
    args: argparse.Namespace,
    probes: list[ProbeSpec],
    items: list[SelectedVlmItem],
    candidate_rows: list[dict[str, Any]],
    predictions: dict[tuple[str, str], tuple[list[PatchRecord], np.ndarray, np.ndarray]],
) -> list[Image.Image]:
    pages = [
        make_page(
            "PER-271 SV40 VLM vs Linear",
            "Actual selector-seeded VLM Stage7 masks compared against PER-270 DINOv3-small linear probes on the same selected crops.",
            draw_vlm_linear_summary(args, probes, items, candidate_rows),
            footer="No SV40-augmented training has been run here.",
            max_body_h=2400,
        )
    ]
    selected_rows_by_case = load_selected_rows_by_case(args.scale500_feature_dir)
    by_case: dict[str, list[SelectedVlmItem]] = defaultdict(list)
    for item in items:
        by_case[item.bundle.case_id].append(item)
    for case_id, case_items in sorted(by_case.items(), key=lambda kv: (kv[1][0].bundle.task, kv[0])):
        bundle = case_items[0].bundle
        slide, _wsi_path = open_openslide_for_bundle(bundle, selected_rows_by_case)
        try:
            crop_rows: list[Image.Image] = []
            for item in sorted(case_items, key=lambda x: candidate_sort_key(x.candidate)):
                vlm_path = item.reviewer_overlay_path if item.reviewer_overlay_path.exists() else item.stage7_overlay_path
                vlm_panel = draw_labeled_image_panel(
                    vlm_path,
                    f"VLM Stage7 | ID {item.candidate.candidate_order}",
                    format_qc(item.reviewer_qc, args),
                    max_dim=args.max_grid_dim,
                )
                model_panels = [prediction_panel_for_item(slide, item, probe, predictions, args) for probe in probes]
                crop_rows.append(make_contact_sheet([vlm_panel, *model_panels], cols=4, gap=16))
        finally:
            slide.close()
        for chunk_idx in range(0, len(crop_rows), 2):
            chunk = crop_rows[chunk_idx : chunk_idx + 2]
            pages.append(
                make_page(
                    f"{case_id} | SV40 task{bundle.task:03d} | selected crop comparison",
                    "Each row is one verifier-selected crop: VLM Stage7 overlay, then scale500_logreg, stress_N50_logreg, stress_N500_logreg.",
                    make_contact_sheet(chunk, cols=1, gap=26),
                    footer="Green/red probe grids are predictions from linear probes; VLM panel is the actual selected-crop Stage7 segmentation/reviewer input.",
                    max_body_h=2650,
                )
            )
    return pages


def compare_vlm_linear_selected(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    compare_dir = args.output_dir / "vlm_stage7_selected_review" / "linear_compare"
    compare_dir.mkdir(parents=True, exist_ok=True)
    bundle_args = make_case_bundle_args(args)
    bundles = load_case_bundles(bundle_args)
    items = load_selected_vlm_items(args, bundles)

    probes = [load_scale500_probe(args, bundles)]
    for sample_size in parse_int_list(args.linear_probe_sample_sizes):
        probes.append(load_stress_probe(args, sample_size))
    backend, model_name = assert_feature_compatible(probes)
    if args.model_backend is None:
        args.model_backend = backend
    if args.model_name is None:
        args.model_name = model_name
    extractor = FeatureExtractor(args)
    extractor.pipeline_mode = args.pipeline_mode
    extractor.prefetch_queue_batches = int(args.prefetch_queue_batches)
    if extractor.backend != backend or extractor.model_name != model_name:
        raise ValueError(
            "Feature extractor model does not match cached training features: "
            f"extractor={extractor.backend}/{extractor.model_name}; cache={backend}/{model_name}"
        )

    candidate_rows, predictions, feature_meta = score_selected_vlm_items(args, items, probes, extractor)
    candidate_path = compare_dir / "selected_vlm_linear_candidate_summary.csv"
    model_path = compare_dir / "model_training_summary.csv"
    write_csv(candidate_path, candidate_rows)
    write_csv(model_path, probe_summary_rows(probes))

    pages = build_vlm_linear_pdf_pages(args, probes, items, candidate_rows, predictions)
    page_dir = compare_dir / "review_pages"
    page_dir.mkdir(parents=True, exist_ok=True)
    page_paths: list[str] = []
    for idx, page in enumerate(pages, start=1):
        path = page_dir / f"page_{idx:03d}.png"
        page.save(path)
        page_paths.append(str(path.resolve()))
    pdf_path = compare_dir / "sv40_32_selected_vlm_vs_linear_probes.pdf"
    pages[0].save(pdf_path, save_all=True, append_images=pages[1:], resolution=150)

    summary = {
        "ticket": args.ticket,
        "mode": args.mode,
        "output_dir": str(args.output_dir.resolve()),
        "comparison_dir": str(compare_dir.resolve()),
        "review_pdf": str(pdf_path.resolve()),
        "candidate_summary_csv": str(candidate_path.resolve()),
        "model_training_summary_csv": str(model_path.resolve()),
        "page_dir": str(page_dir.resolve()),
        "page_count": len(pages),
        "sv40_wsi_count": len({item.bundle.case_id for item in items}),
        "selected_vlm_crop_count": len(items),
        "models": [probe.name for probe in probes],
        "feature_backend": extractor.backend,
        "feature_model": extractor.model_name,
        "feature_meta_by_case": feature_meta,
        "elapsed_seconds": float(time.perf_counter() - started),
        "package_versions": package_versions(),
        "interpretation_boundary": "VLM Stage7 is the label/reviewer route; linear probes are visual comparators only.",
    }
    write_json(compare_dir / "summary.json", summary)
    reproduction_path = write_reproduction(args, summary)
    summary["reproduction_txt"] = str(reproduction_path.resolve())
    write_json(compare_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def write_reproduction(args: argparse.Namespace, summary: dict[str, Any]) -> Path:
    command = [
        sys.executable,
        "scripts/train_sv40_augmented_stress_probe_compare.py",
        str(args.mode),
        "--stress-run-dir",
        str(args.stress_run_dir),
        "--scale500-feature-dir",
        str(args.scale500_feature_dir),
        "--selector-manifest",
        str(args.selector_manifest),
        "--detector-root",
        str(args.detector_root),
        "--per270-run-dir",
        str(args.per270_run_dir),
        "--output-dir",
        str(args.output_dir),
        "--ticket",
        str(args.ticket),
        "--sv40-wsi-count",
        str(args.sv40_wsi_count),
        "--max-crops-per-wsi",
        str(args.max_crops_per_wsi),
        "--patch-size",
        str(args.patch_size),
        "--wsi-reader",
        str(args.wsi_reader),
        "--read-workers",
        str(args.read_workers),
        "--pipeline-mode",
        str(args.pipeline_mode),
        "--prefetch-queue-batches",
        str(args.prefetch_queue_batches),
        "--batch-size",
        str(args.batch_size),
        "--sample-seed",
        str(args.sample_seed),
    ]
    if args.mode == "compare-vlm-linear-selected":
        command.extend(
            [
                "--vlm-stage7-manifest",
                str(default_vlm_stage7_manifest(args)),
                "--selector-run-root",
                str(args.selector_run_root),
                "--reviewer-batch-name",
                str(args.reviewer_batch_name),
                "--linear-probe-sample-sizes",
                str(args.linear_probe_sample_sizes),
                "--max-grid-dim",
                str(args.max_grid_dim),
            ]
        )
    if not args.resume:
        command.append("--no-resume")
    if args.case_limit is not None:
        command.extend(["--case-limit", str(args.case_limit)])
    text = "\n".join(
        [
            "PER-271 SV40-Augmented Stress Probe Comparison",
            "",
            f"Generated timestamp: {datetime.now(timezone.utc).isoformat()}",
            f"Repository: {REPO_ROOT}",
            f"Git commit: {git_commit()}",
            "Git status --short at creation:",
            git_status_short() or "clean\n",
            "DVC status:",
            dvc_status_text(),
            "",
            "Environment:",
            f"Python executable: {sys.executable}",
            "Recommended env: path-agent",
            "HF_TOKEN is required for DINOv3 model access when feature caches are not reused; token value intentionally not recorded.",
            "",
            "Command:",
            shlex.join(command),
            "",
            "Parents:",
            f"PER-269 stress run: {args.stress_run_dir.resolve()}",
            f"PER-270 scale500 comparison run: {args.per270_run_dir.resolve()}",
            f"Scale500 selected feature run: {args.scale500_feature_dir.resolve()}",
            f"Selector manifest: {args.selector_manifest.resolve()}",
            f"Detector root: {args.detector_root.resolve()}",
            "",
            "Method:",
            "prepare-sv40-review excludes the PER-270 SV40 eval cases and selects strict-holdout SV40 WSIs.",
            "Candidate priority is true final unselected red crops, Stage5-unselected backfill, then verifier-selected fallback.",
            "This first packet uses stress_N500_logreg as a foreground-preview model only; Stage7/reviewer labels are not replaced by this preview.",
            "compare-vlm-linear-selected uses the existing all500 selector-seeded VLM Stage7 selected-crop outputs and scores the same selected bbox lattices with scale500_logreg, stress_N50_logreg, and stress_N500_logreg.",
            "All patch records are constrained to the candidate bbox lattice.",
            f"cuCIM/OpenSlide reader setting: {args.wsi_reader}; read_workers={args.read_workers}; batch_size={args.batch_size}.",
            f"Patch extraction pipeline mode: {args.pipeline_mode}; prefetch_queue_batches={args.prefetch_queue_batches}.",
            "",
            "Reviewer/training gate for later phases:",
            "Stage7 morphology policy: fill holes on; binary close off.",
            "Reviewer max_dim=1024; OpenRouter Gemini 3 Flash high thinking calibration reviewer.",
            f"Pass threshold: precision >= {args.reviewer_precision_threshold:.2f} and recall >= {args.reviewer_recall_threshold:.2f}.",
            "",
            "Outputs:",
            json.dumps(summary, indent=2, sort_keys=True),
            "",
        ]
    )
    if args.mode == "compare-vlm-linear-selected":
        path = args.output_dir / "vlm_stage7_selected_review" / "linear_compare" / "reproduction.txt"
    else:
        path = args.output_dir / "reproduction.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def prepare_sv40_review(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "git_status.txt").write_text(git_status_short())
    bundle_args = make_case_bundle_args(args)
    bundles = load_case_bundles(bundle_args)
    if args.case_limit is not None:
        # Keep the ordering deterministic while allowing a fast smoke test.
        args.sv40_wsi_count = min(int(args.sv40_wsi_count), int(args.case_limit))
    review_candidates = select_sv40_review_candidates(args, bundles)

    stress_probe = load_stress_probe(args, 500)
    assert_feature_compatible([stress_probe])
    if args.model_backend is None:
        args.model_backend = stress_probe.feature_backend
    if args.model_name is None:
        args.model_name = stress_probe.feature_model
    extractor = FeatureExtractor(args)
    extractor.pipeline_mode = args.pipeline_mode
    extractor.prefetch_queue_batches = int(args.prefetch_queue_batches)

    started = time.perf_counter()
    candidate_rows, case_predictions = score_review_candidates(args, review_candidates, stress_probe, extractor)
    elapsed = time.perf_counter() - started

    candidate_path = args.output_dir / "sv40_candidate_crops.csv"
    template_path = args.output_dir / "selected_sv40_crops.csv"
    write_csv(candidate_path, candidate_rows)
    write_csv(template_path, candidate_template_rows(candidate_rows))

    pages = build_review_pdf_pages(args, stress_probe, review_candidates, candidate_rows, case_predictions)
    pdf_path = args.output_dir / "sv40_review_packet.pdf"
    png_dir = args.output_dir / "visuals" / "sv40_review_packet_pages"
    png_dir.mkdir(parents=True, exist_ok=True)
    png_paths: list[str] = []
    for idx, page in enumerate(pages, start=1):
        png_path = png_dir / f"page_{idx:03d}.png"
        page.save(png_path)
        png_paths.append(str(png_path.resolve()))
    pages[0].save(pdf_path, save_all=True, append_images=pages[1:], resolution=150)

    source_counts = Counter(str(row["candidate_source"]) for row in candidate_rows)
    case_source_counts = Counter()
    for case_id in {str(row["case_id"]) for row in candidate_rows}:
        sources = {str(row["candidate_source"]) for row in candidate_rows if str(row["case_id"]) == case_id}
        case_source_counts.update(sorted(sources))
    feature_meta = {
        case_id: meta
        for case_id, (_records, _pred, _prob, meta) in case_predictions.items()
    }
    summary: dict[str, Any] = {
        "ticket": args.ticket,
        "mode": args.mode,
        "output_dir": str(args.output_dir.resolve()),
        "sv40_review_packet_pdf": str(pdf_path.resolve()),
        "sv40_candidate_crops_csv": str(candidate_path.resolve()),
        "selected_sv40_crops_template_csv": str(template_path.resolve()),
        "page_png_dir": str(png_dir.resolve()),
        "page_count": len(pages),
        "preview_model": stress_probe.name,
        "sv40_wsi_count": len({str(row["case_id"]) for row in candidate_rows}),
        "candidate_crop_count": len(candidate_rows),
        "candidate_source_counts": dict(source_counts),
        "case_source_counts": dict(case_source_counts),
        "per270_sv40_holdout_count": len(load_per270_sv40_holdout(args.per270_run_dir)),
        "extract_seconds": float(elapsed),
        "feature_backend": extractor.backend,
        "feature_model": extractor.model_name,
        "feature_meta_by_case": feature_meta,
        "package_versions": package_versions(),
        "training_has_run": False,
        "pause_reason": "Await user crop approval in selected_sv40_crops.csv.",
    }
    summary_path = args.output_dir / "summary.json"
    write_json(summary_path, summary)
    reproduction_path = write_reproduction(args, summary)
    print(json.dumps({**summary, "reproduction": str(reproduction_path.resolve())}, indent=2, sort_keys=True))


def train_augmented_probe(args: argparse.Namespace) -> None:
    selection_path = args.selected_sv40_crops or args.output_dir / "selected_sv40_crops.csv"
    if not selection_path.exists():
        raise FileNotFoundError(
            f"{selection_path} does not exist. Run prepare-sv40-review first and approve crops before training."
        )
    rows = read_csv(selection_path)
    approved = [row for row in rows if str(row.get("approved_for_training", "")).strip().lower() in {"1", "true", "yes", "y"}]
    if not approved:
        raise ValueError(
            f"No approved rows found in {selection_path}. Set approved_for_training=true for crops to include."
        )
    raise NotImplementedError(
        "SV40 Stage7/reviewer labeling and stress_N500_sv40_logreg training are intentionally gated until crop approval. "
        "Approved crop rows are present; implement/run this phase next."
    )


def compare(args: argparse.Namespace) -> None:
    raise NotImplementedError(
        "The 4-model scale500/stress32 comparison is the post-approval phase and has not been run by prepare-sv40-review."
    )


def main() -> None:
    args = parse_args()
    if args.mode == "prepare-sv40-review":
        prepare_sv40_review(args)
    elif args.mode == "compare-vlm-linear-selected":
        compare_vlm_linear_selected(args)
    elif args.mode == "train-augmented-probe":
        train_augmented_probe(args)
    elif args.mode == "compare":
        compare(args)
    else:  # pragma: no cover
        raise ValueError(args.mode)


if __name__ == "__main__":
    main()
