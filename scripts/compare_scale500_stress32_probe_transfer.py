#!/usr/bin/env python3
"""Compare stress-trained and scale500-trained DINOv3 FG/BG probes on scale500 crops."""

from __future__ import annotations

import argparse
import csv
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
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from build_per_wsi_probe_unselected_transfer_demo import (  # noqa: E402
    PatchRecord,
    draw_prediction_grid_panel,
    draw_wrapped_text,
    get_font,
    read_csv,
    resize_to_fit,
    write_csv,
)
from train_per_wsi_dinov3_fg_bg_probe import FeatureExtractor, package_versions  # noqa: E402
from train_pooled_dinov3_probe_transfer import (  # noqa: E402
    CaseBundle,
    draw_detector_overview_with_stats,
    extract_case_unselected,
    load_case_bundles,
    pool_selected_features,
)


DEFAULT_STRESS_RUN_DIR = REPO_ROOT / "runs/stress32_gt_overlay_sample_efficiency_probe_v1"
DEFAULT_SCALE500_FEATURE_DIR = REPO_ROOT / "runs/scale500_selected_dinov3small_features_sample100_per_wsi_v1"
DEFAULT_SELECTOR_MANIFEST = (
    REPO_ROOT
    / "runs/auto_context_scale500_selector_all500_v1/manifests/completed_cases_500_20260604_openrouter_review_current.csv"
)
DEFAULT_DETECTOR_ROOT = REPO_ROOT / "runs/detector_pipeline_scale500_v1"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "runs/scale500_stress32_probe_transfer_compare_v1"


@dataclass(frozen=True)
class ProbeSpec:
    name: str
    source: str
    sample_size_per_wsi: int | None
    model: Any
    train_count: int
    train_fg: int
    train_bg: int
    feature_backend: str
    feature_model: str
    extra: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stress-run-dir", type=Path, default=DEFAULT_STRESS_RUN_DIR)
    parser.add_argument("--scale500-feature-dir", type=Path, default=DEFAULT_SCALE500_FEATURE_DIR)
    parser.add_argument("--selector-manifest", type=Path, default=DEFAULT_SELECTOR_MANIFEST)
    parser.add_argument("--detector-root", type=Path, default=DEFAULT_DETECTOR_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ticket", default="PER-270")
    parser.add_argument("--stains", default="EVG,H&E,JONES,PAS,SV40")
    parser.add_argument("--cases-per-stain", type=int, default=10)
    parser.add_argument("--stress-sample-sizes", default="50,500")
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--probe-threshold", type=float, default=0.5)
    parser.add_argument("--model-backend", choices=["transformers", "timm"], default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--fallback-model-name", default="vit_small_patch14_dinov2")
    parser.add_argument("--allow-timm-fallback", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--input-size", type=int, default=None)
    parser.add_argument("--wsi-reader", choices=["cucim", "openslide"], default="cucim")
    parser.add_argument("--read-workers", type=int, default=16)
    parser.add_argument("--max-overview-width", type=int, default=1240)
    parser.add_argument("--max-grid-dim", type=int, default=520)
    parser.add_argument("--sample-seed", type=int, default=270)
    parser.add_argument("--case-limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
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


def parse_int_list(text: str) -> list[int]:
    values = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError(f"No integer values parsed from {text!r}")
    return values


def parse_str_list(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def fit_linear_probe(x: np.ndarray, y: np.ndarray, seed: int) -> Any:
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, solver="lbfgs", class_weight="balanced", random_state=seed),
    )
    model.fit(x, y)
    return model


def predict_prob(model: Any, x: np.ndarray) -> np.ndarray:
    return model.predict_proba(x)[:, 1]


def load_stress_probe(args: argparse.Namespace, sample_size: int) -> ProbeSpec:
    manifest_path = args.stress_run_dir / "sampled_manifests" / f"N{sample_size:03d}.csv"
    rows = read_csv(manifest_path)
    by_case: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("split") == "train":
            by_case[row["case_id"]].append(row)

    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    backends: set[str] = set()
    models: set[str] = set()
    missing: list[str] = []
    bucket_counts: Counter[str] = Counter()
    per_case_counts: dict[str, int] = {}
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
                    continue
                idxs.append(idx)
                bucket_counts.update([row["bucket"]])
            if idxs:
                arr_idx = np.asarray(idxs, dtype="int64")
                features.append(data["features"][arr_idx].astype("float32"))
                labels.append(data["label_fg"][arr_idx].astype("int64"))
                per_case_counts[case_id] = int(len(arr_idx))
            backends.add(str(data["model_backend"]))
            models.add(str(data["model_name"]))
    if missing:
        raise ValueError(f"{len(missing)} sampled stress records were missing cached features; first={missing[0]}")
    if not features:
        raise ValueError(f"No stress training features loaded from {manifest_path}")
    if len(backends) != 1 or len(models) != 1:
        raise ValueError(f"Mixed stress feature models: backends={sorted(backends)} models={sorted(models)}")

    x = np.concatenate(features, axis=0)
    y = np.concatenate(labels, axis=0)
    model = fit_linear_probe(x, y, int(args.sample_seed) + sample_size)
    return ProbeSpec(
        name=f"stress_N{sample_size}_logreg",
        source="stress32_gt_overlay",
        sample_size_per_wsi=sample_size,
        model=model,
        train_count=int(len(y)),
        train_fg=int((y == 1).sum()),
        train_bg=int((y == 0).sum()),
        feature_backend=next(iter(backends)),
        feature_model=next(iter(models)),
        extra={
            "sample_manifest": str(manifest_path.resolve()),
            "case_count": len(per_case_counts),
            "bucket_counts": dict(bucket_counts),
            "min_patches_per_case": min(per_case_counts.values()),
            "max_patches_per_case": max(per_case_counts.values()),
        },
    )


def load_scale500_probe(args: argparse.Namespace, bundles: list[CaseBundle]) -> ProbeSpec:
    x, y, _case_indices = pool_selected_features(bundles)
    backends = {bundle.feature_backend for bundle in bundles}
    models = {bundle.feature_model for bundle in bundles}
    if len(backends) != 1 or len(models) != 1:
        raise ValueError(f"Mixed scale500 feature models: backends={sorted(backends)} models={sorted(models)}")
    model = fit_linear_probe(x, y, int(args.sample_seed) + 5000)
    return ProbeSpec(
        name="scale500_logreg",
        source="scale500_selected_stage7_pseudolabels",
        sample_size_per_wsi=None,
        model=model,
        train_count=int(len(y)),
        train_fg=int((y == 1).sum()),
        train_bg=int((y == 0).sum()),
        feature_backend=next(iter(backends)),
        feature_model=next(iter(models)),
        extra={
            "feature_dir": str(args.scale500_feature_dir.resolve()),
            "case_count": len(bundles),
        },
    )


def assert_feature_compatible(probes: list[ProbeSpec]) -> tuple[str, str]:
    backends = {probe.feature_backend for probe in probes}
    models = {probe.feature_model for probe in probes}
    if len(backends) != 1 or len(models) != 1:
        raise ValueError(f"Probe feature mismatch: backends={sorted(backends)} models={sorted(models)}")
    return next(iter(backends)), next(iter(models))


def unselected_count(bundle: CaseBundle) -> int:
    return sum(not candidate.selected_for_train for candidate in bundle.candidates)


def select_transfer_bundles(args: argparse.Namespace, bundles: list[CaseBundle]) -> list[CaseBundle]:
    desired_stains = parse_str_list(args.stains)
    by_stain: dict[str, list[CaseBundle]] = defaultdict(list)
    for bundle in bundles:
        if bundle.stain in desired_stains and unselected_count(bundle) > 0:
            by_stain[bundle.stain].append(bundle)

    selected: list[CaseBundle] = []
    for stain in desired_stains:
        stain_bundles = sorted(
            by_stain.get(stain, []),
            key=lambda item: (
                int(item.selector_row.get("selection_index_within_stain", item.task)),
                item.task,
                item.case_id,
            ),
        )
        if len(stain_bundles) < int(args.cases_per_stain):
            raise ValueError(
                f"Need {args.cases_per_stain} {stain} cases with unselected bboxes; found {len(stain_bundles)}"
            )
        selected.extend(stain_bundles[: int(args.cases_per_stain)])
    selected.sort(key=lambda item: (desired_stains.index(item.stain), int(item.selector_row["selection_index_within_stain"])))
    if args.case_limit is not None:
        selected = selected[: int(args.case_limit)]
    return selected


def case_selection_rows(bundles: list[CaseBundle]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for bundle in bundles:
        rows.append(
            {
                "case_id": bundle.case_id,
                "task": bundle.task,
                "stain": bundle.stain,
                "selection_index_within_stain": bundle.selector_row.get("selection_index_within_stain", ""),
                "source_wsi_path": bundle.selector_row.get("source_wsi_path", ""),
                "wsi_path": bundle.selector_row.get("wsi_path", ""),
                "selected_detector_ids": sorted(bundle.selected_ids),
                "candidate_count": len(bundle.candidates),
                "unselected_candidate_count": unselected_count(bundle),
                "selected_patch_count": int(len(bundle.selected_labels)),
                "selected_fg": int((bundle.selected_labels == 1).sum()),
                "selected_bg": int((bundle.selected_labels == 0).sum()),
            }
        )
    return rows


def probe_summary_rows(probes: list[ProbeSpec]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for probe in probes:
        row = {
            "model": probe.name,
            "source": probe.source,
            "sample_size_per_wsi": probe.sample_size_per_wsi if probe.sample_size_per_wsi is not None else "",
            "train_count": probe.train_count,
            "train_fg": probe.train_fg,
            "train_bg": probe.train_bg,
            "feature_backend": probe.feature_backend,
            "feature_model": probe.feature_model,
        }
        row.update({f"extra_{key}": value for key, value in probe.extra.items() if not isinstance(value, dict)})
        rows.append(row)
    return rows


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
    max_body_h: int = 2300,
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


def draw_summary_table(
    probes: list[ProbeSpec],
    selected_bundles: list[CaseBundle],
    total_patch_rows: int,
    total_candidate_rows: int,
) -> Image.Image:
    stain_counts = Counter(bundle.stain for bundle in selected_bundles)
    lines = [
        "Scale500 stress-probe transfer comparison",
        "",
        "Cases: " + ", ".join(f"{stain}={stain_counts[stain]}" for stain in sorted(stain_counts)),
        f"Unselected candidate-summary rows: {total_candidate_rows}",
        f"Patch prediction rows: {total_patch_rows}",
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
            "Scale500 has no GT labels in this packet. Treat the output as visual transfer comparison only.",
            "Predictions are generated only for unselected detector bboxes; selected bboxes are context.",
        ]
    )
    font = get_font(22)
    bold = get_font(28, bold=True)
    width = 1500
    line_h = 34
    height = 34 + len(lines) * line_h
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    y = 14
    for idx, line in enumerate(lines):
        draw.text((16, y), line, fill=(0, 0, 0), font=bold if idx == 0 else font)
        y += line_h
    return image


def patch_records_by_candidate(records: list[PatchRecord]) -> dict[int, list[tuple[int, PatchRecord]]]:
    by_candidate: dict[int, list[tuple[int, PatchRecord]]] = defaultdict(list)
    for idx, record in enumerate(records):
        by_candidate[record.candidate_order].append((idx, record))
    return by_candidate


def validate_records_inside_unselected(bundle: CaseBundle, records: list[PatchRecord]) -> None:
    unselected = {candidate.candidate_order: candidate for candidate in bundle.candidates if not candidate.selected_for_train}
    selected = {candidate.candidate_order for candidate in bundle.candidates if candidate.selected_for_train}
    for record in records:
        if record.candidate_order in selected:
            raise ValueError(f"Selected bbox {record.candidate_order} leaked into predictions for {bundle.case_id}")
        candidate = unselected.get(record.candidate_order)
        if candidate is None:
            raise ValueError(f"Unknown/unfinal bbox {record.candidate_order} in predictions for {bundle.case_id}")
        x0, y0, x1, y1 = candidate.bbox_level0
        if not (x0 <= record.x < x1 and y0 <= record.y < y1):
            raise ValueError(f"Patch origin outside bbox for {bundle.case_id} detector {record.candidate_order}")
        if record.x + record.width > x1 + 1 or record.y + record.height > y1 + 1:
            raise ValueError(f"Patch extent outside bbox for {bundle.case_id} detector {record.candidate_order}")


def build_pdf_pages(
    args: argparse.Namespace,
    probes: list[ProbeSpec],
    bundles: list[CaseBundle],
    candidate_rows: list[dict[str, Any]],
    patch_predictions: dict[tuple[str, str], tuple[list[PatchRecord], np.ndarray, np.ndarray]],
) -> list[Image.Image]:
    pages: list[Image.Image] = [
        make_page(
            "Scale500 Stress-Probe Transfer Comparison",
            "Three logistic DINOv3-small probes scored the same unselected detector bboxes.",
            draw_summary_table(
                probes,
                bundles,
                sum(len(records) for records, _pred, _prob in patch_predictions.values()),
                len(candidate_rows),
            ),
            footer="No precision/recall is reported for scale500 because this packet has no GT labels.",
        )
    ]
    rows_by_case_model: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        rows_by_case_model[(str(row["case_id"]), str(row["model"]))].append(row)

    for bundle in bundles:
        overview_panels: list[Image.Image] = []
        for probe in probes:
            stats = {
                int(row["candidate_order"]): row
                for row in rows_by_case_model[(bundle.case_id, probe.name)]
            }
            panel = draw_detector_overview_with_stats(
                bundle,
                probe.name,
                stats,
                patch_predictions.get((bundle.case_id, probe.name)),
                args.max_overview_width,
            )
            overview_panels.append(panel)
        pages.append(
            make_page(
                f"{bundle.case_id} | {bundle.stain} | thumbnail overview",
                "Selected bboxes are green context; red unselected bboxes are scored. Green squares show each model's foreground predictions inside unselected boxes.",
                make_contact_sheet(overview_panels, cols=1, gap=24),
                footer="Each overview repeats the same detector bboxes and changes only the probe used for patch predictions.",
                max_body_h=2500,
            )
        )

    for bundle in bundles:
        slide_path = bundle.selector_row.get("source_wsi_path") or bundle.selector_row.get("wsi_path")
        unselected_candidates = [candidate for candidate in bundle.candidates if not candidate.selected_for_train]
        candidate_sheets: list[Image.Image] = []
        slide = openslide.OpenSlide(slide_path)
        try:
            for candidate in unselected_candidates:
                model_panels: list[Image.Image] = []
                for probe in probes:
                    key = (bundle.case_id, probe.name)
                    if key not in patch_predictions:
                        continue
                    records, pred, prob = patch_predictions[key]
                    pairs = [
                        (idx, record)
                        for idx, record in enumerate(records)
                        if record.candidate_order == candidate.candidate_order
                    ]
                    if not pairs:
                        continue
                    local_records = [record for _idx, record in pairs]
                    pred_by_local = {local: int(pred[global_idx]) for local, (global_idx, _record) in enumerate(pairs)}
                    prob_by_local = {local: float(prob[global_idx]) for local, (global_idx, _record) in enumerate(pairs)}
                    model_panels.append(
                        draw_prediction_grid_panel(
                            slide,
                            candidate,
                            local_records,
                            pred_by_local,
                            prob_by_local,
                            title=f"{probe.name} | detector ID {candidate.candidate_order}",
                            max_dim=args.max_grid_dim,
                        )
                    )
                if model_panels:
                    candidate_sheets.append(make_contact_sheet(model_panels, cols=len(model_panels), gap=18))
        finally:
            slide.close()
        for chunk_idx in range(0, len(candidate_sheets), 3):
            chunk = candidate_sheets[chunk_idx : chunk_idx + 3]
            if not chunk:
                continue
            pages.append(
                make_page(
                    f"{bundle.case_id} | {bundle.stain} | crop-level model comparison",
                    "Each row is one unselected detector bbox; columns compare scale500, stress N=50, and stress N=500 probes.",
                    make_contact_sheet(chunk, cols=1, gap=28),
                    footer="Green patches are predicted foreground; red patches are predicted background. Predictions are confined to the detector bbox crop.",
                    max_body_h=2600,
                )
            )
    return pages


def write_reproduction(args: argparse.Namespace, summary: dict[str, Any]) -> Path:
    command = [
        sys.executable,
        "scripts/compare_scale500_stress32_probe_transfer.py",
        "--stress-run-dir",
        str(args.stress_run_dir),
        "--scale500-feature-dir",
        str(args.scale500_feature_dir),
        "--selector-manifest",
        str(args.selector_manifest),
        "--detector-root",
        str(args.detector_root),
        "--output-dir",
        str(args.output_dir),
        "--ticket",
        str(args.ticket),
        "--stains",
        str(args.stains),
        "--cases-per-stain",
        str(args.cases_per_stain),
        "--stress-sample-sizes",
        str(args.stress_sample_sizes),
        "--patch-size",
        str(args.patch_size),
        "--wsi-reader",
        str(args.wsi_reader),
        "--read-workers",
        str(args.read_workers),
        "--batch-size",
        str(args.batch_size),
        "--sample-seed",
        str(args.sample_seed),
    ]
    if not args.resume:
        command.append("--no-resume")
    if args.case_limit is not None:
        command.extend(["--case-limit", str(args.case_limit)])
    text = "\n".join(
        [
            "PER-270 Scale500 Stress-Probe Transfer Comparison",
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
            f"Scale500 selected feature run: {args.scale500_feature_dir.resolve()}",
            f"Selector manifest: {args.selector_manifest.resolve()}",
            f"Detector root: {args.detector_root.resolve()}",
            "",
            "Method:",
            "Reconstruct stress_N50_logreg and stress_N500_logreg from PER-269 sampled manifests and cached DINOv3-small features.",
            "Reconstruct scale500_logreg from selected scale500 Stage 7 pseudo-label DINOv3-small features.",
            "Score only 512px level-0 patches inside unselected final detector bboxes.",
            f"cuCIM/OpenSlide reader setting: {args.wsi_reader}; read_workers={args.read_workers}; batch_size={args.batch_size}.",
            "",
            "Interpretation boundary:",
            "Scale500 has no GT labels in this packet. Outputs are visual transfer comparison and FG-fraction summaries only.",
            "",
            "Outputs:",
            json.dumps(summary, indent=2, sort_keys=True),
            "",
        ]
    )
    path = args.output_dir / "reproduction.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def main() -> None:
    started = time.perf_counter()
    args = parse_args()
    # Reused scale500 transfer helpers expect this historical argument name.
    args.probe_run_dir = args.scale500_feature_dir
    args.output_dir.mkdir(parents=True, exist_ok=True)

    scale_args = argparse.Namespace(
        probe_run_dir=args.scale500_feature_dir,
        selector_manifest=args.selector_manifest,
        detector_root=args.detector_root,
        case_ids="",
        case_limit=None,
    )
    all_scale_bundles = load_case_bundles(scale_args)
    selected_bundles = select_transfer_bundles(args, all_scale_bundles)
    write_csv(args.output_dir / "case_selection_manifest.csv", case_selection_rows(selected_bundles))

    probes = [load_scale500_probe(args, all_scale_bundles)]
    for sample_size in parse_int_list(args.stress_sample_sizes):
        probes.append(load_stress_probe(args, sample_size))
    backend, model_name = assert_feature_compatible(probes)
    if args.model_backend is None:
        args.model_backend = backend
    if args.model_name is None:
        args.model_name = model_name
    extractor = FeatureExtractor(args)
    if extractor.backend != backend or extractor.model_name != model_name:
        raise ValueError(
            "Feature extractor model does not match cached training features: "
            f"extractor={extractor.backend}/{extractor.model_name}; cache={backend}/{model_name}"
        )
    write_csv(args.output_dir / "model_training_summary.csv", probe_summary_rows(probes))

    candidate_rows: list[dict[str, Any]] = []
    patch_rows: list[dict[str, Any]] = []
    feature_cache_meta: dict[str, Any] = {}
    patch_predictions_for_pdf: dict[tuple[str, str], tuple[list[PatchRecord], np.ndarray, np.ndarray]] = {}

    for bundle in selected_bundles:
        features, records, meta = extract_case_unselected(args, bundle, extractor)
        validate_records_inside_unselected(bundle, records)
        feature_cache_meta[bundle.case_id] = meta
        if not records:
            continue
        by_candidate = patch_records_by_candidate(records)
        for probe in probes:
            prob = predict_prob(probe.model, features)
            pred = (prob >= float(args.probe_threshold)).astype("int64")
            patch_predictions_for_pdf[(bundle.case_id, probe.name)] = (records, pred, prob)
            for idx, record in enumerate(records):
                patch_rows.append(
                    {
                        "case_id": bundle.case_id,
                        "task": bundle.task,
                        "stain": bundle.stain,
                        "model": probe.name,
                        "candidate_order": record.candidate_order,
                        "candidate_id": record.candidate_id,
                        "selected_for_train": False,
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
            for candidate in [c for c in bundle.candidates if not c.selected_for_train]:
                idxs = [idx for idx, _record in by_candidate.get(candidate.candidate_order, [])]
                if not idxs:
                    continue
                probs = prob[np.asarray(idxs, dtype="int64")]
                preds = pred[np.asarray(idxs, dtype="int64")]
                candidate_rows.append(
                    {
                        "case_id": bundle.case_id,
                        "task": bundle.task,
                        "stain": bundle.stain,
                        "model": probe.name,
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

    write_csv(args.output_dir / "unselected_patch_predictions.csv", patch_rows)
    write_csv(args.output_dir / "candidate_comparison_summary.csv", candidate_rows)

    pages = build_pdf_pages(args, probes, selected_bundles, candidate_rows, patch_predictions_for_pdf)
    preview_paths: list[str] = []
    page_dir = args.output_dir / "review_pages"
    page_dir.mkdir(parents=True, exist_ok=True)
    for idx, page in enumerate(pages, start=1):
        page_path = page_dir / f"page_{idx:03d}.png"
        page.save(page_path)
        preview_paths.append(str(page_path.resolve()))
    pdf_path = args.output_dir / "scale500_stress32_probe_transfer_compare_review.pdf"
    pages[0].save(pdf_path, save_all=True, append_images=pages[1:])

    stain_counts = Counter(bundle.stain for bundle in selected_bundles)
    selected_orders = {
        bundle.case_id: sorted(bundle.selected_ids)
        for bundle in selected_bundles
    }
    predicted_selected_rows = [
        row
        for row in patch_rows
        if int(row["candidate_order"]) in set(selected_orders[str(row["case_id"])])
    ]
    if predicted_selected_rows:
        raise ValueError(f"{len(predicted_selected_rows)} selected-bbox prediction rows were produced")

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "ticket": args.ticket,
        "git_commit": git_commit(),
        "output_dir": str(args.output_dir.resolve()),
        "stress_run_dir": str(args.stress_run_dir.resolve()),
        "scale500_feature_dir": str(args.scale500_feature_dir.resolve()),
        "selector_manifest": str(args.selector_manifest.resolve()),
        "detector_root": str(args.detector_root.resolve()),
        "case_count": len(selected_bundles),
        "case_selection_policy": "first N cases per stain with at least one unselected final detector bbox",
        "cases_per_stain": int(args.cases_per_stain),
        "stain_counts": dict(stain_counts),
        "probe_threshold": float(args.probe_threshold),
        "models": [probe.name for probe in probes],
        "feature_backend": backend,
        "feature_model": model_name,
        "feature_extractor": extractor.meta,
        "patch_reader": {"wsi_reader": args.wsi_reader, "read_workers": int(args.read_workers)},
        "package_versions": package_versions(),
        "case_selection_manifest_csv": str((args.output_dir / "case_selection_manifest.csv").resolve()),
        "model_training_summary_csv": str((args.output_dir / "model_training_summary.csv").resolve()),
        "candidate_comparison_summary_csv": str((args.output_dir / "candidate_comparison_summary.csv").resolve()),
        "unselected_patch_predictions_csv": str((args.output_dir / "unselected_patch_predictions.csv").resolve()),
        "review_pdf": str(pdf_path.resolve()),
        "review_pages": preview_paths,
        "feature_cache_meta": feature_cache_meta,
        "elapsed_seconds": float(time.perf_counter() - started),
        "interpretation_boundary": "scale500 qualitative only; no GT precision/recall claims",
    }
    write_json(args.output_dir / "summary.json", summary)
    reproduction_path = write_reproduction(args, summary)
    summary["reproduction_txt"] = str(reproduction_path.resolve())
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
