#!/usr/bin/env python3
"""Render per-WSI DINO probe train-crop and held-out-crop predictions."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import textwrap
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import openslide
from PIL import Image, ImageDraw, ImageFont
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = REPO_ROOT / "runs/per_wsi_dinov3_fg_bg_probe_baseline_v1_dinov3small"
DEFAULT_OUTPUT_DIR = DEFAULT_RUN_DIR / "visuals/per_wsi_probe_fold_demo_case023_budget20_seed0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--case-id", default="anon_02665c40_cc43_42f3_8ab1_fb9a1416e3e6")
    parser.add_argument("--budget-per-class", type=int, default=20)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument(
        "--heldout-bbox-index",
        default="all",
        help="Held-out selected bbox index, or 'all' to render every selected crop fold.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-panel-height", type=int, default=900)
    parser.add_argument("--max-train-panel-height", type=int, default=420)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def git_status_short() -> str:
    try:
        return subprocess.check_output(["git", "status", "--short"], cwd=REPO_ROOT, text=True)
    except Exception as exc:
        return f"git status failed: {exc}\n"


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unknown"


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


def safe_case_filename(case_id: str) -> str:
    return case_id.replace("/", "_")


def feature_path(run_dir: Path, case_id: str) -> Path:
    return run_dir / "features" / f"{safe_case_filename(case_id)}_features.npz"


def resolve_wsi_path(rows: list[dict[str, str]]) -> Path:
    for key in ("wsi_path", "source_wsi_path"):
        for row in rows:
            raw = row.get(key, "")
            if raw and Path(raw).exists():
                return Path(raw)
    raise FileNotFoundError("No readable WSI path found in selected patch manifest rows")


def sample_train_indices(labels: np.ndarray, pool: np.ndarray, budget: int, seed: int) -> np.ndarray:
    fg = pool[labels[pool] == 1]
    bg = pool[labels[pool] == 0]
    if len(fg) < budget or len(bg) < budget:
        raise ValueError(f"insufficient train patches for budget={budget}: fg={len(fg)} bg={len(bg)}")
    rng = np.random.default_rng(seed)
    chosen_fg = rng.choice(fg, size=budget, replace=False)
    chosen_bg = rng.choice(bg, size=budget, replace=False)
    chosen = np.concatenate([chosen_fg, chosen_bg])
    rng.shuffle(chosen)
    return chosen


def metric_summary(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y,
        pred,
        labels=[1],
        average="binary",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision_fg": float(precision),
        "recall_fg": float(recall),
        "f1_fg": float(f1),
    }


def bbox_grid(rows: list[dict[str, str]]) -> tuple[int, int, int, int, int, int]:
    min_x = min(int(r["x_level0"]) for r in rows)
    min_y = min(int(r["y_level0"]) for r in rows)
    max_col = max(int(r["col"]) for r in rows)
    max_row = max(int(r["row"]) for r in rows)
    # Patch stride is 512 in this experiment; patch_w/patch_h may shrink on
    # right/bottom edge but the lattice spacing remains 512.
    stride = 512
    return min_x, min_y, max_col + 1, max_row + 1, stride, stride


def resize_to_height(image: Image.Image, max_height: int) -> Image.Image:
    if image.height <= max_height:
        return image
    scale = max_height / image.height
    resample = getattr(Image, "Resampling", Image).LANCZOS
    return image.resize((max(1, int(round(image.width * scale))), max_height), resample)


def read_lattice_thumbnail(slide: openslide.OpenSlide, rows: list[dict[str, str]], max_height: int) -> tuple[Image.Image, float]:
    x0, y0, cols, grid_rows, stride_x, stride_y = bbox_grid(rows)
    width = cols * stride_x
    height = grid_rows * stride_y
    image = slide.read_region((x0, y0), 0, (width, height)).convert("RGB")
    scale = min(max_height / image.height, 1.0)
    if scale < 1.0:
        resample = getattr(Image, "Resampling", Image).LANCZOS
        image = image.resize((max(1, int(round(image.width * scale))), max_height), resample)
    return image.convert("RGBA"), scale


def draw_label_overlay(
    slide: openslide.OpenSlide,
    rows: list[dict[str, str]],
    label_by_local: dict[int, int],
    title: str,
    max_height: int,
    *,
    sampled_by_local: dict[int, str] | None = None,
    pred_by_local: dict[int, int] | None = None,
    true_by_local: dict[int, int] | None = None,
) -> Image.Image:
    base, scale = read_lattice_thumbnail(slide, rows, max_height)
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    _, _, _cols, _grid_rows, stride_x, stride_y = bbox_grid(rows)
    for row in rows:
        local_idx = int(row["_local_record_index"])
        rr = int(row["row"])
        cc = int(row["col"])
        x1 = int(round(cc * stride_x * scale))
        y1 = int(round(rr * stride_y * scale))
        x2 = int(round((cc + 1) * stride_x * scale))
        y2 = int(round((rr + 1) * stride_y * scale))
        value = int(label_by_local.get(local_idx, 0))
        color = (34, 197, 94, 82) if value else (239, 68, 68, 34)
        draw.rectangle([x1, y1, x2, y2], fill=color)
        outline = (45, 45, 45, 190)
        width = 2
        if sampled_by_local and local_idx in sampled_by_local:
            outline = (250, 204, 21, 255) if sampled_by_local[local_idx] == "fg" else (56, 189, 248, 255)
            width = 5
        if pred_by_local is not None and true_by_local is not None:
            pred = int(pred_by_local.get(local_idx, 0))
            true = int(true_by_local.get(local_idx, 0))
            if pred == 1 and true == 0:
                outline = (249, 115, 22, 255)
                width = 5
            elif pred == 0 and true == 1:
                outline = (37, 99, 235, 255)
                width = 5
        draw.rectangle([x1, y1, x2, y2], outline=outline, width=width)
    combined = Image.alpha_composite(base, overlay).convert("RGB")
    pad = 54
    panel_w = max(combined.width, 330)
    panel = Image.new("RGB", (panel_w, combined.height + pad), "white")
    text = ImageDraw.Draw(panel)
    text.text((12, 12), title, fill=(0, 0, 0), font=get_font(24, bold=True))
    panel.paste(combined, ((panel_w - combined.width) // 2, pad))
    return panel


def make_train_contact_sheet(panels: list[Image.Image], width: int) -> Image.Image:
    if not panels:
        return Image.new("RGB", (width, 120), "white")
    resized: list[Image.Image] = []
    col_w = max(260, (width - 30) // 2)
    for panel in panels:
        if panel.width > col_w:
            scale = col_w / panel.width
            resample = getattr(Image, "Resampling", Image).LANCZOS
            panel = panel.resize((col_w, max(1, int(round(panel.height * scale)))), resample)
        resized.append(panel)
    rows = []
    for i in range(0, len(resized), 2):
        row_panels = resized[i : i + 2]
        row_h = max(p.height for p in row_panels)
        row = Image.new("RGB", (width, row_h), "white")
        x = 0
        for p in row_panels:
            row.paste(p, (x, 0))
            x += col_w + 30
        rows.append(row)
    total_h = sum(r.height for r in rows) + 18 * (len(rows) - 1)
    sheet = Image.new("RGB", (width, total_h), "white")
    y = 0
    for row in rows:
        sheet.paste(row, (0, y))
        y += row.height + 18
    return sheet


def page_for_fold(
    slide: openslide.OpenSlide,
    rows: list[dict[str, str]],
    labels: np.ndarray,
    bbox_indices: np.ndarray,
    record_indices: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    pred: np.ndarray,
    heldout_bbox: int,
    metrics: dict[str, float],
    args: argparse.Namespace,
) -> Image.Image:
    by_bbox: dict[int, list[dict[str, str]]] = defaultdict(list)
    rows_by_local = {int(row["_local_record_index"]): row for row in rows}
    true_by_local = {int(idx): int(label) for idx, label in zip(record_indices.tolist(), labels.tolist())}
    feature_by_local = {int(local): i for i, local in enumerate(record_indices.tolist())}
    sampled_by_local = {
        int(record_indices[i]): ("fg" if labels[i] == 1 else "bg")
        for i in train_idx.tolist()
    }
    for row in rows:
        by_bbox[int(row["bbox_index"])].append(row)

    train_panels: list[Image.Image] = []
    for bbox_index in sorted(by_bbox):
        if bbox_index == heldout_bbox:
            continue
        train_panels.append(
            draw_label_overlay(
                slide,
                by_bbox[bbox_index],
                true_by_local,
                f"train bbox {bbox_index}",
                args.max_train_panel_height,
                sampled_by_local=sampled_by_local,
            )
        )
    heldout_rows = by_bbox[heldout_bbox]
    pred_by_local: dict[int, int] = {}
    for feature_index, pred_value in zip(test_idx.tolist(), pred.tolist()):
        pred_by_local[int(record_indices[feature_index])] = int(pred_value)

    truth_panel = draw_label_overlay(
        slide,
        heldout_rows,
        true_by_local,
        f"held-out bbox {heldout_bbox}: truth",
        args.max_panel_height,
    )
    pred_panel = draw_label_overlay(
        slide,
        heldout_rows,
        pred_by_local,
        f"held-out bbox {heldout_bbox}: prediction",
        args.max_panel_height,
        pred_by_local=pred_by_local,
        true_by_local=true_by_local,
    )
    panel_gap = 34
    body_w = max(1100, truth_panel.width + pred_panel.width + panel_gap)
    train_sheet = make_train_contact_sheet(train_panels, body_w)
    # Limit train sheet height so pages stay readable.
    train_sheet = resize_to_height(train_sheet, 720)
    heldout_h = max(truth_panel.height, pred_panel.height)
    heldout_w = truth_panel.width + pred_panel.width + panel_gap
    page_w = max(body_w, heldout_w) + 80
    page_h = 210 + train_sheet.height + 42 + heldout_h + 120
    page = Image.new("RGB", (page_w, page_h), "white")
    d = ImageDraw.Draw(page)
    title = f"{args.case_id} | held-out bbox {heldout_bbox} | budget {args.budget_per_class}/class | seed {args.sample_seed}"
    d.text((40, 30), "Per-WSI DINOv3 FG/BG Probe Fold", fill=(0, 0, 0), font=get_font(38, bold=True))
    y_text = draw_wrapped_text(d, (40, 82), title, font=get_font(24), fill=(30, 30, 30), width_chars=92)
    y_text = draw_wrapped_text(
        d,
        (40, y_text + 8),
        "Train: sampled patches from other verifier-selected bbox crops. Test: every patch in the held-out selected crop.",
        font=get_font(22),
        fill=(50, 50, 50),
        width_chars=110,
    )
    metric_text = (
        f"held-out metrics: balanced_acc={metrics['balanced_accuracy']:.3f}, "
        f"FG precision={metrics['precision_fg']:.3f}, FG recall={metrics['recall_fg']:.3f}, FG F1={metrics['f1_fg']:.3f}"
    )
    y_text = draw_wrapped_text(d, (40, y_text + 8), metric_text, font=get_font(22), fill=(50, 50, 50), width_chars=110)
    y_text = draw_wrapped_text(
        d,
        (40, y_text + 8),
        "Legend: green=FG, red=BG; train sampled FG border=yellow, train sampled BG border=cyan; held-out prediction FP border=orange, FN border=blue.",
        font=get_font(20),
        fill=(50, 50, 50),
        width_chars=96,
    )
    y = max(232, y_text + 28)
    d.text((40, y), "1. Training selected crops", fill=(0, 0, 0), font=get_font(30, bold=True))
    y += 44
    page.paste(train_sheet, (40, y))
    y += train_sheet.height + 42
    d.text((40, y), "2. Held-out selected crop", fill=(0, 0, 0), font=get_font(30, bold=True))
    y += 44
    page.paste(truth_panel, (40, y))
    page.paste(pred_panel, (40 + truth_panel.width + panel_gap, y))
    return page


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "git_status.txt").write_text(git_status_short())

    selected_rows = read_csv(args.probe_run_dir / "manifests/selected_wsis.csv")
    case_info = next((row for row in selected_rows if row["case_id"] == args.case_id), None)
    if case_info is None:
        raise ValueError(f"case_id not found in selected_wsis.csv: {args.case_id}")

    patch_rows_all = read_csv(args.probe_run_dir / "manifests/selected_patch_manifest.csv")
    patch_rows = [dict(row) for row in patch_rows_all if row["case_id"] == args.case_id]
    if not patch_rows:
        raise ValueError(f"No patches found for case_id={args.case_id}")
    for idx, row in enumerate(patch_rows):
        row["_local_record_index"] = str(idx)

    with np.load(feature_path(args.probe_run_dir, args.case_id), allow_pickle=False) as data:
        features = data["features"]
        labels = data["labels"].astype("int64")
        bbox_indices = data["bbox_indices"].astype("int64")
        record_indices = data["record_indices"].astype("int64")
        model_backend = str(data["model_backend"])
        model_name = str(data["model_name"])

    heldouts = sorted(set(int(x) for x in bbox_indices.tolist()))
    if args.heldout_bbox_index != "all":
        heldouts = [int(args.heldout_bbox_index)]

    slide_path = resolve_wsi_path(patch_rows)
    slide = openslide.OpenSlide(str(slide_path))
    pages: list[Image.Image] = []
    fold_rows: list[dict[str, Any]] = []
    try:
        for heldout_bbox in heldouts:
            train_pool = np.where(bbox_indices != heldout_bbox)[0]
            test_idx = np.where(bbox_indices == heldout_bbox)[0]
            seed = args.sample_seed + 100000 * int(case_info["task"]) + heldout_bbox
            train_idx = sample_train_indices(labels, train_pool, args.budget_per_class, seed)
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=2000, solver="liblinear", random_state=args.sample_seed),
            )
            clf.fit(features[train_idx], labels[train_idx])
            pred = clf.predict(features[test_idx]).astype("int64")
            metrics = metric_summary(labels[test_idx], pred)
            row = {
                "case_id": args.case_id,
                "task": int(case_info["task"]),
                "stain": case_info["stain"],
                "heldout_bbox_index": heldout_bbox,
                "budget_per_class": args.budget_per_class,
                "sample_seed": args.sample_seed,
                "train_n": int(len(train_idx)),
                "train_fg": int((labels[train_idx] == 1).sum()),
                "train_bg": int((labels[train_idx] == 0).sum()),
                "test_n": int(len(test_idx)),
                "test_fg": int((labels[test_idx] == 1).sum()),
                "test_bg": int((labels[test_idx] == 0).sum()),
                **metrics,
            }
            fold_rows.append(row)
            page = page_for_fold(
                slide,
                patch_rows,
                labels,
                bbox_indices,
                record_indices,
                train_idx,
                test_idx,
                pred,
                heldout_bbox,
                metrics,
                args,
            )
            page_path = args.output_dir / f"heldout_bbox{heldout_bbox:02d}.png"
            page.save(page_path)
            pages.append(page)
    finally:
        slide.close()

    pdf_path = args.output_dir / f"{args.case_id}_budget{args.budget_per_class}_seed{args.sample_seed}_fold_demo.pdf"
    if pages:
        first, rest = pages[0], pages[1:]
        first.save(pdf_path, save_all=True, append_images=rest)

    with (args.output_dir / "fold_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fold_rows[0].keys()))
        writer.writeheader()
        writer.writerows(fold_rows)

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "ticket": "PER-250",
        "git_commit": git_commit(),
        "command": ["scripts/build_per_wsi_probe_fold_demo.py", *shlex.split(" ".join(shlex.quote(x) for x in []))],
        "case": case_info,
        "probe_run_dir": str(args.probe_run_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "model_backend": model_backend,
        "model_name": model_name,
        "budget_per_class": args.budget_per_class,
        "sample_seed": args.sample_seed,
        "heldout_bbox_indices": heldouts,
        "wsi_path": str(slide_path),
        "pdf": str(pdf_path.resolve()),
        "fold_metrics": fold_rows,
    }
    write_json(args.output_dir / "summary.json", summary)
    (args.output_dir / "reproduction.txt").write_text(
        "\n".join(
            [
                "Per-WSI DINOv3 Probe Fold Demo",
                "================================",
                "",
                f"Created: {summary['created_at']}",
                "Ticket: PER-250",
                f"Git commit: {summary['git_commit']}",
                f"Dirty git state captured in: {args.output_dir / 'git_status.txt'}",
                "",
                "Command:",
                " ".join(shlex.quote(part) for part in [
                    "python",
                    "scripts/build_per_wsi_probe_fold_demo.py",
                    "--probe-run-dir",
                    str(args.probe_run_dir),
                    "--case-id",
                    args.case_id,
                    "--budget-per-class",
                    str(args.budget_per_class),
                    "--sample-seed",
                    str(args.sample_seed),
                    "--heldout-bbox-index",
                    str(args.heldout_bbox_index),
                    "--output-dir",
                    str(args.output_dir),
                ]),
                "",
                "Inputs:",
                f"- Probe run: {args.probe_run_dir.resolve()}",
                f"- Feature cache: {feature_path(args.probe_run_dir, args.case_id).resolve()}",
                f"- Patch manifest: {(args.probe_run_dir / 'manifests/selected_patch_manifest.csv').resolve()}",
                f"- WSI path: {slide_path}",
                "",
                "Split semantics:",
                "- Selected crops are verifier-selected bbox crops within one WSI.",
                "- For each rendered page, the held-out selected crop is excluded from training.",
                "- The linear probe trains on sampled Stage 7 FG/BG patches from the other selected crops.",
                "- The trained probe is then applied to every patch in the held-out selected crop.",
                "",
                "Outputs:",
                f"- PDF: {pdf_path.resolve()}",
                f"- Per-fold metrics: {(args.output_dir / 'fold_metrics.csv').resolve()}",
                f"- Summary: {(args.output_dir / 'summary.json').resolve()}",
                "",
            ]
        )
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
