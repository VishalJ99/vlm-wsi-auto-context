#!/usr/bin/env python3
"""Report recall-first operating thresholds from saved detector predictions."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class Box:
    xyxy: tuple[float, float, float, float]
    score: float = 1.0

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.xyxy
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


@dataclass(frozen=True)
class Case:
    image_id: int
    case_id: str
    stain: str
    width: int
    height: int
    image_path: Path
    gt_boxes: tuple[Box, ...]


@dataclass(frozen=True)
class DetectorSpec:
    name: str
    family: str
    predictions_path: Path
    thresholds: tuple[float, ...]
    low_threshold: float | None = None
    high_threshold: float | None = None
    make_pdf: bool = True


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def run_text(cmd: list[str], cwd: Path | None = None) -> str:
    try:
        return subprocess.check_output(cmd, cwd=cwd, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_thresholds(raw: str) -> tuple[float, ...]:
    values: set[float] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        values.add(round(float(part), 6))
    if not values:
        raise ValueError("At least one threshold is required")
    return tuple(sorted(values))


def bbox_xywh_to_xyxy(bbox: list[float]) -> tuple[float, float, float, float]:
    x, y, w, h = [float(value) for value in bbox]
    return (x, y, x + w, y + h)


def clip_xyxy(box: tuple[float, float, float, float], width: int, height: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    x1 = min(max(0.0, float(x1)), float(width))
    y1 = min(max(0.0, float(y1)), float(height))
    x2 = min(max(0.0, float(x2)), float(width))
    y2 = min(max(0.0, float(y2)), float(height))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return (x1, y1, x2, y2)


def box_iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a.xyxy
    bx1, by1, bx2, by2 = b.xyxy
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a.area + b.area - inter
    return inter / union if union > 0.0 else 0.0


def load_cases(dataset_dir: Path, split: str) -> list[Case]:
    coco_path = dataset_dir / "annotations" / f"instances_{split}.json"
    coco = read_json(coco_path)
    gt_by_image: dict[int, list[Box]] = defaultdict(list)
    for ann in coco.get("annotations", []):
        image_id = int(ann["image_id"])
        gt_by_image[image_id].append(Box(bbox_xywh_to_xyxy(ann["bbox"])))

    cases: list[Case] = []
    for image in sorted(coco.get("images", []), key=lambda row: int(row["id"])):
        image_id = int(image["id"])
        image_path = dataset_dir / image["file_name"]
        cases.append(
            Case(
                image_id=image_id,
                case_id=str(image.get("case_id", Path(image["file_name"]).stem)),
                stain=str(image.get("stain", "unknown")),
                width=int(image["width"]),
                height=int(image["height"]),
                image_path=image_path,
                gt_boxes=tuple(gt_by_image.get(image_id, [])),
            )
        )
    if not cases:
        raise ValueError(f"No cases found in {coco_path}")
    return cases


def load_yolo_predictions(path: Path) -> dict[str, list[Box]]:
    raw = read_json(path)
    predictions: dict[str, list[Box]] = {}
    for case_id, records in raw.items():
        boxes: list[Box] = []
        for record in records:
            boxes.append(
                Box(
                    tuple(float(value) for value in record["xyxy"]),  # type: ignore[arg-type]
                    score=float(record.get("score", 1.0)),
                )
            )
        predictions[str(case_id)] = boxes
    return predictions


def load_rfdetr_predictions(path: Path, cases: list[Case]) -> dict[str, list[Box]]:
    raw = read_json(path)
    case_by_image_id = {case.image_id: case for case in cases}
    predictions: dict[str, list[Box]] = defaultdict(list)
    for record in raw:
        image_id = int(record["image_id"])
        case = case_by_image_id[image_id]
        box = clip_xyxy(bbox_xywh_to_xyxy(record["bbox"]), case.width, case.height)
        predictions[case.case_id].append(Box(box, score=float(record.get("score", 1.0))))
    return dict(predictions)


def greedy_match(gt_boxes: tuple[Box, ...], pred_boxes: list[Box], iou_threshold: float) -> tuple[int, list[int], list[int]]:
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    for pred_idx in sorted(range(len(pred_boxes)), key=lambda idx: pred_boxes[idx].score, reverse=True):
        best_gt = -1
        best_iou = 0.0
        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_idx in matched_gt:
                continue
            iou = box_iou(pred_boxes[pred_idx], gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt = gt_idx
        if best_gt >= 0 and best_iou >= iou_threshold:
            matched_gt.add(best_gt)
            matched_pred.add(pred_idx)
    unmatched_gt = [idx for idx in range(len(gt_boxes)) if idx not in matched_gt]
    unmatched_pred = [idx for idx in range(len(pred_boxes)) if idx not in matched_pred]
    return len(matched_gt), unmatched_gt, unmatched_pred


def evaluate_threshold(
    cases: list[Case],
    predictions: dict[str, list[Box]],
    *,
    threshold: float,
    match_iou: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    per_stain: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    totals: dict[str, float] = defaultdict(float)
    for case in cases:
        pred_boxes = [box for box in predictions.get(case.case_id, []) if box.score >= threshold]
        matched, unmatched_gt, unmatched_pred = greedy_match(case.gt_boxes, pred_boxes, match_iou)
        row = {
            "case_id": case.case_id,
            "stain": case.stain,
            "gt_count": len(case.gt_boxes),
            "prediction_count": len(pred_boxes),
            "matched_count": matched,
            "missed_count": len(unmatched_gt),
            "false_count": len(unmatched_pred),
        }
        rows.append(row)
        for bucket in (totals, per_stain[case.stain]):
            bucket["images"] += 1
            bucket["gt_count"] += row["gt_count"]
            bucket["prediction_count"] += row["prediction_count"]
            bucket["matched_count"] += row["matched_count"]
            bucket["missed_count"] += row["missed_count"]
            bucket["false_count"] += row["false_count"]

    def finalize(bucket: dict[str, float]) -> dict[str, Any]:
        images = int(bucket.get("images", 0))
        gt = int(bucket.get("gt_count", 0))
        pred = int(bucket.get("prediction_count", 0))
        matched = int(bucket.get("matched_count", 0))
        precision = matched / pred if pred else 0.0
        recall = matched / gt if gt else 0.0
        return {
            "images": images,
            "gt_count": gt,
            "prediction_count": pred,
            "matched_count": matched,
            "missed_count": int(bucket.get("missed_count", 0)),
            "false_count": int(bucket.get("false_count", 0)),
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "false_per_slide": bucket.get("false_count", 0.0) / images if images else 0.0,
            "missed_per_slide": bucket.get("missed_count", 0.0) / images if images else 0.0,
        }

    summary = finalize(totals)
    summary["threshold"] = threshold
    summary["match_iou"] = match_iou
    summary["per_stain"] = {stain: finalize(bucket) for stain, bucket in sorted(per_stain.items())}
    return summary, rows


def sweep_detector(
    cases: list[Case],
    spec: DetectorSpec,
    *,
    match_iou: float,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[float, list[dict[str, Any]]], dict[str, list[Box]]]:
    if spec.family == "yolo":
        predictions = load_yolo_predictions(spec.predictions_path)
    elif spec.family == "rfdetr":
        predictions = load_rfdetr_predictions(spec.predictions_path, cases)
    else:
        raise ValueError(f"Unknown detector family: {spec.family}")

    rows: list[dict[str, Any]] = []
    rows_by_threshold: dict[float, list[dict[str, Any]]] = {}
    details: dict[str, Any] = {
        "name": spec.name,
        "family": spec.family,
        "predictions_path": str(spec.predictions_path),
        "thresholds": list(spec.thresholds),
        "threshold_metrics": {},
    }
    for threshold in spec.thresholds:
        metrics, per_case_rows = evaluate_threshold(cases, predictions, threshold=threshold, match_iou=match_iou)
        rows_by_threshold[threshold] = per_case_rows
        flat = flatten_threshold_metrics(spec.name, spec.family, metrics)
        rows.append(flat)
        details["threshold_metrics"][f"{threshold:.3f}"] = metrics
    return rows, details, rows_by_threshold, predictions


def flatten_threshold_metrics(name: str, family: str, metrics: dict[str, Any]) -> dict[str, Any]:
    sv40 = metrics.get("per_stain", {}).get("SV40", {})
    return {
        "model": name,
        "family": family,
        "threshold": f"{float(metrics['threshold']):.3f}",
        "images": metrics["images"],
        "gt_count": metrics["gt_count"],
        "prediction_count": metrics["prediction_count"],
        "matched_count": metrics["matched_count"],
        "missed_count": metrics["missed_count"],
        "false_count": metrics["false_count"],
        "recall": f"{metrics['recall']:.6f}",
        "precision": f"{metrics['precision']:.6f}",
        "f1": f"{metrics['f1']:.6f}",
        "false_per_slide": f"{metrics['false_per_slide']:.6f}",
        "missed_per_slide": f"{metrics['missed_per_slide']:.6f}",
        "sv40_recall": f"{sv40.get('recall', 0.0):.6f}",
        "sv40_gt_count": sv40.get("gt_count", 0),
        "sv40_matched_count": sv40.get("matched_count", 0),
    }


def draw_box(draw: ImageDraw.ImageDraw, box: Box, color: tuple[int, int, int], label: str, width: int, y_offset: int) -> None:
    x1, y1, x2, y2 = box.xyxy
    shifted = (x1, y1 + y_offset, x2, y2 + y_offset)
    for offset in range(width):
        draw.rectangle(
            (shifted[0] - offset, shifted[1] - offset, shifted[2] + offset, shifted[3] + offset),
            outline=color,
        )
    draw.text((shifted[0] + 2, max(0, shifted[1] - 13)), label, fill=color)


def write_overlay_pdf(
    cases: list[Case],
    predictions: dict[str, list[Box]],
    per_case_rows: list[dict[str, Any]],
    output_dir: Path,
    pdf_path: Path,
    *,
    threshold: float,
    max_pages: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_by_case = {row["case_id"]: row for row in per_case_rows}
    ordered_cases = sorted(
        cases,
        key=lambda case: (
            rows_by_case[case.case_id]["missed_count"] == 0,
            rows_by_case[case.case_id]["false_count"] == 0,
            -rows_by_case[case.case_id]["gt_count"],
            case.case_id,
        ),
    )
    image_paths: list[Path] = []
    font = ImageFont.load_default()
    for case in ordered_cases[:max_pages]:
        with Image.open(case.image_path).convert("RGB") as base:
            header_h = 96
            canvas = Image.new("RGB", (base.width, base.height + header_h), (255, 255, 255))
            canvas.paste(base, (0, header_h))
        draw = ImageDraw.Draw(canvas)
        row = rows_by_case[case.case_id]
        title = (
            f"{case.case_id} | {case.stain} | threshold={threshold:.3f} | "
            f"GT={row['gt_count']} pred={row['prediction_count']} match={row['matched_count']} "
            f"miss={row['missed_count']} false={row['false_count']}"
        )
        draw.text((10, 8), title, fill=(0, 0, 0), font=font)
        draw.text((10, 30), "green=ground truth, red=prediction", fill=(0, 0, 0), font=font)
        for idx, gt_box in enumerate(case.gt_boxes, start=1):
            draw_box(draw, gt_box, (0, 150, 0), f"G{idx}", 3, header_h)
        kept = [box for box in predictions.get(case.case_id, []) if box.score >= threshold]
        for idx, pred_box in enumerate(sorted(kept, key=lambda box: box.score, reverse=True), start=1):
            draw_box(draw, pred_box, (220, 0, 0), f"P{idx} {pred_box.score:.2f}", 3, header_h)
        if canvas.width > 1800:
            scale = 1800 / canvas.width
            canvas = canvas.resize((1800, int(canvas.height * scale)))
        path = output_dir / f"{case.case_id}_threshold_{threshold:.3f}.png"
        canvas.save(path)
        image_paths.append(path)

    pages = [Image.open(path).convert("RGB") for path in image_paths]
    try:
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        if pages:
            pages[0].save(pdf_path, save_all=True, append_images=pages[1:])
    finally:
        for page in pages:
            page.close()


def threshold_lookup(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    key = f"{threshold:.3f}"
    for row in rows:
        if row["threshold"] == key:
            return row
    raise ValueError(f"No row found for threshold {threshold}")


def render_report(
    output_dir: Path,
    *,
    generated_at: str,
    cases: list[Case],
    detector_rows: dict[str, list[dict[str, Any]]],
    low_high_rows: list[dict[str, Any]],
    yolo_name: str,
    rfdetr_name: str,
) -> None:
    gt_total = sum(len(case.gt_boxes) for case in cases)
    best_yolo = max(detector_rows[yolo_name], key=lambda row: int(row["matched_count"]))
    best_rfdetr = max(detector_rows[rfdetr_name], key=lambda row: int(row["matched_count"]))
    all_rows = [row for rows in detector_rows.values() for row in rows]
    reaches_all = [row for row in all_rows if int(row["matched_count"]) == gt_total]

    lines = [
        "# Recall-first probability threshold readout",
        "",
        f"Generated: {generated_at}",
        "",
        (
            "Match rule: greedy one-to-one IoU >= 0.50 against the exported "
            f"pilot-100 test labels. Test set has {len(cases)} thumbnails and {gt_total} GT boxes."
        ),
        "",
        (
            "Important floor limitation: YOLO saved predictions start at confidence 0.10; "
            "RF-DETR saved predictions start at score 0.05. No lower-score predictions exist in these artifacts."
        ),
        "",
        "## Answer",
        "",
    ]
    if reaches_all:
        reached = ", ".join(f"{row['model']}@{row['threshold']}" for row in reaches_all)
        lines.append(f"- Yes, saved thresholds with all {gt_total} GT boxes detected: {reached}.")
    else:
        lines.append(f"- No saved probability threshold reaches all {gt_total} GT boxes.")
    lines.extend(
        [
            (
                f"- YOLO best saved recall is {best_yolo['matched_count']}/{gt_total} "
                f"at confidence {best_yolo['threshold']}."
            ),
            (
                f"- RF-DETR Large best saved recall is {best_rfdetr['matched_count']}/{gt_total} "
                f"at score {best_rfdetr['threshold']}, but this produces "
                f"{best_rfdetr['prediction_count']} predictions and "
                f"{float(best_rfdetr['false_per_slide']):.1f} false boxes per slide."
            ),
            "",
            "## Low/high thresholds",
            "",
            (
                "| model | threshold | matched/gt | recall | precision | false/slide | "
                "missed/slide | SV40 recall |"
            ),
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in low_high_rows:
        lines.append(
            f"| {row['model']} | {float(row['threshold']):.3f} | "
            f"{row['matched_count']}/{row['gt_count']} | {float(row['recall']):.3f} | "
            f"{float(row['precision']):.3f} | {float(row['false_per_slide']):.1f} | "
            f"{float(row['missed_per_slide']):.1f} | {float(row['sv40_recall']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Review PDFs",
            "",
            f"- YOLO low: `{yolo_name}/overlays_threshold_0.100.pdf`",
            f"- YOLO high: `{yolo_name}/overlays_threshold_0.500.pdf`",
            f"- RF-DETR Large low: `{rfdetr_name}/overlays_threshold_0.050.pdf`",
            f"- RF-DETR Large high: `{rfdetr_name}/overlays_threshold_0.500.pdf`",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_reproduction(output_dir: Path, args: argparse.Namespace, generated_at: str) -> None:
    command = " ".join([sys.executable, "scripts/report_detector_recall_thresholds.py"] + sys.argv[1:])
    text = f"""# Recall threshold readout reproduction

Generated: {generated_at}
Repository: {repo_root()}
Git commit: {run_text(["git", "rev-parse", "HEAD"], cwd=repo_root())}
Python executable: {sys.executable}

Command:
{command}

Input dataset: {args.dataset_dir.resolve()}
YOLO predictions: {args.yolo_predictions.resolve()}
RF-DETR Large predictions: {args.rfdetr_large_predictions.resolve()}
RF-DETR Small predictions: {args.rfdetr_small_predictions.resolve()}

This artifact is generated from saved prediction JSON files without model inference or retraining.
YOLO saved prediction floor is 0.10; RF-DETR saved prediction floor is 0.05.
"""
    (output_dir / "reproduction.txt").write_text(text, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--yolo-name", default="yolo11n_default_1024")
    parser.add_argument("--yolo-predictions", required=True, type=Path)
    parser.add_argument("--rfdetr-large-name", default="rfdetr_large_recall_best")
    parser.add_argument("--rfdetr-large-predictions", required=True, type=Path)
    parser.add_argument("--rfdetr-small-name", default="rfdetr_small_map_best")
    parser.add_argument("--rfdetr-small-predictions", required=True, type=Path)
    parser.add_argument("--yolo-thresholds", default="0.10,0.25,0.50,0.90")
    parser.add_argument("--rfdetr-thresholds", default="0.05,0.10,0.25,0.50,0.90")
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--max-pages", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists; pass --overwrite to replace: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_at = now_iso()

    cases = load_cases(args.dataset_dir.resolve(), args.split)
    specs = [
        DetectorSpec(
            name=args.yolo_name,
            family="yolo",
            predictions_path=args.yolo_predictions.resolve(),
            thresholds=parse_thresholds(args.yolo_thresholds),
            low_threshold=0.10,
            high_threshold=0.50,
        ),
        DetectorSpec(
            name=args.rfdetr_large_name,
            family="rfdetr",
            predictions_path=args.rfdetr_large_predictions.resolve(),
            thresholds=parse_thresholds(args.rfdetr_thresholds),
            low_threshold=0.05,
            high_threshold=0.50,
        ),
        DetectorSpec(
            name=args.rfdetr_small_name,
            family="rfdetr",
            predictions_path=args.rfdetr_small_predictions.resolve(),
            thresholds=parse_thresholds(args.rfdetr_thresholds),
            make_pdf=False,
        ),
    ]

    yolo_sweep_rows: list[dict[str, Any]] = []
    rfdetr_sweep_rows: list[dict[str, Any]] = []
    detector_rows: dict[str, list[dict[str, Any]]] = {}
    low_high_rows: list[dict[str, Any]] = []

    for spec in specs:
        rows, details, rows_by_threshold, predictions = sweep_detector(cases, spec, match_iou=args.match_iou)
        detector_rows[spec.name] = rows
        model_dir = output_dir / spec.name
        write_json(model_dir / "threshold_sweep.json", details)
        write_csv(model_dir / "threshold_sweep.csv", rows)
        if spec.family == "yolo":
            yolo_sweep_rows.extend(rows)
        else:
            rfdetr_sweep_rows.extend(rows)
        if spec.low_threshold is not None:
            low_high_rows.append(threshold_lookup(rows, spec.low_threshold))
        if spec.name == args.rfdetr_large_name:
            low_high_rows.append(threshold_lookup(rows, 0.25))
        if spec.high_threshold is not None:
            low_high_rows.append(threshold_lookup(rows, spec.high_threshold))
        if spec.name == args.yolo_name:
            low_high_rows.append(threshold_lookup(rows, 0.90))
        if spec.name == args.rfdetr_large_name:
            low_high_rows.append(threshold_lookup(rows, 0.90))
        if spec.make_pdf:
            for threshold in (spec.low_threshold, spec.high_threshold):
                if threshold is None:
                    continue
                write_overlay_pdf(
                    cases,
                    predictions,
                    rows_by_threshold[threshold],
                    model_dir / f"overlays_threshold_{threshold:.3f}",
                    model_dir / f"overlays_threshold_{threshold:.3f}.pdf",
                    threshold=threshold,
                    max_pages=args.max_pages,
                )

    write_csv(output_dir / "yolo_threshold_sweep.csv", yolo_sweep_rows)
    write_csv(output_dir / "rfdetr_threshold_sweep.csv", rfdetr_sweep_rows)
    write_csv(output_dir / "low_high_probability_summary.csv", low_high_rows)
    render_report(
        output_dir,
        generated_at=generated_at,
        cases=cases,
        detector_rows=detector_rows,
        low_high_rows=low_high_rows,
        yolo_name=args.yolo_name,
        rfdetr_name=args.rfdetr_large_name,
    )
    write_reproduction(output_dir, args, generated_at)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
