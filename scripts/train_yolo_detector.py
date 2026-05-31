#!/usr/bin/env python3
"""Train and evaluate an Ultralytics YOLO detector on exported WSI thumbnails."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont
import yaml


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


@dataclass(frozen=True)
class CaseRecord:
    case_id: str
    split: str
    stain: str
    group_id: str
    image_path: Path
    label_path: Path
    width: int
    height: int


@dataclass(frozen=True)
class BoxRecord:
    xyxy: tuple[float, float, float, float]
    cls: int = 0
    score: float = 1.0

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.xyxy
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _run_text(cmd: list[str], cwd: Path | None = None) -> str:
    try:
        return subprocess.check_output(cmd, cwd=cwd, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
    return value


def yolo_xywhn_to_xyxy_pixels(
    cls_and_box: list[float],
    width: int,
    height: int,
) -> BoxRecord:
    cls = int(cls_and_box[0])
    x_center, y_center, box_width, box_height = cls_and_box[1:5]
    x1 = (x_center - box_width / 2.0) * width
    y1 = (y_center - box_height / 2.0) * height
    x2 = (x_center + box_width / 2.0) * width
    y2 = (y_center + box_height / 2.0) * height
    return BoxRecord(
        (
            max(0.0, min(float(width), x1)),
            max(0.0, min(float(height), y1)),
            max(0.0, min(float(width), x2)),
            max(0.0, min(float(height), y2)),
        ),
        cls=cls,
        score=1.0,
    )


def box_iou(a: BoxRecord, b: BoxRecord) -> float:
    ax1, ay1, ax2, ay2 = a.xyxy
    bx1, by1, bx2, by2 = b.xyxy
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    union = a.area + b.area - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def read_yolo_labels(label_path: Path, width: int, height: int) -> list[BoxRecord]:
    boxes: list[BoxRecord] = []
    if not label_path.exists():
        return boxes
    for line_no, line in enumerate(label_path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{label_path}:{line_no} expected 5 YOLO fields, got {len(parts)}")
        values = [float(part) for part in parts]
        if any(value < 0.0 or value > 1.0 for value in values[1:]):
            raise ValueError(f"{label_path}:{line_no} YOLO coordinates outside [0,1]")
        if values[3] <= 0.0 or values[4] <= 0.0:
            raise ValueError(f"{label_path}:{line_no} YOLO width/height must be positive")
        boxes.append(yolo_xywhn_to_xyxy_pixels(values, width, height))
    return boxes


def greedy_match(
    gt_boxes: list[BoxRecord],
    pred_boxes: list[BoxRecord],
    iou_threshold: float,
) -> tuple[list[dict[str, float | int]], list[int], list[int]]:
    matches: list[dict[str, float | int]] = []
    used_gt: set[int] = set()
    ordered_pred_indices = sorted(
        range(len(pred_boxes)),
        key=lambda idx: pred_boxes[idx].score,
        reverse=True,
    )
    unmatched_pred: list[int] = []
    for pred_idx in ordered_pred_indices:
        best_gt_idx = -1
        best_iou = 0.0
        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_idx in used_gt:
                continue
            iou = box_iou(pred_boxes[pred_idx], gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        if best_gt_idx >= 0 and best_iou >= iou_threshold:
            used_gt.add(best_gt_idx)
            matches.append({"pred_idx": pred_idx, "gt_idx": best_gt_idx, "iou": best_iou})
        else:
            unmatched_pred.append(pred_idx)
    unmatched_gt = [idx for idx in range(len(gt_boxes)) if idx not in used_gt]
    return matches, unmatched_gt, unmatched_pred


def _load_dataset_yaml(dataset_dir: Path) -> dict[str, Any]:
    yaml_path = dataset_dir / "dataset.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"Missing dataset.yaml: {yaml_path}")
    data = yaml.safe_load(yaml_path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"dataset.yaml must parse to a mapping: {yaml_path}")
    return data


def _resolve_dataset_path(dataset_dir: Path, value: str | os.PathLike[str]) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    yaml_root = Path(_load_dataset_yaml(dataset_dir).get("path", dataset_dir))
    if not yaml_root.is_absolute():
        yaml_root = dataset_dir / yaml_root
    return yaml_root / path


def _image_paths(images_dir: Path) -> list[Path]:
    if not images_dir.exists():
        return []
    return sorted(path for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


def verify_dataset(dataset_dir: Path) -> dict[str, Any]:
    dataset_yaml = _load_dataset_yaml(dataset_dir)
    names = dataset_yaml.get("names", {})
    summary: dict[str, Any] = {
        "dataset_dir": str(dataset_dir),
        "dataset_yaml": str(dataset_dir / "dataset.yaml"),
        "names": names,
        "splits": {},
        "image_count": 0,
        "label_file_count": 0,
        "label_row_count": 0,
        "errors": [],
    }
    for split in ("train", "val", "test"):
        rel = dataset_yaml.get(split)
        if not rel:
            continue
        images_dir = _resolve_dataset_path(dataset_dir, rel)
        labels_dir = dataset_dir / "labels" / split
        split_summary = {
            "images_dir": str(images_dir),
            "labels_dir": str(labels_dir),
            "image_count": 0,
            "label_file_count": 0,
            "label_row_count": 0,
            "missing_labels": [],
        }
        for image_path in _image_paths(images_dir):
            split_summary["image_count"] += 1
            label_path = labels_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                split_summary["missing_labels"].append(str(label_path))
                continue
            with Image.open(image_path) as img:
                width, height = img.size
            rows = read_yolo_labels(label_path, width, height)
            split_summary["label_file_count"] += 1
            split_summary["label_row_count"] += len(rows)
        if split_summary["missing_labels"]:
            summary["errors"].extend(split_summary["missing_labels"])
        summary["splits"][split] = split_summary
        summary["image_count"] += split_summary["image_count"]
        summary["label_file_count"] += split_summary["label_file_count"]
        summary["label_row_count"] += split_summary["label_row_count"]
    cases_csv = dataset_dir / "manifests" / "cases.csv"
    if cases_csv.exists():
        groups_by_split: dict[str, set[str]] = defaultdict(set)
        with cases_csv.open(newline="") as handle:
            for row in csv.DictReader(handle):
                groups_by_split[row["group_id"]].add(row["split"])
        leaking_groups = sorted(group for group, splits in groups_by_split.items() if len(splits) > 1)
        summary["group_count"] = len(groups_by_split)
        summary["group_leak_count"] = len(leaking_groups)
        summary["group_leaks"] = leaking_groups
        if leaking_groups:
            summary["errors"].extend(f"group split leak: {group}" for group in leaking_groups)
    summary["status"] = "ok" if not summary["errors"] else "error"
    return summary


def load_cases(dataset_dir: Path, split: str) -> list[CaseRecord]:
    cases_csv = dataset_dir / "manifests" / "cases.csv"
    if not cases_csv.exists():
        raise FileNotFoundError(f"Missing cases manifest: {cases_csv}")
    cases: list[CaseRecord] = []
    with cases_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["split"] != split:
                continue
            image_path = Path(row["image_path"])
            label_path = dataset_dir / "labels" / split / f"{image_path.stem}.txt"
            cases.append(
                CaseRecord(
                    case_id=row["case_id"],
                    split=row["split"],
                    stain=row.get("stain") or row["case_id"].split("_patient_")[0].upper(),
                    group_id=row.get("group_id", ""),
                    image_path=image_path,
                    label_path=label_path,
                    width=int(float(row["width"])),
                    height=int(float(row["height"])),
                )
            )
    if not cases:
        raise ValueError(f"No cases found for split={split!r} in {cases_csv}")
    return cases


def evaluate_project_metrics(
    cases: list[CaseRecord],
    predictions_by_case: dict[str, list[BoxRecord]],
    *,
    match_iou: float,
    fragment_iou: float,
    oversized_area_ratio: float,
    large_area_fraction: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    per_case_rows: list[dict[str, Any]] = []
    aggregate = _empty_metric_bucket()
    per_stain: dict[str, dict[str, Any]] = defaultdict(_empty_metric_bucket)

    for case in cases:
        gt_boxes = read_yolo_labels(case.label_path, case.width, case.height)
        pred_boxes = predictions_by_case.get(case.case_id, [])
        matches, unmatched_gt, unmatched_pred = greedy_match(gt_boxes, pred_boxes, match_iou)
        image_area = float(case.width * case.height)
        matched_ious = [float(match["iou"]) for match in matches]
        oversized_count = 0
        overmerged_count = 0
        for match in matches:
            pred = pred_boxes[int(match["pred_idx"])]
            gt = gt_boxes[int(match["gt_idx"])]
            area_ratio = pred.area / gt.area if gt.area > 0.0 else math.inf
            if area_ratio >= oversized_area_ratio:
                oversized_count += 1
            overlapping_gt = sum(1 for other_gt in gt_boxes if box_iou(pred, other_gt) >= fragment_iou)
            if overlapping_gt > 1:
                overmerged_count += 1
        large_pred_count = sum(1 for pred in pred_boxes if pred.area / image_area >= large_area_fraction)
        duplicate_pred_count = 0
        fragment_pred_count = 0
        for pred_idx in unmatched_pred:
            pred = pred_boxes[pred_idx]
            max_iou_any_gt = max((box_iou(pred, gt) for gt in gt_boxes), default=0.0)
            if max_iou_any_gt >= match_iou:
                duplicate_pred_count += 1
            elif max_iou_any_gt >= fragment_iou:
                fragment_pred_count += 1
        row = {
            "case_id": case.case_id,
            "split": case.split,
            "stain": case.stain,
            "group_id": case.group_id,
            "gt_boxes": len(gt_boxes),
            "pred_boxes": len(pred_boxes),
            "matched_boxes": len(matches),
            "missed_gt_boxes": len(unmatched_gt),
            "false_pred_boxes": len(unmatched_pred),
            "mean_matched_iou": sum(matched_ious) / len(matched_ious) if matched_ious else 0.0,
            "oversized_matched_predictions": oversized_count,
            "large_area_predictions": large_pred_count,
            "duplicate_predictions": duplicate_pred_count,
            "fragment_predictions": fragment_pred_count,
            "overmerged_predictions": overmerged_count,
        }
        per_case_rows.append(row)
        _add_case_to_bucket(aggregate, row)
        _add_case_to_bucket(per_stain[case.stain], row)

    summary = _finalize_bucket(aggregate)
    summary["per_stain"] = {stain: _finalize_bucket(bucket) for stain, bucket in sorted(per_stain.items())}
    summary["match_iou"] = match_iou
    summary["fragment_iou"] = fragment_iou
    summary["oversized_area_ratio"] = oversized_area_ratio
    summary["large_area_fraction"] = large_area_fraction
    return summary, per_case_rows


def parse_thresholds(raw: str, prediction_floor: float) -> list[float]:
    thresholds: list[float] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        value = float(part)
        if value < prediction_floor:
            raise ValueError(
                f"metric threshold {value} is lower than --conf {prediction_floor}; "
                "rerun prediction with a lower --conf first"
            )
        thresholds.append(value)
    thresholds.append(prediction_floor)
    return sorted(set(round(value, 6) for value in thresholds))


def filter_predictions_by_conf(
    predictions_by_case: dict[str, list[BoxRecord]],
    threshold: float,
) -> dict[str, list[BoxRecord]]:
    return {
        case_id: [box for box in boxes if box.score >= threshold]
        for case_id, boxes in predictions_by_case.items()
    }


def _empty_metric_bucket() -> dict[str, Any]:
    return {
        "images": 0,
        "gt_boxes": 0,
        "pred_boxes": 0,
        "matched_boxes": 0,
        "missed_gt_boxes": 0,
        "false_pred_boxes": 0,
        "slides_with_missed_gt": 0,
        "slides_with_false_pred": 0,
        "oversized_matched_predictions": 0,
        "large_area_predictions": 0,
        "duplicate_predictions": 0,
        "fragment_predictions": 0,
        "overmerged_predictions": 0,
        "_matched_iou_sum": 0.0,
        "_matched_iou_count": 0,
    }


def _add_case_to_bucket(bucket: dict[str, Any], row: dict[str, Any]) -> None:
    bucket["images"] += 1
    for key in (
        "gt_boxes",
        "pred_boxes",
        "matched_boxes",
        "missed_gt_boxes",
        "false_pred_boxes",
        "oversized_matched_predictions",
        "large_area_predictions",
        "duplicate_predictions",
        "fragment_predictions",
        "overmerged_predictions",
    ):
        bucket[key] += int(row[key])
    if row["missed_gt_boxes"]:
        bucket["slides_with_missed_gt"] += 1
    if row["false_pred_boxes"]:
        bucket["slides_with_false_pred"] += 1
    if row["matched_boxes"]:
        bucket["_matched_iou_sum"] += float(row["mean_matched_iou"]) * int(row["matched_boxes"])
        bucket["_matched_iou_count"] += int(row["matched_boxes"])


def _finalize_bucket(bucket: dict[str, Any]) -> dict[str, Any]:
    out = dict(bucket)
    matched = out["matched_boxes"]
    pred = out["pred_boxes"]
    gt = out["gt_boxes"]
    out["precision_at_match_iou"] = matched / pred if pred else 0.0
    out["recall_at_match_iou"] = matched / gt if gt else 0.0
    p = out["precision_at_match_iou"]
    r = out["recall_at_match_iou"]
    out["f1_at_match_iou"] = 2 * p * r / (p + r) if p + r else 0.0
    out["false_boxes_per_slide"] = out["false_pred_boxes"] / out["images"] if out["images"] else 0.0
    out["missed_boxes_per_slide"] = out["missed_gt_boxes"] / out["images"] if out["images"] else 0.0
    out["mean_matched_iou"] = (
        out["_matched_iou_sum"] / out["_matched_iou_count"] if out["_matched_iou_count"] else 0.0
    )
    out.pop("_matched_iou_sum", None)
    out.pop("_matched_iou_count", None)
    return out


def _draw_box(
    draw: ImageDraw.ImageDraw,
    box: BoxRecord,
    color: tuple[int, int, int],
    label: str,
    width: int,
    font: ImageFont.ImageFont,
) -> None:
    x1, y1, x2, y2 = box.xyxy
    for offset in range(width):
        draw.rectangle((x1 - offset, y1 - offset, x2 + offset, y2 + offset), outline=color)
    text_bbox = draw.textbbox((x1, max(0, y1 - 12)), label, font=font)
    draw.rectangle(text_bbox, fill=(255, 255, 255))
    draw.text((x1, max(0, y1 - 12)), label, fill=color, font=font)


def write_overlay_pages(
    cases: list[CaseRecord],
    predictions_by_case: dict[str, list[BoxRecord]],
    per_case_rows: list[dict[str, Any]],
    output_dir: Path,
    *,
    max_pages: int,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_by_case = {row["case_id"]: row for row in per_case_rows}
    ordered_cases = sorted(
        cases,
        key=lambda case: (
            rows_by_case[case.case_id]["missed_gt_boxes"] == 0,
            rows_by_case[case.case_id]["false_pred_boxes"] == 0,
            -rows_by_case[case.case_id]["gt_boxes"],
            case.case_id,
        ),
    )
    font = ImageFont.load_default()
    written: list[Path] = []
    for case in ordered_cases[:max_pages]:
        with Image.open(case.image_path).convert("RGB") as img:
            canvas = Image.new("RGB", (img.width, img.height + 88), (255, 255, 255))
            canvas.paste(img, (0, 88))
        draw = ImageDraw.Draw(canvas)
        row = rows_by_case[case.case_id]
        title = (
            f"{case.case_id} | {case.stain} | "
            f"GT={row['gt_boxes']} pred={row['pred_boxes']} match={row['matched_boxes']} "
            f"miss={row['missed_gt_boxes']} false={row['false_pred_boxes']}"
        )
        draw.text((10, 8), title, fill=(0, 0, 0), font=font)
        draw.text((10, 28), "green=ground truth, red=YOLO prediction", fill=(0, 0, 0), font=font)
        shifted_gt = [
            BoxRecord((box.xyxy[0], box.xyxy[1] + 88, box.xyxy[2], box.xyxy[3] + 88), box.cls, box.score)
            for box in read_yolo_labels(case.label_path, case.width, case.height)
        ]
        shifted_pred = [
            BoxRecord((box.xyxy[0], box.xyxy[1] + 88, box.xyxy[2], box.xyxy[3] + 88), box.cls, box.score)
            for box in predictions_by_case.get(case.case_id, [])
        ]
        for idx, gt_box in enumerate(shifted_gt, start=1):
            _draw_box(draw, gt_box, (0, 150, 0), f"G{idx}", 3, font)
        for idx, pred_box in enumerate(shifted_pred, start=1):
            _draw_box(draw, pred_box, (220, 0, 0), f"P{idx} {pred_box.score:.2f}", 3, font)
        max_width = 1800
        if canvas.width > max_width:
            scale = max_width / canvas.width
            canvas = canvas.resize((max_width, int(canvas.height * scale)))
        out_path = output_dir / f"{case.case_id}_overlay.png"
        canvas.save(out_path)
        written.append(out_path)
    return written


def write_overlay_pdf(overlay_paths: list[Path], pdf_path: Path) -> None:
    if not overlay_paths:
        return
    pages = [Image.open(path).convert("RGB") for path in overlay_paths]
    try:
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        pages[0].save(pdf_path, save_all=True, append_images=pages[1:])
    finally:
        for page in pages:
            page.close()


def collect_environment() -> dict[str, str]:
    env = {
        "python": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
        "git_commit": _run_text(["git", "rev-parse", "HEAD"], cwd=_repo_root()),
    }
    for package_name in ("ultralytics", "torch", "PIL", "yaml"):
        try:
            module = __import__(package_name)
            env[package_name] = str(getattr(module, "__version__", "installed"))
        except Exception as exc:
            env[package_name] = f"unavailable: {exc}"
    try:
        import torch

        env["torch_cuda_available"] = str(torch.cuda.is_available())
        env["torch_cuda_device_count"] = str(torch.cuda.device_count())
        if torch.cuda.is_available():
            env["torch_cuda_device_0"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return env


def write_reproduction(
    output_root: Path,
    args: argparse.Namespace,
    validation: dict[str, Any],
    env: dict[str, str],
    train_dir: Path,
    best_weights: Path,
) -> None:
    command = " ".join(_shell_quote(part) for part in sys.argv)
    text = f"""# PER-241 Ultralytics YOLO Detector Distillation

Created: {_now_iso()}
Ticket: PER-241
Repository: {_repo_root()}
Git commit: {env.get("git_commit", "unknown")}

## Command

```bash
{command}
```

## Dataset

- Dataset root: {Path(args.dataset_dir).resolve()}
- Dataset YAML: {Path(args.dataset_dir).resolve() / "dataset.yaml"}
- Verification status: {validation.get("status")}
- Images: {validation.get("image_count")}
- YOLO label rows: {validation.get("label_row_count")}

## Environment

```json
{json.dumps(env, indent=2, sort_keys=True)}
```

## Outputs

- Output root: {output_root.resolve()}
- Ultralytics train directory: {train_dir.resolve()}
- Best weights: {best_weights.resolve()}
- Metrics summary: {(output_root / "metrics_summary.json").resolve()}
- Project error metrics: {(output_root / "project_error_metrics.json").resolve()}
- Threshold sweep: {(output_root / "project_error_metrics_by_conf.json").resolve()}
- Per-case metrics: {(output_root / "per_case_project_metrics.csv").resolve()}
- Prediction overlays: {(output_root / "review" / "prediction_overlays.pdf").resolve()}

## Notes

The project-specific metrics compare YOLO predictions on the selected split
against the exported detector-pipeline labels. This measures distillation
agreement with the VLM-derived labels, not independent pathologist truth.
"""
    (output_root / "reproduction.txt").write_text(text)


def _shell_quote(value: str) -> str:
    if not value:
        return "''"
    safe = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_+-=.,/:")
    if all(char in safe for char in value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


def train_and_evaluate(args: argparse.Namespace) -> dict[str, Any]:
    dataset_dir = Path(args.dataset_dir).resolve()
    output_root = Path(args.output_root).resolve()
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output root exists; pass --overwrite to replace: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    validation = verify_dataset(dataset_dir)
    (output_root / "dataset_validation.json").write_text(json.dumps(validation, indent=2, sort_keys=True))
    if validation["status"] != "ok":
        raise ValueError(f"Dataset validation failed; see {output_root / 'dataset_validation.json'}")

    config = vars(args).copy()
    config["created_at"] = _now_iso()
    config["dataset_dir"] = str(dataset_dir)
    config["output_root"] = str(output_root)
    (output_root / "config.json").write_text(json.dumps(_jsonable(config), indent=2, sort_keys=True))

    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError(
            "Ultralytics is not importable. Install it in the active environment, "
            "for example: python -m pip install ultralytics"
        ) from exc

    dataset_yaml = dataset_dir / "dataset.yaml"
    model = YOLO(args.model)
    train_results = model.train(
        data=str(dataset_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        seed=args.seed,
        deterministic=True,
        cache=args.cache,
        project=str(output_root / "ultralytics"),
        name="train",
        exist_ok=True,
        plots=True,
        verbose=True,
    )
    train_dir = Path(getattr(model.trainer, "save_dir", output_root / "ultralytics" / "train"))
    if not train_dir.exists() and hasattr(train_results, "save_dir"):
        train_dir = Path(train_results.save_dir)
    best_weights = train_dir / "weights" / "best.pt"
    if not best_weights.exists():
        best_weights = train_dir / "weights" / "last.pt"
    if not best_weights.exists():
        raise FileNotFoundError(f"Could not find trained weights under {train_dir / 'weights'}")

    trained = YOLO(str(best_weights))
    val_metrics = trained.val(
        data=str(dataset_yaml),
        split=args.eval_split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(output_root / "ultralytics"),
        name=f"val_{args.eval_split}",
        exist_ok=True,
        plots=True,
        save_json=False,
        verbose=True,
    )
    ultralytics_metrics = getattr(val_metrics, "results_dict", {})
    (output_root / "ultralytics_metrics.json").write_text(
        json.dumps(_jsonable(ultralytics_metrics), indent=2, sort_keys=True)
    )

    cases = load_cases(dataset_dir, args.eval_split)
    predictions_by_case = predict_cases(
        trained,
        cases,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.nms_iou,
        max_det=args.max_det,
        device=args.device,
    )
    write_predictions_json(output_root / "predictions.json", predictions_by_case)
    threshold_metrics: dict[str, Any] = {}
    project_metrics: dict[str, Any] | None = None
    per_case_rows: list[dict[str, Any]] | None = None
    for threshold in parse_thresholds(args.metric_conf_thresholds, args.conf):
        filtered_predictions = filter_predictions_by_conf(predictions_by_case, threshold)
        metrics_for_threshold, rows_for_threshold = evaluate_project_metrics(
            cases,
            filtered_predictions,
            match_iou=args.match_iou,
            fragment_iou=args.fragment_iou,
            oversized_area_ratio=args.oversized_area_ratio,
            large_area_fraction=args.large_area_fraction,
        )
        metrics_for_threshold["prediction_conf_threshold"] = threshold
        threshold_metrics[f"{threshold:.3f}"] = metrics_for_threshold
        if abs(threshold - args.conf) < 1e-9:
            project_metrics = metrics_for_threshold
            per_case_rows = rows_for_threshold
    if project_metrics is None or per_case_rows is None:
        raise RuntimeError("Internal error: did not compute project metrics at --conf")
    (output_root / "project_error_metrics.json").write_text(
        json.dumps(_jsonable(project_metrics), indent=2, sort_keys=True)
    )
    (output_root / "project_error_metrics_by_conf.json").write_text(
        json.dumps(_jsonable(threshold_metrics), indent=2, sort_keys=True)
    )
    write_per_case_csv(output_root / "per_case_project_metrics.csv", per_case_rows)
    overlay_paths = write_overlay_pages(
        cases,
        predictions_by_case,
        per_case_rows,
        output_root / "review" / "overlays",
        max_pages=args.max_review_pages,
    )
    write_overlay_pdf(overlay_paths, output_root / "review" / "prediction_overlays.pdf")

    env = collect_environment()
    metrics_summary = {
        "created_at": _now_iso(),
        "ticket": "PER-241",
        "dataset_dir": str(dataset_dir),
        "output_root": str(output_root),
        "model": args.model,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "eval_split": args.eval_split,
        "prediction_conf": args.conf,
        "nms_iou": args.nms_iou,
        "best_weights": str(best_weights),
        "train_dir": str(train_dir),
        "dataset_validation": validation,
        "ultralytics_metrics": _jsonable(ultralytics_metrics),
        "project_metrics": project_metrics,
        "project_metrics_by_conf": threshold_metrics,
        "environment": env,
    }
    (output_root / "metrics_summary.json").write_text(
        json.dumps(_jsonable(metrics_summary), indent=2, sort_keys=True)
    )
    write_reproduction(output_root, args, validation, env, train_dir, best_weights)
    return metrics_summary


def predict_cases(
    model: Any,
    cases: list[CaseRecord],
    *,
    imgsz: int,
    conf: float,
    iou: float,
    max_det: int,
    device: str,
) -> dict[str, list[BoxRecord]]:
    predictions: dict[str, list[BoxRecord]] = {}
    results = model.predict(
        source=[str(case.image_path) for case in cases],
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        max_det=max_det,
        device=device,
        save=False,
        stream=True,
        verbose=False,
    )
    for case, result in zip(cases, results):
        boxes: list[BoxRecord] = []
        if getattr(result, "boxes", None) is not None and len(result.boxes) > 0:
            xyxy_rows = result.boxes.xyxy.cpu().tolist()
            conf_rows = result.boxes.conf.cpu().tolist()
            cls_rows = result.boxes.cls.cpu().tolist()
            for xyxy, score, cls in zip(xyxy_rows, conf_rows, cls_rows):
                x1, y1, x2, y2 = [float(value) for value in xyxy]
                boxes.append(
                    BoxRecord(
                        (
                            max(0.0, min(case.width, x1)),
                            max(0.0, min(case.height, y1)),
                            max(0.0, min(case.width, x2)),
                            max(0.0, min(case.height, y2)),
                        ),
                        cls=int(cls),
                        score=float(score),
                    )
                )
        predictions[case.case_id] = boxes
    return predictions


def write_predictions_json(path: Path, predictions_by_case: dict[str, list[BoxRecord]]) -> None:
    payload = {
        case_id: [
            {"xyxy": list(box.xyxy), "class": box.cls, "score": box.score}
            for box in boxes
        ]
        for case_id, boxes in sorted(predictions_by_case.items())
    }
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True))


def write_per_case_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train/evaluate an Ultralytics YOLO detector on exported detector thumbnails.",
    )
    parser.add_argument("--dataset-dir", required=True, help="Detector training dataset export root.")
    parser.add_argument("--output-root", required=True, help="Output root for this YOLO run.")
    parser.add_argument("--model", default="yolov8n.pt", help="Ultralytics model or YAML, e.g. yolov8n.pt.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=8, help="Ultralytics batch setting; pass -1 for auto batch.")
    parser.add_argument("--device", default="0", help="Ultralytics device, e.g. 0, 1, cpu.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--cache", action="store_true", help="Enable Ultralytics image caching.")
    parser.add_argument("--eval-split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--conf", type=float, default=0.10, help="Prediction confidence for project metrics.")
    parser.add_argument(
        "--metric-conf-thresholds",
        default="0.10,0.25,0.50",
        help="Comma-separated prediction confidence thresholds for project metric sweep.",
    )
    parser.add_argument("--nms-iou", type=float, default=0.70, help="NMS IoU for project-metric predictions.")
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--fragment-iou", type=float, default=0.10)
    parser.add_argument("--oversized-area-ratio", type=float, default=2.0)
    parser.add_argument("--large-area-fraction", type=float, default=0.35)
    parser.add_argument("--max-review-pages", type=int, default=40)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    train_and_evaluate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
