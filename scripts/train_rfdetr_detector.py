#!/usr/bin/env python3
"""Train and evaluate RF-DETR on exported detector thumbnail datasets."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


SPLIT_TO_RFDETR = {"train": "train", "val": "valid", "test": "test"}
RFDETR_TO_SPLIT = {value: key for key, value in SPLIT_TO_RFDETR.items()}
AP_THRESHOLDS = [round(0.50 + 0.05 * idx, 2) for idx in range(10)]


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def run_git_commit(cwd: Path) -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd, text=True).strip()
        return out
    except Exception:
        return "unknown"


def materialize_image(src: Path, dst: Path, mode: str) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "symlink":
        os.symlink(src, dst)
        return "symlink"
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return "hardlink"
        except OSError:
            shutil.copy2(src, dst)
            return "copy_fallback"
    if mode == "copy":
        shutil.copy2(src, dst)
        return "copy"
    raise ValueError(f"Unknown image materialization mode: {mode}")


def prepare_rfdetr_dataset(
    source_dataset_dir: Path,
    rfdetr_dataset_dir: Path,
    *,
    image_mode: str = "hardlink",
    overwrite: bool = False,
) -> dict[str, Any]:
    source_dataset_dir = source_dataset_dir.resolve()
    rfdetr_dataset_dir = rfdetr_dataset_dir.resolve()
    if overwrite and rfdetr_dataset_dir.exists():
        shutil.rmtree(rfdetr_dataset_dir)
    rfdetr_dataset_dir.mkdir(parents=True, exist_ok=True)

    split_summaries: dict[str, Any] = {}
    materialization_counts: dict[str, int] = defaultdict(int)
    for source_split, rfdetr_split in SPLIT_TO_RFDETR.items():
        coco_path = source_dataset_dir / "annotations" / f"instances_{source_split}.json"
        if not coco_path.exists():
            raise FileNotFoundError(f"Missing COCO split file: {coco_path}")
        coco = read_json(coco_path)
        split_dir = rfdetr_dataset_dir / rfdetr_split
        split_dir.mkdir(parents=True, exist_ok=True)

        seen_names: set[str] = set()
        converted_images: list[dict[str, Any]] = []
        for image in coco.get("images", []):
            src_image = source_dataset_dir / image["file_name"]
            if not src_image.exists():
                raise FileNotFoundError(f"Missing image referenced by {coco_path}: {src_image}")
            dst_name = Path(image["file_name"]).name
            if dst_name in seen_names:
                dst_name = f"{image['id']}_{dst_name}"
            seen_names.add(dst_name)
            mode_used = materialize_image(src_image.resolve(), split_dir / dst_name, image_mode)
            materialization_counts[mode_used] += 1

            converted = dict(image)
            converted["source_file_name"] = image["file_name"]
            converted["file_name"] = dst_name
            converted_images.append(converted)

        converted_coco = {
            "info": coco.get("info", {}),
            "licenses": coco.get("licenses", []),
            "images": converted_images,
            "annotations": coco.get("annotations", []),
            "categories": coco.get("categories", []),
        }
        write_json(split_dir / "_annotations.coco.json", converted_coco)
        split_summaries[source_split] = {
            "rfdetr_split": rfdetr_split,
            "image_count": len(converted_images),
            "annotation_count": len(converted_coco["annotations"]),
            "annotation_path": str((split_dir / "_annotations.coco.json").resolve()),
        }

    summary = {
        "source_dataset_dir": str(source_dataset_dir),
        "rfdetr_dataset_dir": str(rfdetr_dataset_dir),
        "image_mode_requested": image_mode,
        "image_materialization_counts": dict(sorted(materialization_counts.items())),
        "splits": split_summaries,
        "format_contract": "Roboflow/RF-DETR COCO layout: train|valid|test/_annotations.coco.json plus images in each split folder.",
    }
    write_json(rfdetr_dataset_dir / "dataset_adapter_summary.json", summary)
    return summary


def validate_rfdetr_dataset(dataset_dir: Path) -> dict[str, Any]:
    dataset_dir = dataset_dir.resolve()
    issues: list[str] = []
    splits: dict[str, Any] = {}
    for rfdetr_split in ["train", "valid", "test"]:
        split_dir = dataset_dir / rfdetr_split
        ann_path = split_dir / "_annotations.coco.json"
        if not ann_path.exists():
            issues.append(f"missing {ann_path}")
            continue
        coco = read_json(ann_path)
        missing = []
        bad_boxes = 0
        image_ids = {image["id"] for image in coco.get("images", [])}
        for image in coco.get("images", []):
            if not (split_dir / image["file_name"]).exists():
                missing.append(image["file_name"])
        for ann in coco.get("annotations", []):
            bbox = ann.get("bbox", [])
            if ann.get("image_id") not in image_ids:
                bad_boxes += 1
            elif len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0:
                bad_boxes += 1
        if missing:
            issues.append(f"{rfdetr_split} missing {len(missing)} images")
        if bad_boxes:
            issues.append(f"{rfdetr_split} has {bad_boxes} bad annotations")
        splits[rfdetr_split] = {
            "image_count": len(coco.get("images", [])),
            "annotation_count": len(coco.get("annotations", [])),
            "category_count": len(coco.get("categories", [])),
            "missing_images": len(missing),
            "bad_annotations": bad_boxes,
        }
    result = {
        "dataset_dir": str(dataset_dir),
        "status": "ok" if not issues else "error",
        "issues": issues,
        "splits": splits,
    }
    write_json(dataset_dir / "rfdetr_validation.json", result)
    return result


def bbox_xywh_to_xyxy(bbox: list[float]) -> list[float]:
    x, y, w, h = [float(v) for v in bbox]
    return [x, y, x + w, y + h]


def bbox_xyxy_to_xywh(box: list[float]) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def box_area_xyxy(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def iou_xyxy(a: list[float], b: list[float]) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = box_area_xyxy([ix1, iy1, ix2, iy2])
    if inter <= 0:
        return 0.0
    union = box_area_xyxy(a) + box_area_xyxy(b) - inter
    return inter / union if union > 0 else 0.0


def clip_xyxy(box: list[float], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = box
    x1 = min(max(0.0, float(x1)), float(width))
    y1 = min(max(0.0, float(y1)), float(height))
    x2 = min(max(0.0, float(x2)), float(width))
    y2 = min(max(0.0, float(y2)), float(height))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


@dataclass
class Match:
    pred_index: int
    gt_index: int
    iou: float


def greedy_match(
    gt_boxes: list[list[float]],
    pred_boxes: list[list[float]],
    pred_scores: list[float],
    iou_threshold: float,
) -> list[Match]:
    matched_gt: set[int] = set()
    matches: list[Match] = []
    order = sorted(range(len(pred_boxes)), key=lambda idx: pred_scores[idx], reverse=True)
    for pred_idx in order:
        best_gt = -1
        best_iou = 0.0
        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_idx in matched_gt:
                continue
            iou = iou_xyxy(pred_boxes[pred_idx], gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt = gt_idx
        if best_gt >= 0 and best_iou >= iou_threshold:
            matched_gt.add(best_gt)
            matches.append(Match(pred_index=pred_idx, gt_index=best_gt, iou=best_iou))
    return matches


def ap_for_threshold(
    gt_by_image: dict[int, list[list[float]]],
    predictions: list[dict[str, Any]],
    iou_threshold: float,
) -> float:
    npos = sum(len(boxes) for boxes in gt_by_image.values())
    if npos == 0:
        return 0.0
    sorted_preds = sorted(predictions, key=lambda pred: float(pred.get("score", 0.0)), reverse=True)
    matched: dict[int, set[int]] = defaultdict(set)
    tp: list[float] = []
    fp: list[float] = []
    for pred in sorted_preds:
        image_id = int(pred["image_id"])
        pred_box = bbox_xywh_to_xyxy(pred["bbox"])
        best_gt = -1
        best_iou = 0.0
        for gt_idx, gt_box in enumerate(gt_by_image.get(image_id, [])):
            if gt_idx in matched[image_id]:
                continue
            iou = iou_xyxy(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt = gt_idx
        if best_gt >= 0 and best_iou >= iou_threshold:
            matched[image_id].add(best_gt)
            tp.append(1.0)
            fp.append(0.0)
        else:
            tp.append(0.0)
            fp.append(1.0)

    cum_tp: list[float] = []
    cum_fp: list[float] = []
    running_tp = 0.0
    running_fp = 0.0
    for tpi, fpi in zip(tp, fp):
        running_tp += tpi
        running_fp += fpi
        cum_tp.append(running_tp)
        cum_fp.append(running_fp)

    precisions = []
    recalls = []
    for tpi, fpi in zip(cum_tp, cum_fp):
        recalls.append(tpi / npos)
        precisions.append(tpi / max(tpi + fpi, 1e-12))

    ap = 0.0
    for recall_target in [idx / 100 for idx in range(101)]:
        candidates = [p for p, r in zip(precisions, recalls) if r >= recall_target]
        ap += max(candidates) if candidates else 0.0
    return ap / 101.0


def load_split_coco(rfdetr_dataset_dir: Path, split: str) -> tuple[dict[str, Any], Path]:
    rfdetr_split = SPLIT_TO_RFDETR.get(split, split)
    split_dir = rfdetr_dataset_dir / rfdetr_split
    return read_json(split_dir / "_annotations.coco.json"), split_dir


def evaluate_predictions(
    coco: dict[str, Any],
    predictions: list[dict[str, Any]],
    *,
    score_threshold: float,
    match_iou: float,
    oversized_area_ratio: float,
    large_area_fraction: float,
    near_full_fraction: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    images = {int(image["id"]): image for image in coco.get("images", [])}
    gt_by_image: dict[int, list[list[float]]] = defaultdict(list)
    for ann in coco.get("annotations", []):
        gt_by_image[int(ann["image_id"])].append(bbox_xywh_to_xyxy(ann["bbox"]))

    pred_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for pred in predictions:
        pred_by_image[int(pred["image_id"])].append(pred)

    ap_by_threshold = {f"ap_{int(t * 100):02d}": ap_for_threshold(gt_by_image, predictions, t) for t in AP_THRESHOLDS}
    ap_values = list(ap_by_threshold.values())

    per_image_rows: list[dict[str, Any]] = []
    aggregate = defaultdict(float)
    stain_aggregate: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    for image_id, image in sorted(images.items()):
        width = int(image["width"])
        height = int(image["height"])
        gt_boxes = gt_by_image.get(image_id, [])
        filtered_preds = [pred for pred in pred_by_image.get(image_id, []) if float(pred.get("score", 0.0)) >= score_threshold]
        pred_boxes = [clip_xyxy(bbox_xywh_to_xyxy(pred["bbox"]), width, height) for pred in filtered_preds]
        pred_scores = [float(pred.get("score", 0.0)) for pred in filtered_preds]
        matches = greedy_match(gt_boxes, pred_boxes, pred_scores, match_iou)
        matched_pred = {match.pred_index for match in matches}
        matched_gt = {match.gt_index for match in matches}

        duplicate_count = 0
        fragment_count = 0
        oversized_count = 0
        near_full_count = 0
        for pred_idx, pred_box in enumerate(pred_boxes):
            pred_area = box_area_xyxy(pred_box)
            image_area = max(float(width * height), 1.0)
            area_fraction = pred_area / image_area
            if (
                (pred_box[2] - pred_box[0]) / max(width, 1) >= near_full_fraction
                or (pred_box[3] - pred_box[1]) / max(height, 1) >= near_full_fraction
                or area_fraction >= large_area_fraction
            ):
                near_full_count += 1

            best_iou_any = max((iou_xyxy(pred_box, gt_box) for gt_box in gt_boxes), default=0.0)
            if pred_idx not in matched_pred and best_iou_any >= match_iou:
                duplicate_count += 1
            elif pred_idx not in matched_pred and best_iou_any >= 0.10:
                fragment_count += 1
            if area_fraction >= large_area_fraction:
                oversized_count += 1

        for match in matches:
            pred_area = box_area_xyxy(pred_boxes[match.pred_index])
            gt_area = max(box_area_xyxy(gt_boxes[match.gt_index]), 1.0)
            if pred_area / gt_area >= oversized_area_ratio:
                oversized_count += 1

        tp = len(matches)
        gt_count = len(gt_boxes)
        pred_count = len(pred_boxes)
        missed = gt_count - len(matched_gt)
        false_boxes = pred_count - len(matched_pred)
        precision = tp / pred_count if pred_count else (1.0 if gt_count == 0 else 0.0)
        recall = tp / gt_count if gt_count else 1.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        row = {
            "image_id": image_id,
            "case_id": image.get("case_id", Path(image.get("file_name", "")).stem),
            "stain": image.get("stain", "unknown"),
            "file_name": image.get("file_name", ""),
            "gt_count": gt_count,
            "prediction_count": pred_count,
            "true_positive_count": tp,
            "missed_tissue_count": missed,
            "false_box_count": false_boxes,
            "duplicate_or_extra_fragment_count": duplicate_count,
            "partial_fragment_or_near_miss_count": fragment_count,
            "oversized_or_whitespace_proxy_count": oversized_count,
            "near_full_box_count": near_full_count,
            "precision_at_threshold": precision,
            "recall_at_threshold": recall,
            "f1_at_threshold": f1,
        }
        per_image_rows.append(row)
        for key in [
            "gt_count",
            "prediction_count",
            "true_positive_count",
            "missed_tissue_count",
            "false_box_count",
            "duplicate_or_extra_fragment_count",
            "partial_fragment_or_near_miss_count",
            "oversized_or_whitespace_proxy_count",
            "near_full_box_count",
        ]:
            aggregate[key] += float(row[key])
            stain_aggregate[str(row["stain"])][key] += float(row[key])
        aggregate["image_count"] += 1.0
        stain_aggregate[str(row["stain"])]["image_count"] += 1.0

    def finalize(counts: dict[str, float]) -> dict[str, Any]:
        image_count = max(counts.get("image_count", 0.0), 1.0)
        tp = counts.get("true_positive_count", 0.0)
        pred = counts.get("prediction_count", 0.0)
        gt = counts.get("gt_count", 0.0)
        precision = tp / pred if pred else (1.0 if gt == 0 else 0.0)
        recall = tp / gt if gt else 1.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {
            **{key: int(value) if float(value).is_integer() else value for key, value in counts.items()},
            "missed_tissue_per_slide": counts.get("missed_tissue_count", 0.0) / image_count,
            "false_boxes_per_slide": counts.get("false_box_count", 0.0) / image_count,
            "precision_at_threshold": precision,
            "recall_at_threshold": recall,
            "f1_at_threshold": f1,
        }

    per_stain_rows = []
    for stain, counts in sorted(stain_aggregate.items()):
        finalized = finalize(counts)
        finalized["stain"] = stain
        per_stain_rows.append(finalized)

    metrics = {
        "image_count": len(images),
        "annotation_count": sum(len(v) for v in gt_by_image.values()),
        "prediction_count_unthresholded": len(predictions),
        "score_threshold_for_error_metrics": score_threshold,
        "match_iou_for_error_metrics": match_iou,
        "ap": {
            **ap_by_threshold,
            "map_50_95": sum(ap_values) / len(ap_values) if ap_values else 0.0,
        },
        "project_metrics": finalize(aggregate),
        "stain_metrics": per_stain_rows,
    }
    return metrics, per_image_rows, per_stain_rows


def get_model_class(model_size: str):
    from rfdetr import RFDETRBase, RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall

    mapping = {
        "base": RFDETRBase,
        "nano": RFDETRNano,
        "small": RFDETRSmall,
        "medium": RFDETRMedium,
        "large": RFDETRLarge,
    }
    try:
        return mapping[model_size]
    except KeyError as exc:
        raise ValueError(f"Unsupported RF-DETR model size {model_size!r}") from exc


def train_one(args: argparse.Namespace) -> dict[str, Any]:
    model_class = get_model_class(args.model_size)
    train_output_dir = args.output_root / "runs" / args.run_name / "train"
    train_output_dir.mkdir(parents=True, exist_ok=True)
    model_kwargs: dict[str, Any] = {}
    if args.pretrain_weights:
        model_kwargs["pretrain_weights"] = str(args.pretrain_weights)
    if args.device:
        model_kwargs["device"] = args.device
    model = model_class(**model_kwargs)
    train_kwargs: dict[str, Any] = {
        "dataset_dir": str(args.rfdetr_dataset_dir.resolve()),
        "output_dir": str(train_output_dir.resolve()),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "lr": args.lr,
        "lr_encoder": args.lr_encoder,
        "weight_decay": args.weight_decay,
        "num_workers": args.num_workers,
        "checkpoint_interval": args.checkpoint_interval,
        "tensorboard": args.tensorboard,
        "wandb": False,
        "early_stopping": args.early_stopping,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_min_delta": args.early_stopping_min_delta,
        "use_ema": args.use_ema,
        "run_test": False,
        "progress_bar": args.progress_bar,
        "seed": args.seed,
        "notes": {
            "ticket": "PER-242",
            "source_dataset": str(args.source_dataset_dir.resolve()) if args.source_dataset_dir else None,
            "rfdetr_dataset": str(args.rfdetr_dataset_dir.resolve()),
            "run_name": args.run_name,
        },
    }
    if args.device:
        train_kwargs["device"] = args.device
    if args.resolution is not None:
        train_kwargs["resolution"] = args.resolution
    model.train(**train_kwargs)
    checkpoint = train_output_dir / "checkpoint_best_total.pth"
    if not checkpoint.exists():
        fallback = train_output_dir / "checkpoint.pth"
        checkpoint = fallback if fallback.exists() else checkpoint
    summary = {
        "run_name": args.run_name,
        "model_size": args.model_size,
        "train_output_dir": str(train_output_dir.resolve()),
        "checkpoint": str(checkpoint.resolve()) if checkpoint.exists() else None,
        "train_kwargs": {k: str(v) if isinstance(v, Path) else v for k, v in train_kwargs.items()},
        "model_kwargs": model_kwargs,
    }
    write_json(args.output_root / "runs" / args.run_name / "train_summary.json", summary)
    return summary


def detections_to_coco_records(detections: Any, image: dict[str, Any], category_id: int) -> list[dict[str, Any]]:
    width = int(image["width"])
    height = int(image["height"])
    records: list[dict[str, Any]] = []
    xyxy = getattr(detections, "xyxy", [])
    confidence = getattr(detections, "confidence", None)
    for idx, box in enumerate(xyxy):
        score = float(confidence[idx]) if confidence is not None and idx < len(confidence) else 1.0
        clipped = clip_xyxy([float(v) for v in box], width, height)
        xywh = bbox_xyxy_to_xywh(clipped)
        if xywh[2] <= 0 or xywh[3] <= 0:
            continue
        records.append(
            {
                "image_id": int(image["id"]),
                "category_id": int(category_id),
                "bbox": xywh,
                "score": score,
            }
        )
    return records


def draw_overlay(
    image_path: Path,
    image_record: dict[str, Any],
    gt_boxes: list[list[float]],
    predictions: list[dict[str, Any]],
    output_path: Path,
    *,
    score_threshold: float,
) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    line_width = max(2, int(max(width, height) / 550))
    for idx, gt in enumerate(gt_boxes, start=1):
        box = clip_xyxy(gt, width, height)
        draw.rectangle(box, outline=(0, 160, 80), width=line_width)
        draw.text((box[0] + 2, box[1] + 2), f"G{idx}", fill=(0, 120, 60))
    kept = [pred for pred in predictions if float(pred.get("score", 0.0)) >= score_threshold]
    kept.sort(key=lambda pred: float(pred.get("score", 0.0)), reverse=True)
    for idx, pred in enumerate(kept, start=1):
        box = clip_xyxy(bbox_xywh_to_xyxy(pred["bbox"]), width, height)
        draw.rectangle(box, outline=(220, 40, 40), width=line_width)
        draw.text((box[0] + 2, max(0, box[1] - 14)), f"P{idx} {pred['score']:.2f}", fill=(180, 0, 0))
    draw.text((8, 8), str(image_record.get("case_id", image_record.get("file_name", ""))), fill=(0, 0, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    max_long = 1600
    if max(image.size) > max_long:
        scale = max_long / max(image.size)
        image = image.resize((int(image.size[0] * scale), int(image.size[1] * scale)))
    image.save(output_path)


def save_overlay_pdf(overlay_paths: list[Path], pdf_path: Path) -> None:
    if not overlay_paths:
        return
    images = [Image.open(path).convert("RGB") for path in overlay_paths]
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(pdf_path, save_all=True, append_images=images[1:])


def evaluate_checkpoint(args: argparse.Namespace) -> dict[str, Any]:
    model_class = get_model_class(args.model_size)
    model_kwargs: dict[str, Any] = {"pretrain_weights": str(args.checkpoint)}
    if args.device:
        model_kwargs["device"] = args.device
    model = model_class(**model_kwargs)

    run_eval_dir = args.output_root / "runs" / args.run_name / "eval"
    run_eval_dir.mkdir(parents=True, exist_ok=True)
    all_split_summaries: dict[str, Any] = {}
    category_id = 1
    for split in args.splits:
        coco, split_dir = load_split_coco(args.rfdetr_dataset_dir, split)
        categories = sorted(coco.get("categories", []), key=lambda c: c.get("id", 0))
        if categories:
            category_id = int(categories[0]["id"])
        predictions: list[dict[str, Any]] = []
        gt_by_image: dict[int, list[list[float]]] = defaultdict(list)
        for ann in coco.get("annotations", []):
            gt_by_image[int(ann["image_id"])].append(bbox_xywh_to_xyxy(ann["bbox"]))
        pred_by_image: dict[int, list[dict[str, Any]]] = {}
        for image in sorted(coco.get("images", []), key=lambda item: int(item["id"])):
            image_path = split_dir / image["file_name"]
            detections = model.predict(str(image_path), threshold=args.predict_threshold)
            records = detections_to_coco_records(detections, image, category_id)
            predictions.extend(records)
            pred_by_image[int(image["id"])] = records

        metrics, per_image_rows, per_stain_rows = evaluate_predictions(
            coco,
            predictions,
            score_threshold=args.score_threshold,
            match_iou=args.match_iou,
            oversized_area_ratio=args.oversized_area_ratio,
            large_area_fraction=args.large_area_fraction,
            near_full_fraction=args.near_full_fraction,
        )
        split_eval_dir = run_eval_dir / split
        write_json(split_eval_dir / "predictions.coco.json", predictions)
        write_json(split_eval_dir / "metrics.json", metrics)
        write_csv(
            split_eval_dir / "per_image_metrics.csv",
            per_image_rows,
            [
                "image_id",
                "case_id",
                "stain",
                "file_name",
                "gt_count",
                "prediction_count",
                "true_positive_count",
                "missed_tissue_count",
                "false_box_count",
                "duplicate_or_extra_fragment_count",
                "partial_fragment_or_near_miss_count",
                "oversized_or_whitespace_proxy_count",
                "near_full_box_count",
                "precision_at_threshold",
                "recall_at_threshold",
                "f1_at_threshold",
            ],
        )
        write_csv(
            split_eval_dir / "per_stain_metrics.csv",
            per_stain_rows,
            [
                "stain",
                "image_count",
                "gt_count",
                "prediction_count",
                "true_positive_count",
                "missed_tissue_count",
                "false_box_count",
                "duplicate_or_extra_fragment_count",
                "partial_fragment_or_near_miss_count",
                "oversized_or_whitespace_proxy_count",
                "near_full_box_count",
                "missed_tissue_per_slide",
                "false_boxes_per_slide",
                "precision_at_threshold",
                "recall_at_threshold",
                "f1_at_threshold",
            ],
        )

        sorted_rows = sorted(
            per_image_rows,
            key=lambda row: (
                int(row["missed_tissue_count"]),
                int(row["false_box_count"]),
                int(row["oversized_or_whitespace_proxy_count"]),
                str(row["case_id"]),
            ),
            reverse=True,
        )
        overlay_rows = sorted_rows[: args.overlay_max]
        overlay_paths: list[Path] = []
        images_by_id = {int(image["id"]): image for image in coco.get("images", [])}
        for row in overlay_rows:
            image = images_by_id[int(row["image_id"])]
            overlay_path = split_eval_dir / "overlays" / f"{image.get('case_id', Path(image['file_name']).stem)}.png"
            draw_overlay(
                split_dir / image["file_name"],
                image,
                gt_by_image.get(int(image["id"]), []),
                pred_by_image.get(int(image["id"]), []),
                overlay_path,
                score_threshold=args.score_threshold,
            )
            overlay_paths.append(overlay_path)
        save_overlay_pdf(overlay_paths, split_eval_dir / "prediction_overlays.pdf")
        all_split_summaries[split] = metrics

    summary = {
        "run_name": args.run_name,
        "model_size": args.model_size,
        "checkpoint": str(args.checkpoint.resolve()),
        "rfdetr_dataset_dir": str(args.rfdetr_dataset_dir.resolve()),
        "predict_threshold": args.predict_threshold,
        "score_threshold_for_error_metrics": args.score_threshold,
        "split_metrics": all_split_summaries,
    }
    write_json(run_eval_dir / "summary.json", summary)
    return summary


def summarize_runs(output_root: Path, run_names: list[str]) -> dict[str, Any]:
    rows = []
    for run_name in run_names:
        eval_summary_path = output_root / "runs" / run_name / "eval" / "summary.json"
        train_summary_path = output_root / "runs" / run_name / "train_summary.json"
        if not eval_summary_path.exists():
            continue
        eval_summary = read_json(eval_summary_path)
        train_summary = read_json(train_summary_path) if train_summary_path.exists() else {}
        for split, metrics in eval_summary.get("split_metrics", {}).items():
            project = metrics.get("project_metrics", {})
            rows.append(
                {
                    "run_name": run_name,
                    "model_size": eval_summary.get("model_size"),
                    "split": split,
                    "checkpoint": train_summary.get("checkpoint") or eval_summary.get("checkpoint"),
                    "map_50_95": metrics.get("ap", {}).get("map_50_95"),
                    "ap50": metrics.get("ap", {}).get("ap_50"),
                    "ap75": metrics.get("ap", {}).get("ap_75"),
                    "precision_at_threshold": project.get("precision_at_threshold"),
                    "recall_at_threshold": project.get("recall_at_threshold"),
                    "f1_at_threshold": project.get("f1_at_threshold"),
                    "missed_tissue_per_slide": project.get("missed_tissue_per_slide"),
                    "false_boxes_per_slide": project.get("false_boxes_per_slide"),
                    "oversized_or_whitespace_proxy_count": project.get("oversized_or_whitespace_proxy_count"),
                    "near_full_box_count": project.get("near_full_box_count"),
                }
            )
    fieldnames = [
        "run_name",
        "model_size",
        "split",
        "checkpoint",
        "map_50_95",
        "ap50",
        "ap75",
        "precision_at_threshold",
        "recall_at_threshold",
        "f1_at_threshold",
        "missed_tissue_per_slide",
        "false_boxes_per_slide",
        "oversized_or_whitespace_proxy_count",
        "near_full_box_count",
    ]
    write_csv(output_root / "summary_metrics.csv", rows, fieldnames)
    summary = {"output_root": str(output_root.resolve()), "runs": rows}
    write_json(output_root / "summary.json", summary)
    return summary


def write_reproduction(args: argparse.Namespace, command: str, extra: dict[str, Any] | None = None) -> None:
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    git_commit = run_git_commit(Path.cwd())
    payload = {
        "ticket": "PER-242",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_at_runtime": git_commit,
        "command": command,
        "output_root": str(output_root),
        "source_dataset_dir": str(args.source_dataset_dir.resolve()) if getattr(args, "source_dataset_dir", None) else None,
        "rfdetr_dataset_dir": str(args.rfdetr_dataset_dir.resolve()) if getattr(args, "rfdetr_dataset_dir", None) else None,
        "extra": extra or {},
    }
    write_json(output_root / "reproduction.json", payload)
    history_path = output_root / "command_history.jsonl"
    with history_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")
    command_history = []
    with history_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                command_history.append(json.loads(line))

    lines = [
        "RF-DETR Thumbnail Detector Distillation",
        "======================================",
        "",
        "Ticket: PER-242",
        f"Latest git commit at runtime: {git_commit}",
        "",
        "Latest command:",
        command,
        "",
        f"Output root: {output_root}",
    ]
    if getattr(args, "source_dataset_dir", None):
        lines += ["", f"Source dataset: {args.source_dataset_dir.resolve()}"]
    if getattr(args, "rfdetr_dataset_dir", None):
        lines += ["", f"RF-DETR dataset adapter: {args.rfdetr_dataset_dir.resolve()}"]
    lines += [
        "",
        f"Command history JSONL: {history_path}",
        "",
        "Command history:",
    ]
    for idx, item in enumerate(command_history, start=1):
        lines += [
            f"{idx}. {item.get('timestamp_utc', 'unknown-time')} | {item.get('git_commit_at_runtime', 'unknown-commit')}",
            f"   {item.get('command', '')}",
        ]
    if extra:
        lines += ["", "Latest command summary:", json.dumps(extra, indent=2, sort_keys=True)]
    (output_root / "reproduction.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-dataset-dir", type=Path)
    parser.add_argument("--rfdetr-dataset-dir", type=Path, required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Create RF-DETR-compatible COCO layout.")
    add_common_args(prepare)
    prepare.add_argument("--image-mode", choices=["hardlink", "copy", "symlink"], default="hardlink")
    prepare.add_argument("--overwrite", action="store_true")

    validate = subparsers.add_parser("validate", help="Validate RF-DETR-compatible dataset layout.")
    validate.add_argument("--rfdetr-dataset-dir", type=Path, required=True)
    validate.add_argument("--output-root", type=Path, required=True)

    train = subparsers.add_parser("train", help="Train one RF-DETR run.")
    add_common_args(train)
    train.add_argument("--run-name", required=True)
    train.add_argument("--model-size", choices=["nano", "small", "medium", "large", "base"], default="nano")
    train.add_argument("--pretrain-weights", type=Path)
    train.add_argument("--epochs", type=int, default=10)
    train.add_argument("--batch-size", type=int, default=4)
    train.add_argument("--grad-accum-steps", type=int, default=4)
    train.add_argument("--lr", type=float, default=1e-4)
    train.add_argument("--lr-encoder", type=float, default=1.5e-4)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--resolution", type=int)
    train.add_argument("--device", default="cuda:0")
    train.add_argument("--num-workers", type=int, default=4)
    train.add_argument("--checkpoint-interval", type=int, default=5)
    train.add_argument("--seed", type=int, default=13)
    train.add_argument("--no-tensorboard", action="store_true")
    train.add_argument("--no-ema", action="store_true")
    train.add_argument("--progress-bar", choices=["tqdm", "rich", "none"], default="tqdm")
    train.add_argument("--early-stopping", action="store_true")
    train.add_argument("--early-stopping-patience", type=int, default=5)
    train.add_argument("--early-stopping-min-delta", type=float, default=0.001)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate one RF-DETR checkpoint.")
    add_common_args(evaluate)
    evaluate.add_argument("--run-name", required=True)
    evaluate.add_argument("--model-size", choices=["nano", "small", "medium", "large", "base"], default="nano")
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--splits", nargs="+", choices=["train", "val", "test"], default=["val", "test"])
    evaluate.add_argument("--device", default="cuda:0")
    evaluate.add_argument("--predict-threshold", type=float, default=0.05)
    evaluate.add_argument("--score-threshold", type=float, default=0.25)
    evaluate.add_argument("--match-iou", type=float, default=0.5)
    evaluate.add_argument("--oversized-area-ratio", type=float, default=2.5)
    evaluate.add_argument("--large-area-fraction", type=float, default=0.35)
    evaluate.add_argument("--near-full-fraction", type=float, default=0.80)
    evaluate.add_argument("--overlay-max", type=int, default=20)

    summarize = subparsers.add_parser("summarize", help="Summarize evaluated runs.")
    summarize.add_argument("--output-root", type=Path, required=True)
    summarize.add_argument("--run-names", nargs="+", required=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    command = " ".join([Path(sys.argv[0]).name] + sys.argv[1:])
    if args.command == "prepare":
        if args.source_dataset_dir is None:
            parser.error("prepare requires --source-dataset-dir")
        summary = prepare_rfdetr_dataset(
            args.source_dataset_dir,
            args.rfdetr_dataset_dir,
            image_mode=args.image_mode,
            overwrite=args.overwrite,
        )
        validation = validate_rfdetr_dataset(args.rfdetr_dataset_dir)
        extra = {"adapter_summary": summary, "validation": validation}
        write_reproduction(args, command, extra)
        print(json.dumps(extra, indent=2, sort_keys=True))
        return 0 if validation["status"] == "ok" else 1
    if args.command == "validate":
        result = validate_rfdetr_dataset(args.rfdetr_dataset_dir)
        args.source_dataset_dir = None
        write_reproduction(args, command, {"validation": result})
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["status"] == "ok" else 1
    if args.command == "train":
        args.tensorboard = not args.no_tensorboard
        args.use_ema = not args.no_ema
        if args.progress_bar == "none":
            args.progress_bar = None
        summary = train_one(args)
        write_reproduction(args, command, {"train_summary": summary})
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if args.command == "evaluate":
        summary = evaluate_checkpoint(args)
        write_reproduction(args, command, {"eval_summary": summary})
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if args.command == "summarize":
        summary = summarize_runs(args.output_root, args.run_names)
        args.source_dataset_dir = None
        args.rfdetr_dataset_dir = None
        write_reproduction(args, command, {"summary": summary})
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    parser.error(f"Unknown command {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
