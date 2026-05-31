#!/usr/bin/env python3
"""Aggregate detector tuning runs into a compact variance report."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


KEY_METRICS = [
    "framework_map50",
    "framework_map50_95",
    "project_recall_iou50",
    "project_precision_iou50",
    "project_f1_iou50",
    "false_boxes_per_slide",
    "missed_boxes_per_slide",
    "sv40_recall_iou50",
]


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def run_text(cmd: list[str], cwd: Path | None = None) -> str:
    try:
        return subprocess.check_output(cmd, cwd=cwd, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def maybe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def fmt(value: Any, digits: int = 3) -> str:
    number = maybe_float(value)
    if number is None:
        return ""
    return f"{number:.{digits}f}"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def yolo_model_name(value: str) -> str:
    name = Path(value).name
    return name or value


def yolo_summary_rows(summary_csv: Path, source: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in read_csv(summary_csv):
        project_conf = maybe_float(row.get("project_conf"))
        rows.append(
            {
                "family": "yolo",
                "source": source,
                "run": row.get("run", ""),
                "model": yolo_model_name(row.get("model", "")),
                "model_size": Path(row.get("model", "")).stem,
                "augment_profile": row.get("augment_profile", "default"),
                "imgsz": maybe_float(row.get("imgsz")),
                "epochs": maybe_float(row.get("epochs_requested")),
                "project_conf": project_conf,
                "framework_map50": maybe_float(row.get("test_ultra_map50")),
                "framework_map50_95": maybe_float(row.get("test_ultra_map50_95")),
                "project_precision_iou50": maybe_float(row.get("project_precision_iou50")),
                "project_recall_iou50": maybe_float(row.get("project_recall_iou50")),
                "project_f1_iou50": maybe_float(row.get("project_f1_iou50")),
                "false_boxes_per_slide": maybe_float(row.get("false_boxes_per_slide")),
                "missed_boxes_per_slide": maybe_float(row.get("missed_boxes_per_slide")),
                "mean_matched_iou": maybe_float(row.get("mean_matched_iou")),
                "sv40_recall_iou50": maybe_float(row.get("sv40_recall_iou50")),
                "sv40_false_boxes_per_slide": maybe_float(row.get("sv40_false_boxes_per_slide")),
                "sv40_missed_boxes_per_slide": maybe_float(row.get("sv40_missed_boxes_per_slide")),
                "artifact_path": row.get("output_root", ""),
            }
        )
    return rows


def yolo_metrics_json_rows(root: Path, source: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("*/metrics_summary.json")):
        data = json.loads(path.read_text())
        ultra = data.get("ultralytics_metrics", {})
        conf_metrics = data.get("project_metrics_by_conf", {})
        for conf, metrics in sorted(conf_metrics.items(), key=lambda item: float(item[0])):
            sv40 = metrics.get("per_stain", {}).get("SV40", {})
            rows.append(
                {
                    "family": "yolo",
                    "source": source,
                    "run": Path(data.get("output_root", path.parents[1])).name,
                    "model": yolo_model_name(str(data.get("model", ""))),
                    "model_size": Path(str(data.get("model", ""))).stem,
                    "augment_profile": data.get("augment_profile", "default"),
                    "imgsz": maybe_float(data.get("imgsz")),
                    "epochs": maybe_float(data.get("epochs")),
                    "project_conf": maybe_float(conf),
                    "framework_map50": maybe_float(ultra.get("metrics/mAP50(B)")),
                    "framework_map50_95": maybe_float(ultra.get("metrics/mAP50-95(B)")),
                    "project_precision_iou50": maybe_float(metrics.get("precision_at_match_iou")),
                    "project_recall_iou50": maybe_float(metrics.get("recall_at_match_iou")),
                    "project_f1_iou50": maybe_float(metrics.get("f1_at_match_iou")),
                    "false_boxes_per_slide": maybe_float(metrics.get("false_boxes_per_slide")),
                    "missed_boxes_per_slide": maybe_float(metrics.get("missed_boxes_per_slide")),
                    "mean_matched_iou": maybe_float(metrics.get("mean_matched_iou")),
                    "sv40_recall_iou50": maybe_float(sv40.get("recall_at_match_iou")),
                    "sv40_false_boxes_per_slide": maybe_float(sv40.get("false_boxes_per_slide")),
                    "sv40_missed_boxes_per_slide": maybe_float(sv40.get("missed_boxes_per_slide")),
                    "artifact_path": str(Path(data.get("output_root", path.parents[1])).resolve()),
                }
            )
    return rows


def rfdetr_summary_rows(summary_csv: Path, source: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in read_csv(summary_csv):
        if row.get("split") != "test":
            continue
        run_name = row.get("run_name", "")
        rows.append(
            {
                "family": "rfdetr",
                "source": source,
                "run": run_name,
                "model": f"rf-detr-{row.get('model_size', '')}",
                "model_size": row.get("model_size", ""),
                "augment_profile": "native",
                "imgsz": "",
                "epochs": "",
                "project_conf": 0.25,
                "framework_map50": maybe_float(row.get("ap50")),
                "framework_map50_95": maybe_float(row.get("map_50_95")),
                "project_precision_iou50": maybe_float(row.get("precision_at_threshold")),
                "project_recall_iou50": maybe_float(row.get("recall_at_threshold")),
                "project_f1_iou50": maybe_float(row.get("f1_at_threshold")),
                "false_boxes_per_slide": maybe_float(row.get("false_boxes_per_slide")),
                "missed_boxes_per_slide": maybe_float(row.get("missed_tissue_per_slide")),
                "mean_matched_iou": "",
                "sv40_recall_iou50": "",
                "sv40_false_boxes_per_slide": "",
                "sv40_missed_boxes_per_slide": "",
                "artifact_path": str(summary_csv.parent.resolve() / "runs" / run_name),
            }
        )
    return rows


def attach_rfdetr_sv40(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        if row.get("family") != "rfdetr":
            continue
        summary_path = Path(str(row["artifact_path"])) / "eval" / "summary.json"
        if not summary_path.exists():
            continue
        data = json.loads(summary_path.read_text())
        test = data.get("split_metrics", {}).get("test", {})
        for stain in test.get("stain_metrics", []):
            if stain.get("stain") == "SV40":
                row["sv40_recall_iou50"] = maybe_float(stain.get("recall_at_threshold"))
                row["sv40_false_boxes_per_slide"] = maybe_float(stain.get("false_boxes_per_slide"))
                row["sv40_missed_boxes_per_slide"] = maybe_float(stain.get("missed_tissue_per_slide"))
                break


def stat_block(rows: list[dict[str, Any]], metric: str) -> dict[str, float | int | None]:
    values = [maybe_float(row.get(metric)) for row in rows]
    nums = [value for value in values if value is not None]
    if not nums:
        return {"n": 0, "min": None, "max": None, "range": None, "mean": None, "std": None}
    return {
        "n": len(nums),
        "min": min(nums),
        "max": max(nums),
        "range": max(nums) - min(nums),
        "mean": mean(nums),
        "std": pstdev(nums) if len(nums) > 1 else 0.0,
    }


def variance_groups(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("family") == "yolo" and maybe_float(row.get("imgsz")) == 1024 and maybe_float(row.get("project_conf")) == 0.10:
            groups["YOLO 1024, project conf 0.10"].append(row)
        if row.get("family") == "rfdetr" and row.get("run") != "nano_e8_lr5e5":
            groups["RF-DETR reasonable LR, score conf 0.25"].append(row)
    return {
        name: {
            "runs": [row["run"] for row in group],
            "metrics": {metric: stat_block(group, metric) for metric in KEY_METRICS},
        }
        for name, group in sorted(groups.items())
    }


def top_rows(rows: list[dict[str, Any]], *, family: str, metric: str, limit: int = 8) -> list[dict[str, Any]]:
    candidates = [row for row in rows if row.get("family") == family and maybe_float(row.get(metric)) is not None]
    candidates.sort(key=lambda row: maybe_float(row.get(metric)) or -1.0, reverse=True)
    return candidates[:limit]


def markdown_table(rows: list[dict[str, Any]]) -> str:
    headers = [
        "run",
        "model",
        "aug",
        "conf",
        "mAP50",
        "mAP50-95",
        "recall",
        "precision",
        "F1",
        "false/slide",
        "miss/slide",
        "SV40 recall",
    ]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        values = [
            str(row.get("run", "")),
            str(row.get("model_size", row.get("model", ""))),
            str(row.get("augment_profile", "")),
            fmt(row.get("project_conf"), 2),
            fmt(row.get("framework_map50")),
            fmt(row.get("framework_map50_95")),
            fmt(row.get("project_recall_iou50")),
            fmt(row.get("project_precision_iou50")),
            fmt(row.get("project_f1_iou50")),
            fmt(row.get("false_boxes_per_slide"), 1),
            fmt(row.get("missed_boxes_per_slide"), 1),
            fmt(row.get("sv40_recall_iou50")),
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def summarize_takeaway(groups: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    yolo = groups.get("YOLO 1024, project conf 0.10", {}).get("metrics", {})
    rfdetr = groups.get("RF-DETR reasonable LR, score conf 0.25", {}).get("metrics", {})
    if yolo:
        m = yolo["framework_map50_95"]
        r = yolo["project_recall_iou50"]
        f = yolo["false_boxes_per_slide"]
        lines.append(
            f"- YOLO 1024 variants span mAP50-95 {fmt(m['min'])}-{fmt(m['max'])} "
            f"(range {fmt(m['range'])}), recall {fmt(r['min'])}-{fmt(r['max'])}, "
            f"and false boxes/slide {fmt(f['min'], 1)}-{fmt(f['max'], 1)} at project conf 0.10."
        )
    if rfdetr:
        m = rfdetr["framework_map50_95"]
        r = rfdetr["project_recall_iou50"]
        f = rfdetr["false_boxes_per_slide"]
        lines.append(
            f"- RF-DETR reasonable-LR variants span mAP50-95 {fmt(m['min'])}-{fmt(m['max'])} "
            f"(range {fmt(m['range'])}), recall {fmt(r['min'])}-{fmt(r['max'])}, "
            f"and false boxes/slide {fmt(f['min'], 1)}-{fmt(f['max'], 1)} at score conf 0.25."
        )
    return lines


def build_report(
    *,
    output_dir: Path,
    dataset_dir: Path,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = sorted(
        rows,
        key=lambda row: (
            str(row.get("family", "")),
            str(row.get("model_size", "")),
            str(row.get("augment_profile", "")),
            maybe_float(row.get("project_conf")) or 0.0,
            str(row.get("run", "")),
        ),
    )
    attach_rfdetr_sv40(rows)
    groups = variance_groups(rows)
    write_csv(output_dir / "all_metrics.csv", rows)
    write_json(output_dir / "all_metrics.json", rows)
    write_json(output_dir / "variance_summary.json", groups)

    yolo_main = [
        row
        for row in rows
        if row.get("family") == "yolo" and maybe_float(row.get("imgsz")) == 1024 and maybe_float(row.get("project_conf")) == 0.10
    ]
    yolo_main.sort(key=lambda row: maybe_float(row.get("framework_map50_95")) or -1.0, reverse=True)
    rfdetr_main = [row for row in rows if row.get("family") == "rfdetr" and row.get("run") != "nano_e8_lr5e5"]
    rfdetr_main.sort(key=lambda row: maybe_float(row.get("framework_map50_95")) or -1.0, reverse=True)
    yolo_best = yolo_main[0] if yolo_main else {}
    rfdetr_best_map = rfdetr_main[0] if rfdetr_main else {}
    rfdetr_best_recall = max(
        rfdetr_main,
        key=lambda row: maybe_float(row.get("project_recall_iou50")) or -1.0,
        default={},
    )

    report = [
        "# PER-244 Detector Tuning Variance Report",
        "",
        f"Created: {now_iso()}",
        "",
        "## Scope",
        "",
        "This report quantifies detector-training tuning variance on the pilot-100 thumbnail detector dataset. "
        "It compares compact YOLO model/augmentation settings and RF-DETR model-size settings. "
        "It does not rerun the VLM prompts or test DINOv3 backbones directly.",
        "",
        f"Dataset: `{dataset_dir.resolve()}`",
        "",
        "## Bottom Line",
        "",
        f"- Best YOLO 1024 by mAP50-95 is `{yolo_best.get('run', '')}` "
        f"({fmt(yolo_best.get('framework_map50_95'))}); the new reduced and stain-jitter augmentation variants did not beat that prior default run.",
        f"- Best RF-DETR by mAP50-95 is `{rfdetr_best_map.get('run', '')}` "
        f"({fmt(rfdetr_best_map.get('framework_map50_95'))}); best RF-DETR recall is `{rfdetr_best_recall.get('run', '')}` "
        f"({fmt(rfdetr_best_recall.get('project_recall_iou50'))}) but with {fmt(rfdetr_best_recall.get('false_boxes_per_slide'), 1)} false boxes/slide.",
        *summarize_takeaway(groups),
        "- The variance is real but does not change the practical ranking enough to justify a broad tuning pass before adding more reviewed thumbnails.",
        "- The current sample is only 10 validation and 10 test slides, so this is a direction-setting variance estimate, not a definitive benchmark.",
        "- SV40 remains the main weak spot across variants; that looks more like data/domain coverage than a knob that simple tuning fixes.",
        "",
        "## Main YOLO 1024 Results",
        "",
        markdown_table(yolo_main),
        "",
        "## Main RF-DETR Results",
        "",
        markdown_table(rfdetr_main),
        "",
        "## Best Rows By Framework mAP50-95",
        "",
        "### YOLO",
        "",
        markdown_table(top_rows(rows, family="yolo", metric="framework_map50_95", limit=8)),
        "",
        "### RF-DETR",
        "",
        markdown_table(top_rows(rows, family="rfdetr", metric="framework_map50_95", limit=8)),
        "",
        "## Interpretation",
        "",
        "The YOLO augmentation/model spread is meaningful but not large enough to suggest that a broad augmentation search is the highest-ROI next step. "
        "The RF-DETR size/LR sweep shows larger sensitivity, especially the prior Nano low-LR collapse, but among reasonable LR runs Small is currently the strongest RF-DETR variant.",
        "",
        "Given this spread, the practical next move is to use the best currently observed detector as an active-learning bootstrap, run the reviewer on fresh random samples, and add failure cases back into the training pool. "
        "More labeled thumbnails should likely beat further local tuning on the same 100-slide dataset.",
    ]
    (output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    reproduction = [
        "# Reproduction",
        "",
        f"Generated: {now_iso()}",
        f"Git commit: {run_text(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parents[1])}",
        "",
        "Command:",
        "```bash",
        " ".join(args.command_argv),
        "```",
        "",
        "Inputs:",
        f"- Dataset: {dataset_dir.resolve()}",
        f"- YOLO baseline summary: {args.yolo_baseline_summary}",
        f"- YOLO tuning root: {args.yolo_tuning_root}",
        f"- RF-DETR baseline summary: {args.rfdetr_baseline_summary}",
        f"- RF-DETR tuning summary: {args.rfdetr_tuning_summary}",
        "",
        "Outputs:",
        f"- Report: {(output_dir / 'report.md').resolve()}",
        f"- Metrics CSV: {(output_dir / 'all_metrics.csv').resolve()}",
        f"- Metrics JSON: {(output_dir / 'all_metrics.json').resolve()}",
        f"- Variance JSON: {(output_dir / 'variance_summary.json').resolve()}",
    ]
    (output_dir / "reproduction.txt").write_text("\n".join(reproduction) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--yolo-baseline-summary", type=Path)
    parser.add_argument("--yolo-tuning-root", type=Path)
    parser.add_argument("--rfdetr-baseline-summary", type=Path)
    parser.add_argument("--rfdetr-tuning-summary", type=Path)
    args = parser.parse_args()
    args.command_argv = ["python", str(Path(__file__)), *[arg for arg in __import__("sys").argv[1:]]]
    return args


def main() -> int:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    if args.yolo_baseline_summary:
        rows.extend(yolo_summary_rows(args.yolo_baseline_summary, "per241_yolo_baseline"))
    if args.yolo_tuning_root:
        rows.extend(yolo_metrics_json_rows(args.yolo_tuning_root, "per244_yolo_tuning"))
    if args.rfdetr_baseline_summary:
        rows.extend(rfdetr_summary_rows(args.rfdetr_baseline_summary, "per242_rfdetr_baseline"))
    if args.rfdetr_tuning_summary:
        rows.extend(rfdetr_summary_rows(args.rfdetr_tuning_summary, "per244_rfdetr_size_tuning"))
    if not rows:
        raise SystemExit("No metrics rows found.")
    build_report(output_dir=args.output_dir, dataset_dir=args.dataset_dir, rows=rows, args=args)
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "rows": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
