#!/usr/bin/env python3
"""Build final Stage 1-6 detector pipeline packet after crop TP/FP filtering."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import textwrap
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw

from stage1_detection_review_pilot import _font, _repo_git_commit, _safe_slug, _thumb, _timestamp


REPO_ROOT = Path(__file__).resolve().parents[1]
STAGE1_CASES = (
    REPO_ROOT
    / "runs/stage1_detector_pilot_v1/stage1_detection_review_v1"
    / "high_recall_stage1_rot0_v1/summary/high_recall_stage1_cases.csv"
)
STAGE2B_RESULTS = (
    REPO_ROOT
    / "runs/stage1_detector_pilot_v1/stage1_detection_review_v1"
    / "high_recall_raw_overlay_pilot100_short_reviewer_high_thinking_v1"
    / "stage2b_nonminor_two_pass_gemini_flash_low_v1"
    / "reviews/stage2b_two_pass_results.jsonl"
)
STAGE3_RESULTS = (
    REPO_ROOT
    / "runs/stage1_detector_pilot_v1/stage1_detection_review_v1"
    / "high_recall_raw_overlay_pilot100_short_reviewer_high_thinking_v1"
    / "stage3_refinement_minimal_feedback_gemini_flash_high_v1"
    / "reviews/stage3_refinement_results.jsonl"
)
DEFAULT_CANDIDATES = (
    REPO_ROOT
    / "runs/stage1_detector_pilot_v1/stage1_detection_review_v1"
    / "stage6_all100_final_detector_v1/stage4_inputs/summary/stage4_crop_prompt_packet_candidates.csv"
)
DEFAULT_STAGE6_RESULTS = (
    REPO_ROOT
    / "runs/stage1_detector_pilot_v1/stage1_detection_review_v1"
    / "stage6_all100_final_detector_v1/high_thinking/reviews/stage6_crop_tissue_artifact_high_thinking.jsonl"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "runs/stage1_detector_pilot_v1/stage1_detection_review_v1"
    / "stage6_all100_final_detector_v1/final_packet"
)
PROMPT_VERSION = "stage6_final_detection_packet_2026-05-25"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _wrap(text: str, width: int) -> list[str]:
    lines: list[str] = []
    for paragraph in str(text).splitlines() or [""]:
        lines.extend(textwrap.wrap(paragraph, width=width) or [""])
    return lines


def _draw_wrapped(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, width: int, font: Any, fill: str = "#111111", line_h: int = 23) -> int:
    x, y = xy
    for line in _wrap(text, width):
        draw.text((x, y), line, font=font, fill=fill)
        y += line_h
    return y


def _yxyx_iou(a: list[float], b: list[float]) -> float:
    ay1, ax1, ay2, ax2 = a
    by1, bx1, by2, bx2 = b
    inter_y1, inter_x1 = max(ay1, by1), max(ax1, bx1)
    inter_y2, inter_x2 = min(ay2, by2), min(ax2, bx2)
    inter = max(0.0, inter_y2 - inter_y1) * max(0.0, inter_x2 - inter_x1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ay2 - ay1) * max(0.0, ax2 - ax1)
    area_b = max(0.0, by2 - by1) * max(0.0, bx2 - bx1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def _merge_iou_yxyx(boxes: list[list[float]], threshold: float) -> tuple[list[list[float]], int]:
    merged = [list(map(float, box)) for box in boxes]
    merge_events = 0
    changed = True
    while changed:
        changed = False
        out: list[list[float]] = []
        used: set[int] = set()
        for i, box in enumerate(merged):
            if i in used:
                continue
            hull = list(box)
            for j in range(i + 1, len(merged)):
                if j in used:
                    continue
                if _yxyx_iou(hull, merged[j]) > threshold:
                    other = merged[j]
                    hull = [
                        min(hull[0], other[0]),
                        min(hull[1], other[1]),
                        max(hull[2], other[2]),
                        max(hull[3], other[3]),
                    ]
                    used.add(j)
                    merge_events += 1
                    changed = True
            out.append(hull)
        merged = out
    return merged, merge_events


def _expand_yxyx(box: list[float], frac: float) -> list[float]:
    y1, x1, y2, x2 = box
    h = max(1.0, y2 - y1)
    w = max(1.0, x2 - x1)
    return [
        max(0.0, y1 - h * frac),
        max(0.0, x1 - w * frac),
        min(1000.0, y2 + h * frac),
        min(1000.0, x2 + w * frac),
    ]


def _draw_boxes_on_thumbnail(thumbnail_path: Path, boxes: list[list[float]], output_path: Path, title: str) -> Path:
    image = Image.open(thumbnail_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = _font(max(16, min(34, max(image.size) // 38)))
    line_w = max(3, max(image.size) // 280)
    colors = ["#e31a1c", "#33a02c", "#1f78b4", "#ff7f00", "#6a3d9a", "#b15928", "#00bcd4", "#f781bf"]
    w, h = image.size
    for idx, box in enumerate(boxes, start=1):
        y1, x1, y2, x2 = box
        px = [int(round(x1 / 1000 * w)), int(round(y1 / 1000 * h)), int(round(x2 / 1000 * w)), int(round(y2 / 1000 * h))]
        color = colors[(idx - 1) % len(colors)]
        draw.rectangle(px, outline=color, width=line_w)
        label = str(idx)
        label_xy = (px[0] + 3, px[1] + 3)
        tb = draw.textbbox(label_xy, label, font=font)
        draw.rectangle(tb, fill="white", outline=color, width=max(1, line_w // 2))
        draw.text(label_xy, label, fill=color, font=font)
    if title:
        title_font = _font(max(20, min(42, max(image.size) // 34)))
        tb = draw.textbbox((10, 8), title, font=title_font)
        draw.rectangle(tb, fill="white")
        draw.text((10, 8), title, font=title_font, fill="#111111")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return output_path


def _load_candidate_rows(path: Path) -> dict[tuple[int, int], dict[str, Any]]:
    rows: dict[tuple[int, int], dict[str, Any]] = {}
    for row in _read_csv(path):
        meta = json.loads(Path(row["metadata_path"]).read_text())
        candidate = meta["candidate"]
        norm = [float(v) for v in candidate["box_2d_yxyx_normalized"]]
        key = (int(row["case_index"]), int(row["candidate_order"]))
        rows[key] = {
            **row,
            "box_2d_yxyx_normalized": norm,
            "source_bbox_record": candidate.get("source_bbox_record", {}),
        }
    return rows


def _load_stage2b(path: Path) -> dict[int, dict[str, Any]]:
    return {int(r["case_index"]): r for r in _read_jsonl(path)}


def _load_stage3(path: Path) -> dict[int, dict[str, Any]]:
    return {int(r["case_index"]): r for r in _read_jsonl(path)}


def _build_case_records(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    stage1_rows = {int(r["case_index"]): r for r in _read_csv(args.stage1_cases)}
    candidates = _load_candidate_rows(args.candidates)
    stage6 = _read_jsonl(args.stage6_results)
    stage2b = _load_stage2b(args.stage2b_results)
    stage3 = _load_stage3(args.stage3_results)
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in stage6:
        key = (int(row["case_index"]), int(row["candidate_order"]))
        if key not in candidates:
            raise SystemExit(f"Missing candidate metadata for {key}")
        grouped.setdefault(key[0], []).append({**row, **candidates[key]})

    case_records: list[dict[str, Any]] = []
    overlay_dir = args.output_root / "overlays"
    for case_index in sorted(stage1_rows):
        s1 = stage1_rows[case_index]
        rows = sorted(grouped.get(case_index, []), key=lambda r: int(r["candidate_order"]))
        yes_rows = [r for r in rows if r["tissue_focus_decision"] == "yes"]
        boxes = [r["box_2d_yxyx_normalized"] for r in yes_rows]
        merged, merge_events = _merge_iou_yxyx(boxes, args.merge_iou_threshold)
        expanded = [_expand_yxyx(box, args.expand_frac) for box in merged]
        final_overlay = _draw_boxes_on_thumbnail(
            Path(s1["thumbnail_path"]),
            expanded,
            overlay_dir / f"{case_index:03d}_{_safe_slug(s1['case_display'])}_final_overlay.png",
            f"Final filtered+merged boxes: {len(expanded)}",
        )
        s2 = stage2b.get(case_index, {})
        s3 = stage3.get(case_index, {})
        stage3_used = bool(s2.get("final_non_minor_detection_failure") and s3.get("stage3_overlay_path"))
        record = {
            "case_index": case_index,
            "case_display": s1["case_display"],
            "thumbnail_path": s1["thumbnail_path"],
            "stage1_raw_overlay_path": s1["raw_overlay_path"],
            "stage1_raw_count": int(s1.get("raw_rot0_count") or 0),
            "stage1_final_count": int(s1.get("final_count") or 0),
            "stage1_raw_response_status": s1.get("raw_response_status", ""),
            "stage2b_final_non_minor_detection_failure": bool(s2.get("final_non_minor_detection_failure")),
            "stage2b_final_answer": s2.get("final_answer", ""),
            "stage2b_final_justification": s2.get("final_justification", ""),
            "stage2a_review_text": s2.get("source_review_text", ""),
            "stage3_used": stage3_used,
            "stage3_overlay_path": s3.get("stage3_overlay_path", ""),
            "stage3_detection_count": len(s3.get("detections", [])) if s3 else 0,
            "stage4_candidate_count": len(rows),
            "stage6_yes_count": len(yes_rows),
            "stage6_no_count": sum(1 for r in rows if r["tissue_focus_decision"] == "no"),
            "stage6_unknown_count": sum(1 for r in rows if r["tissue_focus_decision"] == "unknown"),
            "pre_merge_yes_boxes": boxes,
            "post_filter_merge_iou_threshold": args.merge_iou_threshold,
            "post_filter_merge_events": merge_events,
            "post_merge_boxes_yxyx_normalized": merged,
            "final_expand_fraction": args.expand_frac,
            "final_boxes_yxyx_normalized": expanded,
            "final_overlay_path": str(final_overlay),
            "stage6_candidates": rows,
        }
        case_records.append(record)
    summary = {
        "created_at": _timestamp(),
        "ticket": "PER-207",
        "git_commit": _repo_git_commit(),
        "prompt_version": PROMPT_VERSION,
        "cases": len(case_records),
        "stage4_candidates": sum(r["stage4_candidate_count"] for r in case_records),
        "stage6_yes": sum(r["stage6_yes_count"] for r in case_records),
        "stage6_no": sum(r["stage6_no_count"] for r in case_records),
        "stage6_unknown": sum(r["stage6_unknown_count"] for r in case_records),
        "final_boxes": sum(len(r["final_boxes_yxyx_normalized"]) for r in case_records),
        "stage3_used_cases": [r["case_index"] for r in case_records if r["stage3_used"]],
        "merge_iou_threshold": args.merge_iou_threshold,
        "final_expand_fraction": args.expand_frac,
    }
    return case_records, summary


def _paste_fit(page: Image.Image, image_path: str | Path, box: tuple[int, int, int, int]) -> None:
    x, y, w, h = box
    if not image_path:
        draw = ImageDraw.Draw(page)
        draw.rectangle((x, y, x + w, y + h), fill="#f4f4f4", outline="#cccccc")
        draw.text((x + 20, y + h // 2), "Not run", font=_font(26), fill="#555555")
        return
    path = Path(image_path)
    if not path.exists():
        draw = ImageDraw.Draw(page)
        draw.rectangle((x, y, x + w, y + h), fill="#f4f4f4", outline="#cccccc")
        draw.text((x + 20, y + h // 2), "Missing image", font=_font(26), fill="#aa0000")
        return
    image = _thumb(path, (w, h))
    page.paste(image, (x, y))


def _draw_crop_grid(page: Image.Image, rows: list[dict[str, Any]], xy: tuple[int, int], cell: tuple[int, int], cols: int) -> int:
    draw = ImageDraw.Draw(page)
    font = _font(15)
    small = _font(13)
    x0, y0 = xy
    cell_w, cell_h = cell
    for idx, row in enumerate(rows[:16]):
        x = x0 + (idx % cols) * cell_w
        y = y0 + (idx // cols) * cell_h
        decision = row["tissue_focus_decision"]
        color = {"yes": "#188038", "no": "#d93025", "unknown": "#5f6368"}.get(decision, "#5f6368")
        draw.text((x, y), f"{int(row['candidate_order']):02d} {decision}", font=font, fill=color)
        _paste_fit(page, row["selected_overlay_path"], (x, y + 22, cell_w - 12, cell_h - 58))
        raw = str(row.get("raw_response", "")).replace("\n", " ")
        draw.text((x, y + cell_h - 31), raw[:52], font=small, fill="#333333")
    if len(rows) > 16:
        draw.text((x0, y0 + 4 * cell_h + 10), f"{len(rows) - 16} more crop reviews in CSV/JSONL", font=font, fill="#111111")
    return y0 + ((min(len(rows), 16) + cols - 1) // cols) * cell_h


def _draw_cover(summary: dict[str, Any], args: argparse.Namespace) -> Image.Image:
    page = Image.new("RGB", (2400, 3200), "white")
    draw = ImageDraw.Draw(page)
    title = _font(44)
    header = _font(28)
    body = _font(21)
    small = _font(17)
    y = 60
    draw.text((70, y), "PER-207 Stage 1-6 Final Detector Review Packet", font=title, fill="black")
    y += 62
    lines = [
        f"Cases={summary['cases']} | Stage4 crop candidates={summary['stage4_candidates']} | Stage6 yes={summary['stage6_yes']} no={summary['stage6_no']} unknown={summary['stage6_unknown']}",
        f"Final boxes={summary['final_boxes']} | Stage3-used cases={summary['stage3_used_cases']}",
        f"Postprocess: filter Stage6 no/unknown, merge standard IoU > {args.merge_iou_threshold:.2f}, expand final boxes by {args.expand_frac:.2f}.",
    ]
    for line in lines:
        draw.text((70, y), line, font=body, fill="#111111")
        y += 34
    y += 22
    draw.text((70, y), "Active Pipeline", font=header, fill="black")
    y += 40
    pipeline = [
        "1. Stage 1 high-recall raw rot0 detector.",
        "2. Stage 2a high-thinking thumbnail reviewer for missed tissue / overcoverage.",
        "3. Stage 2b low-thinking text router for non-minor failure.",
        "4. Stage 3 feedback redetection only for Stage 2b-positive cases.",
        "5. Stage 4 crop export: 30% padded WSI rereads near 1024 px max dimension.",
        "6. Stage 6 high-thinking crop tissue-vs-artifact yes/no filter.",
        "7. Deterministic postprocess: no agentic bbox refinement; artifact filter, IoU merge, 10% margin.",
    ]
    for line in pipeline:
        y = _draw_wrapped(draw, (90, y), line, 140, small)
    return page


def _draw_case_page(record: dict[str, Any]) -> Image.Image:
    page = Image.new("RGB", (2400, 3200), "white")
    draw = ImageDraw.Draw(page)
    title = _font(34)
    header = _font(23)
    body = _font(17)
    small = _font(14)
    y = 42
    draw.text((50, y), record["case_display"], font=title, fill="black")
    y += 46
    summary = (
        f"stage1_raw={record['stage1_raw_count']} | stage2b={record['stage2b_final_answer']} "
        f"({record['stage2b_final_non_minor_detection_failure']}) | stage3_used={record['stage3_used']} "
        f"stage3_boxes={record['stage3_detection_count']} | stage4_candidates={record['stage4_candidate_count']} | "
        f"stage6 yes/no/unk={record['stage6_yes_count']}/{record['stage6_no_count']}/{record['stage6_unknown_count']} | "
        f"final_boxes={len(record['final_boxes_yxyx_normalized'])} | merges={record['post_filter_merge_events']}"
    )
    y = _draw_wrapped(draw, (50, y), summary, 185, body)
    y += 18
    col_w = 740
    row_h = 520
    labels = [
        ("Source thumbnail", record["thumbnail_path"]),
        ("Stage 1 raw overlay", record["stage1_raw_overlay_path"]),
        ("Stage 3 feedback redetection" if record["stage3_used"] else "Stage 3 not used", record["stage3_overlay_path"] if record["stage3_used"] else ""),
    ]
    for idx, (label, path) in enumerate(labels):
        x = 50 + idx * 780
        draw.text((x, y), label, font=header, fill="black")
        _paste_fit(page, path, (x, y + 32, col_w, row_h))
    y += row_h + 72
    draw.text((50, y), "Stage 6 crop IO: selected-candidate overlays with tissue-vs-artifact output", font=header, fill="black")
    draw.text((1240, y), "Final filtered / merged / expanded overlay", font=header, fill="black")
    y += 32
    _draw_crop_grid(page, record["stage6_candidates"], (50, y), (280, 205), 4)
    _paste_fit(page, record["final_overlay_path"], (1240, y, 1080, 820))
    y += 870
    draw.text((50, y), "Stage 2a reviewer text", font=header, fill="black")
    y += 30
    y = _draw_wrapped(draw, (70, y), str(record["stage2a_review_text"])[:1100], 165, small, "#111111", 18)
    y += 16
    draw.text((50, y), "Stage 2b final justification", font=header, fill="black")
    y += 30
    _draw_wrapped(draw, (70, y), str(record["stage2b_final_justification"])[:700], 165, small, "#111111", 18)
    return page


def _write_pdf(output_root: Path, case_records: list[dict[str, Any]], summary: dict[str, Any], args: argparse.Namespace) -> Path:
    pages = [_draw_cover(summary, args)]
    pages.extend(_draw_case_page(record) for record in case_records)
    pdf_path = output_root / "visuals" / "stage1_to_stage6_final_detector_all100.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pages[0].save(pdf_path, "PDF", resolution=150, save_all=True, append_images=pages[1:])
    return pdf_path


def _write_outputs(case_records: list[dict[str, Any]], summary: dict[str, Any], pdf_path: Path, args: argparse.Namespace) -> None:
    output_root = args.output_root
    _write_json(output_root / "summary/stage1_to_stage6_final_summary.json", {**summary, "pdf": str(pdf_path.resolve())})
    _write_jsonl(output_root / "final_detections/stage1_to_stage6_final_cases.jsonl", case_records)
    rows = []
    for record in case_records:
        for idx, box in enumerate(record["final_boxes_yxyx_normalized"], start=1):
            rows.append(
                {
                    "case_index": record["case_index"],
                    "case_display": record["case_display"],
                    "final_box_index": idx,
                    "box_2d_yxyx_normalized": json.dumps([round(v, 3) for v in box]),
                    "stage6_yes_count": record["stage6_yes_count"],
                    "stage6_no_count": record["stage6_no_count"],
                    "post_filter_merge_events": record["post_filter_merge_events"],
                    "final_overlay_path": record["final_overlay_path"],
                }
            )
    _write_csv(
        output_root / "final_detections/stage1_to_stage6_final_boxes.csv",
        rows,
        [
            "case_index",
            "case_display",
            "final_box_index",
            "box_2d_yxyx_normalized",
            "stage6_yes_count",
            "stage6_no_count",
            "post_filter_merge_events",
            "final_overlay_path",
        ],
    )


def _write_reproduction(pdf_path: Path, args: argparse.Namespace) -> None:
    command = " ".join(
        shlex.quote(part)
        for part in [
            "python",
            "scripts/stage6_final_detection_packet.py",
            "--candidates",
            str(args.candidates.resolve()),
            "--stage6-results",
            str(args.stage6_results.resolve()),
            "--output-root",
            str(args.output_root.resolve()),
            "--merge-iou-threshold",
            str(args.merge_iou_threshold),
            "--expand-frac",
            str(args.expand_frac),
        ]
    )
    text = f"""\
Stage 1-6 final detector packet
===============================

Created: {_timestamp()}
Ticket: PER-207
Git commit: {_repo_git_commit()}
Prompt version: {PROMPT_VERSION}

Objective:
Create the all-100 review packet after Stage 6 crop tissue-vs-artifact
filtering. The final deterministic postprocess filters artifact/no/unknown crop
detections, merges remaining boxes with standard IoU > {args.merge_iou_threshold:.2f}, and
expands final boxes by {args.expand_frac:.2f}.

Inputs:
- Stage 1 cases: {args.stage1_cases.resolve()}
- Stage 2b results: {args.stage2b_results.resolve()}
- Stage 3 results: {args.stage3_results.resolve()}
- Stage 4 candidate manifest: {args.candidates.resolve()}
- Stage 6 crop review results: {args.stage6_results.resolve()}

Command:
{command}

Outputs:
- PDF: {pdf_path.resolve()}
- Summary JSON: {(args.output_root / 'summary/stage1_to_stage6_final_summary.json').resolve()}
- Final cases JSONL: {(args.output_root / 'final_detections/stage1_to_stage6_final_cases.jsonl').resolve()}
- Final boxes CSV: {(args.output_root / 'final_detections/stage1_to_stage6_final_boxes.csv').resolve()}
"""
    (args.output_root / "reproduction.txt").write_text(text)


def run(args: argparse.Namespace) -> int:
    args.output_root.mkdir(parents=True, exist_ok=True)
    case_records, summary = _build_case_records(args)
    pdf_path = _write_pdf(args.output_root, case_records, summary, args)
    _write_outputs(case_records, summary, pdf_path, args)
    _write_reproduction(pdf_path, args)
    print(json.dumps({**summary, "pdf": str(pdf_path)}, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-cases", type=Path, default=STAGE1_CASES)
    parser.add_argument("--stage2b-results", type=Path, default=STAGE2B_RESULTS)
    parser.add_argument("--stage3-results", type=Path, default=STAGE3_RESULTS)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--stage6-results", type=Path, default=DEFAULT_STAGE6_RESULTS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--merge-iou-threshold", type=float, default=0.40)
    parser.add_argument("--expand-frac", type=float, default=0.10)
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
