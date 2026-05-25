#!/usr/bin/env python3
"""Build a high-resolution Stage 6 debug PDF for selected pilot cases."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import textwrap
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw

from stage1_detection_review_pilot import _font, _repo_git_commit, _thumb, _timestamp


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATES = (
    REPO_ROOT
    / "runs/stage1_detector_pilot_v1/stage1_detection_review_v1"
    / "stage6_all100_final_detector_v1/stage4_inputs/summary/stage4_crop_prompt_packet_candidates.csv"
)
DEFAULT_OLD_RESULTS = (
    REPO_ROOT
    / "runs/stage1_detector_pilot_v1/stage1_detection_review_v1"
    / "stage6_all100_final_detector_v1/high_thinking/reviews/stage6_crop_tissue_artifact_high_thinking.jsonl"
)
DEFAULT_NEW_RESULTS = (
    REPO_ROOT
    / "runs/stage1_detector_pilot_v1/stage1_detection_review_v1"
    / "stage6_selected_case_debug_v1/high_thinking/reviews/stage6_crop_tissue_artifact_high_thinking.jsonl"
)
DEFAULT_FINAL_CASES = (
    REPO_ROOT
    / "runs/stage1_detector_pilot_v1/stage1_detection_review_v1"
    / "stage6_all100_final_detector_v1/final_packet/final_detections/stage1_to_stage6_final_cases.jsonl"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "runs/stage1_detector_pilot_v1/stage1_detection_review_v1/stage6_selected_case_debug_v1/debug_packet"
)
PROMPT_VERSION = "stage6_selected_case_debug_packet_2026-05-25"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


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


def _draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    width: int,
    font: Any,
    fill: str = "#111111",
    line_h: int = 24,
) -> int:
    x, y = xy
    for line in _wrap(text, width):
        draw.text((x, y), line, font=font, fill=fill)
        y += line_h
    return y


def _paste_fit(page: Image.Image, image_path: str | Path, box: tuple[int, int, int, int]) -> None:
    x, y, w, h = box
    path = Path(image_path)
    if not image_path or not path.exists():
        draw = ImageDraw.Draw(page)
        draw.rectangle((x, y, x + w, y + h), fill="#f4f4f4", outline="#cccccc")
        draw.text((x + 20, y + h // 2), "Missing image", font=_font(26), fill="#aa0000")
        return
    image = _thumb(path, (w, h))
    page.paste(image, (x, y))


def _box_metrics(a: list[float], b: list[float]) -> dict[str, float]:
    ay1, ax1, ay2, ax2 = a
    by1, bx1, by2, bx2 = b
    iy1, ix1 = max(ay1, by1), max(ax1, bx1)
    iy2, ix2 = min(ay2, by2), min(ax2, bx2)
    inter = max(0.0, iy2 - iy1) * max(0.0, ix2 - ix1)
    area_a = max(0.0, ay2 - ay1) * max(0.0, ax2 - ax1)
    area_b = max(0.0, by2 - by1) * max(0.0, bx2 - bx1)
    union = area_a + area_b - inter
    return {
        "intersection": inter,
        "iou": inter / union if union > 0 else 0.0,
        "intersection_over_min_area": inter / min(area_a, area_b) if min(area_a, area_b) > 0 else 0.0,
    }


def _candidate_key(row: dict[str, Any]) -> tuple[int, int]:
    return int(row["case_index"]), int(row["candidate_order"])


def _load_candidates(path: Path, indices: set[int]) -> dict[tuple[int, int], dict[str, Any]]:
    rows: dict[tuple[int, int], dict[str, Any]] = {}
    for row in _read_csv(path):
        case_index = int(row["case_index"])
        if case_index not in indices:
            continue
        metadata = json.loads(Path(row["metadata_path"]).read_text())
        candidate = metadata["candidate"]
        read_info = candidate.get("read_info", {})
        row.update(
            {
                "box_2d_yxyx_normalized": candidate["box_2d_yxyx_normalized"],
                "wsi_path": metadata.get("wsi_path", ""),
                "selected_level": read_info.get("selected_level", ""),
                "selected_downsample": read_info.get("selected_downsample", ""),
                "crop_size": read_info.get("crop_size", ""),
                "source_bbox_in_crop": read_info.get("source_bbox_in_crop", ""),
                "padded_bbox_level0": read_info.get("padded_bbox_level0", ""),
                "source_bbox_level0": read_info.get("source_bbox_level0", ""),
            }
        )
        rows[(case_index, int(row["candidate_order"]))] = row
    return rows


def _load_results(path: Path, indices: set[int]) -> dict[tuple[int, int], dict[str, Any]]:
    rows = {}
    for row in _read_jsonl(path):
        case_index = int(row["case_index"])
        if case_index in indices:
            rows[_candidate_key(row)] = row
    return rows


def _load_final_cases(path: Path, indices: set[int]) -> dict[int, dict[str, Any]]:
    return {int(row["case_index"]): row for row in _read_jsonl(path) if int(row["case_index"]) in indices}


def _build_records(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    indices = set(args.indices)
    candidates = _load_candidates(args.candidates, indices)
    old = _load_results(args.old_results, indices)
    new = _load_results(args.new_results, indices)
    final_cases = _load_final_cases(args.final_cases, indices)
    records = []
    for key in sorted(new):
        candidate = candidates[key]
        old_row = old.get(key, {})
        new_row = new[key]
        records.append(
            {
                **candidate,
                "old_decision": old_row.get("tissue_focus_decision", ""),
                "old_raw_response": old_row.get("raw_response", ""),
                "new_decision": new_row.get("tissue_focus_decision", ""),
                "new_raw_response": new_row.get("raw_response", ""),
                "new_parser_route": new_row.get("parser_route", ""),
                "flip": old_row.get("tissue_focus_decision", "") != new_row.get("tissue_focus_decision", ""),
            }
        )
    case_summary = []
    for case_index in sorted(indices):
        rows = [r for r in records if int(r["case_index"]) == case_index]
        final = final_cases.get(case_index, {})
        overlap_rows = []
        boxes = [r["box_2d_yxyx_normalized"] for r in rows if r["new_decision"] == "yes"]
        for i, box_a in enumerate(boxes):
            for j, box_b in enumerate(boxes[i + 1 :], start=i + 1):
                metrics = _box_metrics([float(v) for v in box_a], [float(v) for v in box_b])
                if metrics["iou"] > 0 or metrics["intersection_over_min_area"] > 0:
                    overlap_rows.append({"a": i + 1, "b": j + 1, **metrics})
        case_summary.append(
            {
                "case_index": case_index,
                "case_display": rows[0]["case_display"] if rows else final.get("case_display", ""),
                "candidate_count": len(rows),
                "old_counts": dict(Counter(r["old_decision"] for r in rows)),
                "new_counts": dict(Counter(r["new_decision"] for r in rows)),
                "flip_count": sum(1 for r in rows if r["flip"]),
                "final_boxes": len(final.get("final_boxes_yxyx_normalized", [])),
                "merge_events": final.get("post_filter_merge_events", ""),
                "overlap_rows": overlap_rows,
                "final_overlay_path": final.get("final_overlay_path", ""),
                "stage1_raw_overlay_path": final.get("stage1_raw_overlay_path", ""),
                "thumbnail_path": final.get("thumbnail_path", ""),
            }
        )
    summary = {
        "created_at": _timestamp(),
        "ticket": "PER-207",
        "git_commit": _repo_git_commit(),
        "prompt_version": PROMPT_VERSION,
        "indices": sorted(indices),
        "candidate_count": len(records),
        "old_counts": dict(Counter(r["old_decision"] for r in records)),
        "new_counts": dict(Counter(r["new_decision"] for r in records)),
        "flip_count": sum(1 for r in records if r["flip"]),
    }
    return records, case_summary, summary


def _draw_cover(summary: dict[str, Any], case_summary: list[dict[str, Any]]) -> Image.Image:
    page = Image.new("RGB", (2400, 3200), "white")
    draw = ImageDraw.Draw(page)
    title = _font(42)
    header = _font(28)
    body = _font(20)
    small = _font(17)
    y = 55
    draw.text((65, y), "Stage 6 Selected Case Debug Packet", font=title, fill="black")
    y += 58
    y = _draw_wrapped(
        draw,
        (65, y),
        (
            f"Cases={summary['indices']} | crops={summary['candidate_count']} | "
            f"old={summary['old_counts']} | rerun={summary['new_counts']} | flips={summary['flip_count']}"
        ),
        160,
        body,
    )
    y += 30
    draw.text((65, y), "Purpose", font=header, fill="black")
    y += 36
    y = _draw_wrapped(
        draw,
        (85, y),
        (
            "High-resolution crop-level inspection for cases flagged during manual review. "
            "Each candidate page shows the raw reread crop, the selected-candidate overlay sent to Stage 6, "
            "the previous all-100 Stage 6 decision, and the focused rerun decision."
        ),
        150,
        small,
    )
    y += 30
    draw.text((65, y), "Case Summary", font=header, fill="black")
    y += 36
    for row in case_summary:
        text = (
            f"{row['case_index']}/100 | old={row['old_counts']} rerun={row['new_counts']} "
            f"flips={row['flip_count']} final_boxes={row['final_boxes']} merges={row['merge_events']}"
        )
        y = _draw_wrapped(draw, (85, y), text, 150, small)
        overlaps = [r for r in row["overlap_rows"] if r["intersection_over_min_area"] >= 0.5]
        for ov in overlaps[:5]:
            y = _draw_wrapped(
                draw,
                (115, y),
                f"overlap pair {ov['a']}-{ov['b']}: IoU={ov['iou']:.3f}, intersection/min-area={ov['intersection_over_min_area']:.3f}",
                145,
                small,
                "#444444",
            )
    return page


def _draw_case_page(row: dict[str, Any]) -> Image.Image:
    page = Image.new("RGB", (2400, 3200), "white")
    draw = ImageDraw.Draw(page)
    title = _font(34)
    header = _font(24)
    body = _font(17)
    y = 42
    draw.text((55, y), f"{row['case_index']}/100 | {row['case_display']}", font=title, fill="black")
    y += 50
    draw.text(
        (55, y),
        f"old={row['old_counts']} | rerun={row['new_counts']} | flips={row['flip_count']} | final_boxes={row['final_boxes']} | merges={row['merge_events']}",
        font=body,
        fill="#111111",
    )
    y += 42
    for idx, (label, path) in enumerate(
        [
            ("Source thumbnail", row["thumbnail_path"]),
            ("Stage 1 raw overlay", row["stage1_raw_overlay_path"]),
            ("Final all-100 overlay", row["final_overlay_path"]),
        ]
    ):
        x = 55 + idx * 775
        draw.text((x, y), label, font=header, fill="black")
        _paste_fit(page, path, (x, y + 32, 720, 560))
    y += 635
    draw.text((55, y), "Overlap diagnostics among rerun yes boxes", font=header, fill="black")
    y += 32
    overlaps = [r for r in row["overlap_rows"] if r["intersection_over_min_area"] >= 0.1 or r["iou"] >= 0.1]
    if not overlaps:
        draw.text((75, y), "No overlapping rerun-yes boxes.", font=body, fill="#111111")
    else:
        for ov in overlaps[:18]:
            y = _draw_wrapped(
                draw,
                (75, y),
                (
                    f"pair {ov['a']}-{ov['b']}: IoU={ov['iou']:.3f}, "
                    f"intersection/min-area={ov['intersection_over_min_area']:.3f}, intersection={ov['intersection']:.0f}"
                ),
                170,
                body,
                "#111111",
                24,
            )
    return page


def _draw_candidate_page(row: dict[str, Any]) -> Image.Image:
    page = Image.new("RGB", (2400, 3200), "white")
    draw = ImageDraw.Draw(page)
    title = _font(34)
    header = _font(25)
    body = _font(18)
    small = _font(15)
    y = 38
    color = "#188038" if row["new_decision"] == "yes" else "#d93025" if row["new_decision"] == "no" else "#5f6368"
    draw.text(
        (55, y),
        f"{int(row['case_index']):03d} candidate {int(row['candidate_order']):02d} | old={row['old_decision']} rerun={row['new_decision']}",
        font=title,
        fill=color,
    )
    y += 48
    y = _draw_wrapped(draw, (55, y), row["case_display"], 170, body)
    y += 18
    read = (
        f"crop_size={row['crop_size']} | level={row['selected_level']} | downsample={row['selected_downsample']} | "
        f"source_bbox_in_crop={row['source_bbox_in_crop']} | parser={row['new_parser_route']} | flip={row['flip']}"
    )
    y = _draw_wrapped(draw, (55, y), read, 170, small, "#111111", 20)
    y += 28
    draw.text((55, y), "Raw higher-resolution reread crop", font=header, fill="black")
    draw.text((1235, y), "Selected-candidate overlay sent to Stage 6", font=header, fill="black")
    y += 38
    _paste_fit(page, row["crop_path"], (55, y, 1080, 1080))
    _paste_fit(page, row["selected_overlay_path"], (1235, y, 1080, 1080))
    y += 1130
    draw.text((55, y), "Previous all-100 Stage 6 output", font=header, fill="black")
    y += 32
    y = _draw_wrapped(draw, (75, y), row["old_raw_response"], 175, small, "#111111", 20)
    y += 22
    draw.text((55, y), "Focused rerun Stage 6 output", font=header, fill="black")
    y += 32
    y = _draw_wrapped(draw, (75, y), row["new_raw_response"], 175, small, "#111111", 20)
    y += 22
    draw.text((55, y), "WSI read details", font=header, fill="black")
    y += 32
    details = (
        f"wsi={row['wsi_path']} | source_bbox_level0={row['source_bbox_level0']} | "
        f"padded_bbox_level0={row['padded_bbox_level0']}"
    )
    _draw_wrapped(draw, (75, y), details, 175, small, "#111111", 20)
    return page


def _write_pdf(output_root: Path, records: list[dict[str, Any]], case_summary: list[dict[str, Any]], summary: dict[str, Any]) -> Path:
    pages = [_draw_cover(summary, case_summary)]
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[int(record["case_index"])].append(record)
    case_by_index = {row["case_index"]: row for row in case_summary}
    for case_index in sorted(grouped):
        pages.append(_draw_case_page(case_by_index[case_index]))
        for row in grouped[case_index]:
            pages.append(_draw_candidate_page(row))
    output_root.mkdir(parents=True, exist_ok=True)
    pdf_path = output_root / "visuals/stage6_selected_case_highres_debug.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pages[0].save(pdf_path, "PDF", resolution=150, save_all=True, append_images=pages[1:])
    return pdf_path


def _write_outputs(args: argparse.Namespace, records: list[dict[str, Any]], case_summary: list[dict[str, Any]], summary: dict[str, Any], pdf_path: Path) -> None:
    summary = {**summary, "pdf": str(pdf_path.resolve()), "output_root": str(args.output_root.resolve())}
    _write_json(args.output_root / "summary/stage6_selected_case_debug_summary.json", summary)
    _write_json(args.output_root / "summary/stage6_selected_case_debug_case_summary.json", case_summary)
    rows = []
    for row in records:
        rows.append(
            {
                "case_index": row["case_index"],
                "case_display": row["case_display"],
                "candidate_order": row["candidate_order"],
                "candidate_id": row["candidate_id"],
                "old_decision": row["old_decision"],
                "new_decision": row["new_decision"],
                "flip": row["flip"],
                "old_raw_response": row["old_raw_response"],
                "new_raw_response": row["new_raw_response"],
                "crop_path": row["crop_path"],
                "selected_overlay_path": row["selected_overlay_path"],
                "metadata_path": row["metadata_path"],
            }
        )
    _write_csv(
        args.output_root / "summary/stage6_selected_case_debug_candidates.csv",
        rows,
        [
            "case_index",
            "case_display",
            "candidate_order",
            "candidate_id",
            "old_decision",
            "new_decision",
            "flip",
            "old_raw_response",
            "new_raw_response",
            "crop_path",
            "selected_overlay_path",
            "metadata_path",
        ],
    )


def _write_reproduction(args: argparse.Namespace, pdf_path: Path) -> None:
    command = " ".join(
        shlex.quote(part)
        for part in [
            "python",
            "scripts/stage6_selected_case_debug_packet.py",
            "--candidates",
            str(args.candidates.resolve()),
            "--old-results",
            str(args.old_results.resolve()),
            "--new-results",
            str(args.new_results.resolve()),
            "--final-cases",
            str(args.final_cases.resolve()),
            "--output-root",
            str(args.output_root.resolve()),
            "--indices",
            *[str(i) for i in args.indices],
        ]
    )
    text = f"""\
Stage 6 selected case high-resolution debug packet
=================================================

Created: {_timestamp()}
Ticket: PER-207
Git commit: {_repo_git_commit()}
Prompt version: {PROMPT_VERSION}

Objective:
Compare the original pilot-100 Stage 6 crop tissue-vs-artifact decisions
against a focused rerun for selected manually reviewed cases. The PDF shows the
raw higher-resolution WSI reread crop, the selected-candidate overlay sent to
Stage 6, and the old/new free-text outputs for each candidate.

Inputs:
- Candidate manifest: {args.candidates.resolve()}
- Original Stage 6 results: {args.old_results.resolve()}
- Focused rerun Stage 6 results: {args.new_results.resolve()}
- Final all-100 case records: {args.final_cases.resolve()}

Command:
{command}

Outputs:
- PDF: {pdf_path.resolve()}
- Summary JSON: {(args.output_root / 'summary/stage6_selected_case_debug_summary.json').resolve()}
- Candidate CSV: {(args.output_root / 'summary/stage6_selected_case_debug_candidates.csv').resolve()}
"""
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "reproduction.txt").write_text(text)


def run(args: argparse.Namespace) -> int:
    records, case_summary, summary = _build_records(args)
    pdf_path = _write_pdf(args.output_root, records, case_summary, summary)
    _write_outputs(args, records, case_summary, summary, pdf_path)
    _write_reproduction(args, pdf_path)
    print(json.dumps({**summary, "pdf": str(pdf_path)}, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--old-results", type=Path, default=DEFAULT_OLD_RESULTS)
    parser.add_argument("--new-results", type=Path, default=DEFAULT_NEW_RESULTS)
    parser.add_argument("--final-cases", type=Path, default=DEFAULT_FINAL_CASES)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--indices", type=int, nargs="+", default=[47, 49, 74, 80, 84, 99])
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
