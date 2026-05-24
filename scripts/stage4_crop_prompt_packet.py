#!/usr/bin/env python3
"""Build a Stage 4+ crop-input and prompt packet for selected pilot cases."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import textwrap
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont

from stage1_detection_review_pilot import (
    _font,
    _load_raw_orientation_bboxes,
    _repo_git_commit,
    _safe_slug,
    _thumb,
    _timestamp,
)
from utils.wsi_backend import close_wsi, get_pyramid_info, load_wsi, read_region_rgb


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
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "runs/stage1_detector_pilot_v1/stage1_detection_review_v1"
    / "stage4_crop_prompt_packet_v1"
)
PROMPT_DIR = REPO_ROOT / "prompts/stage1_detector_oracle"
PROMPT_FILES = {
    "stage4_crop_export_spec": PROMPT_DIR / "stage4_crop_export_spec.txt",
    "stage5a_crop_split_review": PROMPT_DIR / "stage5a_crop_split_review.txt",
    "stage6_crop_true_false_positive": PROMPT_DIR / "stage6_crop_true_false_positive.txt",
    "stage7_crop_bbox_adjustment": PROMPT_DIR / "stage7_crop_bbox_adjustment.txt",
}
DEFAULT_INDICES = [3, 11, 15, 33, 74]
PROMPT_VERSION = "stage4_crop_prompt_packet_2026-05-24"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
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


def _selected_stage1_rows(path: Path, indices: list[int]) -> list[dict[str, str]]:
    by_index = {int(row["case_index"]): row for row in _read_csv(path)}
    missing = [idx for idx in indices if idx not in by_index]
    if missing:
        raise SystemExit(f"Stage 1 case rows missing: {missing}")
    return [by_index[idx] for idx in indices]


def _boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "yes", "1"}


def _normalised_yxyx_to_level0(norm_yxyx: list[float], wsi_size: tuple[int, int]) -> list[int]:
    wsi_w, wsi_h = wsi_size
    y1, x1, y2, x2 = [float(v) for v in norm_yxyx]
    cy1, cy2 = sorted((max(0.0, min(1000.0, y1)), max(0.0, min(1000.0, y2))))
    cx1, cx2 = sorted((max(0.0, min(1000.0, x1)), max(0.0, min(1000.0, x2))))
    return [
        int(round(cx1 / 1000.0 * wsi_w)),
        int(round(cy1 / 1000.0 * wsi_h)),
        int(round(cx2 / 1000.0 * wsi_w)),
        int(round(cy2 / 1000.0 * wsi_h)),
    ]


def _pad_level0_bbox(bbox: list[int], wsi_size: tuple[int, int], padding_frac: float) -> list[int]:
    wsi_w, wsi_h = wsi_size
    x1, y1, x2, y2 = [int(v) for v in bbox]
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    pad_x = int(round(bw * padding_frac))
    pad_y = int(round(bh * padding_frac))
    return [
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(wsi_w, x2 + pad_x),
        min(wsi_h, y2 + pad_y),
    ]


def _choose_read_level(pyramid: dict[str, Any], bbox_w: int, bbox_h: int, target_max_dim: int) -> tuple[int, float, float]:
    best_above: tuple[float, int, float, float] | None = None
    best_any: tuple[float, int, float, float] | None = None
    for level, downsample in enumerate(pyramid["level_downsamples"]):
        projected = max(bbox_w / float(downsample), bbox_h / float(downsample))
        any_candidate = (abs(projected - target_max_dim), int(level), float(downsample), float(projected))
        if best_any is None or any_candidate < best_any:
            best_any = any_candidate
        if projected >= target_max_dim:
            above_candidate = (projected - target_max_dim, int(level), float(downsample), float(projected))
            if best_above is None or above_candidate < best_above:
                best_above = above_candidate
    _, level, downsample, projected = best_above or best_any or (0.0, 0, 1.0, float(max(bbox_w, bbox_h)))
    return level, downsample, projected


def _read_padded_crop(
    wsi: Any,
    reader: str,
    pyramid: dict[str, Any],
    source_bbox_level0: list[int],
    padded_bbox_level0: list[int],
    target_max_dim: int,
) -> tuple[Image.Image, dict[str, Any]]:
    px1, py1, px2, py2 = [int(v) for v in padded_bbox_level0]
    padded_w = max(1, px2 - px1)
    padded_h = max(1, py2 - py1)
    level, downsample, projected = _choose_read_level(pyramid, padded_w, padded_h, target_max_dim)
    read_w = max(1, int(math.ceil(padded_w / downsample)))
    read_h = max(1, int(math.ceil(padded_h / downsample)))
    arr = read_region_rgb(wsi, reader, x=px1, y=py1, width=read_w, height=read_h, level=level)
    crop = Image.fromarray(arr).convert("RGB")
    before_size = crop.size
    scale = 1.0
    resized = False
    if max(crop.size) > target_max_dim:
        scale = target_max_dim / float(max(crop.size))
        resampling = getattr(Image, "Resampling", Image).LANCZOS
        crop = crop.resize(
            (
                max(1, int(round(crop.size[0] * scale))),
                max(1, int(round(crop.size[1] * scale))),
            ),
            resampling,
        )
        resized = True

    sx1, sy1, sx2, sy2 = [int(v) for v in source_bbox_level0]
    box_in_crop = [
        int(round(((sx1 - px1) / downsample) * scale)),
        int(round(((sy1 - py1) / downsample) * scale)),
        int(round(((sx2 - px1) / downsample) * scale)),
        int(round(((sy2 - py1) / downsample) * scale)),
    ]
    read_info = {
        "source_bbox_level0": [sx1, sy1, sx2, sy2],
        "source_bbox_level0_size": [max(1, sx2 - sx1), max(1, sy2 - sy1)],
        "padded_bbox_level0": [px1, py1, px2, py2],
        "padded_bbox_level0_size": [padded_w, padded_h],
        "padding_fraction": None,
        "selected_level": int(level),
        "selected_downsample": float(downsample),
        "projected_long_edge_at_level": round(float(projected), 2),
        "read_size_at_level": [read_w, read_h],
        "crop_size_before_resize": list(before_size),
        "crop_size": list(crop.size),
        "resized_after_read": resized,
        "resize_scale_after_read": round(scale, 6),
        "source_bbox_in_crop": box_in_crop,
        "target_max_dim": int(target_max_dim),
    }
    return crop, read_info


def _draw_selected_overlay(crop: Image.Image, box: list[int], label: str) -> Image.Image:
    overlay = crop.copy()
    draw = ImageDraw.Draw(overlay)
    font = _font(max(22, min(44, max(crop.size) // 24)))
    line_width = max(3, max(crop.size) // 180)
    x1, y1, x2, y2 = [int(v) for v in box]
    color = "#e31a1c"
    draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)
    label_text = str(label)
    label_bbox = draw.textbbox((x1 + 5, y1 + 5), label_text, font=font)
    draw.rectangle(label_bbox, fill="white", outline=color, width=max(2, line_width // 2))
    draw.text((x1 + 5, y1 + 5), label_text, fill=color, font=font)
    return overlay


def _wrap_lines(text: str, width: int) -> list[str]:
    lines: list[str] = []
    for paragraph in str(text).splitlines() or [""]:
        lines.extend(textwrap.wrap(paragraph, width=width) or [""])
    return lines


def _draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    width_chars: int,
    fill: str = "#111111",
    line_height: int | None = None,
) -> int:
    x, y = xy
    line_height = line_height or max(18, int(font.size * 1.25) if hasattr(font, "size") else 20)
    for line in _wrap_lines(text, width_chars):
        draw.text((x, y), line, font=font, fill=fill)
        y += line_height
    return y


def _paste_fit(page: Image.Image, image_path: Path, box: tuple[int, int, int, int]) -> None:
    x, y, w, h = box
    image = _thumb(image_path, (w, h))
    page.paste(image, (x, y))


def _prompt_texts() -> dict[str, str]:
    return {name: path.read_text().strip() for name, path in PROMPT_FILES.items()}


def _load_stage2b_flags(path: Path) -> dict[int, bool]:
    return {
        int(row["case_index"]): _boolish(row.get("final_non_minor_detection_failure"))
        for row in _read_jsonl(path)
        if "case_index" in row
    }


def _load_stage3_by_case(path: Path) -> dict[int, dict[str, Any]]:
    return {
        int(row["case_index"]): row
        for row in _read_jsonl(path)
        if "case_index" in row and not row.get("error")
    }


def _case_bboxes(
    row: dict[str, str],
    stage2b_flags: dict[int, bool],
    stage3_by_case: dict[int, dict[str, Any]],
    use_stage3_when_available: bool,
    rotation: int,
) -> tuple[str, Path, list[dict[str, Any]]]:
    case_index = int(row["case_index"])
    if use_stage3_when_available and stage2b_flags.get(case_index) and case_index in stage3_by_case:
        stage3 = stage3_by_case[case_index]
        return "stage3_feedback_redetection", Path(stage3["stage3_overlay_path"]), list(stage3.get("detections", []))

    thumbnail_path = Path(row["thumbnail_path"])
    with Image.open(thumbnail_path) as image:
        thumbnail_size = image.size
    bboxes, note = _load_raw_orientation_bboxes(Path(row["bboxes_json_path"]), thumbnail_size, rotation)
    if note:
        raise SystemExit(f"Could not load raw rotation {rotation} bboxes for case {case_index}: {note}")
    return f"stage1_raw_rot{rotation}", Path(row["raw_overlay_path"]), bboxes


def _collect_case_inputs(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, str]]:
    rows = _selected_stage1_rows(args.stage1_cases, args.indices)
    stage2b_flags = _load_stage2b_flags(args.stage2b_results)
    stage3_by_case = _load_stage3_by_case(args.stage3_results)
    prompts = _prompt_texts()
    case_records: list[dict[str, Any]] = []

    for row in rows:
        case_index = int(row["case_index"])
        metadata = json.loads(Path(row["metadata_path"]).read_text())
        wsi_path = metadata["wsi_path"]
        wsi_size = (int(metadata["wsi_dimensions"]["width"]), int(metadata["wsi_dimensions"]["height"]))
        bbox_source, detector_overlay_path, bboxes = _case_bboxes(
            row,
            stage2b_flags,
            stage3_by_case,
            args.use_stage3_when_available,
            args.rotation,
        )
        case_slug = _safe_slug(f"{case_index:03d}_{row['case_display']}")
        case_dir = args.output_root / "cases" / case_slug
        crops_dir = case_dir / "candidates"
        wsi, reader = load_wsi(wsi_path, args.wsi_reader)
        try:
            pyramid = get_pyramid_info(wsi, reader)
            candidates: list[dict[str, Any]] = []
            for order, bbox in enumerate(bboxes, start=1):
                norm = [float(v) for v in bbox["box_2d_yxyx_normalized"]]
                source_bbox_l0 = _normalised_yxyx_to_level0(norm, wsi_size)
                padded_bbox_l0 = _pad_level0_bbox(source_bbox_l0, wsi_size, args.padding_frac)
                crop, read_info = _read_padded_crop(
                    wsi,
                    reader,
                    pyramid,
                    source_bbox_l0,
                    padded_bbox_l0,
                    args.max_dim,
                )
                read_info["padding_fraction"] = float(args.padding_frac)
                label = str(bbox.get("label") or f"candidate_{order:02d}")
                candidate_id = f"{order:02d}_{_safe_slug(label)}"
                candidate_dir = crops_dir / candidate_id
                candidate_dir.mkdir(parents=True, exist_ok=True)
                crop_path = candidate_dir / "crop.png"
                overlay_path = candidate_dir / "selected_candidate_overlay.png"
                metadata_path = candidate_dir / "metadata.json"
                crop.save(crop_path)
                overlay = _draw_selected_overlay(crop, read_info["source_bbox_in_crop"], str(order))
                overlay.save(overlay_path)
                candidate = {
                    "candidate_id": candidate_id,
                    "candidate_order": order,
                    "label": label,
                    "source_bbox_record": bbox,
                    "box_2d_yxyx_normalized": norm,
                    "bbox_source": bbox_source,
                    "crop_path": str(crop_path),
                    "selected_overlay_path": str(overlay_path),
                    "metadata_path": str(metadata_path),
                    "read_info": read_info,
                    "stage_inputs": {
                        "stage5a_crop_split_review": {
                            "image": str(overlay_path),
                            "prompt_file": str(PROMPT_FILES["stage5a_crop_split_review"]),
                        },
                        "stage6_crop_true_false_positive": {
                            "image": str(overlay_path),
                            "prompt_file": str(PROMPT_FILES["stage6_crop_true_false_positive"]),
                        },
                        "stage7_crop_bbox_adjustment": {
                            "image": str(overlay_path),
                            "prompt_file": str(PROMPT_FILES["stage7_crop_bbox_adjustment"]),
                        },
                    },
                }
                _write_json(
                    metadata_path,
                    {
                        "case_index": case_index,
                        "case_display": row["case_display"],
                        "wsi_path": wsi_path,
                        "wsi_reader": reader,
                        "pyramid": pyramid,
                        "candidate": candidate,
                    },
                )
                candidates.append(candidate)
        finally:
            close_wsi(wsi, reader)

        case_record = {
            "case_index": case_index,
            "case_display": row["case_display"],
            "case_slug": case_slug,
            "bbox_source": bbox_source,
            "wsi_path": wsi_path,
            "wsi_reader": reader,
            "thumbnail_path": row["thumbnail_path"],
            "detector_overlay_path": str(detector_overlay_path),
            "candidate_count": len(candidates),
            "candidates": candidates,
            "case_dir": str(case_dir),
        }
        _write_json(case_dir / "case_stage4_packet.json", case_record)
        case_records.append(case_record)

    return case_records, prompts


def _draw_cover_page(case_records: list[dict[str, Any]], prompts: dict[str, str], args: argparse.Namespace) -> Image.Image:
    page = Image.new("RGB", (2400, 3200), "white")
    draw = ImageDraw.Draw(page)
    title_font = _font(44)
    header_font = _font(28)
    body_font = _font(22)
    small_font = _font(18)
    y = 55
    draw.text((70, y), "Stage 4+ Crop-Level Input And Prompt Packet", font=title_font, fill="black")
    y += 64
    draw.text(
        (70, y),
        f"Prompt version={PROMPT_VERSION} | padding={args.padding_frac:.2f} | target max dim={args.max_dim}px",
        font=body_font,
        fill="#222222",
    )
    y += 46
    draw.text((70, y), "Subset", font=header_font, fill="black")
    y += 36
    for record in case_records:
        line = f"{record['case_display']} | source={record['bbox_source']} | candidates={record['candidate_count']}"
        y = _draw_wrapped(draw, (90, y), line, small_font, 170, "#111111", 24)
    y += 20
    for key in (
        "stage4_crop_export_spec",
        "stage5a_crop_split_review",
        "stage6_crop_true_false_positive",
        "stage7_crop_bbox_adjustment",
    ):
        draw.text((70, y), key, font=header_font, fill="black")
        y += 36
        y = _draw_wrapped(draw, (90, y), prompts[key], small_font, 150, "#111111", 23)
        y += 24
    return page


def _draw_case_summary_page(record: dict[str, Any]) -> Image.Image:
    page = Image.new("RGB", (2400, 3200), "white")
    draw = ImageDraw.Draw(page)
    title_font = _font(38)
    header_font = _font(28)
    body_font = _font(20)
    small_font = _font(17)
    y = 50
    draw.text((60, y), record["case_display"], font=title_font, fill="black")
    y += 52
    draw.text(
        (60, y),
        f"bbox source={record['bbox_source']} | candidates={record['candidate_count']} | reader={record['wsi_reader']}",
        font=body_font,
        fill="#111111",
    )
    y += 48
    draw.text((60, y), "Source thumbnail", font=header_font, fill="black")
    draw.text((1240, y), "Detector overlay used for Stage 4", font=header_font, fill="black")
    y += 38
    _paste_fit(page, Path(record["thumbnail_path"]), (60, y, 1080, 650))
    _paste_fit(page, Path(record["detector_overlay_path"]), (1240, y, 1080, 650))
    y += 720
    draw.text((60, y), "Candidate crop inputs", font=header_font, fill="black")
    y += 36
    x0 = 60
    cell_w = 560
    cell_h = 430
    cols = 4
    for idx, candidate in enumerate(record["candidates"][:16]):
        x = x0 + (idx % cols) * cell_w
        cy = y + (idx // cols) * cell_h
        draw.text((x, cy), f"{candidate['candidate_order']:02d} | {candidate['label']}", font=small_font, fill="#111111")
        _paste_fit(page, Path(candidate["selected_overlay_path"]), (x, cy + 28, cell_w - 30, cell_h - 78))
        info = candidate["read_info"]
        footer = f"L{info['selected_level']} d={info['selected_downsample']:.1f} {info['crop_size'][0]}x{info['crop_size'][1]}"
        draw.text((x, cy + cell_h - 42), footer, font=small_font, fill="#333333")
    if len(record["candidates"]) > 16:
        draw.text((60, 3000), f"Additional candidates shown on per-candidate pages: {len(record['candidates']) - 16}", font=body_font, fill="#111111")
    return page


def _draw_candidate_page(record: dict[str, Any], candidate: dict[str, Any]) -> Image.Image:
    page = Image.new("RGB", (2400, 3200), "white")
    draw = ImageDraw.Draw(page)
    title_font = _font(34)
    header_font = _font(26)
    body_font = _font(19)
    small_font = _font(16)
    y = 45
    draw.text(
        (55, y),
        f"{record['case_index']:03d} candidate {candidate['candidate_order']:02d} | {candidate['label']}",
        font=title_font,
        fill="black",
    )
    y += 48
    y = _draw_wrapped(draw, (55, y), record["case_display"], body_font, 150, "#111111", 25)
    y += 18
    read = candidate["read_info"]
    meta_line = (
        f"source={candidate['bbox_source']} | padded level0={read['padded_bbox_level0']} | "
        f"selected_level={read['selected_level']} | downsample={read['selected_downsample']:.3f} | "
        f"crop={read['crop_size'][0]}x{read['crop_size'][1]}"
    )
    y = _draw_wrapped(draw, (55, y), meta_line, body_font, 155, "#111111", 25)
    y += 24
    draw.text((55, y), "Raw padded crop", font=header_font, fill="black")
    draw.text((1240, y), "Selected-candidate overlay sent to crop-level VLMs", font=header_font, fill="black")
    y += 38
    _paste_fit(page, Path(candidate["crop_path"]), (55, y, 1080, 1080))
    _paste_fit(page, Path(candidate["selected_overlay_path"]), (1240, y, 1080, 1080))
    y += 1140
    draw.text((55, y), "Stage inputs using this overlay", font=header_font, fill="black")
    y += 36
    stage_lines = [
        "Stage 5a split review: image=selected_candidate_overlay.png, prompt=stage5a_crop_split_review.txt, Gemini 3 Flash high thinking.",
        "Stage 5b if split is flagged: split boxes are converted back to WSI coordinates, padded by 30%, reread, and reviewed again.",
        "Stage 6 crop TP/FP classifier: image=selected_candidate_overlay.png, prompt=stage6_crop_true_false_positive.txt, Gemini 3 Flash high thinking.",
        "Stage 7 bbox adjustment loop: image=selected_candidate_overlay.png, prompt=stage7_crop_bbox_adjustment.txt, max 3 iterations; small=10%, medium=25%.",
    ]
    for line in stage_lines:
        y = _draw_wrapped(draw, (75, y), line, body_font, 150, "#111111", 25)
        y += 6
    y += 14
    draw.text((55, y), "Crop metadata", font=header_font, fill="black")
    y += 34
    compact_meta = {
        "source_bbox_level0": read["source_bbox_level0"],
        "padded_bbox_level0": read["padded_bbox_level0"],
        "source_bbox_in_crop": read["source_bbox_in_crop"],
        "projected_long_edge_at_level": read["projected_long_edge_at_level"],
        "read_size_at_level": read["read_size_at_level"],
        "crop_size_before_resize": read["crop_size_before_resize"],
        "crop_size": read["crop_size"],
    }
    y = _draw_wrapped(draw, (75, y), json.dumps(compact_meta, sort_keys=True), small_font, 180, "#111111", 21)
    return page


def _write_pdf(output_root: Path, case_records: list[dict[str, Any]], prompts: dict[str, str], args: argparse.Namespace) -> Path:
    pages: list[Image.Image] = [_draw_cover_page(case_records, prompts, args)]
    for record in case_records:
        pages.append(_draw_case_summary_page(record))
        for candidate in record["candidates"]:
            pages.append(_draw_candidate_page(record, candidate))
    pdf_path = output_root / "visuals" / "stage4_crop_prompt_packet_subset.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pages[0].save(pdf_path, "PDF", resolution=150, save_all=True, append_images=pages[1:])
    return pdf_path


def _write_reproduction(output_root: Path, args: argparse.Namespace, pdf_path: Path) -> None:
    command = " ".join(
        [
            "python",
            "scripts/stage4_crop_prompt_packet.py",
            "--output-root",
            str(args.output_root.resolve()),
            "--indices",
            *[str(v) for v in args.indices],
            "--padding-frac",
            str(args.padding_frac),
            "--max-dim",
            str(args.max_dim),
            "--wsi-reader",
            args.wsi_reader,
            "--rotation",
            str(args.rotation),
        ]
    )
    text = f"""\
Stage 4+ crop prompt packet
===========================

Created: {_timestamp()}
Ticket: PER-207
Git commit: {_repo_git_commit()}
Prompt version: {PROMPT_VERSION}

Objective:
Create a reviewable packet for the first crop-level stages after thumbnail
detector review/refinement. The packet contains the deterministic Stage 4 crop
inputs and the prompts intended for Stage 5a split review, Stage 6 crop
true/false-positive classification, and Stage 7 bbox adjustment.

Inputs:
- Stage 1 case table: {args.stage1_cases.resolve()}
- Stage 2b router results: {args.stage2b_results.resolve()}
- Stage 3 redetection results: {args.stage3_results.resolve()}
- Prompt files: {PROMPT_DIR.resolve()}
- Selected case indices: {args.indices}

Command:
{command}

Output policy:
- Cases with a Stage 2b final non-minor detection failure and available Stage 3
  redetection results use Stage 3 boxes.
- Other cases use raw Stage 1 rotation-{args.rotation} boxes.
- Each selected bbox is converted back to level-0 WSI coordinates, padded by
  {args.padding_frac:.2f} on each side, clipped to WSI bounds, read from the WSI
  pyramid at a level chosen to keep the saved crop near max dimension
  {args.max_dim}px, and saved with a selected-candidate overlay.

Outputs:
- PDF: {pdf_path.resolve()}
- Manifest JSON: {(output_root / 'summary/stage4_crop_prompt_packet_manifest.json').resolve()}
- Manifest CSV: {(output_root / 'summary/stage4_crop_prompt_packet_candidates.csv').resolve()}
- Per-candidate crop inputs: {(output_root / 'cases').resolve()}
"""
    (output_root / "reproduction.txt").write_text(text)


def _candidate_summary_rows(case_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in case_records:
        for candidate in record["candidates"]:
            read = candidate["read_info"]
            rows.append(
                {
                    "case_index": record["case_index"],
                    "case_display": record["case_display"],
                    "bbox_source": record["bbox_source"],
                    "candidate_order": candidate["candidate_order"],
                    "candidate_id": candidate["candidate_id"],
                    "label": candidate["label"],
                    "crop_path": candidate["crop_path"],
                    "selected_overlay_path": candidate["selected_overlay_path"],
                    "metadata_path": candidate["metadata_path"],
                    "selected_level": read["selected_level"],
                    "selected_downsample": read["selected_downsample"],
                    "projected_long_edge_at_level": read["projected_long_edge_at_level"],
                    "crop_width": read["crop_size"][0],
                    "crop_height": read["crop_size"][1],
                    "source_bbox_level0": json.dumps(read["source_bbox_level0"]),
                    "padded_bbox_level0": json.dumps(read["padded_bbox_level0"]),
                    "source_bbox_in_crop": json.dumps(read["source_bbox_in_crop"]),
                }
            )
    return rows


def _git_diff_name_only() -> list[str]:
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only"],
            cwd=REPO_ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return [line for line in result.stdout.splitlines() if line.strip()]
    except Exception:
        return []


def run(args: argparse.Namespace) -> int:
    args.output_root.mkdir(parents=True, exist_ok=True)
    case_records, prompts = _collect_case_inputs(args)
    rows = _candidate_summary_rows(case_records)
    manifest = {
        "created_at": _timestamp(),
        "ticket": "PER-207",
        "prompt_version": PROMPT_VERSION,
        "git_commit": _repo_git_commit(),
        "git_diff_name_only_at_creation": _git_diff_name_only(),
        "indices": args.indices,
        "padding_fraction": float(args.padding_frac),
        "target_max_dim": int(args.max_dim),
        "wsi_reader_requested": args.wsi_reader,
        "prompt_files": {key: str(path) for key, path in PROMPT_FILES.items()},
        "cases": case_records,
    }
    _write_json(args.output_root / "summary/stage4_crop_prompt_packet_manifest.json", manifest)
    _write_csv(
        args.output_root / "summary/stage4_crop_prompt_packet_candidates.csv",
        rows,
        [
            "case_index",
            "case_display",
            "bbox_source",
            "candidate_order",
            "candidate_id",
            "label",
            "crop_path",
            "selected_overlay_path",
            "metadata_path",
            "selected_level",
            "selected_downsample",
            "projected_long_edge_at_level",
            "crop_width",
            "crop_height",
            "source_bbox_level0",
            "padded_bbox_level0",
            "source_bbox_in_crop",
        ],
    )
    pdf_path = _write_pdf(args.output_root, case_records, prompts, args)
    _write_reproduction(args.output_root, args, pdf_path)
    print(
        json.dumps(
            {
                "output_root": str(args.output_root),
                "pdf": str(pdf_path),
                "cases": len(case_records),
                "candidates": len(rows),
                "candidate_counts": {
                    str(record["case_index"]): record["candidate_count"] for record in case_records
                },
            },
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-cases", type=Path, default=STAGE1_CASES)
    parser.add_argument("--stage2b-results", type=Path, default=STAGE2B_RESULTS)
    parser.add_argument("--stage3-results", type=Path, default=STAGE3_RESULTS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--indices", type=int, nargs="+", default=DEFAULT_INDICES)
    parser.add_argument("--padding-frac", type=float, default=0.30)
    parser.add_argument("--max-dim", type=int, default=1024)
    parser.add_argument("--wsi-reader", default="auto", choices=["auto", "openslide", "cucim", "isyntax"])
    parser.add_argument("--rotation", type=int, default=0)
    parser.add_argument("--use-stage3-when-available", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
