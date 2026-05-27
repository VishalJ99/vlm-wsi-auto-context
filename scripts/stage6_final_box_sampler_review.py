#!/usr/bin/env python3
"""Ask a VLM to choose representative postprocessed final-box crops."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import sys
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw

from stage1_detection_review_pilot import _chat_with_images, _font, _repo_git_commit, _safe_slug, _thumb, _timestamp
from stage4_crop_prompt_packet import (
    _normalised_yxyx_to_level0,
    _pad_level0_bbox,
    _read_padded_crop,
)
from utils.wsi_backend import close_wsi, get_pyramid_info, load_wsi


REPO_ROOT = Path(__file__).resolve().parents[1]
STAGE1_CASES = (
    REPO_ROOT
    / "runs/stage1_detector_pilot_v1/stage1_detection_review_v1"
    / "high_recall_stage1_rot0_v1/summary/high_recall_stage1_cases.csv"
)
FINAL_BOXES_CSV = (
    REPO_ROOT
    / "runs/stage1_detector_pilot_v1/stage1_detection_review_v1"
    / "stage6_all100_contains_explain_hires_prompt_v1/final_packet_containment_merge_v1"
    / "final_detections/stage1_to_stage6_final_boxes.csv"
)
DEFAULT_PROMPT = REPO_ROOT / "prompts/stage1_detector_oracle/stage6_final_box_sampler_raw.txt"
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "runs/stage1_detector_pilot_v1/stage1_detection_review_v1"
    / "stage6_final_box_sampler_case99_plus10_raw_prompt_v1"
)
DEFAULT_MODEL = "google/gemini-3-flash-preview"
DEFAULT_INDICES = [23, 31, 49, 70, 74, 80, 84, 85, 95, 99, 100]
PROMPT_VERSION = "stage6_final_box_sampler_raw_2026-05-27"


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


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for record in records:
            fh.write(json.dumps(record, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _api_settings(args: argparse.Namespace) -> tuple[str, str]:
    api_key = args.api_key
    if args.api_key_stdin:
        api_key = sys.stdin.readline().strip()
    api_key = api_key or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("No API key found. Set --api-key, --api-key-stdin, OPENROUTER_API_KEY, or OPENAI_API_KEY.")
    return args.api_base or "https://openrouter.ai/api/v1", api_key


def _case_rows_by_index(path: Path) -> dict[int, dict[str, str]]:
    return {int(row["case_index"]): row for row in _read_csv(path)}


def _final_box_rows_by_case(path: Path, indices: list[int]) -> dict[int, list[dict[str, Any]]]:
    wanted = set(indices)
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in _read_csv(path):
        case_index = int(row["case_index"])
        if case_index not in wanted:
            continue
        parsed = dict(row)
        parsed["case_index"] = case_index
        parsed["final_box_index"] = int(row["final_box_index"])
        parsed["box_2d_yxyx_normalized"] = [float(v) for v in json.loads(row["box_2d_yxyx_normalized"])]
        grouped.setdefault(case_index, []).append(parsed)
    for rows in grouped.values():
        rows.sort(key=lambda r: int(r["final_box_index"]))
    return dict(sorted(grouped.items()))


def _draw_region_overlay(crop: Image.Image, box: list[int], label: str) -> Image.Image:
    overlay = crop.copy()
    draw = ImageDraw.Draw(overlay)
    font = _font(max(22, min(44, max(crop.size) // 24)))
    line_width = max(3, max(crop.size) // 180)
    x1, y1, x2, y2 = [int(v) for v in box]
    color = "#e31a1c"
    draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)
    label_bbox = draw.textbbox((x1 + 5, y1 + 5), label, font=font)
    draw.rectangle(label_bbox, fill="white", outline=color, width=max(2, line_width // 2))
    draw.text((x1 + 5, y1 + 5), label, fill=color, font=font)
    return overlay


def _save_vlm_jpeg(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path, quality=92, optimize=True)


def _build_case_inputs(args: argparse.Namespace, prompt: str) -> list[dict[str, Any]]:
    case_rows = _case_rows_by_index(args.stage1_cases)
    final_rows = _final_box_rows_by_case(args.final_boxes, args.indices)
    missing = [idx for idx in args.indices if idx not in case_rows or idx not in final_rows]
    if missing:
        raise SystemExit(f"Missing selected cases or final boxes: {missing}")

    records: list[dict[str, Any]] = []
    crops_root = args.output_root / "final_box_crops"
    for case_index in args.indices:
        case_row = case_rows[case_index]
        metadata = json.loads(Path(case_row["metadata_path"]).read_text())
        wsi_path = metadata["wsi_path"]
        wsi_size = (int(metadata["wsi_dimensions"]["width"]), int(metadata["wsi_dimensions"]["height"]))
        case_slug = _safe_slug(f"{case_index:03d}_{case_row['case_display']}")
        case_dir = crops_root / "cases" / case_slug
        regions: list[dict[str, Any]] = []
        wsi, reader = load_wsi(wsi_path, args.wsi_reader)
        try:
            pyramid = get_pyramid_info(wsi, reader)
            for box_row in final_rows[case_index]:
                order = int(box_row["final_box_index"])
                norm = [float(v) for v in box_row["box_2d_yxyx_normalized"]]
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
                region_dir = case_dir / "regions" / f"{order:02d}"
                region_dir.mkdir(parents=True, exist_ok=True)
                crop_path = region_dir / "crop.png"
                overlay_path = region_dir / "selected_region_overlay.png"
                vlm_image_path = region_dir / "selected_region_overlay_vlm.jpg"
                metadata_path = region_dir / "metadata.json"
                crop.save(crop_path)
                overlay = _draw_region_overlay(crop, read_info["source_bbox_in_crop"], str(order))
                overlay.save(overlay_path)
                _save_vlm_jpeg(overlay, vlm_image_path)
                region = {
                    "region_index": order,
                    "case_index": case_index,
                    "case_display": case_row["case_display"],
                    "box_2d_yxyx_normalized": norm,
                    "source_bbox_level0": read_info["source_bbox_level0"],
                    "padded_bbox_level0": read_info["padded_bbox_level0"],
                    "source_bbox_in_crop": read_info["source_bbox_in_crop"],
                    "read_info": read_info,
                    "crop_path": str(crop_path),
                    "selected_overlay_path": str(overlay_path),
                    "vlm_image_path": str(vlm_image_path),
                    "metadata_path": str(metadata_path),
                }
                _write_json(
                    metadata_path,
                    {
                        "case_index": case_index,
                        "case_display": case_row["case_display"],
                        "wsi_path": wsi_path,
                        "wsi_reader": reader,
                        "pyramid": pyramid,
                        "region": region,
                    },
                )
                regions.append(region)
        finally:
            close_wsi(wsi, reader)

        record = {
            "task_id": f"stage6_sampler_select_{case_index:03d}",
            "case_index": case_index,
            "case_display": case_row["case_display"],
            "case_slug": case_slug,
            "wsi_path": wsi_path,
            "wsi_reader": reader,
            "region_count": len(regions),
            "regions": regions,
            "final_overlay_path": final_rows[case_index][0]["final_overlay_path"],
            "prompt": prompt,
            "prompt_version": PROMPT_VERSION,
            "model": args.model,
            "created_at": _timestamp(),
        }
        _write_json(case_dir / "case_sampler_input.json", record)
        records.append(record)
    return records


def _run_one(
    case_record: dict[str, Any],
    effort: str,
    args: argparse.Namespace,
    base_url: str,
    api_key: str,
) -> dict[str, Any]:
    record = {
        key: case_record[key]
        for key in (
            "task_id",
            "case_index",
            "case_display",
            "case_slug",
            "wsi_path",
            "wsi_reader",
            "region_count",
            "final_overlay_path",
            "prompt_version",
            "model",
            "created_at",
        )
    }
    record.update(
        {
            "reasoning_effort": effort,
            "region_indices": [int(region["region_index"]) for region in case_record["regions"]],
            "vlm_image_paths": [region["vlm_image_path"] for region in case_record["regions"]],
            "raw_response": "",
            "error": "",
            "usage": {},
            "response_model": "",
            "parse_status": "unparsed",
        }
    )
    try:
        raw, usage, response_model = _chat_with_images(
            model=args.model,
            prompt_text=case_record["prompt"],
            image_paths=[Path(path) for path in record["vlm_image_paths"]],
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            base_url=base_url,
            api_key=api_key,
            reasoning_effort=effort,
        )
        record.update({"raw_response": raw, "usage": usage, "response_model": response_model})
    except Exception as exc:
        record["error"] = repr(exc)
    return record


def _draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    width: int,
    font: Any,
    fill: str = "#111111",
    line_height: int = 23,
) -> int:
    x, y = xy
    for paragraph in str(text).splitlines() or [""]:
        for line in textwrap.wrap(paragraph, width=width) or [""]:
            draw.text((x, y), line, font=font, fill=fill)
            y += line_height
    return y


def _draw_cover(case_records: list[dict[str, Any]], prompt: str, args: argparse.Namespace) -> Image.Image:
    page = Image.new("RGB", (2400, 3200), "white")
    draw = ImageDraw.Draw(page)
    title = _font(44)
    header = _font(30)
    body = _font(24)
    small = _font(20)
    y = 60
    draw.text((70, y), "Stage 6 Final-Box Sampler Raw Review", font=title, fill="black")
    y += 70
    draw.text(
        (70, y),
        f"model={args.model} | efforts={', '.join(args.reasoning_efforts)} | cases={len(case_records)} | regions={sum(r['region_count'] for r in case_records)}",
        font=body,
        fill="#222222",
    )
    y += 54
    draw.text((70, y), f"padding={args.padding_frac:.2f} | target max dim={args.max_dim}px", font=body, fill="#222222")
    y += 58
    draw.text((70, y), "Prompt", font=header, fill="black")
    y += 40
    y = _draw_wrapped(draw, (95, y), prompt, 145, body, "#111111", 29)
    y += 40
    draw.text((70, y), "Selected Cases", font=header, fill="black")
    y += 40
    for record in case_records:
        y = _draw_wrapped(
            draw,
            (95, y),
            f"{record['case_index']:03d} | regions={record['region_count']} | {record['case_display']}",
            150,
            small,
            "#111111",
            24,
        )
    return page


def _draw_crop_grid(page: Image.Image, regions: list[dict[str, Any]], x0: int, y0: int) -> int:
    draw = ImageDraw.Draw(page)
    small = _font(18)
    panel_w, panel_h = 330, 245
    gap_x, gap_y = 25, 48
    cols = 3
    max_y = y0
    for idx, region in enumerate(regions):
        col = idx % cols
        row = idx // cols
        x = x0 + col * (panel_w + gap_x)
        y = y0 + row * (panel_h + gap_y)
        image = _thumb(Path(region["selected_overlay_path"]), (panel_w, panel_h))
        page.paste(image, (x, y))
        draw.text((x, y + image.height + 5), f"region {region['region_index']}", font=small, fill="#111111")
        max_y = max(max_y, y + panel_h + gap_y)
    return max_y


def _draw_case_page(
    case_record: dict[str, Any],
    low: dict[str, Any] | None,
    high: dict[str, Any] | None,
) -> Image.Image:
    page = Image.new("RGB", (2400, 3200), "white")
    draw = ImageDraw.Draw(page)
    title = _font(40)
    header = _font(28)
    body = _font(22)
    small = _font(19)
    y = 50
    draw.text((60, y), f"{case_record['case_index']:03d} sampler choice raw output", font=title, fill="black")
    y += 50
    y = _draw_wrapped(draw, (60, y), case_record["case_display"], 155, body, "#111111", 27)
    y += 20
    draw.text((60, y), "Final postprocessed overlay", font=header, fill="black")
    draw.text((1240, y), "Final-box crops sent to VLM", font=header, fill="black")
    y += 36
    final_overlay = _thumb(Path(case_record["final_overlay_path"]), (1120, 720))
    page.paste(final_overlay, (60, y))
    grid_bottom = _draw_crop_grid(page, case_record["regions"], 1240, y)
    y = max(y + final_overlay.height, grid_bottom) + 45
    for label, row in (("low thinking", low), ("high thinking", high)):
        draw.text((60, y), label, font=header, fill="black")
        y += 35
        if row is None:
            y = _draw_wrapped(draw, (80, y), "missing result", 170, body)
        else:
            meta = f"error={row.get('error', '')} | parse={row.get('parse_status', '')}"
            y = _draw_wrapped(draw, (80, y), meta, 170, small, "#333333", 22)
            y += 10
            y = _draw_wrapped(draw, (80, y), row.get("raw_response", ""), 170, body, "#111111", 27)
        y += 28
    return page


def _write_pdf(
    output_root: Path,
    case_records: list[dict[str, Any]],
    all_results: dict[str, list[dict[str, Any]]],
    prompt: str,
    args: argparse.Namespace,
) -> Path:
    by_effort = {
        effort: {int(row["case_index"]): row for row in rows}
        for effort, rows in all_results.items()
    }
    pages = [_draw_cover(case_records, prompt, args)]
    for case_record in case_records:
        pages.append(
            _draw_case_page(
                case_record,
                by_effort.get("low", {}).get(int(case_record["case_index"])),
                by_effort.get("high", {}).get(int(case_record["case_index"])),
            )
        )
    pdf_path = output_root / "visuals/stage6_final_box_sampler_low_vs_high.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pages[0].save(pdf_path, "PDF", resolution=150, save_all=True, append_images=pages[1:])
    return pdf_path


def _write_outputs(output_root: Path, effort: str, results: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    reviews_path = output_root / f"{effort}_thinking/reviews/stage6_final_box_sampler_{effort}_thinking.jsonl"
    _write_jsonl(reviews_path, results)
    csv_path = output_root / f"{effort}_thinking/summary/stage6_final_box_sampler_{effort}_thinking.csv"
    _write_csv(
        csv_path,
        results,
        [
            "case_index",
            "case_display",
            "region_count",
            "region_indices",
            "reasoning_effort",
            "parse_status",
            "error",
            "raw_response",
            "final_overlay_path",
            "vlm_image_paths",
        ],
    )
    summary = {
        "created_at": _timestamp(),
        "ticket": "PER-207",
        "git_commit": _repo_git_commit(),
        "prompt_version": PROMPT_VERSION,
        "prompt_file": str(args.prompt.resolve()),
        "final_boxes_csv": str(args.final_boxes.resolve()),
        "model": args.model,
        "reasoning_effort": effort,
        "cases": len(results),
        "errors": sum(1 for row in results if row.get("error")),
        "max_concurrent": args.max_concurrent,
        "max_tokens": args.max_tokens,
        "known_usage_cost_if_reported": sum(float((row.get("usage") or {}).get("cost") or 0.0) for row in results),
        "results_jsonl": str(reviews_path.resolve()),
        "summary_csv": str(csv_path.resolve()),
    }
    _write_json(output_root / f"{effort}_thinking/summary/stage6_final_box_sampler_{effort}_thinking_summary.json", summary)
    return summary


def _write_reproduction(
    output_root: Path,
    args: argparse.Namespace,
    prompt: str,
    case_records: list[dict[str, Any]],
    summaries: dict[str, dict[str, Any]],
    pdf_path: Path,
) -> None:
    command_parts = [
        "python",
        "scripts/stage6_final_box_sampler_review.py",
        "--stage1-cases",
        str(args.stage1_cases.resolve()),
        "--final-boxes",
        str(args.final_boxes.resolve()),
        "--prompt",
        str(args.prompt.resolve()),
        "--output-root",
        str(args.output_root.resolve()),
        "--model",
        args.model,
        "--reasoning-efforts",
        *args.reasoning_efforts,
        "--indices",
        *(str(v) for v in args.indices),
        "--padding-frac",
        str(args.padding_frac),
        "--max-dim",
        str(args.max_dim),
        "--wsi-reader",
        args.wsi_reader,
        "--max-concurrent",
        str(args.max_concurrent),
        "--max-tokens",
        str(args.max_tokens),
        "--temperature",
        str(args.temperature),
    ]
    if args.reuse_existing:
        command_parts.append("--reuse-existing")
    command = " ".join(shlex.quote(part) for part in command_parts)
    output_lines = [f"- Low-vs-high PDF: {pdf_path.resolve()}"]
    for effort, summary in summaries.items():
        output_lines.append(f"- {effort} results JSONL: {summary['results_jsonl']}")
        output_lines.append(f"- {effort} summary CSV: {summary['summary_csv']}")
    if args.reuse_existing:
        output_lines.append("- Reused existing model outputs; remove `--reuse-existing` to regenerate API responses.")
    text = f"""\
Stage 6 final-box sampler raw review
====================================

Created: {_timestamp()}
Ticket: PER-207
Git commit: {_repo_git_commit()}
Prompt version: {PROMPT_VERSION}
Model: {args.model}
Reasoning efforts: {', '.join(args.reasoning_efforts)}
Backend: OpenRouter-compatible chat completions
Reuse existing model outputs: {args.reuse_existing}

Objective:
Ask Gemini 3 Flash to choose up to three postprocessed tissue regions per case
for a downstream dataset, using raw free-text output from the exact prompt file.

Prompt:
{prompt}

Input policy:
- Source final boxes: {args.final_boxes.resolve()}
- Selected cases: {', '.join(str(v) for v in args.indices)}
- Each final postprocessed box is converted back to level-0 WSI coordinates,
  padded by {args.padding_frac:.2f} on each side, clipped to WSI bounds, and
  reread from the WSI pyramid at the level closest to {args.max_dim}px max
  dimension when possible.
- Each VLM call receives all labeled final-box crop overlays for one case.

Command:
{command}

Outputs:
{chr(10).join(output_lines)}

Cases and region counts:
{chr(10).join(f"- {record['case_index']:03d}: {record['region_count']} regions | {record['case_display']}" for record in case_records)}
"""
    (output_root / "reproduction.txt").write_text(text)


def run(args: argparse.Namespace) -> int:
    args.output_root.mkdir(parents=True, exist_ok=True)
    prompt = args.prompt.read_text().strip()
    case_records = _build_case_inputs(args, prompt)
    _write_jsonl(args.output_root / "tasks/stage6_final_box_sampler_cases.jsonl", case_records)
    region_rows = []
    for record in case_records:
        for region in record["regions"]:
            read = region["read_info"]
            region_rows.append(
                {
                    "case_index": record["case_index"],
                    "case_display": record["case_display"],
                    "region_index": region["region_index"],
                    "crop_path": region["crop_path"],
                    "selected_overlay_path": region["selected_overlay_path"],
                    "vlm_image_path": region["vlm_image_path"],
                    "metadata_path": region["metadata_path"],
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
    _write_csv(
        args.output_root / "summary/stage6_final_box_sampler_regions.csv",
        region_rows,
        [
            "case_index",
            "case_display",
            "region_index",
            "crop_path",
            "selected_overlay_path",
            "vlm_image_path",
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
    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "cases": len(case_records),
                    "regions": len(region_rows),
                    "case_region_counts": {str(r["case_index"]): r["region_count"] for r in case_records},
                    "prompt": prompt,
                },
                indent=2,
            )
        )
        return 0

    base_url = api_key = ""
    if not args.reuse_existing:
        base_url, api_key = _api_settings(args)
    all_results: dict[str, list[dict[str, Any]]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    for effort in args.reasoning_efforts:
        results_path = args.output_root / f"{effort}_thinking/reviews/stage6_final_box_sampler_{effort}_thinking.jsonl"
        if args.reuse_existing and results_path.exists():
            results = _read_jsonl(results_path)
        elif args.reuse_existing:
            raise SystemExit(f"Missing existing results for --reuse-existing: {results_path}")
        elif args.max_concurrent > 1:
            results = []
            with ThreadPoolExecutor(max_workers=args.max_concurrent) as pool:
                futures = [pool.submit(_run_one, case_record, effort, args, base_url, api_key) for case_record in case_records]
                for future in as_completed(futures):
                    results.append(future.result())
        else:
            results = [_run_one(case_record, effort, args, base_url, api_key) for case_record in case_records]
        results.sort(key=lambda row: int(row["case_index"]))
        all_results[effort] = results
        summaries[effort] = _write_outputs(args.output_root, effort, results, args)
    pdf_path = _write_pdf(args.output_root, case_records, all_results, prompt, args)
    root_summary = {
        "created_at": _timestamp(),
        "ticket": "PER-207",
        "git_commit": _repo_git_commit(),
        "prompt_version": PROMPT_VERSION,
        "prompt_file": str(args.prompt.resolve()),
        "final_boxes_csv": str(args.final_boxes.resolve()),
        "output_root": str(args.output_root.resolve()),
        "selected_cases": args.indices,
        "cases": len(case_records),
        "regions": len(region_rows),
        "model": args.model,
        "reasoning_efforts": args.reasoning_efforts,
        "comparison_pdf": str(pdf_path.resolve()),
        "effort_summaries": summaries,
    }
    _write_json(args.output_root / "summary/stage6_final_box_sampler_summary.json", root_summary)
    _write_reproduction(args.output_root, args, prompt, case_records, summaries, pdf_path)
    print(json.dumps(root_summary, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-cases", type=Path, default=STAGE1_CASES)
    parser.add_argument("--final-boxes", type=Path, default=FINAL_BOXES_CSV)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--reasoning-efforts", nargs="+", default=["low", "high"], choices=["low", "medium", "high"])
    parser.add_argument("--indices", type=int, nargs="+", default=DEFAULT_INDICES)
    parser.add_argument("--padding-frac", type=float, default=0.30)
    parser.add_argument("--max-dim", type=int, default=1024)
    parser.add_argument("--wsi-reader", default="auto", choices=["auto", "openslide", "cucim", "isyntax"])
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-key-stdin", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1600)
    parser.add_argument("--max-concurrent", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reuse-existing", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
