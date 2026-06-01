#!/usr/bin/env python3
"""Rerun tissue detection after raw-orientation coverage-review feedback."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import re
import subprocess
import textwrap
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "runs" / "stage1_detector_pilot_v1" / "review_packet" / "all_detections_manifest.csv"
DEFAULT_COVERAGE_RESULTS = (
    REPO_ROOT
    / "runs"
    / "stage1_detector_pilot_v1"
    / "stage1_detection_review_v1"
    / "raw_rot0_coverage_review_missing_cases_v1"
    / "reviews"
    / "raw_orientation_coverage_rot0_results.jsonl"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "runs"
    / "stage1_detector_pilot_v1"
    / "stage1_detection_review_v1"
    / "raw_rot0_feedback_redetect_missing_cases_v1"
)
DEFAULT_MODEL = "google/gemini-3-flash-preview"
DEFAULT_API_BASE = "https://openrouter.ai/api/v1"
PROMPT_VERSION = "stage1_raw_orientation_feedback_redetect_v1_2026-05-23"

PROMPT_TEMPLATE = """\
You are looking at a whole slide image containing tissue core biopsies at low magnification.

The previous raw detector orientation returned zero bounding boxes.
A reviewer then inspected the same thumbnail and reported that visible tissue-like foreground signal was missed.

Reviewer feedback:
{reviewer_feedback}

Task:
Rerun the tissue detection from the source thumbnail. Draw a bounding box around each visible tissue-like foreground region that should have been detected.

Inputs:
- Image 1 is the source thumbnail.
- Image 2 is the previous raw detector overlay showing zero boxes.

Output a JSON array of bounding boxes in normalized 0-1000 coordinates:
[{{"box_2d": [y_min, x_min, y_max, x_max], "label": "tissue_1"}}]

Rules:
- Use the reviewer feedback to pay attention to subtle missed tissue-like signal.
- Output only boxes around visible tissue-like foreground material.
- Ignore glass edges, pen marks, bubbles, dust, debris, and blank background.
- Do not use pathology domain knowledge.
- Do not infer control tissue, diagnosis, specimen type, or downstream handling.
- Output JSON only. Do not include prose.
"""


def _timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _repo_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
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


def _safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()


def _font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    width: int,
    fill: str,
) -> int:
    x, y = xy
    for para in str(text or "").splitlines() or [""]:
        for line in textwrap.wrap(para, width=width) or [""]:
            draw.text((x, y), line, font=font, fill=fill)
            y += 20
    return y


def _thumb(path: Path, max_size: tuple[int, int]) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail(max_size)
    return image


def _image_to_data_url(path: Path) -> str:
    mime = "image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _api_key(args: argparse.Namespace) -> str:
    key = args.api_key or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("No API key found. Set --api-key, OPENROUTER_API_KEY, or OPENAI_API_KEY.")
    return key


def _chat_with_images(args: argparse.Namespace, prompt_text: str, image_paths: list[Path]) -> tuple[str, dict[str, Any], str]:
    payload = {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt_text}]
                + [
                    {"type": "image_url", "image_url": {"url": _image_to_data_url(path)}}
                    for path in image_paths
                ],
            }
        ],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }
    req = urllib.request.Request(
        (args.api_base or DEFAULT_API_BASE).rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {_api_key(args)}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=args.timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    choice = data.get("choices", [{}])[0]
    message = choice.get("message", {}) if isinstance(choice, dict) else {}
    raw = message.get("content", "") if isinstance(message, dict) else ""
    return raw, data.get("usage", {}) or {}, str(data.get("model", ""))


def _detection_coords(item: dict[str, Any]) -> Any:
    for key, value in item.items():
        normalized_key = re.sub(r"\s+", "", str(key)).lower()
        if normalized_key in {"box_2d", "bbox_2d", "bbox"}:
            return value
        if not isinstance(value, list) or len(value) != 4:
            continue
        if re.fullmatch(r"(?:bbox|box)(?:[_-]?\d+)?", normalized_key):
            return value
        if re.fullmatch(r".*(?:bbox|box)[_-]?\d+", normalized_key):
            return value
    return None


def _is_detection_payload(payload: Any) -> bool:
    if isinstance(payload, dict):
        return bool(
            _detection_coords(payload) is not None
            or isinstance(payload.get("detected_regions"), list)
            or isinstance(payload.get("bboxes"), list)
            or isinstance(payload.get("boxes"), list)
        )
    if isinstance(payload, list):
        return len(payload) == 0 or any(isinstance(item, dict) for item in payload)
    return False


def _extract_json_payload(text: str) -> Any:
    text = text.strip()
    try:
        payload = json.loads(text)
        if _is_detection_payload(payload):
            return payload
    except json.JSONDecodeError:
        pass
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    if fenced:
        try:
            payload = json.loads(fenced.group(1))
            if _is_detection_payload(payload):
                return payload
        except json.JSONDecodeError:
            pass
    decoder = json.JSONDecoder()
    for idx, char in enumerate(text):
        if char not in "[{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        if _is_detection_payload(payload):
            return payload
    return {"raw_text": text}


def _normalised_detection_items(payload: Any, thumbnail_size: tuple[int, int]) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        if _detection_coords(payload) is not None:
            items = [payload]
        elif isinstance(payload.get("detected_regions"), list):
            items = payload["detected_regions"]
        elif isinstance(payload.get("bboxes"), list):
            items = payload["bboxes"]
        elif isinstance(payload.get("boxes"), list):
            items = payload["boxes"]
        else:
            items = []
    elif isinstance(payload, list):
        items = payload
    else:
        items = []

    detections: list[dict[str, Any]] = []
    for idx, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        coords = _detection_coords(item)
        if not isinstance(coords, list) or len(coords) != 4:
            continue
        try:
            raw_coords = [float(value) for value in coords]
        except (TypeError, ValueError):
            continue
        bbox_thumbnail, normalized_box, interpretation, warnings = _parse_detection_coords(raw_coords, thumbnail_size)
        detections.append(
            {
                "label": str(item.get("label") or f"tissue_{idx}"),
                "raw_box_2d": [round(value, 3) for value in raw_coords],
                "coordinate_interpretation": interpretation,
                "parser_warnings": warnings,
                "box_2d_yxyx_normalized": normalized_box,
                "bbox_thumbnail": bbox_thumbnail,
            }
        )
    return detections


def _parse_detection_coords(
    raw_coords: list[float],
    thumbnail_size: tuple[int, int],
) -> tuple[list[int], list[int], str, list[str]]:
    width, height = thumbnail_size
    warnings: list[str] = []

    def pixel_bbox(x1: float, y1: float, x2: float, y2: float) -> list[int]:
        sx1, sx2 = sorted((max(0.0, min(float(width), x1)), max(0.0, min(float(width), x2))))
        sy1, sy2 = sorted((max(0.0, min(float(height), y1)), max(0.0, min(float(height), y2))))
        return [round(sx1), round(sy1), round(sx2), round(sy2)]

    def normalized_from_pixel(bbox: list[int]) -> list[int]:
        x1, y1, x2, y2 = bbox
        return [
            round(y1 / float(max(1, height)) * 1000.0),
            round(x1 / float(max(1, width)) * 1000.0),
            round(y2 / float(max(1, height)) * 1000.0),
            round(x2 / float(max(1, width)) * 1000.0),
        ]

    y1, x1, y2, x2 = raw_coords
    if all(0.0 <= value <= 1000.0 for value in raw_coords):
        cy1, cy2 = sorted((max(0.0, min(1000.0, y1)), max(0.0, min(1000.0, y2))))
        cx1, cx2 = sorted((max(0.0, min(1000.0, x1)), max(0.0, min(1000.0, x2))))
        if [cy1, cx1, cy2, cx2] != raw_coords:
            warnings.append("coords_sorted_or_clipped")
        bbox = [
            round(cx1 / 1000.0 * width),
            round(cy1 / 1000.0 * height),
            round(cx2 / 1000.0 * width),
            round(cy2 / 1000.0 * height),
        ]
        return bbox, [round(cy1), round(cx1), round(cy2), round(cx2)], "normalized_yxyx", warnings

    px1, py1, px2, py2 = raw_coords
    if 0 <= px1 <= width and 0 <= px2 <= width and 0 <= py1 <= height and 0 <= py2 <= height:
        bbox = pixel_bbox(px1, py1, px2, py2)
        warnings.append("interpreted_as_pixel_xyxy_outside_prompt_schema")
        return bbox, normalized_from_pixel(bbox), "pixel_xyxy", warnings

    cy1, cy2 = sorted((max(0.0, min(1000.0, y1)), max(0.0, min(1000.0, y2))))
    cx1, cx2 = sorted((max(0.0, min(1000.0, x1)), max(0.0, min(1000.0, x2))))
    bbox = [
        round(cx1 / 1000.0 * width),
        round(cy1 / 1000.0 * height),
        round(cx2 / 1000.0 * width),
        round(cy2 / 1000.0 * height),
    ]
    warnings.append("coords_outside_0_1000_clipped_as_normalized_yxyx")
    return bbox, [round(cy1), round(cx1), round(cy2), round(cx2)], "normalized_yxyx_clipped", warnings


def _draw_overlay(thumbnail_path: Path, detections: list[dict[str, Any]], output_path: Path) -> None:
    image = Image.open(thumbnail_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = _font(28)
    colors = ["red", "green", "blue", "orange", "purple", "cyan", "magenta"]
    for idx, detection in enumerate(detections):
        x1, y1, x2, y2 = detection["bbox_thumbnail"]
        label = detection["label"]
        color = colors[idx % len(colors)]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=5)
        label_box = draw.textbbox((x1 + 4, y1 + 4), label, font=font)
        draw.rectangle(label_box, fill="white", outline=color, width=2)
        draw.text((x1 + 4, y1 + 4), label, fill=color, font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def _case_display(row: dict[str, str]) -> str:
    return (
        f"{row['index']}/100 | {row['stain']} | {row['case_id']} | "
        f"{row['Anon_Path_ID']} | {Path(row['wsi_path']).name}"
    )


def build_tasks(args: argparse.Namespace) -> list[dict[str, Any]]:
    manifest_rows = {int(row["index"]): row for row in _read_csv(args.manifest.resolve())}
    coverage_results = _read_jsonl(args.coverage_results.resolve())
    tasks: list[dict[str, Any]] = []
    for result in coverage_results:
        manifest_row = result.get("manifest_row")
        if not isinstance(manifest_row, dict):
            match = re.match(r"raw_orientation_coverage_(\d+)_", str(result.get("task_id", "")))
            if not match:
                continue
            manifest_row = manifest_rows[int(match.group(1))]
        index = int(manifest_row["index"])
        if args.indices and index not in args.indices:
            continue
        parsed = result.get("parsed_response") if isinstance(result.get("parsed_response"), dict) else {}
        review = parsed.get("coverage_review") if isinstance(parsed.get("coverage_review"), dict) else {}
        reviewer_feedback = str(review.get("reasoning") or result.get("raw_response") or "").strip()
        if not reviewer_feedback:
            reviewer_feedback = "The reviewer reported that visible tissue-like signal was missed."
        prompt = PROMPT_TEMPLATE.format(reviewer_feedback=reviewer_feedback)
        tasks.append(
            {
                "task_id": f"raw_orientation_feedback_redetect_{index:03d}_rot{result.get('rotation', args.rotation)}",
                "prompt_version": PROMPT_VERSION,
                "model": args.model,
                "case_display": _case_display(manifest_row),
                "manifest_row": manifest_row,
                "rotation": result.get("rotation", args.rotation),
                "reviewer_feedback": reviewer_feedback,
                "thumbnail_path": manifest_row["thumbnail_path"],
                "previous_overlay_path": result["overlay_path"],
                "prompt": prompt,
                "created_at": _timestamp(),
            }
        )
    return tasks


def run(args: argparse.Namespace) -> int:
    args.output_root = args.output_root.resolve()
    tasks = build_tasks(args)
    tasks_path = args.output_root / "tasks" / "feedback_redetect_tasks.jsonl"
    _write_jsonl(tasks_path, tasks)
    if args.dry_run:
        print(json.dumps({"dry_run": True, "tasks": len(tasks), "tasks_jsonl": str(tasks_path)}, indent=2))
        return 0

    results: list[dict[str, Any]] = []
    for task in tasks:
        record = {
            "task_id": task["task_id"],
            "case_display": task["case_display"],
            "prompt_version": task["prompt_version"],
            "model": args.model,
            "rotation": task["rotation"],
            "reviewer_feedback": task["reviewer_feedback"],
            "thumbnail_path": task["thumbnail_path"],
            "previous_overlay_path": task["previous_overlay_path"],
            "created_at": _timestamp(),
            "error": "",
        }
        try:
            raw, usage, response_model = _chat_with_images(
                args,
                task["prompt"],
                [Path(task["thumbnail_path"]), Path(task["previous_overlay_path"])],
            )
            parsed = _extract_json_payload(raw)
            with Image.open(task["thumbnail_path"]) as image:
                thumbnail_size = image.size
            detections = _normalised_detection_items(parsed, thumbnail_size)
            output_slug = _safe_slug(task["task_id"].replace("raw_orientation_feedback_redetect_", ""))
            overlay_path = args.output_root / "overlays" / f"{output_slug}_feedback_redetect_overlay.png"
            _draw_overlay(Path(task["thumbnail_path"]), detections, overlay_path)
            record.update(
                {
                    "raw_response": raw,
                    "parsed_response": parsed,
                    "detections": detections,
                    "detection_count": len(detections),
                    "overlay_path": str(overlay_path),
                    "usage": usage,
                    "response_model": response_model,
                }
            )
        except Exception as exc:
            record.update(
                {
                    "raw_response": "",
                    "parsed_response": {},
                    "detections": [],
                    "detection_count": 0,
                    "overlay_path": "",
                    "usage": {},
                    "response_model": "",
                    "error": repr(exc),
                }
            )
        results.append(record)

    results_path = args.output_root / "reviews" / "feedback_redetect_results.jsonl"
    _write_jsonl(results_path, results)
    write_outputs(args, tasks_path, results_path, results)
    print(
        json.dumps(
            {
                "tasks": len(tasks),
                "results_jsonl": str(results_path),
                "summary_csv": str(args.output_root / "summary" / "feedback_redetect_summary.csv"),
                "pdf": str(args.output_root / "visuals" / "feedback_redetect_report.pdf"),
            },
            indent=2,
        )
    )
    return 0


def write_outputs(args: argparse.Namespace, tasks_path: Path, results_path: Path, results: list[dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for result in results:
        rows.append(
            {
                "task_id": result.get("task_id", ""),
                "case_display": result.get("case_display", ""),
                "rotation": result.get("rotation", ""),
                "error": result.get("error", ""),
                "detection_count": result.get("detection_count", 0),
                "reviewer_feedback": result.get("reviewer_feedback", ""),
                "overlay_path": result.get("overlay_path", ""),
                "previous_overlay_path": result.get("previous_overlay_path", ""),
            }
        )
    _write_csv(
        args.output_root / "summary" / "feedback_redetect_summary.csv",
        rows,
        [
            "task_id",
            "case_display",
            "rotation",
            "error",
            "detection_count",
            "reviewer_feedback",
            "overlay_path",
            "previous_overlay_path",
        ],
    )
    _write_json(
        args.output_root / "summary" / "feedback_redetect_summary.json",
        {
            "results": len(results),
            "errors": sum(1 for result in results if result.get("error")),
            "with_detections": sum(1 for result in results if int(result.get("detection_count") or 0) > 0),
            "total_detections": sum(int(result.get("detection_count") or 0) for result in results),
        },
    )
    write_pdf(args, results)
    write_reproduction(args, tasks_path, results_path)


def write_pdf(args: argparse.Namespace, results: list[dict[str, Any]]) -> None:
    pages: list[Image.Image] = []
    title_font = _font(28)
    body_font = _font(18)
    small_font = _font(14)
    for result in results:
        page = Image.new("RGB", (2200, 2500), "white")
        draw = ImageDraw.Draw(page)
        y = 30
        draw.text((45, y), result["case_display"], font=title_font, fill="black")
        y += 44
        draw.text(
            (45, y),
            f"feedback redetect boxes={result.get('detection_count')} error={result.get('error', '')}",
            font=body_font,
            fill="black",
        )
        y += 34
        y = _draw_wrapped(draw, (45, y), "Reviewer feedback: " + str(result.get("reviewer_feedback", "")), small_font, 190, "#111111")
        y += 22
        source = _thumb(Path(result["thumbnail_path"]), (680, 430))
        prior = _thumb(Path(result["previous_overlay_path"]), (680, 430))
        redetect = _thumb(Path(result["overlay_path"]), (680, 430)) if result.get("overlay_path") else Image.new("RGB", (680, 430), "white")
        draw.text((45, y), "Source thumbnail", font=body_font, fill="black")
        draw.text((760, y), "Zero-box reviewed overlay", font=body_font, fill="black")
        draw.text((1475, y), "Feedback redetect overlay", font=body_font, fill="black")
        page.paste(source, (45, y + 35))
        page.paste(prior, (760, y + 35))
        page.paste(redetect, (1475, y + 35))
        y += 520
        draw.text((45, y), "Parsed detections", font=body_font, fill="black")
        y += 30
        parsed_text = json.dumps(result.get("detections", []), sort_keys=True)
        y = _draw_wrapped(draw, (45, y), parsed_text[:3000], small_font, 210, "#111111")
        y += 20
        draw.text((45, y), "Raw response", font=body_font, fill="black")
        y += 30
        _draw_wrapped(draw, (45, y), str(result.get("raw_response", ""))[:2600], small_font, 210, "#222222")
        pages.append(page)
    pdf_path = args.output_root / "visuals" / "feedback_redetect_report.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    if pages:
        pages[0].save(pdf_path, "PDF", resolution=150, save_all=True, append_images=pages[1:])


def write_reproduction(args: argparse.Namespace, tasks_path: Path, results_path: Path) -> None:
    text = f"""\
Stage 1 raw-orientation feedback redetection
============================================

Created: {_timestamp()}
Git commit: {_repo_git_commit()}
Ticket: PER-207
Prompt version: {PROMPT_VERSION}
Model: {args.model}
Backend: OpenRouter-compatible chat completions via urllib
Coverage-review input: {args.coverage_results.resolve()}

Command:
python scripts/stage1_raw_orientation_feedback_redetect.py \\
  --coverage-results {args.coverage_results.resolve()} \\
  --output-root {args.output_root.resolve()} \\
  --model {args.model} \\
  --temperature {args.temperature}

Outputs:
- Tasks: {tasks_path}
- Raw/parsed results: {results_path}
- Summary CSV: {args.output_root / 'summary' / 'feedback_redetect_summary.csv'}
- PDF: {args.output_root / 'visuals' / 'feedback_redetect_report.pdf'}
- Overlays: {args.output_root / 'overlays'}

Notes:
- This tests whether reviewer feedback can recover boxes after a raw single-orientation zero-box miss.
- It does not use TTA or the postprocessed merged Stage 1 bboxes.
"""
    (args.output_root / "reproduction.txt").write_text(text)


def parse_indices(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--coverage-results", type=Path, default=DEFAULT_COVERAGE_RESULTS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--indices", type=parse_indices, default=[])
    parser.add_argument("--rotation", type=int, choices=[0, 90, 180, 270], default=0)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
