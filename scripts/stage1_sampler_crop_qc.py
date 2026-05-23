#!/usr/bin/env python3
"""Run medium-power crop QC for Stage 1 bbox detections."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import math
import os
import re
import subprocess
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from utils.wsi_backend import close_wsi, get_pyramid_info, load_wsi, read_region_rgb


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PILOT_ROOT = REPO_ROOT / "runs" / "stage1_detector_pilot_v1"
DEFAULT_MANIFEST = DEFAULT_PILOT_ROOT / "review_packet" / "all_detections_manifest.csv"
DEFAULT_OUTPUT_ROOT = DEFAULT_PILOT_ROOT / "stage1_detection_review_v1" / "sampler_crop_qc"
DEFAULT_MODEL = "google/gemini-3-flash-preview"
PROMPT_VERSION = "stage1_sampler_crop_qc_v1_2026-05-15"

SAMPLER_CROP_QC_PROMPT = """\
You are reviewing medium-power crops from a whole-slide image tissue-core detector.

Each crop corresponds to one Stage 1 bbox. Some bboxes may contain true tissue
cores. Some may contain non-tissue artifacts such as crystalline material,
mounting media, glass-edge marks, bubbles, debris, or staining residue.

Use the higher-resolution crop detail to decide whether each bbox is a real
tissue-core region or should be discarded before the sampler chooses diverse
cores for expensive downstream processing.

Inputs:
- Image 1 is a labeled contact sheet for orientation only.
- Each following image is one full-resolution crop, in the order listed below.

Return only one JSON object with this exact shape:
{
  "crop_reviews": [
    {
      "bbox_id": "tissue_1",
      "is_tissue_core": true,
      "category": "true_tissue_core",
      "artifact_type": "none",
      "sampler_action": "eligible",
      "confidence": "high",
      "reasoning": "short explanation"
    }
  ],
  "non_tissue_bboxes": [],
  "uncertain_bboxes": [],
  "summary": "short explanation"
}

Allowed category values: true_tissue_core, non_tissue_artifact, uncertain.
Allowed artifact_type values: none, crystalline_or_mounting_media, glass_edge_or_slide_mark, bubble, debris_or_dust, stain_or_pen_artifact, other_artifact, uncertain.
Allowed sampler_action values: eligible, discard_non_tissue, manual_review.
Allowed confidence values: low, medium, high.

Every bbox listed below must appear exactly once in crop_reviews.
Set is_tissue_core to false when the crop lacks tissue architecture and is
mainly artifact, crystalline material, mounting medium, glass mark, bubble,
debris, or non-biological material.
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


def _font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _selected_row(manifest: Path, index: int) -> dict[str, str]:
    for row in _read_csv(manifest):
        if int(row["index"]) == index:
            return row
    raise SystemExit(f"Missing manifest index: {index}")


def _case_display(row: dict[str, str]) -> str:
    return (
        f"{row['index']}/100 | {row['stain']} | {row['case_id']} | "
        f"{row['Anon_Path_ID']} | {Path(row['wsi_path']).name}"
    )


def _safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()


def _image_to_data_url(path: Path) -> str:
    ext = path.suffix.lower()
    mime = "image/jpeg" if ext in {".jpg", ".jpeg"} else "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else {"raw_payload": payload}
    except json.JSONDecodeError:
        pass
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        try:
            payload = json.loads(fenced.group(1))
            return payload if isinstance(payload, dict) else {"raw_payload": payload}
        except json.JSONDecodeError:
            pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            payload = json.loads(match.group(0))
            return payload if isinstance(payload, dict) else {"raw_payload": payload}
        except json.JSONDecodeError:
            pass
    return {"raw_text": text}


def _choose_read_level(pyramid: dict[str, Any], bbox_w: int, bbox_h: int, max_dim: int) -> tuple[int, float, float]:
    best_above: tuple[float, int, float, float] | None = None
    best_any: tuple[float, int, float, float] | None = None
    for level, downsample in enumerate(pyramid["level_downsamples"]):
        projected = max(bbox_w / float(downsample), bbox_h / float(downsample))
        any_diff = abs(projected - max_dim)
        any_candidate = (any_diff, int(level), float(downsample), float(projected))
        if best_any is None or any_candidate < best_any:
            best_any = any_candidate
        if projected >= max_dim:
            above_diff = projected - max_dim
            above_candidate = (above_diff, int(level), float(downsample), float(projected))
            if best_above is None or above_candidate < best_above:
                best_above = above_candidate
    _, level, downsample, projected = best_above or best_any or (0.0, 0, 1.0, float(max(bbox_w, bbox_h)))
    return level, downsample, projected


def _read_bbox_crop(
    wsi: Any,
    reader: str,
    pyramid: dict[str, Any],
    bbox_level0: list[int],
    max_dim: int,
) -> tuple[Image.Image, dict[str, Any]]:
    x1, y1, x2, y2 = [int(v) for v in bbox_level0]
    bbox_w = max(1, x2 - x1)
    bbox_h = max(1, y2 - y1)
    level, downsample, projected = _choose_read_level(pyramid, bbox_w, bbox_h, max_dim)
    read_w = max(1, int(math.ceil(bbox_w / downsample)))
    read_h = max(1, int(math.ceil(bbox_h / downsample)))

    arr = read_region_rgb(wsi, reader, x=x1, y=y1, width=read_w, height=read_h, level=level)
    crop = Image.fromarray(arr).convert("RGB")
    original_size = crop.size
    long_edge = max(crop.size)
    resized = False
    if long_edge > max_dim:
        scale = max_dim / float(long_edge)
        resampling = getattr(Image, "Resampling", Image).LANCZOS
        crop = crop.resize(
            (
                max(1, int(round(crop.size[0] * scale))),
                max(1, int(round(crop.size[1] * scale))),
            ),
            resampling,
        )
        resized = True
    read_info = {
        "bbox_level0": [x1, y1, x2, y2],
        "bbox_level0_size": [bbox_w, bbox_h],
        "selected_level": level,
        "selected_downsample": downsample,
        "projected_long_edge_at_level": round(projected, 2),
        "read_size_at_level": [read_w, read_h],
        "crop_size_before_resize": list(original_size),
        "crop_size": list(crop.size),
        "crop_long_edge": max(crop.size),
        "target_max_dim": int(max_dim),
        "resized_after_read": resized,
    }
    return crop, read_info


def _thumb(path: Path, size: tuple[int, int]) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail(size, getattr(Image, "Resampling", Image).LANCZOS)
    return image


def _draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    width_chars: int,
    fill: str,
) -> int:
    x, y = xy
    for paragraph in str(text).splitlines() or [""]:
        lines = textwrap.wrap(paragraph, width=width_chars) or [""]
        for line in lines:
            draw.text((x, y), line, font=font, fill=fill)
            y += 20
    return y


def _write_contact_sheet(crops: list[dict[str, Any]], output_path: Path) -> None:
    font = _font(22)
    small = _font(16)
    cell_w, cell_h = 540, 460
    cols = 3
    rows = math.ceil(len(crops) / cols)
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    for idx, record in enumerate(crops):
        x = (idx % cols) * cell_w
        y = (idx // cols) * cell_h
        draw.rectangle((x + 8, y + 8, x + cell_w - 8, y + cell_h - 8), outline="#aaaaaa", width=2)
        title = f"{record['bbox_id']} | level {record['read_info']['selected_level']} | {record['read_info']['crop_size'][0]}x{record['read_info']['crop_size'][1]}"
        draw.text((x + 20, y + 18), title, font=font, fill="black")
        image = _thumb(Path(record["crop_path"]), (cell_w - 50, cell_h - 95))
        sheet.paste(image, (x + 25, y + 58))
        draw.text(
            (x + 20, y + cell_h - 28),
            f"L0 bbox {record['read_info']['bbox_level0_size'][0]}x{record['read_info']['bbox_level0_size'][1]}",
            font=small,
            fill="#333333",
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def _write_report_pdf(output_dir: Path, record: dict[str, Any]) -> None:
    page = Image.new("RGB", (2300, 2600), "white")
    draw = ImageDraw.Draw(page)
    title = _font(30)
    body = _font(18)
    small = _font(15)
    y = 35
    draw.text((45, y), "Stage 1 Sampler Crop QC", font=title, fill="black")
    y += 42
    draw.text((45, y), record["case_display"], font=body, fill="#222222")
    y += 35
    summary = record.get("parsed_response", {}).get("summary", "")
    non_tissue = record.get("parsed_response", {}).get("non_tissue_bboxes", [])
    uncertain = record.get("parsed_response", {}).get("uncertain_bboxes", [])
    draw.text(
        (45, y),
        f"model={record['model']} | max_dim={record['max_dim']} | non_tissue={non_tissue} | uncertain={uncertain}",
        font=body,
        fill="#111111",
    )
    y += 40
    y = _draw_wrapped(draw, (45, y), summary, small, 180, "#111111")
    y += 15
    thumb = _thumb(Path(record["thumbnail_path"]), (930, 560))
    overlay = _thumb(Path(record["overlay_path"]), (930, 560))
    sheet = _thumb(Path(record["contact_sheet_path"]), (1200, 720))
    draw.text((45, y), "Source thumbnail", font=body, fill="black")
    draw.text((1040, y), "Stage 1 overlay", font=body, fill="black")
    page.paste(thumb, (45, y + 30))
    page.paste(overlay, (1040, y + 30))
    y += 620
    draw.text((45, y), "Medium-power crop contact sheet", font=body, fill="black")
    page.paste(sheet, (45, y + 30))
    y += 790
    draw.text((45, y), "Crop reviews", font=body, fill="black")
    y += 28
    for item in record.get("parsed_response", {}).get("crop_reviews", []):
        y = _draw_wrapped(draw, (65, y), json.dumps(item, sort_keys=True), small, 180, "#111111")
    y += 10
    draw.text((45, y), "Read levels", font=body, fill="black")
    y += 28
    for crop in record.get("crops", []):
        line = {
            "bbox_id": crop["bbox_id"],
            "selected_level": crop["read_info"]["selected_level"],
            "selected_downsample": crop["read_info"]["selected_downsample"],
            "projected_long_edge_at_level": crop["read_info"]["projected_long_edge_at_level"],
            "crop_size": crop["read_info"]["crop_size"],
        }
        y = _draw_wrapped(draw, (65, y), json.dumps(line, sort_keys=True), small, 180, "#111111")
    page.save(output_dir / "sampler_crop_qc_report.pdf", "PDF", resolution=150)


def _api_settings(args: argparse.Namespace) -> tuple[str, str]:
    base_url = args.api_base or "https://openrouter.ai/api/v1"
    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("No API key found. Set --api-key, OPENROUTER_API_KEY, or OPENAI_API_KEY.")
    return base_url, api_key


def _chat_with_images(
    *,
    model: str,
    prompt_text: str,
    image_paths: list[Path],
    temperature: float,
    max_tokens: int,
    base_url: str,
    api_key: str,
) -> tuple[str, dict[str, Any], str]:
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt_text}]
                + [
                    {"type": "image_url", "image_url": {"url": _image_to_data_url(path)}}
                    for path in image_paths
                ],
            }
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    raw = response.choices[0].message.content or ""
    usage = response.usage.model_dump() if getattr(response, "usage", None) else {}
    response_model = getattr(response, "model", "")
    return raw, usage, response_model


def _build_prompt(row: dict[str, str], crops: list[dict[str, Any]]) -> str:
    crop_lines = []
    for offset, crop in enumerate(crops, start=2):
        read = crop["read_info"]
        crop_lines.append(
            "- "
            + json.dumps(
                {
                    "image_number": offset,
                    "bbox_id": crop["bbox_id"],
                    "bbox_thumbnail": crop["bbox_thumbnail"],
                    "bbox_level0": read["bbox_level0"],
                    "crop_size": read["crop_size"],
                    "selected_level": read["selected_level"],
                    "selected_downsample": read["selected_downsample"],
                },
                sort_keys=True,
            )
        )
    return (
        SAMPLER_CROP_QC_PROMPT
        + "\n\nCase:\n"
        + _case_display(row)
        + "\n\nCrop image order:\n"
        + "\n".join(crop_lines)
    )


def _write_reproduction(output_dir: Path, args: argparse.Namespace, record: dict[str, Any]) -> None:
    reproduction = f"""\
Stage 1 sampler medium-power crop QC
====================================

Created: {_timestamp()}
Git commit: {_repo_git_commit()}
Prompt version: {PROMPT_VERSION}
Model: {args.model}
Case: {record['case_display']}
Manifest: {args.manifest.resolve()}
Max crop long edge: {args.max_dim}

Command:
PYTHONPATH=/data2/vj724/python_deps/openai_py310:$PYTHONPATH \\
python scripts/stage1_sampler_crop_qc.py \\
  --manifest {args.manifest.resolve()} \\
  --output-root {args.output_root.resolve()} \\
  --index {args.index} \\
  --model {args.model} \\
  --max-dim {args.max_dim} \\
  --wsi-reader {args.wsi_reader} \\
  --temperature {args.temperature}

Outputs:
- Task: {output_dir / 'sampler_crop_qc_task.json'}
- Result: {output_dir / 'sampler_crop_qc_result.json'}
- Crop metadata: {output_dir / 'crop_metadata.json'}
- Crop contact sheet: {output_dir / 'crop_contact_sheet.png'}
- Report PDF: {output_dir / 'sampler_crop_qc_report.pdf'}

Notes:
- Each Stage 1 bbox was read from the WSI pyramid level whose projected long edge
  was the closest level at or above --max-dim when available, then downsampled
  only if the read image exceeded that max dimension. This avoids undersampling
  thin artifact-like boxes when a finer pyramid level can support an approximately
  1024px crop.
- Image 1 in the VLM call was the contact sheet; subsequent images were the
  individual crop PNGs in bbox order.
"""
    (output_dir / "reproduction.txt").write_text(reproduction)


def run_sampler_crop_qc(args: argparse.Namespace) -> int:
    manifest = args.manifest.resolve()
    row = _selected_row(manifest, args.index)
    bboxes_path = Path(row["bboxes_json_path"])
    bboxes_payload = json.loads(bboxes_path.read_text())
    bboxes = list(bboxes_payload.get("detected_regions", []))
    if not bboxes:
        raise SystemExit(f"No detected_regions in {bboxes_path}")

    case_slug = f"{int(row['index']):03d}_{Path(row['wsi_path']).stem}"
    output_dir = args.output_root.resolve() / case_slug
    crops_dir = output_dir / "crops"
    output_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    wsi, reader = load_wsi(row["wsi_path"], args.wsi_reader)
    try:
        pyramid = get_pyramid_info(wsi, reader)
        crops: list[dict[str, Any]] = []
        for bbox in bboxes:
            label = str(bbox.get("label", f"bbox_{len(crops) + 1}"))
            bbox_level0 = [int(v) for v in bbox["bbox_level0"]]
            crop, read_info = _read_bbox_crop(wsi, reader, pyramid, bbox_level0, args.max_dim)
            crop_path = crops_dir / f"{label}.png"
            crop.save(crop_path)
            crops.append(
                {
                    "bbox_id": label,
                    "crop_path": str(crop_path),
                    "bbox_thumbnail": bbox.get("bbox_thumbnail", []),
                    "bbox_normalized": bbox.get("bbox_normalized", []),
                    "read_info": read_info,
                }
            )
    finally:
        close_wsi(wsi, reader)

    contact_sheet_path = output_dir / "crop_contact_sheet.png"
    _write_contact_sheet(crops, contact_sheet_path)
    _write_json(output_dir / "crop_metadata.json", {"case": row, "reader": reader, "pyramid": pyramid, "crops": crops})

    prompt = _build_prompt(row, crops)
    task = {
        "task_id": f"sampler_crop_qc_{int(row['index']):03d}",
        "prompt_version": PROMPT_VERSION,
        "model": args.model,
        "case_display": _case_display(row),
        "thumbnail_path": row["thumbnail_path"],
        "overlay_path": row["overlay_path"],
        "contact_sheet_path": str(contact_sheet_path),
        "crop_paths": [crop["crop_path"] for crop in crops],
        "prompt": prompt,
        "created_at": _timestamp(),
    }
    _write_json(output_dir / "sampler_crop_qc_task.json", task)

    record: dict[str, Any] = {
        "task_id": task["task_id"],
        "prompt_version": PROMPT_VERSION,
        "case_display": _case_display(row),
        "model": args.model,
        "thumbnail_path": row["thumbnail_path"],
        "overlay_path": row["overlay_path"],
        "bboxes_json_path": str(bboxes_path),
        "max_dim": int(args.max_dim),
        "contact_sheet_path": str(contact_sheet_path),
        "crops": crops,
        "created_at": _timestamp(),
        "error": "",
    }
    if args.dry_run:
        record["dry_run"] = True
        _write_json(output_dir / "sampler_crop_qc_result.json", record)
        _write_reproduction(output_dir, args, record)
        print(json.dumps({"dry_run": True, "output_dir": str(output_dir), "crops": crops}, indent=2))
        return 0

    base_url, api_key = _api_settings(args)
    image_paths = [contact_sheet_path] + [Path(crop["crop_path"]) for crop in crops]
    try:
        raw, usage, response_model = _chat_with_images(
            model=args.model,
            prompt_text=prompt,
            image_paths=image_paths,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            base_url=base_url,
            api_key=api_key,
        )
        parsed = _extract_json_object(raw)
        record.update(
            {
                "raw_response": raw,
                "parsed_response": parsed,
                "usage": usage,
                "response_model": response_model,
            }
        )
    except Exception as exc:
        record["error"] = repr(exc)

    _write_json(output_dir / "sampler_crop_qc_result.json", record)
    _write_report_pdf(output_dir, record)
    _write_reproduction(output_dir, args, record)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "non_tissue_bboxes": record.get("parsed_response", {}).get("non_tissue_bboxes", []),
                "uncertain_bboxes": record.get("parsed_response", {}).get("uncertain_bboxes", []),
                "error": record.get("error", ""),
                "pdf": str(output_dir / "sampler_crop_qc_report.pdf"),
            },
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--index", type=int, default=74)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1800)
    parser.add_argument("--max-dim", type=int, default=1024)
    parser.add_argument("--wsi-reader", default="auto", choices=["auto", "openslide", "cucim", "isyntax"])
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return run_sampler_crop_qc(args)


if __name__ == "__main__":
    raise SystemExit(main())
