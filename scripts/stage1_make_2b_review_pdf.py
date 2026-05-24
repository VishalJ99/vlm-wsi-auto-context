#!/usr/bin/env python3
"""Build a visual PDF joining Stage 1, Stage 2a, and Stage 2b outputs."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import textwrap
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont

from stage1_detection_review_pilot import _repo_git_commit, _timestamp


REPO_ROOT = Path(__file__).resolve().parents[1]
RESAMPLE_LANCZOS = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
DEFAULT_REVIEW_ROOT = (
    REPO_ROOT
    / "runs"
    / "stage1_detector_pilot_v1"
    / "stage1_detection_review_v1"
    / "high_recall_pilot100_short_reviewer_high_thinking_v1"
)
DEFAULT_STAGE1_CASES = (
    REPO_ROOT
    / "runs"
    / "stage1_detector_pilot_v1"
    / "stage1_detection_review_v1"
    / "high_recall_stage1_rot0_v1"
    / "summary"
    / "high_recall_stage1_cases.csv"
)
DEFAULT_STAGE2A_REVIEWS = DEFAULT_REVIEW_ROOT / "reviews" / "edge_review_results.jsonl"
DEFAULT_STAGE2B_RESULTS = (
    DEFAULT_REVIEW_ROOT
    / "stage2b_nonminor_binary_low_thinking_v2"
    / "reviews"
    / "stage2b_trigger_router_results.jsonl"
)
DEFAULT_OUTPUT_DIR = DEFAULT_REVIEW_ROOT / "stage2b_nonminor_binary_low_thinking_v2" / "visuals"


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


TITLE_FONT = _font(34, bold=True)
HEADER_FONT = _font(24, bold=True)
BODY_FONT = _font(22)
SMALL_FONT = _font(19)
TINY_FONT = _font(17)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _read_csv_by_case(path: Path) -> dict[int, dict[str, str]]:
    with path.open(newline="") as fh:
        return {int(row["case_index"]): row for row in csv.DictReader(fh)}


def _rows_by_case(rows: Iterable[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(row["case_index"]): row for row in rows}


def _thumb(path: str | Path, box: tuple[int, int]) -> Image.Image:
    if not str(path).strip():
        image = Image.new("RGB", box, "#f2f2f2")
        ImageDraw.Draw(image).text((24, box[1] // 2 - 12), "No image path", font=SMALL_FONT, fill="#555555")
        return image
    image_path = Path(path)
    if not image_path.exists() or image_path.is_dir():
        image = Image.new("RGB", box, "#f2f2f2")
        draw = ImageDraw.Draw(image)
        draw.text((24, box[1] // 2 - 12), f"Missing image: {image_path.name}", font=SMALL_FONT, fill="#555555")
        return image
    image = Image.open(image_path).convert("RGB")
    image.thumbnail(box, RESAMPLE_LANCZOS)
    canvas = Image.new("RGB", box, "white")
    canvas.paste(image, ((box[0] - image.width) // 2, (box[1] - image.height) // 2))
    return canvas


def _wrap_text(text: str, width: int) -> list[str]:
    lines: list[str] = []
    for raw_line in str(text).splitlines() or [""]:
        if not raw_line.strip():
            lines.append("")
            continue
        lines.extend(textwrap.wrap(raw_line, width=width, replace_whitespace=False) or [""])
    return lines


def _draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    width: int,
    fill: str = "#111111",
    line_gap: int = 7,
    max_y: int = 2650,
) -> int:
    x, y = xy
    line_height = int(font.getbbox("Ag")[3] - font.getbbox("Ag")[1]) + line_gap
    for line in _wrap_text(text, width):
        if y + line_height > max_y:
            draw.text((x, y), "... [truncated]", font=font, fill="#777777")
            return y + line_height
        draw.text((x, y), line, font=font, fill=fill)
        y += line_height
    return y


def _response_text(row: dict[str, Any]) -> str:
    parsed = row.get("parsed_response")
    if isinstance(parsed, dict) and isinstance(parsed.get("raw_text"), str):
        return parsed["raw_text"].strip()
    return str(row.get("raw_response") or "").strip()


def _make_title_page(args: argparse.Namespace, stage2b_rows: dict[int, dict[str, Any]]) -> Image.Image:
    page = Image.new("RGB", (2200, 2700), "white")
    draw = ImageDraw.Draw(page)
    y = 55
    draw.text((55, y), "Stage 1 + Stage 2a + Stage 2b Low-Thinking Review", font=TITLE_FONT, fill="black")
    y += 60
    counts = Counter(str(row.get("non_minor_detection_failure", row.get("trigger_refinement"))) for row in stage2b_rows.values())
    summary = (
        f"Created: {_timestamp()}\n"
        f"Git commit: {_repo_git_commit()}\n"
        f"Cases: {len(stage2b_rows)}\n"
        f"Stage 2b non-minor-failure counts: {dict(counts)}\n"
        f"Stage 1 cases CSV: {args.stage1_cases}\n"
        f"Stage 2a reviews: {args.stage2a_reviews}\n"
        f"Stage 2b low-thinking results: {args.stage2b_results}\n"
    )
    y = _draw_wrapped(draw, (75, y), summary, BODY_FONT, 145)
    y += 35
    draw.text((55, y), "Stage 2b Prompt", font=HEADER_FONT, fill="black")
    y += 34
    prompt_path = Path(stage2b_rows[min(stage2b_rows)].get("prompt_file", ""))
    prompt_text = prompt_path.read_text().strip() if prompt_path.exists() else "Prompt file not found in results."
    _draw_wrapped(draw, (75, y), prompt_text, BODY_FONT, 145)
    return page


def _make_case_page(
    case_index: int,
    stage1: dict[str, str],
    stage2a: dict[str, Any],
    stage2b: dict[str, Any],
) -> Image.Image:
    page = Image.new("RGB", (2200, 2700), "white")
    draw = ImageDraw.Draw(page)
    y = 38
    case_display = stage1.get("case_display") or stage2a.get("case_display") or stage2b.get("case_display") or str(case_index)
    draw.text((45, y), str(case_display), font=TITLE_FONT, fill="black")
    y += 50
    stage1_line = (
        f"Stage 1: raw rot0 boxes={stage1.get('raw_rot0_count', '')} | "
        f"raw response boxes={stage1.get('raw_response_box_count', '')} | "
        f"final boxes={stage1.get('final_count', '')} | status={stage1.get('raw_response_status', '')}"
    )
    stage2a_line = (
        f"Stage 2a: model={stage2a.get('model', '')} | "
        f"thinking={stage2a.get('reasoning_effort', '')} | "
        f"reviewed boxes={stage2a.get('reviewed_bbox_count', '')}"
    )
    stage2b_value = stage2b.get("non_minor_detection_failure", stage2b.get("trigger_refinement", ""))
    stage2b_line = (
        f"Stage 2b low thinking: non_minor_detection_failure={stage2b_value} | "
        f"raw={str(stage2b.get('raw_response', '')).strip()!r}"
    )
    y = _draw_wrapped(draw, (45, y), stage1_line, BODY_FONT, 160)
    y = _draw_wrapped(draw, (45, y), stage2a_line, BODY_FONT, 160)
    y = _draw_wrapped(draw, (45, y), stage2b_line, BODY_FONT, 160)
    y += 18

    images = [
        ("Source thumbnail", stage1.get("thumbnail_path") or stage2a.get("thumbnail_path")),
        ("Stage 1 raw overlay", stage1.get("raw_overlay_path")),
        ("Stage 1 final / Stage 2a input overlay", stage2a.get("review_overlay_path") or stage1.get("final_overlay_path")),
    ]
    for x, label, path in zip((45, 770, 1495), [item[0] for item in images], [item[1] for item in images]):
        draw.text((x, y), label, font=HEADER_FONT, fill="black")
        page.paste(_thumb(path or "", (660, 420)), (x, y + 34))
    y += 500

    draw.text((45, y), "Stage 2a Reviewer Output", font=HEADER_FONT, fill="black")
    y += 34
    if stage2a.get("error"):
        y = _draw_wrapped(draw, (65, y), f"ERROR: {stage2a['error']}", SMALL_FONT, 180, "#aa0000")
    else:
        y = _draw_wrapped(draw, (65, y), _response_text(stage2a), SMALL_FONT, 180)
    y += 24
    draw.text((45, y), "Stage 2b Low-Thinking Output", font=HEADER_FONT, fill="black")
    y += 34
    y = _draw_wrapped(draw, (65, y), str(stage2b.get("raw_response", "")).strip(), BODY_FONT, 160)
    rationale = str(stage2b.get("rationale", "")).strip()
    if rationale and rationale.lower() not in {"yes", "no"}:
        y += 12
        _draw_wrapped(draw, (65, y), f"Parsed rationale: {rationale}", TINY_FONT, 190, "#555555")
    return page


def _write_reproduction(args: argparse.Namespace, pdf_path: Path, page_count: int) -> None:
    command = " ".join(
        shlex.quote(part)
        for part in [
            "python",
            "scripts/stage1_make_2b_review_pdf.py",
            "--stage1-cases",
            str(args.stage1_cases.resolve()),
            "--stage2a-reviews",
            str(args.stage2a_reviews.resolve()),
            "--stage2b-results",
            str(args.stage2b_results.resolve()),
            "--output-pdf",
            str(pdf_path.resolve()),
        ]
    )
    text = f"""\
Stage 1 + Stage 2a + Stage 2b visual PDF
========================================

Created: {_timestamp()}
Git commit: {_repo_git_commit()}
Ticket: PER-207

Command:
{command}

Inputs:
- Stage 1 cases CSV: {args.stage1_cases.resolve()}
- Stage 2a reviews JSONL: {args.stage2a_reviews.resolve()}
- Stage 2b results JSONL: {args.stage2b_results.resolve()}

Output:
- PDF: {pdf_path.resolve()}
- Pages: {page_count}
"""
    (pdf_path.parent / "reproduction.txt").write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-cases", type=Path, default=DEFAULT_STAGE1_CASES)
    parser.add_argument("--stage2a-reviews", type=Path, default=DEFAULT_STAGE2A_REVIEWS)
    parser.add_argument("--stage2b-results", type=Path, default=DEFAULT_STAGE2B_RESULTS)
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "stage1_stage2a_stage2b_low_thinking_pilot100.pdf",
    )
    args = parser.parse_args()

    stage1_rows = _read_csv_by_case(args.stage1_cases)
    stage2a_rows = _rows_by_case(_read_jsonl(args.stage2a_reviews))
    stage2b_rows = _rows_by_case(_read_jsonl(args.stage2b_results))
    case_indices = sorted(set(stage1_rows) & set(stage2a_rows) & set(stage2b_rows))
    if not case_indices:
        raise SystemExit("No overlapping cases found across Stage 1, Stage 2a, and Stage 2b inputs.")

    pages = [_make_title_page(args, {idx: stage2b_rows[idx] for idx in case_indices})]
    for case_index in case_indices:
        pages.append(_make_case_page(case_index, stage1_rows[case_index], stage2a_rows[case_index], stage2b_rows[case_index]))

    args.output_pdf.parent.mkdir(parents=True, exist_ok=True)
    pages[0].save(args.output_pdf, "PDF", resolution=150, save_all=True, append_images=pages[1:])
    _write_reproduction(args, args.output_pdf, len(pages))
    print(json.dumps({"pdf": str(args.output_pdf.resolve()), "pages": len(pages)}, indent=2))


if __name__ == "__main__":
    main()
