#!/usr/bin/env python3
"""Build a focused PDF for raw-vs-postprocessed Stage 2b calibration examples."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import textwrap
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from stage1_detection_review_pilot import _repo_git_commit, _timestamp


REPO_ROOT = Path(__file__).resolve().parents[1]
RESAMPLE_LANCZOS = getattr(getattr(Image, "Resampling", Image), "LANCZOS")

STAGE1_CASES = (
    REPO_ROOT
    / "runs/stage1_detector_pilot_v1/stage1_detection_review_v1"
    / "high_recall_stage1_rot0_v1/summary/high_recall_stage1_cases.csv"
)
RAW_2B_ROOT = (
    REPO_ROOT
    / "runs/stage1_detector_pilot_v1/stage1_detection_review_v1"
    / "high_recall_raw_overlay_pilot100_short_reviewer_high_thinking_v1"
    / "stage2b_nonminor_two_pass_gemini_flash_low_v1"
)
POST_2B_ROOT = (
    REPO_ROOT
    / "runs/stage1_detector_pilot_v1/stage1_detection_review_v1"
    / "high_recall_pilot100_short_reviewer_high_thinking_v1"
    / "stage2b_nonminor_two_pass_gemini_flash_low_v2"
)
DEFAULT_COMPARISON_CSV = RAW_2B_ROOT / "comparison/raw_vs_postprocessed_two_pass_comparison.csv"
DEFAULT_OUTPUT_PDF = RAW_2B_ROOT / "comparison/raw_vs_postprocessed_calibration_examples.pdf"
DEFAULT_RAW_RESULTS = RAW_2B_ROOT / "reviews/stage2b_two_pass_results.jsonl"
DEFAULT_POST_RESULTS = POST_2B_ROOT / "reviews/stage2b_two_pass_results.jsonl"
DEFAULT_AGREEMENT_CASES = [1, 22, 50, 74, 84, 85]


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
BODY_FONT = _font(20)
SMALL_FONT = _font(17)
TINY_FONT = _font(15)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _by_case(rows: list[dict[str, Any]] | list[dict[str, str]]) -> dict[int, dict[str, Any]]:
    return {int(row["case_index"]): row for row in rows}


def _parse_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "yes", "1"}:
        return True
    if text in {"false", "no", "0"}:
        return False
    if text in {"", "none", "null", "not_comparable"}:
        return None
    raise ValueError(f"Cannot parse bool: {value!r}")


def _wrap(text: str, width: int) -> list[str]:
    lines: list[str] = []
    for raw in str(text).splitlines() or [""]:
        if not raw.strip():
            lines.append("")
        else:
            lines.extend(textwrap.wrap(raw, width=width, replace_whitespace=False) or [""])
    return lines


def _draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    width: int,
    fill: str = "#111111",
    max_y: int = 2670,
    gap: int = 6,
) -> int:
    x, y = xy
    line_height = int(font.getbbox("Ag")[3] - font.getbbox("Ag")[1]) + gap
    for line in _wrap(text, width):
        if y + line_height > max_y:
            draw.text((x, y), "... [truncated]", font=font, fill="#777777")
            return y + line_height
        draw.text((x, y), line, font=font, fill=fill)
        y += line_height
    return y


def _thumb(path: str | Path | None, box: tuple[int, int]) -> Image.Image:
    if not path:
        image = Image.new("RGB", box, "#f2f2f2")
        ImageDraw.Draw(image).text((24, box[1] // 2 - 12), "No image path", font=SMALL_FONT, fill="#555555")
        return image
    image_path = Path(path)
    if not image_path.exists():
        image = Image.new("RGB", box, "#f2f2f2")
        ImageDraw.Draw(image).text((24, box[1] // 2 - 12), f"Missing: {image_path.name}", font=SMALL_FONT, fill="#555555")
        return image
    image = Image.open(image_path).convert("RGB")
    image.thumbnail(box, RESAMPLE_LANCZOS)
    canvas = Image.new("RGB", box, "white")
    canvas.paste(image, ((box[0] - image.width) // 2, (box[1] - image.height) // 2))
    return canvas


def _verdict_text(raw_final: bool | None, post_final: bool | None) -> str:
    if raw_final == post_final:
        return f"AGREE: raw={raw_final}, postprocessed={post_final}"
    return f"FLIP: raw={raw_final}, postprocessed={post_final}"


def _final_line(prefix: str, row: dict[str, Any], comparison: dict[str, Any], field_prefix: str) -> str:
    first = row.get("first_non_minor_detection_failure")
    final = row.get("final_non_minor_detection_failure")
    flip = comparison.get(f"{field_prefix}_2b_flip")
    justification = row.get("final_justification") or comparison.get(f"{field_prefix}_final_justification") or ""
    return f"{prefix}: first={first} final={final} pass2_flip={flip} | {justification}"


def _make_cover(
    args: argparse.Namespace,
    comparison_rows: list[dict[str, str]],
    flip_cases: list[int],
    agreement_cases: list[int],
) -> Image.Image:
    page = Image.new("RGB", (2200, 2700), "white")
    draw = ImageDraw.Draw(page)
    y = 55
    draw.text((55, y), "Raw vs Postprocessed Stage 2b Calibration Examples", font=TITLE_FONT, fill="black")
    y += 60
    comparable = [
        row
        for row in comparison_rows
        if _parse_bool(row["raw_final_non_minor"]) is not None and _parse_bool(row["post_final_non_minor"]) is not None
    ]
    raw_true = sum(_parse_bool(row["raw_final_non_minor"]) is True for row in comparison_rows)
    post_true = sum(_parse_bool(row["post_final_non_minor"]) is True for row in comparison_rows)
    post_none = sum(_parse_bool(row["post_final_non_minor"]) is None for row in comparison_rows)
    summary = (
        f"Created: {_timestamp()}\n"
        f"Git commit: {_repo_git_commit()}\n"
        "Scope: existing pilot-100 outputs only; no model rerun.\n"
        "Purpose: document how raw-overlay vs postprocessed-overlay Stage 2a wording changes the simple two-pass "
        "Stage 2b non-minor-failure router, and preserve examples where both routes agree.\n\n"
        f"Comparable cases: {len(comparable)}/100\n"
        f"Raw-overlay final triggers: {raw_true}/100\n"
        f"Postprocessed-overlay final triggers: {post_true}/100; parser blank/None: {post_none}\n"
        f"Raw-vs-postprocessed final flips: {len(flip_cases)} -> {', '.join(map(str, flip_cases))}\n"
        f"Agreement examples: {', '.join(map(str, agreement_cases))}\n\n"
        "Interpretation to carry forward: this is a lightweight calibration layer over a subjective reviewer. "
        "A second cheap text pass can suppress over-sensitive first-pass readings of minor/faint misses. After this, "
        "remaining triggers are best treated as the practical floor of reviewer ability rather than a prompt-wrapper problem; "
        "the pipeline is intentionally high recall."
    )
    y = _draw_wrapped(draw, (75, y), summary, BODY_FONT, 150)
    y += 35
    draw.text((55, y), "Inputs", font=HEADER_FONT, fill="black")
    y += 35
    inputs = (
        f"Comparison CSV: {args.comparison_csv.resolve()}\n"
        f"Stage 1 cases CSV: {args.stage1_cases.resolve()}\n"
        f"Raw-overlay Stage 2b JSONL: {args.raw_results.resolve()}\n"
        f"Postprocessed-overlay Stage 2b JSONL: {args.post_results.resolve()}"
    )
    _draw_wrapped(draw, (75, y), inputs, SMALL_FONT, 175)
    return page


def _make_case_page(
    section: str,
    case_index: int,
    comparison: dict[str, Any],
    stage1: dict[str, Any],
    raw_2b: dict[str, Any],
    post_2b: dict[str, Any],
) -> Image.Image:
    page = Image.new("RGB", (2200, 2700), "white")
    draw = ImageDraw.Draw(page)
    y = 38
    raw_final = _parse_bool(comparison["raw_final_non_minor"])
    post_final = _parse_bool(comparison["post_final_non_minor"])
    case_display = comparison.get("case_display") or stage1.get("case_display") or str(case_index)
    draw.text((45, y), f"{section} | {case_display}", font=TITLE_FONT, fill="black")
    y += 48
    verdict = _verdict_text(raw_final, post_final)
    verdict_fill = "#9b1c1c" if raw_final != post_final else "#14532d"
    y = _draw_wrapped(draw, (45, y), verdict, HEADER_FONT, 130, fill=verdict_fill)
    y += 12
    stage1_line = (
        f"Stage 1 raw boxes={stage1.get('raw_rot0_count')} | raw response boxes={stage1.get('raw_response_box_count')} | "
        f"final boxes={stage1.get('final_count')} | raw status={stage1.get('raw_response_status')}"
    )
    y = _draw_wrapped(draw, (45, y), stage1_line, BODY_FONT, 150)
    y += 18

    images = [
        ("Source thumbnail", stage1.get("thumbnail_path")),
        ("Raw Stage 1 overlay", stage1.get("raw_overlay_path")),
        ("Postprocessed/final overlay", stage1.get("final_overlay_path")),
    ]
    for x, (label, path) in zip((45, 770, 1495), images):
        draw.text((x, y), label, font=HEADER_FONT, fill="black")
        page.paste(_thumb(path, (660, 420)), (x, y + 34))
    y += 510

    draw.text((45, y), "Stage 2b Final Decisions", font=HEADER_FONT, fill="black")
    y += 35
    y = _draw_wrapped(draw, (65, y), _final_line("Raw overlay", raw_2b, comparison, "raw"), BODY_FONT, 155)
    y += 10
    y = _draw_wrapped(draw, (65, y), _final_line("Postprocessed overlay", post_2b, comparison, "post"), BODY_FONT, 155)
    y += 28

    draw.text((45, y), "Raw-Overlay Stage 2a Review Excerpt", font=HEADER_FONT, fill="black")
    y += 35
    y = _draw_wrapped(draw, (65, y), comparison.get("raw_review_excerpt", ""), SMALL_FONT, 182, max_y=1805)
    y += 22
    draw.text((45, y), "Postprocessed-Overlay Stage 2a Review Excerpt", font=HEADER_FONT, fill="black")
    y += 35
    y = _draw_wrapped(draw, (65, y), comparison.get("post_review_excerpt", ""), SMALL_FONT, 182, max_y=2475)
    y += 20

    raw_first = raw_2b.get("first_justification", "")
    post_first = post_2b.get("first_justification", "")
    compact = (
        f"Raw first pass: {raw_2b.get('first_non_minor_detection_failure')} | {raw_first}\n"
        f"Post first pass: {post_2b.get('first_non_minor_detection_failure')} | {post_first}"
    )
    _draw_wrapped(draw, (65, y), compact, TINY_FONT, 205, fill="#555555")
    return page


def _write_reproduction(args: argparse.Namespace, output_pdf: Path, page_count: int, flip_cases: list[int], agreement_cases: list[int]) -> None:
    command = " ".join(
        shlex.quote(part)
        for part in [
            "python",
            "scripts/stage1_make_raw_vs_post_calibration_pdf.py",
            "--comparison-csv",
            str(args.comparison_csv.resolve()),
            "--stage1-cases",
            str(args.stage1_cases.resolve()),
            "--raw-results",
            str(args.raw_results.resolve()),
            "--post-results",
            str(args.post_results.resolve()),
            "--agreement-cases",
            ",".join(map(str, agreement_cases)),
            "--output-pdf",
            str(output_pdf.resolve()),
        ]
    )
    text = f"""\
Raw-vs-postprocessed Stage 2b calibration examples
==================================================

Created: {_timestamp()}
Git commit: {_repo_git_commit()}
Ticket/thread: PER-207

Command:
{command}

Inputs:
- Comparison CSV: {args.comparison_csv.resolve()}
- Stage 1 cases CSV: {args.stage1_cases.resolve()}
- Raw-overlay Stage 2b JSONL: {args.raw_results.resolve()}
- Postprocessed-overlay Stage 2b JSONL: {args.post_results.resolve()}

Selected cases:
- Raw-vs-postprocessed final flips: {', '.join(map(str, flip_cases))}
- Agreement examples: {', '.join(map(str, agreement_cases))}

Output:
- PDF: {output_pdf.resolve()}
- Pages: {page_count}

Notes:
- This builder uses existing run outputs only. It does not call any model.
- The selected agreement examples are the five cases where both raw and
  postprocessed routes finally trigger, plus case 001 where both routes suppress
  a minor/faint-fragment first-pass trigger after adjudication.
"""
    (output_pdf.parent / "raw_vs_postprocessed_calibration_examples_reproduction.txt").write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-csv", type=Path, default=DEFAULT_COMPARISON_CSV)
    parser.add_argument("--stage1-cases", type=Path, default=STAGE1_CASES)
    parser.add_argument("--raw-results", type=Path, default=DEFAULT_RAW_RESULTS)
    parser.add_argument("--post-results", type=Path, default=DEFAULT_POST_RESULTS)
    parser.add_argument("--agreement-cases", default=",".join(map(str, DEFAULT_AGREEMENT_CASES)))
    parser.add_argument("--output-pdf", type=Path, default=DEFAULT_OUTPUT_PDF)
    args = parser.parse_args()

    comparison_rows = _read_csv(args.comparison_csv)
    comparison_by_case = _by_case(comparison_rows)
    stage1_by_case = _by_case(_read_csv(args.stage1_cases))
    raw_by_case = _by_case(_read_jsonl(args.raw_results))
    post_by_case = _by_case(_read_jsonl(args.post_results))

    flip_cases = [
        int(row["case_index"])
        for row in comparison_rows
        if _parse_bool(row["raw_vs_post_final_flip"]) is True
    ]
    agreement_cases = [int(part) for part in args.agreement_cases.split(",") if part.strip()]

    pages = [_make_cover(args, comparison_rows, flip_cases, agreement_cases)]
    for case_index in flip_cases:
        pages.append(
            _make_case_page(
                "Raw-vs-post final flip",
                case_index,
                comparison_by_case[case_index],
                stage1_by_case[case_index],
                raw_by_case[case_index],
                post_by_case[case_index],
            )
        )
    for case_index in agreement_cases:
        pages.append(
            _make_case_page(
                "Agreement example",
                case_index,
                comparison_by_case[case_index],
                stage1_by_case[case_index],
                raw_by_case[case_index],
                post_by_case[case_index],
            )
        )

    args.output_pdf.parent.mkdir(parents=True, exist_ok=True)
    pages[0].save(args.output_pdf, "PDF", resolution=150, save_all=True, append_images=pages[1:])
    _write_reproduction(args, args.output_pdf, len(pages), flip_cases, agreement_cases)
    print(json.dumps({"pdf": str(args.output_pdf.resolve()), "pages": len(pages)}, indent=2))


if __name__ == "__main__":
    main()
