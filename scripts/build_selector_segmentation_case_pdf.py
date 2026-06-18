#!/usr/bin/env python3
"""Render a single-case detector -> selector -> segmentation PDF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from textwrap import wrap
from typing import Iterable, Optional

from PIL import Image, ImageDraw, ImageFont


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise SystemExit(f"Expected JSON object in {path}")
    return data


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def resolve_path(path_text: str | None) -> Optional[Path]:
    if not path_text:
        return None
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path.cwd() / path).resolve()


def find_first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def fit_image(img: Image.Image, box_w: int, box_h: int) -> Image.Image:
    img = img.convert("RGB")
    scale = min(box_w / img.width, box_h / img.height)
    new_size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
    return img.resize(new_size, Image.Resampling.LANCZOS)


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    *,
    fill: str,
    font_obj: ImageFont.ImageFont,
    max_chars: int,
    line_gap: int = 4,
) -> int:
    x, y = xy
    if not text:
        return y
    for paragraph in str(text).splitlines():
        lines = wrap(paragraph, width=max_chars) or [""]
        for line in lines:
            draw.text((x, y), line, fill=fill, font=font_obj)
            bbox = draw.textbbox((x, y), line or " ", font=font_obj)
            y += bbox[3] - bbox[1] + line_gap
    return y


def draw_panel(
    page: Image.Image,
    *,
    title: str,
    image_path: Optional[Path],
    x: int,
    y: int,
    w: int,
    h: int,
    caption: str,
    allow_missing: bool,
) -> None:
    draw = ImageDraw.Draw(page)
    title_font = font(30, bold=True)
    caption_font = font(22)
    draw.text((x, y), title, fill="black", font=title_font)
    image_y = y + 42
    image_h = h - 120
    draw.rectangle([x, image_y, x + w, image_y + image_h], outline=(210, 210, 210), width=2)
    if image_path is None or not image_path.exists():
        if not allow_missing:
            raise SystemExit(f"Missing required panel image for {title}: {image_path}")
        draw.text((x + 24, image_y + 24), "missing", fill=(180, 0, 0), font=title_font)
    else:
        img = fit_image(Image.open(image_path), w - 8, image_h - 8)
        ox = x + (w - img.width) // 2
        oy = image_y + (image_h - img.height) // 2
        page.paste(img, (ox, oy))
    draw_wrapped(
        draw,
        (x, image_y + image_h + 16),
        caption,
        fill=(30, 30, 30),
        font_obj=caption_font,
        max_chars=max(28, w // 15),
    )


def resolve_detector_overlay(args: argparse.Namespace, case_input: dict) -> Optional[Path]:
    explicit = resolve_path(args.detector_overlay)
    if explicit:
        return explicit
    final_overlay = case_input.get("case_row", {}).get("final_overlay_png")
    return resolve_path(final_overlay)


def resolve_selector_overlay(args: argparse.Namespace) -> Optional[Path]:
    explicit = resolve_path(args.selector_overlay)
    if explicit:
        return explicit
    return (
        Path(args.artifact_probe_root)
        / "second_pass_verifier"
        / args.case_id
        / "revised_overlay.png"
    ).resolve()


def resolve_segmentation_overlay(run_dir: Path) -> Optional[Path]:
    candidates = [run_dir / "stage7" / "mask_overlay.png"]
    candidates.extend(sorted(run_dir.glob("bboxes/*/stage7/postprocess_after.png")))
    candidates.extend(sorted(run_dir.glob("bboxes/*/stage7/postprocess_overlay_before_after.png")))
    return find_first_existing(candidates)


def resolve_stage7_metadata(run_dir: Path) -> Optional[Path]:
    candidates = [run_dir / "stage7" / "postprocess_metadata.json"]
    candidates.extend(sorted(run_dir.glob("bboxes/*/stage7/postprocess_metadata.json")))
    return find_first_existing(candidates)


def format_stage7_metadata(path: Optional[Path]) -> Optional[str]:
    if path is None or not path.exists():
        return None
    metadata = read_json(path)
    params = metadata.get("params", {})
    stats = metadata.get("stats", {})
    close_state = "on" if not params.get("skip_close", False) else "off"
    fill_state = "on" if not params.get("skip_fill_holes", False) else "off"
    return (
        f"Stage 7 morphology: binary close={close_state}, fill holes={fill_state}; "
        f"tissue {stats.get('tissue_before', 'unknown')} -> {stats.get('tissue_after', 'unknown')}; "
        f"components {stats.get('components_before', 'unknown')} -> {stats.get('components_after', 'unknown')}."
    )


def format_reviewer_qc(args: argparse.Namespace) -> Optional[str]:
    results_path = resolve_path(args.reviewer_results_jsonl)
    summary_path = resolve_path(args.reviewer_summary)
    if results_path is None and summary_path is not None:
        results_path = summary_path.parent / "results.jsonl"
    if results_path is None or not results_path.exists():
        return None
    with results_path.open("r", encoding="utf-8") as f:
        first = f.readline().strip()
    if not first:
        return None
    result = json.loads(first)
    qc = result.get("qc", {})
    if not qc:
        return None
    return (
        f"Calibration reviewer QC: precision={qc.get('precision', 'unknown')}, "
        f"recall={qc.get('recall', 'unknown')}, "
        f"precision_pass={qc.get('precision_pass', 'unknown')}, "
        f"recall_pass={qc.get('recall_pass', 'unknown')}, "
        f"overall_pass={qc.get('overall_pass', 'unknown')}."
    )


def parse_prompt_specs(specs: list[str]) -> list[tuple[str, Path, str]]:
    prompts: list[tuple[str, Path, str]] = []
    for spec in specs:
        if "=" in spec:
            label, path_text = spec.split("=", 1)
        else:
            path_text = spec
            label = Path(path_text).name
        prompt_path = Path(path_text).expanduser()
        if not prompt_path.is_absolute():
            prompt_path = (Path.cwd() / prompt_path).resolve()
        if not prompt_path.exists():
            raise SystemExit(f"Missing prompt file: {prompt_path}")
        prompts.append((label.strip() or prompt_path.name, prompt_path, read_text(prompt_path)))
    return prompts


def build_prompt_pages(prompt_specs: list[tuple[str, Path, str]]) -> list[Image.Image]:
    if not prompt_specs:
        return []
    pages: list[Image.Image] = []
    page = Image.new("RGB", (2400, 1600), "white")
    pages.append(page)
    draw = ImageDraw.Draw(page)
    title_font = font(38, bold=True)
    section_font = font(26, bold=True)
    path_font = font(18)
    prompt_font = font(20)

    y = 48
    draw.text((56, y), "Prompt Provenance", fill="black", font=title_font)
    y += 62

    def new_page() -> tuple[Image.Image, ImageDraw.ImageDraw, int]:
        next_page = Image.new("RGB", (2400, 1600), "white")
        pages.append(next_page)
        next_draw = ImageDraw.Draw(next_page)
        next_draw.text((56, 48), "Prompt Provenance", fill="black", font=title_font)
        return next_page, next_draw, 118

    for label, prompt_path, text in prompt_specs:
        if y > 1420:
            page, draw, y = new_page()
        draw.text((56, y), label, fill="black", font=section_font)
        y += 36
        y = draw_wrapped(
            draw,
            (56, y),
            str(prompt_path),
            fill=(70, 70, 70),
            font_obj=path_font,
            max_chars=190,
            line_gap=3,
        )
        y += 10
        for raw_line in text.splitlines():
            lines = wrap(raw_line, width=190, replace_whitespace=False) or [""]
            for line in lines:
                if y > 1530:
                    page, draw, y = new_page()
                draw.text((56, y), line, fill=(20, 20, 20), font=prompt_font)
                bbox = draw.textbbox((56, y), line or " ", font=prompt_font)
                y += bbox[3] - bbox[1] + 5
        y += 24
    return pages


def build_pdf(args: argparse.Namespace) -> None:
    case_input = read_json(Path(args.case_input_json))
    run_dir = Path(args.run_dir).expanduser().resolve()
    pipeline_meta_path = run_dir / "pipeline_metadata.json"
    pipeline_meta = read_json(pipeline_meta_path) if pipeline_meta_path.exists() else {}
    stage1_bboxes = read_json(run_dir / "stage1" / "bboxes.json")
    selection = (
        pipeline_meta.get("config", {}).get("selection")
        or stage1_bboxes.get("selection")
        or {}
    )

    detector_overlay = resolve_detector_overlay(args, case_input)
    selector_overlay = resolve_selector_overlay(args)
    segmentation_overlay = resolve_segmentation_overlay(run_dir)
    stage7_line = format_stage7_metadata(resolve_stage7_metadata(run_dir))
    reviewer_line = format_reviewer_qc(args)

    output_pdf = Path(args.output_pdf).expanduser().resolve()
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    page = Image.new("RGB", (2400, 1600), "white")
    draw = ImageDraw.Draw(page)
    title_font = font(42, bold=True)
    header_font = font(30, bold=True)
    body_font = font(23)

    y = 44
    draw.text((52, y), f"{args.case_id} | Selector-Seeded Foreground Pipeline", fill="black", font=title_font)
    y += 64
    summary = (
        f"Detector boxes: {stage1_bboxes.get('source_detection_count', 'unknown')} source final detections. "
        f"Seeded auto-context boxes: {stage1_bboxes.get('regions_count', 'unknown')}. "
        f"Selection policy: {selection.get('selection_policy', 'unknown')} "
        f"({selection.get('selection_source', 'unknown')}); selected ids: {selection.get('selected_box_ids', 'unknown')}."
    )
    y = draw_wrapped(draw, (56, y), summary, fill=(20, 20, 20), font_obj=body_font, max_chars=170)
    y += 18

    panel_y = y
    panel_w = 720
    panel_h = 850
    gap = 55
    draw_panel(
        page,
        title="1. Detector Final Bboxes",
        image_path=detector_overlay,
        x=56,
        y=panel_y,
        w=panel_w,
        h=panel_h,
        caption="Output of the scale500 VLM bbox detector pipeline before selector filtering.",
        allow_missing=False,
    )
    draw_panel(
        page,
        title="2. Verifier-Selected Bboxes",
        image_path=selector_overlay,
        x=56 + panel_w + gap,
        y=panel_y,
        w=panel_w,
        h=panel_h,
        caption="Second-pass verifier output. The revised selected bbox is seeded as the only Stage 1 detection.",
        allow_missing=False,
    )
    draw_panel(
        page,
        title="3. Foreground Segmentation",
        image_path=segmentation_overlay,
        x=56 + 2 * (panel_w + gap),
        y=panel_y,
        w=panel_w,
        h=panel_h,
        caption="Auto-context Stage 7 foreground segmentation result from the selector-seeded bbox layout.",
        allow_missing=args.allow_missing_segmentation,
    )

    footer_y = panel_y + panel_h + 30
    draw.text((56, footer_y), "Pipeline Evidence", fill="black", font=header_font)
    footer_y += 44
    evidence = [
        f"Run dir: {run_dir}",
        f"Detector overlay: {detector_overlay}",
        f"Selector overlay: {selector_overlay}",
        f"Segmentation overlay: {segmentation_overlay}",
        f"Selection reason: {selection.get('verifier_pairs') or selection.get('verifier_confidence') or 'see metadata'}",
    ]
    if stage7_line:
        evidence.append(stage7_line)
    if reviewer_line:
        evidence.append(reviewer_line)
    draw_wrapped(
        draw,
        (56, footer_y),
        "\n".join(evidence),
        fill=(25, 25, 25),
        font_obj=body_font,
        max_chars=180,
    )

    prompt_pages = build_prompt_pages(parse_prompt_specs(args.prompt_file))
    pages = [page, *prompt_pages]
    pages[0].save(output_pdf, "PDF", resolution=150, save_all=True, append_images=pages[1:])
    print(output_pdf)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--case-input-json", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-pdf", required=True, type=Path)
    parser.add_argument("--detector-overlay", default=None)
    parser.add_argument("--selector-overlay", default=None)
    parser.add_argument(
        "--prompt-file",
        action="append",
        default=[],
        help="Prompt provenance entry as LABEL=PATH. May be passed multiple times.",
    )
    parser.add_argument("--reviewer-summary", default=None)
    parser.add_argument("--reviewer-results-jsonl", default=None)
    parser.add_argument(
        "--artifact-probe-root",
        type=Path,
        default=Path("runs/detector_pipeline_scale500_v1/analysis/artifact_redundancy_probe_50case_prohigh_v1"),
    )
    parser.add_argument(
        "--allow-missing-segmentation",
        action="store_true",
        help="Render missing panel placeholders instead of failing.",
    )
    args = parser.parse_args()
    build_pdf(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
