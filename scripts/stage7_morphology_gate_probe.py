#!/usr/bin/env python3
"""Build synthetic Stage 7 morphology-gate tasks from a Stage 6 patch map."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from export_auto_context_reviewer_inputs import render_patch_grid_mask  # noqa: E402
from postprocess_mask import (  # noqa: E402
    apply_closing,
    fill_small_holes,
    remove_small_components,
)
from run_vlm_bbox_inference import (  # noqa: E402
    DEFAULT_OPENROUTER_REFERER,
    DEFAULT_OPENROUTER_URL,
    OpenRouterRunner,
    encode_image_base64,
)
from run_selector_seeded_foreground_pipeline import parse_openrouter_key_from_zshrc  # noqa: E402
from vlm_reviewer import build_green_overlay, parse_json_response  # noqa: E402


CANDIDATES = ("none", "fill_holes", "close", "close_fill")


@dataclass(frozen=True)
class Task:
    task_id: str
    description: str
    input_mask: np.ndarray
    expected_candidate: str | None
    perturbation: dict[str, Any]


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise SystemExit(f"Expected JSON object in {path}")
    return data


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def bbox_from_metadata(meta: dict[str, Any]) -> tuple[int, int, int, int]:
    bbox = meta.get("bbox_level0")
    if isinstance(bbox, dict):
        return tuple(int(bbox[k]) for k in ("x1", "y1", "x2", "y2"))  # type: ignore[return-value]
    if isinstance(bbox, list) and len(bbox) == 4:
        return tuple(int(v) for v in bbox)  # type: ignore[return-value]
    raise SystemExit("Stage 6 metadata does not contain bbox_level0")


def padded_bbox_from_reviewer_meta(meta: dict[str, Any], fallback: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    value = meta.get("padded_bbox_level0")
    if isinstance(value, list) and len(value) == 4:
        return tuple(int(v) for v in value)  # type: ignore[return-value]
    return fallback


def resolve_tissue_id(meta: dict[str, Any]) -> int:
    labels = [str(x).lower() for x in (meta.get("class_order") or meta.get("class_labels") or [])]
    if "tissue" in labels:
        return labels.index("tissue")
    label_map = meta.get("label_map")
    if isinstance(label_map, dict):
        for key, value in label_map.items():
            if str(value).lower() == "tissue":
                return int(key)
    return 1


def component_count(mask: np.ndarray) -> int:
    _, n = ndi.label(mask.astype(bool), structure=ndi.generate_binary_structure(2, 1))
    return int(n)


def component_sizes(mask: np.ndarray) -> list[int]:
    labels, n = ndi.label(mask.astype(bool), structure=ndi.generate_binary_structure(2, 1))
    return [int((labels == i).sum()) for i in range(1, n + 1)]


def postprocess_variant(
    mask: np.ndarray,
    candidate: str,
    min_component_size: int,
    close_kernel: int,
    max_hole_size: int,
) -> np.ndarray:
    out, _, _ = remove_small_components(mask.astype(bool), min_size=min_component_size, connectivity=4)
    if candidate in {"close", "close_fill"}:
        out = apply_closing(out, kernel=close_kernel)
    if candidate in {"fill_holes", "close_fill"}:
        out = fill_small_holes(out, max_hole_size=max_hole_size, connectivity=4)
    return out.astype(bool)


def neighbor_counts(mask: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), dtype=int)
    kernel[1, 1] = 0
    return ndi.convolve(mask.astype(int), kernel, mode="constant", cval=0)


def choose_internal_tissue_cell(mask: np.ndarray) -> tuple[int, int] | None:
    counts = neighbor_counts(mask)
    ys, xs = np.where(mask & (counts >= 8))
    if len(ys) == 0:
        ys, xs = np.where(mask & (counts >= 7))
    if len(ys) == 0:
        return None
    center = np.array(mask.shape) / 2.0
    order = np.argsort((ys - center[0]) ** 2 + (xs - center[1]) ** 2)
    return int(ys[order[0]]), int(xs[order[0]])


def choose_border_tissue_cell(mask: np.ndarray) -> tuple[int, int] | None:
    counts = neighbor_counts(mask)
    ys, xs = np.where(mask & (counts >= 3) & (counts <= 5))
    if len(ys) == 0:
        return None
    center = np.array(mask.shape) / 2.0
    order = np.argsort((ys - center[0]) ** 2 + (xs - center[1]) ** 2)
    return int(ys[order[0]]), int(xs[order[0]])


def build_tasks(raw_mask: np.ndarray, min_component_size: int) -> list[Task]:
    base, _, _ = remove_small_components(raw_mask.astype(bool), min_size=min_component_size, connectivity=4)
    tasks = [
        Task(
            task_id="native_bridge_risk",
            description="Native Stage 6 mask for a case where global closing bridges two large tissue components.",
            input_mask=raw_mask.astype(bool),
            expected_candidate="none",
            perturbation={"type": "native"},
        )
    ]

    internal = choose_internal_tissue_cell(base)
    if internal is not None:
        perturbed = base.copy()
        perturbed[internal] = False
        tasks.append(
            Task(
                task_id="synthetic_internal_hole",
                description="Synthetic one-patch interior hole removed from tissue.",
                input_mask=perturbed,
                expected_candidate="fill_holes",
                perturbation={"type": "remove_internal_patch", "row": internal[0], "col": internal[1]},
            )
        )

    border = choose_border_tissue_cell(base)
    if border is not None:
        perturbed = base.copy()
        perturbed[border] = False
        tasks.append(
            Task(
                task_id="synthetic_border_gap",
                description="Synthetic one-patch tissue-edge gap removed from tissue.",
                input_mask=perturbed,
                expected_candidate=None,
                perturbation={"type": "remove_border_patch", "row": border[0], "col": border[1]},
            )
        )
    return tasks


def render_candidate_overlay(
    *,
    crop: Image.Image,
    mask_patch_grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    padded_bbox: tuple[int, int, int, int],
    patch_size: int,
    out_path: Path,
    mask_path: Path,
) -> None:
    rendered_mask = render_patch_grid_mask(
        tissue_mask=mask_patch_grid.astype(bool),
        bbox=bbox,
        padded_bbox=padded_bbox,
        output_size=crop.size,
        patch_size=patch_size,
    )
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    rendered_mask.save(mask_path)
    overlay = build_green_overlay(crop_img=crop, mask_path=str(mask_path), alpha=0.5, threshold=0)
    overlay.save(out_path)


def resize_for_panel(img: Image.Image, w: int, h: int) -> Image.Image:
    scale = min(w / img.width, h / img.height)
    return img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.Resampling.LANCZOS)


def build_panel(crop: Image.Image, overlays: dict[str, Path], out_path: Path) -> None:
    panel_w, panel_h = 2200, 1700
    page = Image.new("RGB", (panel_w, panel_h), "white")
    draw = ImageDraw.Draw(page)
    title_font = font(34, bold=True)
    label_font = font(24, bold=True)

    draw.text((36, 28), "Stage 7 Morphology Gate Candidates", fill="black", font=title_font)
    cells = [
        ("source", None),
        ("none", overlays["none"]),
        ("fill_holes", overlays["fill_holes"]),
        ("close", overlays["close"]),
        ("close_fill", overlays["close_fill"]),
    ]
    x0, y0 = 36, 90
    cell_w, cell_h = 690, 500
    gap_x, gap_y = 28, 52
    for idx, (label, path) in enumerate(cells):
        row, col = divmod(idx, 3)
        x = x0 + col * (cell_w + gap_x)
        y = y0 + row * (cell_h + gap_y)
        draw.text((x, y), label, fill="black", font=label_font)
        img = crop if path is None else Image.open(path).convert("RGB")
        fitted = resize_for_panel(img, cell_w, cell_h - 36)
        page.paste(fitted, (x + (cell_w - fitted.width) // 2, y + 34))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    page.save(out_path)


def build_parts(prompt: str, crop: Image.Image, panel: Image.Image) -> list[dict[str, Any]]:
    return [
        {"type": "text", "text": prompt},
        {"type": "text", "text": "Input 1: Source histology crop"},
        {"type": "image", "image": {"pil": crop, "b64": encode_image_base64(crop, resize=False)}},
        {"type": "text", "text": "Input 2: Candidate morphology overlay panel"},
        {"type": "image", "image": {"pil": panel, "b64": encode_image_base64(panel, resize=False)}},
    ]


def run_openrouter(args: argparse.Namespace, crop: Image.Image, panel: Image.Image) -> dict[str, Any]:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key and args.load_openrouter_from_zshrc:
        api_key = parse_openrouter_key_from_zshrc(args.zshrc)
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY missing; export it or keep --load-openrouter-from-zshrc enabled")
    runner = OpenRouterRunner(
        model=args.model,
        api_key=api_key,
        url=args.openrouter_url,
        timeout=args.timeout,
        temperature=0.0,
        max_tokens=args.max_tokens,
        max_retries=args.max_retries,
        referer=DEFAULT_OPENROUTER_REFERER,
        reasoning_effort=args.reasoning_effort,
    )
    prompt = args.prompt_file.read_text(encoding="utf-8").strip()
    start = time.time()
    result = runner.run_full(build_parts(prompt, crop, panel))
    text = (result.get("text") or "").strip()
    return {
        "raw_text": text,
        "parsed_json": parse_json_response(text),
        "elapsed_seconds": round(time.time() - start, 3),
        "usage": result.get("usage") or {},
        "finish_reason": result.get("finish_reason"),
        "error": result.get("error"),
        "attempts": result.get("attempts"),
    }


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage6-dir", required=True, type=Path)
    parser.add_argument("--reviewer-stage3-dir", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--prompt-file", type=Path, default=REPO_ROOT / "prompts" / "stage7_morphology_gate.txt")
    parser.add_argument("--min-component-size", type=int, default=3)
    parser.add_argument("--close-kernel", type=int, default=3)
    parser.add_argument(
        "--max-hole-size",
        type=int,
        default=1,
        help="Fill enclosed background holes up to this many patch-grid cells; 0 fills all holes.",
    )
    parser.add_argument("--execute-vlm", action="store_true")
    parser.add_argument("--model", default="google/gemini-3-flash-preview")
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--openrouter-url", default=DEFAULT_OPENROUTER_URL)
    parser.add_argument("--zshrc", type=Path, default=Path.home() / ".zshrc")
    parser.add_argument("--load-openrouter-from-zshrc", action="store_true", default=True)
    parser.add_argument("--no-zshrc-openrouter-key", dest="load_openrouter_from_zshrc", action="store_false")
    return parser


def main() -> int:
    args = create_parser().parse_args()
    stage6_dir = args.stage6_dir.resolve()
    reviewer_stage3_dir = args.reviewer_stage3_dir.resolve()
    out_dir = args.output_root.resolve() / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    stage6_meta = read_json(stage6_dir / "metadata.json")
    reviewer_meta = read_json(reviewer_stage3_dir / "metadata.json")
    class_map = np.load(stage6_dir / "class_map.npy")
    tissue_id = resolve_tissue_id(stage6_meta)
    raw_mask = class_map == tissue_id
    crop = Image.open(reviewer_stage3_dir / "crop.png").convert("RGB")
    bbox = bbox_from_metadata(stage6_meta)
    padded_bbox = padded_bbox_from_reviewer_meta(reviewer_meta, bbox)
    patch_size = int(stage6_meta.get("patch_size_level0") or stage6_meta.get("patch_size") or 512)

    rows: list[dict[str, Any]] = []
    for task in build_tasks(raw_mask=raw_mask, min_component_size=args.min_component_size):
        task_dir = out_dir / "tasks" / task.task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        overlays: dict[str, Path] = {}
        candidate_stats: dict[str, Any] = {}
        for candidate in CANDIDATES:
            mask = postprocess_variant(
                task.input_mask,
                candidate=candidate,
                min_component_size=args.min_component_size,
                close_kernel=args.close_kernel,
                max_hole_size=max(0, args.max_hole_size),
            )
            mask_path = task_dir / f"{candidate}_mask.png"
            overlay_path = task_dir / f"{candidate}_overlay_green50.png"
            render_candidate_overlay(
                crop=crop,
                mask_patch_grid=mask,
                bbox=bbox,
                padded_bbox=padded_bbox,
                patch_size=patch_size,
                out_path=overlay_path,
                mask_path=mask_path,
            )
            overlays[candidate] = overlay_path
            candidate_stats[candidate] = {
                "tissue": int(mask.sum()),
                "components": component_count(mask),
                "component_sizes": component_sizes(mask),
                "added_vs_input": int((mask & ~task.input_mask).sum()),
                "removed_vs_input": int((task.input_mask & ~mask).sum()),
            }

        panel_path = task_dir / "candidate_panel.png"
        build_panel(crop=crop, overlays=overlays, out_path=panel_path)
        vlm_result = run_openrouter(args, crop=crop, panel=Image.open(panel_path).convert("RGB")) if args.execute_vlm else None
        selected = None
        if isinstance(vlm_result, dict) and isinstance(vlm_result.get("parsed_json"), dict):
            selected = vlm_result["parsed_json"].get("selected_candidate")
        row = {
            "task_id": task.task_id,
            "description": task.description,
            "expected_candidate": task.expected_candidate,
            "selected_candidate": selected,
            "matches_expected": None if task.expected_candidate is None or selected is None else selected == task.expected_candidate,
            "input_tissue": int(task.input_mask.sum()),
            "input_components": component_count(task.input_mask),
            "input_component_sizes": component_sizes(task.input_mask),
            "perturbation": task.perturbation,
            "candidate_stats": candidate_stats,
            "panel_path": str(panel_path),
            "vlm_result": vlm_result,
        }
        write_json(task_dir / "task.json", row)
        rows.append(row)
        print(f"{task.task_id}: expected={task.expected_candidate} selected={selected} panel={panel_path}")

    manifest = {
        "created_at": datetime.now().isoformat(),
        "run_id": args.run_id,
        "stage6_dir": str(stage6_dir),
        "reviewer_stage3_dir": str(reviewer_stage3_dir),
        "prompt_file": str(args.prompt_file.resolve()),
        "max_hole_size": int(max(0, args.max_hole_size)),
        "execute_vlm": bool(args.execute_vlm),
        "tasks": rows,
    }
    write_json(out_dir / "summary.json", manifest)
    with (out_dir / "summary.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    (out_dir / "reproduction.txt").write_text(
        "\n".join(
            [
                "PER-250 Stage 7 morphology gate probe",
                "",
                f"Working directory: {REPO_ROOT}",
                f"Command: {' '.join(sys.argv)}",
                f"Stage 6 dir: {stage6_dir}",
                f"Reviewer stage3 dir: {reviewer_stage3_dir}",
                f"Prompt file: {args.prompt_file.resolve()}",
                f"Max hole size: {int(max(0, args.max_hole_size))}",
                f"Executed VLM: {bool(args.execute_vlm)}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"summary: {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
