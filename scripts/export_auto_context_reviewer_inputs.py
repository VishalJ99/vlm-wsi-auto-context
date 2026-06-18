#!/usr/bin/env python3
# ABOUTME: Export auto-context Stage 7 bbox masks as high-resolution reviewer inputs.
# ABOUTME: Writes Stage3-compatible crop/mask/overlay files for run_vlm_reviewer_batch.py.

from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.wsi_backend import (  # noqa: E402
    close_wsi,
    get_level0_dimensions,
    get_pyramid_info,
    load_wsi,
    read_region_rgb,
)


BBox = Tuple[int, int, int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create run_vlm_reviewer_batch-compatible crop/mask/overlay inputs "
            "from an auto-context run's final per-bbox Stage 7 masks."
        )
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Auto-context run directory containing stage1/ and bboxes/.",
    )
    parser.add_argument(
        "--wsi",
        default=None,
        help="Source WSI path. If omitted, read from pipeline_status.json or stage1 metadata.",
    )
    parser.add_argument(
        "--output-root",
        default="runs/auto_context_reviewer_inputs",
        help="Root for Stage3-compatible reviewer inputs.",
    )
    parser.add_argument("--case-id", default=None, help="Output case ID. Defaults to source case ID.")
    parser.add_argument(
        "--run-id",
        default=None,
        help="Output run ID. Defaults to <source_run_id>_stage7_l0_review.",
    )
    parser.add_argument(
        "--bbox",
        action="append",
        default=None,
        help="Optional bbox ID to export. May be passed multiple times.",
    )
    parser.add_argument("--max-items", type=int, default=None, help="Optional cap on exported bboxes.")
    parser.add_argument(
        "--wsi-reader",
        "--reader",
        dest="wsi_reader",
        choices=["auto", "openslide", "cucim", "isyntax"],
        default="auto",
        help="WSI reader backend.",
    )
    parser.add_argument(
        "--max-dim",
        type=int,
        default=1024,
        help="Maximum long edge for exported reviewer crop/overlay.",
    )
    parser.add_argument(
        "--padding-frac",
        type=float,
        default=0.0,
        help="Padding around each bbox as a fraction of bbox long edge.",
    )
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.45,
        help="Foreground overlay opacity in [0, 1].",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing per-bbox exported files.",
    )
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def current_git_commit() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            text=True,
            capture_output=True,
        )
        return proc.stdout.strip()
    except Exception:
        return "unknown"


def command_line() -> str:
    return shlex.join([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]])


def bbox_id_from_level0(bbox: Sequence[int]) -> str:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    return f"{x1}_{y1}_{x2}_{y2}"


def clamp_bbox(bbox: BBox, wsi_w: int, wsi_h: int) -> BBox:
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(int(x1), wsi_w - 1))
    y1 = max(0, min(int(y1), wsi_h - 1))
    x2 = max(x1 + 1, min(int(x2), wsi_w))
    y2 = max(y1 + 1, min(int(y2), wsi_h))
    return x1, y1, x2, y2


def pad_bbox(bbox: BBox, padding_frac: float, wsi_w: int, wsi_h: int) -> BBox:
    x1, y1, x2, y2 = bbox
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    pad = int(round(max(width, height) * max(0.0, padding_frac)))
    return clamp_bbox((x1 - pad, y1 - pad, x2 + pad, y2 + pad), wsi_w, wsi_h)


def choose_read_level(pyramid: dict, bbox_w: int, bbox_h: int, max_dim: int) -> Tuple[int, float]:
    best_level = 0
    best_diff = float("inf")
    for level, downsample in enumerate(pyramid["level_downsamples"]):
        projected = max(bbox_w / float(downsample), bbox_h / float(downsample))
        diff = abs(projected - max_dim)
        if diff < best_diff:
            best_level = level
            best_diff = diff
    return best_level, float(pyramid["level_downsamples"][best_level])


def read_bbox_crop(wsi, reader: str, pyramid: dict, bbox: BBox, max_dim: int) -> Tuple[Image.Image, int, float]:
    x1, y1, x2, y2 = bbox
    bbox_w = max(1, x2 - x1)
    bbox_h = max(1, y2 - y1)
    level, downsample = choose_read_level(pyramid, bbox_w, bbox_h, max_dim)
    read_w = max(1, int(math.ceil(bbox_w / downsample)))
    read_h = max(1, int(math.ceil(bbox_h / downsample)))

    arr = read_region_rgb(wsi, reader, x=x1, y=y1, width=read_w, height=read_h, level=level)
    crop = Image.fromarray(arr).convert("RGB")
    long_edge = max(crop.size)
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
    return crop, level, downsample


def resolve_wsi_path(run_dir: Path, explicit_wsi: Optional[str]) -> Path:
    if explicit_wsi:
        path = Path(explicit_wsi)
        if not path.is_file():
            raise SystemExit(f"WSI not found: {path}")
        return path

    candidates: List[Optional[str]] = []
    status_path = run_dir / "pipeline_status.json"
    if status_path.exists():
        status = json.loads(status_path.read_text())
        candidates.extend([status.get("wsi_path"), status.get("wsi_input")])

    stage1_meta_path = run_dir / "stage1" / "metadata.json"
    if stage1_meta_path.exists():
        stage1_meta = json.loads(stage1_meta_path.read_text())
        candidates.append(stage1_meta.get("wsi_path"))

    for candidate in candidates:
        if not candidate:
            continue
        path = Path(str(candidate))
        if path.is_file():
            return path
    raise SystemExit("Could not resolve WSI path; pass --wsi explicitly.")


def load_stage1_bboxes(run_dir: Path) -> Dict[str, dict]:
    bboxes_path = run_dir / "stage1" / "bboxes.json"
    if not bboxes_path.is_file():
        raise SystemExit(f"Missing Stage 1 bboxes: {bboxes_path}")
    payload = json.loads(bboxes_path.read_text())
    regions = payload.get("detected_regions", [])
    if not isinstance(regions, list):
        raise SystemExit(f"Malformed detected_regions in {bboxes_path}")

    out: Dict[str, dict] = {}
    for region in regions:
        bbox = region.get("bbox_level0")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        bbox_tuple = tuple(int(v) for v in bbox)
        out[bbox_id_from_level0(bbox_tuple)] = {
            "bbox_level0": bbox_tuple,
            "label": region.get("label"),
            "bbox_thumbnail": region.get("bbox_thumbnail"),
            "bbox_normalized": region.get("bbox_normalized"),
        }
    return out


def iter_bbox_items(run_dir: Path, requested: Optional[Iterable[str]]) -> List[Tuple[str, dict]]:
    by_id = load_stage1_bboxes(run_dir)
    requested_set = set(requested or [])
    items: List[Tuple[str, dict]] = []

    bboxes_dir = run_dir / "bboxes"
    if not bboxes_dir.is_dir():
        raise SystemExit(f"Missing bboxes directory: {bboxes_dir}")

    for bbox_dir in sorted(p for p in bboxes_dir.iterdir() if p.is_dir()):
        bbox_id = bbox_dir.name
        if requested_set and bbox_id not in requested_set:
            continue
        if bbox_id not in by_id:
            try:
                by_id[bbox_id] = {
                    "bbox_level0": tuple(int(v) for v in bbox_id.split("_")),
                    "label": None,
                    "bbox_thumbnail": None,
                    "bbox_normalized": None,
                }
            except Exception:
                continue
        info = dict(by_id[bbox_id])
        info["bbox_dir"] = bbox_dir
        items.append((bbox_id, info))

    return items


def read_patch_size(bbox_dir: Path) -> int:
    meta_path = bbox_dir / "stage6" / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        value = meta.get("patch_size_level0") or meta.get("patch_size")
        if value:
            return int(value)
    return 512


def load_tissue_mask(bbox_dir: Path) -> np.ndarray:
    mask_path = bbox_dir / "stage7" / "tissue_mask_post.npy"
    if not mask_path.is_file():
        raise FileNotFoundError(f"Missing Stage 7 tissue mask: {mask_path}")
    return np.load(mask_path).astype(bool)


def render_patch_grid_mask(
    tissue_mask: np.ndarray,
    bbox: BBox,
    padded_bbox: BBox,
    output_size: Tuple[int, int],
    patch_size: int,
) -> Image.Image:
    mask = Image.new("L", output_size, 0)
    draw = ImageDraw.Draw(mask)

    bx1, by1, bx2, by2 = bbox
    px1, py1, px2, py2 = padded_bbox
    out_w, out_h = output_size
    sx = out_w / float(max(1, px2 - px1))
    sy = out_h / float(max(1, py2 - py1))

    for row, col in np.argwhere(tissue_mask):
        cell_x1 = bx1 + int(col) * patch_size
        cell_y1 = by1 + int(row) * patch_size
        cell_x2 = min(cell_x1 + patch_size, bx2)
        cell_y2 = min(cell_y1 + patch_size, by2)

        ix1 = max(cell_x1, px1)
        iy1 = max(cell_y1, py1)
        ix2 = min(cell_x2, px2)
        iy2 = min(cell_y2, py2)
        if ix2 <= ix1 or iy2 <= iy1:
            continue

        ox1 = max(0, min(out_w - 1, int(math.floor((ix1 - px1) * sx))))
        oy1 = max(0, min(out_h - 1, int(math.floor((iy1 - py1) * sy))))
        ox2 = max(0, min(out_w, int(math.ceil((ix2 - px1) * sx))))
        oy2 = max(0, min(out_h, int(math.ceil((iy2 - py1) * sy))))
        if ox2 > ox1 and oy2 > oy1:
            draw.rectangle((ox1, oy1, ox2 - 1, oy2 - 1), fill=255)

    return mask


def build_overlay(crop: Image.Image, mask: Image.Image, alpha: float) -> Image.Image:
    alpha_u8 = int(round(max(0.0, min(1.0, alpha)) * 255))
    overlay = Image.new("RGBA", crop.size, (0, 190, 70, 0))
    overlay.putalpha(mask.point(lambda value: alpha_u8 if value > 0 else 0))
    return Image.alpha_composite(crop.convert("RGBA"), overlay).convert("RGB")


def write_reproduction_file(path: Path) -> None:
    content = "\n".join(
        [
            "Reproduce this auto-context reviewer-input export",
            "",
            f"Working directory: {REPO_ROOT}",
            f"Git commit: {current_git_commit()}",
            f"Command: {command_line()}",
            "",
        ]
    )
    path.write_text(content)


def main() -> int:
    args = parse_args()
    if args.max_dim < 1:
        raise SystemExit("--max-dim must be >= 1")
    if not 0.0 <= args.overlay_alpha <= 1.0:
        raise SystemExit("--overlay-alpha must be in [0, 1]")
    if args.max_items is not None and args.max_items < 1:
        raise SystemExit("--max-items must be >= 1")

    source_run_dir = Path(args.run_dir).resolve()
    if not source_run_dir.is_dir():
        raise SystemExit(f"Run directory not found: {source_run_dir}")

    wsi_path = resolve_wsi_path(source_run_dir, args.wsi)
    source_case_id = source_run_dir.parent.name
    source_run_id = source_run_dir.name
    case_id = args.case_id or source_case_id
    run_id = args.run_id or f"{source_run_id}_stage7_l0_review"
    run_dir = Path(args.output_root) / case_id / run_id
    bboxes_out = run_dir / "bboxes"
    bboxes_out.mkdir(parents=True, exist_ok=True)

    items = iter_bbox_items(source_run_dir, args.bbox)
    if args.max_items is not None:
        items = items[: args.max_items]
    if not items:
        raise SystemExit("No bbox items selected for export.")

    manifest_rows: List[dict] = []
    wsi, reader = load_wsi(str(wsi_path), args.wsi_reader)
    try:
        wsi_w, wsi_h = get_level0_dimensions(wsi, reader)
        pyramid = get_pyramid_info(wsi, reader)
        exported = 0

        for bbox_id, info in items:
            bbox_dir = Path(info["bbox_dir"])
            raw_bbox = clamp_bbox(tuple(int(v) for v in info["bbox_level0"]), wsi_w, wsi_h)
            padded_bbox = pad_bbox(raw_bbox, args.padding_frac, wsi_w, wsi_h)
            stage3_dir = bboxes_out / bbox_id / "stage3"

            if stage3_dir.exists() and not args.overwrite:
                print(f"Skipping existing: {stage3_dir}")
                continue
            stage3_dir.mkdir(parents=True, exist_ok=True)

            crop, read_level, read_downsample = read_bbox_crop(
                wsi=wsi,
                reader=reader,
                pyramid=pyramid,
                bbox=padded_bbox,
                max_dim=args.max_dim,
            )
            tissue_mask = load_tissue_mask(bbox_dir)
            patch_size = read_patch_size(bbox_dir)
            mask = render_patch_grid_mask(
                tissue_mask=tissue_mask,
                bbox=raw_bbox,
                padded_bbox=padded_bbox,
                output_size=crop.size,
                patch_size=patch_size,
            )
            overlay = build_overlay(crop, mask, args.overlay_alpha)

            crop_path = stage3_dir / "crop.png"
            mask_path = stage3_dir / "mask.png"
            overlay_path = stage3_dir / "overlay.png"
            crop.save(crop_path)
            mask.save(mask_path)
            overlay.save(overlay_path)

            meta = {
                "source": "auto_context_stage7",
                "source_run_dir": str(source_run_dir),
                "source_bbox_dir": str(bbox_dir),
                "wsi_path": str(wsi_path),
                "case_id": case_id,
                "run_id": run_id,
                "bbox_id": bbox_id,
                "label": info.get("label"),
                "bbox_level0": list(raw_bbox),
                "padded_bbox_level0": list(padded_bbox),
                "bbox_thumbnail": info.get("bbox_thumbnail"),
                "bbox_normalized": info.get("bbox_normalized"),
                "crop_size": list(crop.size),
                "read_level": int(read_level),
                "read_downsample": float(read_downsample),
                "resolved_wsi_reader": reader,
                "mask_source": "stage7/tissue_mask_post.npy",
                "tissue_mask_shape": list(tissue_mask.shape),
                "tissue_cells": int(tissue_mask.sum()),
                "patch_size_level0": int(patch_size),
                "overlay_alpha": float(args.overlay_alpha),
            }
            write_json(stage3_dir / "metadata.json", meta)

            row = {
                "case_id": case_id,
                "run_id": run_id,
                "bbox_id": bbox_id,
                "bbox_level0": " ".join(str(v) for v in raw_bbox),
                "padded_bbox_level0": " ".join(str(v) for v in padded_bbox),
                "crop_path": str(crop_path),
                "mask_path": str(mask_path),
                "overlay_path": str(overlay_path),
                "metadata_path": str(stage3_dir / "metadata.json"),
            }
            manifest_rows.append(row)
            exported += 1
            print(f"Exported [{exported}/{len(items)}]: {stage3_dir}")
    finally:
        close_wsi(wsi, reader)

    run_meta = {
        "source": "auto_context_stage7",
        "created_at": datetime.now().isoformat(),
        "source_run_dir": str(source_run_dir),
        "source_case_id": source_case_id,
        "source_run_id": source_run_id,
        "wsi_path": str(wsi_path),
        "case_id": case_id,
        "run_id": run_id,
        "output_root": str(args.output_root),
        "exported_items": len(manifest_rows),
        "max_dim": int(args.max_dim),
        "padding_frac": float(args.padding_frac),
        "overlay_alpha": float(args.overlay_alpha),
        "git_commit_hash": current_git_commit(),
        "reviewer_batch_command": (
            f"python run_vlm_reviewer_batch.py --baseline-dir {shlex.quote(str(args.output_root))} "
            f"--run-selection latest --case-pattern {shlex.quote(case_id)} "
            f"--output-root runs/reviewer --batch-name auto_context_review_{case_id}_{run_id}"
        ),
    }
    write_json(run_dir / "metadata.json", run_meta)
    write_reproduction_file(run_dir / "reproduction.txt")

    manifest_csv = run_dir / "manifest.csv"
    fieldnames = [
        "case_id",
        "run_id",
        "bbox_id",
        "bbox_level0",
        "padded_bbox_level0",
        "crop_path",
        "mask_path",
        "overlay_path",
        "metadata_path",
    ]
    with manifest_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    manifest_jsonl = run_dir / "manifest.jsonl"
    with manifest_jsonl.open("w") as f:
        for row in manifest_rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    print(f"Exported {len(manifest_rows)} reviewer item(s)")
    print(f"Run dir: {run_dir}")
    print(f"Manifest: {manifest_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
