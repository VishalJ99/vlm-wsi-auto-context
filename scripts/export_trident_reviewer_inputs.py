#!/usr/bin/env python3
# ABOUTME: Export TRIDENT GeoJSON tissue contours as Stage3-style reviewer inputs.
# ABOUTME: Writes per-contour crop, mask, overlay, and metadata for VLM reviewer QC.

from __future__ import annotations

import argparse
import csv
import json
import re
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

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


Point = Tuple[float, float]
Ring = List[Point]
PolygonRings = List[Ring]
BBox = Tuple[int, int, int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create run_vlm_reviewer_batch-compatible crop/mask/overlay inputs "
            "from a TRIDENT contours_geojson slide output."
        )
    )
    parser.add_argument("--wsi", required=True, help="Path to the source WSI.")
    parser.add_argument(
        "--geojson",
        help="TRIDENT contours_geojson/<slide>.geojson. If omitted, use --trident-job-dir.",
    )
    parser.add_argument(
        "--trident-job-dir",
        help="TRIDENT job dir containing contours_geojson/<slide_stem>.geojson.",
    )
    parser.add_argument(
        "--output-root",
        default="runs/trident_reviewer_inputs",
        help="Root for Stage3-compatible reviewer inputs.",
    )
    parser.add_argument(
        "--case-id",
        default=None,
        help="Case directory name. Defaults to WSI stem.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Run directory name. Defaults to current YYYYMMDD_HHMMSS.",
    )
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
        default=2048,
        help="Maximum long edge for exported reviewer crop/overlay.",
    )
    parser.add_argument(
        "--padding-frac",
        type=float,
        default=0.08,
        help="Padding around each contour bbox as a fraction of bbox long edge.",
    )
    parser.add_argument(
        "--min-bbox-area",
        type=float,
        default=1024.0,
        help="Skip contours with unpadded level-0 bbox area below this many pixels.",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Optional cap on exported contours after area filtering.",
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
        help="Overwrite existing per-core files.",
    )
    return parser.parse_args()


def resolve_geojson(wsi_path: Path, geojson: Optional[str], trident_job_dir: Optional[str]) -> Path:
    if geojson:
        path = Path(geojson)
    elif trident_job_dir:
        path = Path(trident_job_dir) / "contours_geojson" / f"{wsi_path.stem}.geojson"
    else:
        raise SystemExit("Provide --geojson or --trident-job-dir.")

    if not path.is_file():
        raise SystemExit(f"GeoJSON not found: {path}")
    return path


def iter_polygon_rings(geometry: dict) -> Iterable[PolygonRings]:
    geom_type = geometry.get("type")
    coordinates = geometry.get("coordinates")
    if geom_type == "Polygon":
        yield normalize_polygon(coordinates)
    elif geom_type == "MultiPolygon":
        for polygon in coordinates or []:
            yield normalize_polygon(polygon)


def normalize_polygon(coordinates: object) -> PolygonRings:
    rings: PolygonRings = []
    if not isinstance(coordinates, list):
        return rings
    for ring in coordinates:
        if not isinstance(ring, list):
            continue
        points: Ring = []
        for pair in ring:
            if not isinstance(pair, list) or len(pair) < 2:
                continue
            points.append((float(pair[0]), float(pair[1])))
        if len(points) >= 3:
            rings.append(points)
    return rings


def flatten_points(polygons: Sequence[PolygonRings]) -> List[Point]:
    points: List[Point] = []
    for rings in polygons:
        for ring in rings:
            points.extend(ring)
    return points


def clamp_bbox(bbox: BBox, wsi_w: int, wsi_h: int) -> BBox:
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(int(x1), wsi_w - 1))
    y1 = max(0, min(int(y1), wsi_h - 1))
    x2 = max(x1 + 1, min(int(x2), wsi_w))
    y2 = max(y1 + 1, min(int(y2), wsi_h))
    return x1, y1, x2, y2


def polygon_bbox(polygons: Sequence[PolygonRings]) -> BBox:
    points = flatten_points(polygons)
    if not points:
        raise ValueError("empty geometry")
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (
        int(np.floor(min(xs))),
        int(np.floor(min(ys))),
        int(np.ceil(max(xs))),
        int(np.ceil(max(ys))),
    )


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


def read_bbox_crop(
    wsi,
    reader: str,
    pyramid: dict,
    bbox: BBox,
    max_dim: int,
) -> Tuple[Image.Image, int, float]:
    x1, y1, x2, y2 = bbox
    bbox_w = max(1, x2 - x1)
    bbox_h = max(1, y2 - y1)
    level, downsample = choose_read_level(pyramid, bbox_w, bbox_h, max_dim)
    read_w = max(1, int(np.ceil(bbox_w / downsample)))
    read_h = max(1, int(np.ceil(bbox_h / downsample)))

    arr = read_region_rgb(
        wsi,
        reader,
        x=x1,
        y=y1,
        width=read_w,
        height=read_h,
        level=level,
    )
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


def transform_ring(ring: Ring, bbox: BBox, output_size: Tuple[int, int]) -> List[Tuple[int, int]]:
    x1, y1, x2, y2 = bbox
    out_w, out_h = output_size
    sx = out_w / float(max(1, x2 - x1))
    sy = out_h / float(max(1, y2 - y1))
    return [
        (
            int(round((x - x1) * sx)),
            int(round((y - y1) * sy)),
        )
        for x, y in ring
    ]


def build_mask(polygons: Sequence[PolygonRings], bbox: BBox, output_size: Tuple[int, int]) -> Image.Image:
    mask = Image.new("L", output_size, 0)
    draw = ImageDraw.Draw(mask)
    for rings in polygons:
        if not rings:
            continue
        exterior = transform_ring(rings[0], bbox, output_size)
        if len(exterior) >= 3:
            draw.polygon(exterior, fill=255)
        for hole in rings[1:]:
            transformed_hole = transform_ring(hole, bbox, output_size)
            if len(transformed_hole) >= 3:
                draw.polygon(transformed_hole, fill=0)
    return mask


def build_overlay(crop: Image.Image, mask: Image.Image, alpha: float) -> Image.Image:
    alpha_u8 = int(round(max(0.0, min(1.0, alpha)) * 255))
    overlay = Image.new("RGBA", crop.size, (0, 190, 70, 0))
    overlay.putalpha(mask.point(lambda value: alpha_u8 if value > 0 else 0))
    return Image.alpha_composite(crop.convert("RGBA"), overlay).convert("RGB")


def load_features(geojson_path: Path) -> List[dict]:
    obj = json.loads(geojson_path.read_text())
    if obj.get("type") != "FeatureCollection":
        raise SystemExit(f"Expected GeoJSON FeatureCollection: {geojson_path}")
    features = obj.get("features", [])
    if not isinstance(features, list):
        raise SystemExit(f"GeoJSON features is not a list: {geojson_path}")
    return features


def safe_core_id(index: int, properties: dict, bbox: BBox) -> str:
    tissue_id = properties.get("tissue_id", index)
    tissue_id_safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(tissue_id)).strip("_") or str(index)
    x1, y1, x2, y2 = bbox
    return f"core_{index:04d}_tid{tissue_id_safe}_{x1}_{y1}_{x2}_{y2}"


def write_json(path: Path, payload: dict) -> None:
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


def write_reproduction_file(path: Path) -> None:
    command = shlex.join([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]])
    content = "\n".join(
        [
            "Reproduce this TRIDENT reviewer-input export",
            "",
            f"Working directory: {REPO_ROOT}",
            f"Git commit: {current_git_commit()}",
            f"Command: {command}",
            "",
        ]
    )
    path.write_text(content)


def main() -> int:
    args = parse_args()
    if args.max_dim < 1:
        raise SystemExit("--max-dim must be >= 1")
    if args.max_items is not None and args.max_items < 1:
        raise SystemExit("--max-items must be >= 1")

    wsi_path = Path(args.wsi)
    if not wsi_path.is_file():
        raise SystemExit(f"WSI not found: {wsi_path}")
    geojson_path = resolve_geojson(wsi_path, args.geojson, args.trident_job_dir)

    case_id = args.case_id or wsi_path.stem
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_root) / case_id / run_id
    bboxes_dir = run_dir / "bboxes"
    bboxes_dir.mkdir(parents=True, exist_ok=True)

    features = load_features(geojson_path)
    wsi, reader = load_wsi(str(wsi_path), args.wsi_reader)
    manifest_rows = []
    try:
        wsi_w, wsi_h = get_level0_dimensions(wsi, reader)
        pyramid = get_pyramid_info(wsi, reader)
        exported = 0

        for feature_idx, feature in enumerate(features):
            geometry = feature.get("geometry") or {}
            properties = feature.get("properties") or {}
            polygons = list(iter_polygon_rings(geometry))
            if not polygons:
                continue

            raw_bbox = clamp_bbox(polygon_bbox(polygons), wsi_w, wsi_h)
            raw_area = float(max(1, raw_bbox[2] - raw_bbox[0]) * max(1, raw_bbox[3] - raw_bbox[1]))
            if raw_area < args.min_bbox_area:
                continue

            padded_bbox = pad_bbox(raw_bbox, args.padding_frac, wsi_w, wsi_h)
            core_id = safe_core_id(feature_idx, properties, raw_bbox)
            stage3_dir = bboxes_dir / core_id / "stage3"
            if stage3_dir.exists() and not args.overwrite:
                print(f"Skipping existing: {stage3_dir}")
                continue
            stage3_dir.mkdir(parents=True, exist_ok=True)

            crop, read_level, read_downsample = read_bbox_crop(
                wsi,
                reader,
                pyramid,
                padded_bbox,
                args.max_dim,
            )
            mask = build_mask(polygons, padded_bbox, crop.size)
            overlay = build_overlay(crop, mask, args.overlay_alpha)

            crop_path = stage3_dir / "crop.png"
            mask_path = stage3_dir / "mask.png"
            overlay_path = stage3_dir / "overlay.png"
            crop.save(crop_path)
            mask.save(mask_path)
            overlay.save(overlay_path)

            metadata = {
                "source": "trident_geojson",
                "wsi_path": str(wsi_path),
                "geojson_path": str(geojson_path),
                "case_id": case_id,
                "run_id": run_id,
                "bbox_id": core_id,
                "feature_index": feature_idx,
                "properties": properties,
                "bbox_level0": list(raw_bbox),
                "padded_bbox_level0": list(padded_bbox),
                "crop_size": list(crop.size),
                "read_level": int(read_level),
                "read_downsample": float(read_downsample),
                "resolved_wsi_reader": reader,
                "overlay_alpha": float(args.overlay_alpha),
            }
            write_json(stage3_dir / "metadata.json", metadata)
            manifest_rows.append(
                {
                    "case_id": case_id,
                    "run_id": run_id,
                    "bbox_id": core_id,
                    "feature_index": feature_idx,
                    "bbox_level0": " ".join(str(v) for v in raw_bbox),
                    "padded_bbox_level0": " ".join(str(v) for v in padded_bbox),
                    "crop_path": str(crop_path),
                    "mask_path": str(mask_path),
                    "overlay_path": str(overlay_path),
                    "metadata_path": str(stage3_dir / "metadata.json"),
                }
            )

            exported += 1
            if args.max_items is not None and exported >= args.max_items:
                break

        run_meta = {
            "source": "trident_geojson",
            "created_at": datetime.now().isoformat(),
            "wsi_path": str(wsi_path),
            "geojson_path": str(geojson_path),
            "case_id": case_id,
            "run_id": run_id,
            "output_root": str(args.output_root),
            "exported_items": len(manifest_rows),
            "wsi_dimensions_level0": [int(wsi_w), int(wsi_h)],
            "resolved_wsi_reader": reader,
            "reviewer_batch_command": (
                f"python run_vlm_reviewer_batch.py --baseline-dir {args.output_root} "
                f"--run-selection latest --output-root runs/reviewer "
                f"--batch-name trident_review_{case_id}_{run_id}"
            ),
        }
        write_json(run_dir / "metadata.json", run_meta)
        write_reproduction_file(run_dir / "reproduction.txt")
    finally:
        close_wsi(wsi, reader)

    manifest_csv = run_dir / "manifest.csv"
    fieldnames = [
        "case_id",
        "run_id",
        "bbox_id",
        "feature_index",
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
