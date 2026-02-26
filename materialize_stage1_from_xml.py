#!/usr/bin/env python3
# ABOUTME: Materialize Stage 1 outputs from ROI XML annotations instead of VLM detection.
# ABOUTME: Produces thumbnail.png, bboxes.json, metadata.json, and bbox_overlay.png compatible with downstream stages.
"""
Materialize Stage 1 output from ROI XML annotations.

This script bypasses VLM Stage 1 and converts XML ROI annotations into the
same Stage 1 contract consumed by Stage 2+:
  - thumbnail.png
  - bboxes.json
  - metadata.json

Typical usage:
  python materialize_stage1_from_xml.py --wsi slide.svs --xml roi.xml --output-dir /tmp/run/stage1

Or write under stage1_output/{wsi_id}/{model_tag}/{timestamp}:
  python materialize_stage1_from_xml.py --wsi slide.svs --xml roi.xml
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw

from utils.wsi_backend import close_wsi, get_pyramid_info, load_wsi, read_region_rgb

DEFAULT_MAX_DIM = 1024
DEFAULT_OUTPUT_BASE = Path("stage1_output")


@dataclass
class RawAnnotation:
    annotation_index: int
    annotation_type: str
    annotation_name: str
    group: str
    bbox_float: Tuple[float, float, float, float]
    points_count: int


def sanitize_model_name(name: str) -> str:
    return name.replace("/", "_").replace(":", "_").replace("-", "_")

def read_wsi_thumbnail(
    wsi_path: Path,
    max_dim: int,
    wsi_reader: str = "auto",
) -> Tuple[Image.Image, int, int, Dict[str, object]]:
    """
    Open WSI and generate thumbnail using requested reader.
    """
    wsi, resolved_reader = load_wsi(str(wsi_path), wsi_reader)
    try:
        pyramid = get_pyramid_info(wsi, resolved_reader)
        wsi_w, wsi_h = [int(v) for v in pyramid["level_dimensions"][0]]

        if resolved_reader == "openslide":
            thumb = wsi.get_thumbnail((max_dim, max_dim)).convert("RGB")
            return thumb, wsi_w, wsi_h, {"reader": "openslide"}

        level = int(pyramid["level_count"]) - 1
        level_w, level_h = [int(v) for v in pyramid["level_dimensions"][level]]
        arr = read_region_rgb(
            wsi,
            resolved_reader,
            x=0,
            y=0,
            width=level_w,
            height=level_h,
            level=level,
        )
        thumb = Image.fromarray(arr).convert("RGB")
        current_max = max(thumb.size)
        if current_max > max_dim:
            scale = max_dim / float(current_max)
            target = (max(1, int(thumb.size[0] * scale)), max(1, int(thumb.size[1] * scale)))
            thumb = thumb.resize(target, Image.LANCZOS)
        return thumb, wsi_w, wsi_h, {"reader": resolved_reader, "level": level}
    finally:
        close_wsi(wsi, resolved_reader)


def parse_xml_annotations(
    xml_path: Path,
    group_name: str,
    include_non_rect: bool,
) -> Tuple[List[RawAnnotation], int]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    annotations = root.findall(".//Annotations/Annotation")
    total_annotations = len(annotations)

    out: List[RawAnnotation] = []
    for ann_idx, ann in enumerate(annotations):
        ann_group = ann.attrib.get("PartOfGroup", "")
        if ann_group != group_name:
            continue
        ann_type = ann.attrib.get("Type", "")
        if not include_non_rect and ann_type.lower() != "rectangle":
            continue

        coords = ann.findall(".//Coordinate")
        if not coords:
            continue
        points: List[Tuple[float, float]] = []
        for coord in coords:
            try:
                x = float(coord.attrib["X"])
                y = float(coord.attrib["Y"])
            except (KeyError, TypeError, ValueError):
                continue
            points.append((x, y))
        if len(points) < 2:
            continue

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        out.append(
            RawAnnotation(
                annotation_index=ann_idx,
                annotation_type=ann_type,
                annotation_name=ann.attrib.get("Name", ""),
                group=ann_group,
                bbox_float=(min(xs), min(ys), max(xs), max(ys)),
                points_count=len(points),
            )
        )
    return out, total_annotations


def clamp_bbox_to_slide(
    bbox: Sequence[float],
    wsi_w: int,
    wsi_h: int,
) -> Optional[List[int]]:
    x1f, y1f, x2f, y2f = bbox
    x1f, x2f = (x1f, x2f) if x1f <= x2f else (x2f, x1f)
    y1f, y2f = (y1f, y2f) if y1f <= y2f else (y2f, y1f)

    x1 = int(math.floor(x1f))
    y1 = int(math.floor(y1f))
    x2 = int(math.ceil(x2f))
    y2 = int(math.ceil(y2f))

    x1 = max(0, min(wsi_w - 1, x1))
    y1 = max(0, min(wsi_h - 1, y1))
    x2 = max(1, min(wsi_w, x2))
    y2 = max(1, min(wsi_h, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def bbox_level0_to_thumbnail(
    bbox_level0: Sequence[int],
    wsi_w: int,
    wsi_h: int,
    thumb_w: int,
    thumb_h: int,
) -> List[int]:
    sx = thumb_w / float(wsi_w)
    sy = thumb_h / float(wsi_h)
    x1 = int(math.floor(int(bbox_level0[0]) * sx))
    y1 = int(math.floor(int(bbox_level0[1]) * sy))
    x2 = int(math.ceil(int(bbox_level0[2]) * sx))
    y2 = int(math.ceil(int(bbox_level0[3]) * sy))
    x1 = max(0, min(max(thumb_w - 1, 0), x1))
    y1 = max(0, min(max(thumb_h - 1, 0), y1))
    x2 = max(x1 + 1, min(thumb_w, x2))
    y2 = max(y1 + 1, min(thumb_h, y2))
    return [x1, y1, x2, y2]


def bbox_level0_to_normalized(
    bbox_level0: Sequence[int],
    wsi_w: int,
    wsi_h: int,
) -> List[int]:
    x1, y1, x2, y2 = [int(v) for v in bbox_level0]
    vals = [
        int(round((x1 / float(wsi_w)) * 1000.0)),
        int(round((y1 / float(wsi_h)) * 1000.0)),
        int(round((x2 / float(wsi_w)) * 1000.0)),
        int(round((y2 / float(wsi_h)) * 1000.0)),
    ]
    return [max(0, min(1000, v)) for v in vals]


def draw_overlay(thumbnail: Image.Image, regions: List[Dict[str, object]], out_path: Path) -> None:
    img = thumbnail.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    for region in regions:
        x1, y1, x2, y2 = [int(v) for v in region["bbox_thumbnail"]]
        draw.rectangle((x1, y1, x2, y2), outline=(255, 0, 0), width=2)
    img.save(out_path)


def create_output_dir(
    *,
    output_dir: Optional[Path],
    output_base: Path,
    wsi_path: Path,
    model_tag: str,
) -> Path:
    if output_dir is not None:
        return output_dir
    wsi_id = wsi_path.stem
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_base / wsi_id / sanitize_model_name(model_tag) / ts


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize Stage 1 outputs from ROI XML annotations.",
    )
    parser.add_argument("--wsi", type=Path, required=True, help="Path to WSI file.")
    parser.add_argument("--xml", type=Path, required=True, help="Path to ROI XML file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Direct output directory for stage1 files. If omitted, writes under --output-base.",
    )
    parser.add_argument(
        "--output-base",
        type=Path,
        default=DEFAULT_OUTPUT_BASE,
        help="Base output directory when --output-dir is omitted (default: stage1_output).",
    )
    parser.add_argument("--xml-group", type=str, default="biopsy", help="ROI group to extract.")
    parser.add_argument(
        "--include-non-rect",
        action="store_true",
        help="Use coordinate extents for non-Rectangle annotations in the selected group.",
    )
    parser.add_argument("--model-tag", type=str, default="xml_roi", help="Model tag for metadata/output path.")
    parser.add_argument("--max-dim", type=int, default=DEFAULT_MAX_DIM, help="Thumbnail max dimension.")
    parser.add_argument(
        "--wsi-reader",
        choices=["auto", "openslide", "cucim"],
        default="auto",
        help="WSI reader backend (default: auto).",
    )
    parser.add_argument(
        "--skip-dvc-check",
        action="store_true",
        help="Accepted for pipeline CLI compatibility; no-op in this script.",
    )
    return parser


def main() -> int:
    parser = create_parser()
    args = parser.parse_args()

    wsi_path = args.wsi.expanduser().resolve()
    xml_path = args.xml.expanduser().resolve()
    if not wsi_path.exists():
        raise FileNotFoundError(f"WSI not found: {wsi_path}")
    if not xml_path.exists():
        raise FileNotFoundError(f"XML not found: {xml_path}")
    if args.max_dim < 32:
        raise ValueError("--max-dim must be >= 32")

    out_dir = create_output_dir(
        output_dir=args.output_dir,
        output_base=args.output_base.expanduser().resolve(),
        wsi_path=wsi_path,
        model_tag=args.model_tag,
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading WSI: {wsi_path}")
    thumbnail, wsi_w, wsi_h, wsi_info = read_wsi_thumbnail(
        wsi_path,
        args.max_dim,
        args.wsi_reader,
    )
    thumb_w, thumb_h = thumbnail.size
    print(f"WSI dimensions: {wsi_w}x{wsi_h}")
    print(f"Thumbnail dimensions: {thumb_w}x{thumb_h}")

    print(f"Parsing XML: {xml_path}")
    raw_regions, total_annotations = parse_xml_annotations(
        xml_path=xml_path,
        group_name=args.xml_group,
        include_non_rect=bool(args.include_non_rect),
    )
    if not raw_regions:
        raise RuntimeError(
            f"No usable annotations found in group '{args.xml_group}' "
            f"(include_non_rect={bool(args.include_non_rect)})"
        )

    regions: List[Dict[str, object]] = []
    seen = set()
    for raw_idx, ann in enumerate(raw_regions):
        bbox_level0 = clamp_bbox_to_slide(ann.bbox_float, wsi_w=wsi_w, wsi_h=wsi_h)
        if bbox_level0 is None:
            continue
        key = tuple(int(v) for v in bbox_level0)
        if key in seen:
            continue
        seen.add(key)
        regions.append(
            {
                "label": f"tissue_{len(regions) + 1}",
                "bbox_level0": bbox_level0,
                "bbox_thumbnail": bbox_level0_to_thumbnail(
                    bbox_level0=bbox_level0,
                    wsi_w=wsi_w,
                    wsi_h=wsi_h,
                    thumb_w=thumb_w,
                    thumb_h=thumb_h,
                ),
                "bbox_normalized": bbox_level0_to_normalized(
                    bbox_level0=bbox_level0,
                    wsi_w=wsi_w,
                    wsi_h=wsi_h,
                ),
                "xml_source": {
                    "annotation_index": ann.annotation_index,
                    "annotation_type": ann.annotation_type,
                    "annotation_name": ann.annotation_name,
                    "points_count": ann.points_count,
                    "bbox_float": [float(v) for v in ann.bbox_float],
                    "raw_match_index": raw_idx,
                },
            }
        )

    if not regions:
        raise RuntimeError("XML annotations were found but all bboxes were invalid after clipping.")

    thumbnail.save(out_dir / "thumbnail.png")
    draw_overlay(thumbnail, regions, out_dir / "bbox_overlay.png")

    bboxes_payload = {
        "detected_regions": regions,
        "regions_count": len(regions),
        "source": "xml_roi",
        "xml_path": str(xml_path),
        "xml_group": args.xml_group,
        "include_non_rect": bool(args.include_non_rect),
    }
    with (out_dir / "bboxes.json").open("w", encoding="utf-8") as f:
        json.dump(bboxes_payload, f, indent=2)

    metadata = {
        "wsi_path": str(wsi_path),
        "wsi_dimensions": {"width": int(wsi_w), "height": int(wsi_h)},
        "thumbnail_dimensions": {"width": int(thumb_w), "height": int(thumb_h)},
        "model": args.model_tag,
        "backend": "xml",
        "bbox_coord_order": "xyxy",
        "resolved_bbox_coord_order": "xyxy",
        "prompt": "N/A (XML ROI input)",
        "max_dim": int(args.max_dim),
        "padding": 0.0,
        "merge_overlap_threshold": None,
        "detected_regions": regions,
        "regions_count": len(regions),
        "xml_path": str(xml_path),
        "xml_group": args.xml_group,
        "xml_include_non_rect": bool(args.include_non_rect),
        "xml_total_annotations": int(total_annotations),
        "wsi_reader": wsi_info,
        "created_at": datetime.now().isoformat(),
        "git_hash": "unknown",
        "reproducibility_bypassed": True,
        "source": "materialize_stage1_from_xml",
    }
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved Stage 1 materialization: {out_dir}")
    print(f"Regions: {len(regions)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
