#!/usr/bin/env python3
"""Build a Stage 6-like patch grid from reviewer crop masks.

This adapter is for one-off distilled-student checks on existing reviewer
inputs. It does not call a VLM. It converts each reviewer `stage3/mask.png`
inside a bbox into a `stage6/patches.csv` grid that the packaged student
inference script can consume.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewer-input-root", required=True, type=Path)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--wsi-manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--split-manifest-output", required=True, type=Path)
    parser.add_argument("--manifest-csv-output", required=True, type=Path)
    parser.add_argument("--manifest-json-output", required=True, type=Path)
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--label-threshold", type=float, default=0.5)
    parser.add_argument("--overlay-max-dim-default", type=int, default=2048)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_case_to_wsi(manifest_csv: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with manifest_csv.open(newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            path = row[0].strip()
            if not path or path.lower() in {"path", "wsi_path"}:
                continue
            case_id = Path(path).stem
            if case_id.startswith("anon_"):
                mapping[case_id] = path
    if not mapping:
        raise RuntimeError(f"no anon_<uuid> WSI paths found in {manifest_csv}")
    return mapping


def bbox_dict_to_list(value: object) -> list[int]:
    if isinstance(value, dict):
        return [int(value["x1"]), int(value["y1"]), int(value["x2"]), int(value["y2"])]
    if isinstance(value, list) and len(value) == 4:
        return [int(v) for v in value]
    raise ValueError(f"cannot parse bbox: {value!r}")


def iter_stage3_dirs(reviewer_root: Path, run_name: str) -> Iterable[Path]:
    pattern = f"*/{run_name}/bboxes/*/stage3/metadata.json"
    for meta_path in sorted(reviewer_root.glob(pattern)):
        yield meta_path.parent


def tissue_fraction(mask: np.ndarray, x0: float, y0: float, x1: float, y1: float) -> float:
    height, width = mask.shape
    ix0 = max(0, min(width, int(math.floor(x0))))
    iy0 = max(0, min(height, int(math.floor(y0))))
    ix1 = max(0, min(width, int(math.ceil(x1))))
    iy1 = max(0, min(height, int(math.ceil(y1))))
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    return float((mask[iy0:iy1, ix0:ix1] > 0).mean())


def copy_if_present(src: Path, dst: Path) -> None:
    if src.is_file():
        shutil.copy2(src, dst)


def main() -> None:
    args = parse_args()
    if args.output_root.exists() and not args.overwrite:
        raise RuntimeError(f"output root exists; pass --overwrite to replace files: {args.output_root}")

    case_to_wsi = load_case_to_wsi(args.wsi_manifest)
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.split_manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_csv_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_json_output.parent.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, object]] = []
    cases: set[str] = set()

    for stage3_dir in iter_stage3_dirs(args.reviewer_input_root, args.run_name):
        source_meta = json.loads((stage3_dir / "metadata.json").read_text())
        case_id = str(source_meta["case_id"])
        bbox_id = str(source_meta["bbox_id"])
        wsi_path = case_to_wsi.get(case_id) or str(source_meta.get("wsi_path", ""))
        if not wsi_path:
            raise RuntimeError(f"could not resolve WSI path for {case_id}")

        padded_bbox = bbox_dict_to_list(source_meta.get("padded_bbox_level0") or source_meta["bbox_level0"])
        original_bbox = bbox_dict_to_list(source_meta["bbox_level0"])
        x1, y1, x2, y2 = padded_bbox
        bbox_w = max(1, x2 - x1)
        bbox_h = max(1, y2 - y1)
        rows = int(math.ceil(bbox_h / args.patch_size))
        cols = int(math.ceil(bbox_w / args.patch_size))

        mask_img = Image.open(stage3_dir / "mask.png").convert("L")
        mask_np = np.asarray(mask_img)
        mask_h, mask_w = mask_np.shape
        x_scale = mask_w / bbox_w
        y_scale = mask_h / bbox_h

        stage6_dir = args.output_root / case_id / "bboxes" / bbox_id / "stage6"
        stage6_dir.mkdir(parents=True, exist_ok=True)
        copy_if_present(stage3_dir / "crop.png", stage6_dir / "trident_reference_crop.png")
        copy_if_present(stage3_dir / "mask.png", stage6_dir / "trident_reference_mask.png")
        copy_if_present(stage3_dir / "overlay.png", stage6_dir / "trident_reference_overlay.png")
        copy_if_present(stage3_dir / "overlay.png", stage6_dir / "class_overlay.png")

        class_map = np.zeros((rows, cols), dtype=np.int16)
        patch_rows: list[dict[str, object]] = []
        for row in range(rows):
            for col in range(cols):
                wsi_x = x1 + col * args.patch_size
                wsi_y = y1 + row * args.patch_size
                mx0 = (wsi_x - x1) * x_scale
                my0 = (wsi_y - y1) * y_scale
                mx1 = (wsi_x + args.patch_size - x1) * x_scale
                my1 = (wsi_y + args.patch_size - y1) * y_scale
                frac = tissue_fraction(mask_np, mx0, my0, mx1, my1)
                is_tissue = frac >= args.label_threshold
                class_map[row, col] = 1 if is_tissue else 0
                patch_rows.append(
                    {
                        "row": row,
                        "col": col,
                        "wsi_x": wsi_x,
                        "wsi_y": wsi_y,
                        "patch_w": args.patch_size,
                        "patch_h": args.patch_size,
                        "pred_label_canonical": "tissue" if is_tissue else "background",
                        "trident_mask_tissue_fraction": frac,
                        "case_id": case_id,
                        "bbox_id": bbox_id,
                        "wsi_path": wsi_path,
                    }
                )

        with (stage6_dir / "patches.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(patch_rows[0]))
            writer.writeheader()
            writer.writerows(patch_rows)
        np.save(stage6_dir / "class_map.npy", class_map)

        metadata = {
            "case_id": case_id,
            "bbox_id": bbox_id,
            "source": "synthetic_stage6_grid_from_trident_reviewer_mask",
            "source_reviewer_stage3": str(stage3_dir),
            "wsi_path": wsi_path,
            "wsi_manifest": str(args.wsi_manifest),
            "bbox_level0": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "original_stage1_bbox_level0": original_bbox,
            "padded_bbox_level0": padded_bbox,
            "patch_size": args.patch_size,
            "grid_shape": {"rows": rows, "cols": cols},
            "overlay_read": {"target_max_dim": args.overlay_max_dim_default},
            "label_source": "trident_reviewer_mask_patch_fraction",
            "label_threshold_tissue_fraction": args.label_threshold,
            "note": "padded bbox is used to match the existing reviewer crop/mask extent",
        }
        (stage6_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

        manifest_rows.append(
            {
                "case_id": case_id,
                "bbox_id": bbox_id,
                "stage6_dir": str(stage6_dir),
                "n_patches": len(patch_rows),
                "grid_rows": rows,
                "grid_cols": cols,
                "wsi_path": wsi_path,
                "source_stage3": str(stage3_dir),
            }
        )
        cases.add(case_id)

    if not manifest_rows:
        raise RuntimeError(f"no reviewer stage3 metadata found for run {args.run_name}")

    args.split_manifest_output.write_text(json.dumps({"test": sorted(cases)}, indent=2) + "\n")
    args.manifest_json_output.write_text(json.dumps(manifest_rows, indent=2) + "\n")
    with args.manifest_csv_output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"wrote {len(manifest_rows)} bbox grids across {len(cases)} cases")
    print(args.output_root)


if __name__ == "__main__":
    main()
