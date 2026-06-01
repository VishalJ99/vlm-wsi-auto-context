from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import export_detector_training_dataset as exporter  # noqa: E402


def test_normalized_yxyx_conversions() -> None:
    box = [100, 200, 300, 500]

    assert exporter.normalized_yxyx_to_pixel_xywh(box, width=100, height=50) == (
        20.0,
        5.0,
        30.0,
        10.0,
    )
    assert exporter.normalized_yxyx_to_yolo_xywh(box) == (0.35, 0.2, 0.3, 0.2)


def _write_tiny_pipeline_case(root: Path, case_id: str, wsi_path: str) -> dict:
    thumbnail_path = root / case_id / "intermediate_stage_artifacts" / "stage1_thumbnail_detection" / "thumbnail.png"
    thumbnail_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (100, 50), color=(240, 240, 240)).save(thumbnail_path)
    return {
        "case_id": case_id,
        "case_display": f"{case_id}.svs",
        "wsi_path": wsi_path,
        "ticket": "PER-207",
        "git_commit": "abc123",
        "pipeline_version": "test",
        "coordinate_system": "normalized_0_1000_y_min_x_min_y_max_x_max",
        "paths": {"thumbnail_path": str(thumbnail_path)},
        "detections": [
            {
                "box_2d": [100, 200, 300, 500],
                "classification_decision": "yes",
                "odd_one_out_flagged": False,
                "source_candidate_order": 1,
            }
        ],
    }


def test_export_dataset_writes_coco_yolo_and_group_safe_manifests(tmp_path: Path) -> None:
    pipeline_root = tmp_path / "pipeline"
    records = []
    for patient in ("001", "002", "003", "004"):
        for stain in ("evg", "he"):
            case_id = f"{stain}_patient_{patient}_slide_001"
            records.append(
                _write_tiny_pipeline_case(
                    pipeline_root,
                    case_id,
                    f"/data2/example/{stain.upper()}/{case_id}.svs",
                )
            )
    (pipeline_root / "all_detections.json").write_text(json.dumps(records))

    output_dir = tmp_path / "dataset"
    args = argparse.Namespace(
        pipeline_output_roots=[pipeline_root],
        output_dir=output_dir,
        ticket="PER-248",
        image_mode="copy",
        group_by="auto",
        split_fractions=(0.5, 0.25, 0.25),
        seed=13,
        class_name="tissue_candidate",
        overwrite=True,
        skip_validation=False,
    )
    summary = exporter.export_dataset(args)

    assert summary["case_count"] == 8
    assert summary["box_count"] == 8
    assert summary["group_count"] == 4
    assert summary["ticket"] == "PER-248"
    assert summary["source_pipeline_roots"] == [str(pipeline_root.resolve())]
    assert (output_dir / "annotations" / "instances_all.json").is_file()
    assert (output_dir / "dataset.yaml").is_file()
    assert (output_dir / "validation.json").is_file()
    assert (output_dir / "reproduction.txt").is_file()

    coco = json.loads((output_dir / "annotations" / "instances_all.json").read_text())
    assert len(coco["images"]) == 8
    assert len(coco["annotations"]) == 8
    assert coco["annotations"][0]["bbox"] == [20.0, 5.0, 30.0, 10.0]
    assert coco["annotations"][0]["attributes"]["box_2d"] == [100.0, 200.0, 300.0, 500.0]

    group_splits: dict[str, set[str]] = {}
    with (output_dir / "manifests" / "cases.csv").open() as handle:
        for row in csv.DictReader(handle):
            group_splits.setdefault(row["group_id"], set()).add(row["split"])
    assert group_splits
    assert all(len(splits) == 1 for splits in group_splits.values())

    label_paths = sorted((output_dir / "labels").glob("*/*.txt"))
    assert len(label_paths) == 8
    assert label_paths[0].read_text().strip() == "0 0.35 0.2 0.3 0.2"


def test_export_dataset_accepts_multiple_pipeline_roots(tmp_path: Path) -> None:
    evg_root = tmp_path / "pipeline_evg"
    sv40_root = tmp_path / "pipeline_sv40"
    evg_record = _write_tiny_pipeline_case(
        evg_root,
        "evg_patient_001_slide_001",
        "/data2/example/EVG/evg_patient_001_slide_001.svs",
    )
    sv40_record = _write_tiny_pipeline_case(
        sv40_root,
        "sv40_patient_001_slide_001",
        "/data2/example/SV40/sv40_patient_001_slide_001.svs",
    )
    (evg_root / "all_detections.json").write_text(json.dumps([evg_record]))
    (sv40_root / "all_detections.json").write_text(json.dumps([sv40_record]))

    output_dir = tmp_path / "dataset"
    args = argparse.Namespace(
        pipeline_output_roots=[evg_root, sv40_root],
        output_dir=output_dir,
        ticket="PER-248",
        image_mode="copy",
        group_by="case",
        split_fractions=(1.0, 0.0, 0.0),
        seed=13,
        class_name="tissue_candidate",
        overwrite=True,
        skip_validation=False,
    )

    summary = exporter.export_dataset(args)

    assert summary["case_count"] == 2
    assert summary["box_count"] == 2
    assert summary["source_pipeline_roots"] == [str(evg_root.resolve()), str(sv40_root.resolve())]
    assert summary["validation"]["status"] == "ok"
    coco = json.loads((output_dir / "annotations" / "instances_all.json").read_text())
    assert coco["info"]["ticket"] == "PER-248"
    assert sorted(image["case_id"] for image in coco["images"]) == [
        "evg_patient_001_slide_001",
        "sv40_patient_001_slide_001",
    ]
