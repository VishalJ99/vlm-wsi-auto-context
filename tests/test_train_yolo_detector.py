from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import train_yolo_detector as yolo  # noqa: E402


def test_yolo_box_conversion_iou_and_matching() -> None:
    gt = [
        yolo.BoxRecord((10.0, 10.0, 50.0, 50.0)),
        yolo.BoxRecord((100.0, 100.0, 160.0, 160.0)),
    ]
    preds = [
        yolo.BoxRecord((11.0, 11.0, 49.0, 49.0), score=0.9),
        yolo.BoxRecord((200.0, 200.0, 240.0, 240.0), score=0.8),
    ]

    assert yolo.box_iou(gt[0], preds[0]) > 0.9
    matches, unmatched_gt, unmatched_pred = yolo.greedy_match(gt, preds, iou_threshold=0.5)

    assert matches == [{"pred_idx": 0, "gt_idx": 0, "iou": yolo.box_iou(preds[0], gt[0])}]
    assert unmatched_gt == [1]
    assert unmatched_pred == [1]


def test_verify_dataset_accepts_exporter_layout(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    image_dir = dataset_dir / "images" / "train"
    label_dir = dataset_dir / "labels" / "train"
    manifest_dir = dataset_dir / "manifests"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    manifest_dir.mkdir(parents=True)

    image_path = image_dir / "case_001.png"
    Image.new("RGB", (100, 50), color=(255, 255, 255)).save(image_path)
    (label_dir / "case_001.txt").write_text("0 0.5 0.5 0.2 0.4\n")
    (dataset_dir / "dataset.yaml").write_text(
        f"path: {dataset_dir}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        "  0: tissue_candidate\n"
    )
    (manifest_dir / "cases.csv").write_text(
        "case_id,case_display,split,group_id,stain,patient_id,slide_id,wsi_path,"
        "source_thumbnail_path,image_path,image_relpath,width,height,box_count,"
        "pipeline_ticket,pipeline_git_commit,pipeline_version\n"
        f"case_001,case_001.svs,train,patient_001_slide_001,EVG,001,001,/tmp/a.svs,"
        f"{image_path},{image_path},images/train/case_001.png,100,50,1,PER-207,abc,test\n"
    )

    summary = yolo.verify_dataset(dataset_dir)

    assert summary["status"] == "ok"
    assert summary["image_count"] == 1
    assert summary["label_file_count"] == 1
    assert summary["label_row_count"] == 1
    assert summary["group_leak_count"] == 0


def test_parse_thresholds_includes_prediction_floor() -> None:
    assert yolo.parse_thresholds("0.25, 0.50", prediction_floor=0.10) == [0.1, 0.25, 0.5]


def test_yolo_augmentation_profiles_are_explicit() -> None:
    assert yolo.yolo_augmentation_kwargs("default") == {}
    reduced = yolo.yolo_augmentation_kwargs("reduced")
    assert reduced["mosaic"] == 0.0
    assert reduced["hsv_s"] == 0.0
    stain = yolo.yolo_augmentation_kwargs("stain-jitter")
    assert stain["hsv_s"] > reduced["hsv_s"]
    assert stain["mosaic"] > reduced["mosaic"]


def test_yolo_hyperparameter_kwargs_only_passes_overrides() -> None:
    defaults = argparse.Namespace(
        optimizer=None,
        lr0=None,
        lrf=None,
        weight_decay=None,
        cos_lr=False,
    )
    assert yolo.yolo_hyperparameter_kwargs(defaults) == {}

    tuned = argparse.Namespace(
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.1,
        weight_decay=0.0005,
        cos_lr=True,
    )
    assert yolo.yolo_hyperparameter_kwargs(tuned) == {
        "optimizer": "AdamW",
        "lr0": 0.001,
        "lrf": 0.1,
        "weight_decay": 0.0005,
        "cos_lr": True,
    }
