import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import train_rfdetr_detector as rfdetr_runner  # noqa: E402


def _write_export_dataset(root: Path) -> None:
    (root / "annotations").mkdir(parents=True)
    for split in ["train", "val", "test"]:
        image_dir = root / "images" / split
        image_dir.mkdir(parents=True)
        image_name = f"{split}_case.png"
        Image.new("RGB", (100, 80), "white").save(image_dir / image_name)
        coco = {
            "images": [
                {
                    "id": 1,
                    "file_name": f"images/{split}/{image_name}",
                    "width": 100,
                    "height": 80,
                    "case_id": f"{split}_case",
                    "stain": "PAS",
                }
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [10, 20, 30, 20],
                    "area": 600,
                    "iscrowd": 0,
                }
            ],
            "categories": [{"id": 1, "name": "tissue_candidate"}],
        }
        (root / "annotations" / f"instances_{split}.json").write_text(json.dumps(coco), encoding="utf-8")


def test_prepare_rfdetr_dataset_layout(tmp_path):
    source = tmp_path / "source"
    _write_export_dataset(source)
    dest = tmp_path / "rfdetr"

    summary = rfdetr_runner.prepare_rfdetr_dataset(source, dest, image_mode="copy")
    validation = rfdetr_runner.validate_rfdetr_dataset(dest)

    assert summary["splits"]["val"]["rfdetr_split"] == "valid"
    assert validation["status"] == "ok"
    assert (dest / "train" / "_annotations.coco.json").exists()
    assert (dest / "valid" / "val_case.png").exists()
    valid_coco = json.loads((dest / "valid" / "_annotations.coco.json").read_text())
    assert valid_coco["images"][0]["file_name"] == "val_case.png"
    assert valid_coco["images"][0]["source_file_name"] == "images/val/val_case.png"


def test_evaluate_predictions_counts_miss_false_and_ap():
    coco = {
        "images": [
            {"id": 1, "file_name": "a.png", "width": 100, "height": 100, "case_id": "a", "stain": "EVG"},
            {"id": 2, "file_name": "b.png", "width": 100, "height": 100, "case_id": "b", "stain": "SV40"},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20]},
            {"id": 2, "image_id": 2, "category_id": 1, "bbox": [60, 60, 20, 20]},
        ],
        "categories": [{"id": 1, "name": "tissue_candidate"}],
    }
    predictions = [
        {"image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.9},
        {"image_id": 1, "category_id": 1, "bbox": [12, 12, 20, 20], "score": 0.8},
        {"image_id": 2, "category_id": 1, "bbox": [0, 0, 10, 10], "score": 0.7},
    ]

    metrics, per_image, per_stain = rfdetr_runner.evaluate_predictions(
        coco,
        predictions,
        score_threshold=0.25,
        match_iou=0.5,
        oversized_area_ratio=2.5,
        large_area_fraction=0.35,
        near_full_fraction=0.8,
    )

    project = metrics["project_metrics"]
    assert project["true_positive_count"] == 1
    assert project["missed_tissue_count"] == 1
    assert project["false_box_count"] == 2
    assert project["duplicate_or_extra_fragment_count"] == 1
    assert metrics["ap"]["ap_50"] > 0
    assert {row["case_id"] for row in per_image} == {"a", "b"}
    assert {row["stain"] for row in per_stain} == {"EVG", "SV40"}
