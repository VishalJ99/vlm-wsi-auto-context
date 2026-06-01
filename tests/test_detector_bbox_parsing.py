from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from detect_foreground_regions_from_wsi_thumbnail import parse_bboxes_response  # noqa: E402
from stage1_detection_review_pilot import _normalised_detection_items  # noqa: E402


def test_stage1_detector_accepts_numbered_box_key() -> None:
    parsed = parse_bboxes_response(
        """
        ```json
        [
          {"box_2d": [10, 20, 30, 40]},
          {"Box_2": [206, 597, 720, 895]}
        ]
        ```
        """,
        coord_order="yxxy",
    )

    assert parsed == [
        {"bbox_2d": [10, 20, 30, 40], "label": "tissue_1"},
        {"bbox_2d": [206, 597, 720, 895], "label": "tissue_2"},
    ]


def test_pipeline_detection_items_accept_numbered_box_key() -> None:
    detections = _normalised_detection_items(
        [
            {"box_2d": [10, 20, 30, 40]},
            {"Box_2": [206, 597, 720, 895]},
        ],
        (2000, 1000),
    )

    assert [row["box_2d_yxyx_normalized"] for row in detections] == [
        [10, 20, 30, 40],
        [206, 597, 720, 895],
    ]
