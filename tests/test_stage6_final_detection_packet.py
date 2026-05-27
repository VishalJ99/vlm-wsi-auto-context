import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from stage6_final_detection_packet import _merge_duplicate_yxyx, _yxyx_overlap_metrics  # noqa: E402


def test_containment_overlap_merges_case49_like_nested_boxes() -> None:
    box_1 = [245.0, 75.0, 792.0, 237.0]
    box_2 = [423.0, 131.0, 747.0, 248.0]

    iou, overlap_over_smaller = _yxyx_overlap_metrics(box_1, box_2)
    assert round(iou, 3) == 0.373
    assert round(overlap_over_smaller, 3) == 0.906

    iou_only, iou_counts = _merge_duplicate_yxyx([box_1, box_2], 0.40, 0.0)
    assert iou_only == [box_1, box_2]
    assert iou_counts == {"total": 0, "iou": 0, "containment": 0}

    containment_aware, containment_counts = _merge_duplicate_yxyx([box_1, box_2], 0.40, 0.80)
    assert containment_aware == [[245.0, 75.0, 792.0, 248.0]]
    assert containment_counts == {"total": 1, "iou": 0, "containment": 1}
