from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import refresh_detector_final_outputs as refresh  # noqa: E402


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _make_case(root: Path, case_id: str, final_boxes: int, odd_flagged: list[int]) -> dict:
    case_dir = root / case_id
    thumb = case_dir / "intermediate_stage_artifacts" / "stage1_thumbnail_detection" / "thumbnail.png"
    overlay = case_dir / "final_detected_bboxes.png"
    thumb.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (120, 80), "white").save(thumb)
    Image.new("RGB", (120, 80), "white").save(overlay)
    record = {
        "case_id": case_id,
        "case_display": f"{case_id}.svs",
        "wsi_path": f"/tmp/{case_id}.svs",
        "pipeline_status": "ok",
        "coordinate_system": "normalized_0_1000_y_min_x_min_y_max_x_max",
        "stage_contract": [
            {
                "stage": "stage7_comparative_thumbnail_filter",
                "prompt": "/tmp/odd_prompt.txt",
                "input_image": "thumbnail crops",
                "output": "odd-one-out flags",
            }
        ],
        "stage_counts": {
            "classification_yes": final_boxes + len(odd_flagged),
            "final_boxes": final_boxes,
            "odd_one_out_filter_skipped": False,
            "odd_one_out_flagged": len(odd_flagged),
        },
        "paths": {
            "thumbnail_path": str(thumb),
            "final_overlay_png": str(overlay),
            "odd_one_out_sheet_path": str(case_dir / "odd.png"),
        },
        "odd_one_out": {
            "ran": True,
            "filter_skipped": False,
            "filter_applied": True,
            "parse_status": "ok",
            "flagged_candidate_orders": odd_flagged,
            "raw_flagged_candidate_orders": odd_flagged,
            "skipped": {},
            "error": "",
        },
        "detections": [
            {
                "box_2d": [10, 10, 20, 20],
                "source_candidate_order": 1,
                "classification_decision": "yes",
                "odd_one_out_flagged": False,
            }
        ][:final_boxes],
        "errors": [],
        "degraded_fallbacks": [],
        "stage_artifacts_saved": True,
        "stage_artifacts_dir": str(case_dir / "intermediate_stage_artifacts"),
    }
    _write_json(case_dir / "detections.json", record)
    (case_dir / "reproduction.txt").write_text("original reproduction\n")
    aggregate = dict(record)
    aggregate["case_dir"] = str(case_dir)
    return aggregate


def test_refresh_sv40_skips_odd_filter_without_touching_non_sv40(tmp_path: Path) -> None:
    root = tmp_path / "run"
    sv40 = _make_case(root, "sv40_patient_004_slide_001", final_boxes=1, odd_flagged=[2])
    evg = _make_case(root, "evg_patient_004_slide_001", final_boxes=1, odd_flagged=[])
    _write_json(root / "all_detections.json", [sv40, evg])
    _write_json(
        root / "summary.json",
        {
            "final_boxes": 2,
            "case_outputs": [
                {"case_id": sv40["case_id"], "final_box_count": 1},
                {"case_id": evg["case_id"], "final_box_count": 1},
            ],
        },
    )
    (root / "reproduction.txt").write_text("root reproduction\n")
    stage6 = root / "intermediate_stage_artifacts" / "stage6_classification_results.jsonl"
    stage6.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "case_id": sv40["case_id"],
            "candidate_order": 1,
            "box_2d_yxyx_normalized": [10, 10, 20, 20],
            "tissue_focus_decision": "yes",
            "error": "",
        },
        {
            "case_id": sv40["case_id"],
            "candidate_order": 2,
            "box_2d_yxyx_normalized": [30, 30, 40, 40],
            "tissue_focus_decision": "yes",
            "error": "",
        },
    ]
    stage6.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")

    evg_json = root / evg["case_id"] / "detections.json"
    evg_overlay = root / evg["case_id"] / "final_detected_bboxes.png"
    evg_json_before = evg_json.read_text()
    evg_overlay_before = _sha256(evg_overlay)
    evg_aggregate_before = evg.copy()

    args = refresh.build_parser().parse_args(
        [
            str(root),
            "--case-glob",
            "sv40_*",
            "--skip-odd-one-out-filter",
            "--no-write-review-pdf",
            "--no-backup",
        ]
    )
    refresh._run(args)

    refreshed = json.loads((root / sv40["case_id"] / "detections.json").read_text())
    assert refreshed["stage_counts"]["final_boxes"] == 2
    assert refreshed["stage_counts"]["odd_one_out_filter_skipped"] is True
    assert refreshed["odd_one_out"]["ran"] is False
    assert refreshed["odd_one_out"]["filter_skipped"] is True
    assert refreshed["paths"]["odd_one_out_sheet_path"] == ""
    assert refreshed["stage_contract"][0]["prompt"] is None

    assert evg_json.read_text() == evg_json_before
    assert _sha256(evg_overlay) == evg_overlay_before
    aggregate = json.loads((root / "all_detections.json").read_text())
    assert aggregate[1] == evg_aggregate_before
    summary = json.loads((root / "summary.json").read_text())
    assert summary["final_boxes"] == 3
    assert summary["case_outputs"][0]["final_box_count"] == 2
