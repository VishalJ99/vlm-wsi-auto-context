from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import export_scale500_detector_to_auto_context as exporter  # noqa: E402


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_verifier_policy_uses_revised_ids(tmp_path: Path) -> None:
    selection_path = tmp_path / "results.jsonl"
    _write_jsonl(
        selection_path,
        [
            {
                "case_id": "anon_a",
                "baseline_selected_box_ids": [1, 2],
                "direct_selected_box_ids": [1, 2],
                "verifier_selected_box_ids": [2],
                "verifier_added_vs_baseline_box_ids": [],
                "verifier_dropped_vs_baseline_box_ids": [1],
                "verifier_parse_status": "ok",
            }
        ],
    )

    loaded = exporter.load_selection_map(selection_path, "verifier")

    assert loaded is not None
    assert loaded["anon_a"]["selected_box_ids"] == [2]
    assert loaded["anon_a"]["selection_source"] == "verifier_selected_box_ids"


def test_verifier_policy_fails_without_clean_verifier_ids(tmp_path: Path) -> None:
    selection_path = tmp_path / "results.jsonl"
    _write_jsonl(
        selection_path,
        [
            {
                "case_id": "anon_a",
                "baseline_selected_box_ids": [1, 2],
                "direct_selected_box_ids": [1, 2],
                "verifier_selected_box_ids": [],
                "verifier_parse_status": "schema_warning",
            }
        ],
    )

    try:
        exporter.load_selection_map(selection_path, "verifier")
    except exporter.ExportError as exc:
        assert "strict verifier selection" in str(exc)
    else:
        raise AssertionError("Expected strict verifier policy to fail")


def test_verifier_or_baseline_policy_keeps_legacy_fallback(tmp_path: Path) -> None:
    selection_path = tmp_path / "results.jsonl"
    _write_jsonl(
        selection_path,
        [
            {
                "case_id": "anon_a",
                "baseline_selected_box_ids": [1, 2],
                "direct_selected_box_ids": [1, 2],
                "verifier_selected_box_ids": [],
                "verifier_parse_status": "schema_warning",
            }
        ],
    )

    loaded = exporter.load_selection_map(selection_path, "verifier-or-baseline")

    assert loaded is not None
    assert loaded["anon_a"]["selected_box_ids"] == [1, 2]
    assert loaded["anon_a"]["selection_source"] == "baseline_fallback_verifier_not_ok_or_empty"


def test_conservative_policy_keeps_baseline_for_replacements(tmp_path: Path) -> None:
    selection_path = tmp_path / "results.jsonl"
    _write_jsonl(
        selection_path,
        [
            {
                "case_id": "anon_a",
                "baseline_selected_box_ids": [1, 2],
                "direct_selected_box_ids": [3, 4],
                "verifier_selected_box_ids": [3, 4],
                "verifier_added_vs_baseline_box_ids": [3, 4],
                "verifier_dropped_vs_baseline_box_ids": [1, 2],
                "verifier_parse_status": "ok",
            }
        ],
    )

    loaded = exporter.load_selection_map(selection_path, "conservative-verifier-drop-only")

    assert loaded is not None
    assert loaded["anon_a"]["selected_box_ids"] == [1, 2]
    assert loaded["anon_a"]["selection_source"] == "baseline_fallback_not_drop_only"


def test_case_input_root_resolves_mixed_detector_subroot(tmp_path: Path) -> None:
    case_id = "anon_a"
    case_dir = tmp_path / "detector_root" / "non_sv40" / case_id
    case_dir.mkdir(parents=True)
    detections_path = case_dir / "detections.json"
    detections_path.write_text('{"detections": []}', encoding="utf-8")
    case_input_dir = tmp_path / "selection_probe" / "cases" / case_id
    case_input_dir.mkdir(parents=True)
    (case_input_dir / "case_input.json").write_text(
        json.dumps({"detections_path": str(detections_path)}),
        encoding="utf-8",
    )

    resolved = exporter.case_dir_from_case_input(tmp_path / "selection_probe", case_id)

    assert resolved == case_dir.resolve()


def test_stage2_crop_falls_back_to_synced_candidate_dir_when_saved_path_is_stale(tmp_path: Path) -> None:
    candidate_dir = tmp_path / "synced_candidate"
    candidate_dir.mkdir()
    Image.new("RGB", (12, 12), (200, 10, 10)).save(candidate_dir / "crop.png")
    thumbnail_path = tmp_path / "thumbnail.png"
    Image.new("RGB", (12, 12), (10, 10, 200)).save(thumbnail_path)
    output_path = tmp_path / "stage2" / "bbox_region.png"

    crop_info = exporter.save_stage2_crop(
        dst_png=output_path,
        candidate_meta={
            "candidate": {
                "crop_path": str(tmp_path / "stale_data2_path" / "crop.png"),
                "read_info": {},
            }
        },
        candidate_dir=candidate_dir,
        thumbnail_path=thumbnail_path,
        bbox_thumb=[0, 0, 12, 12],
        crop_mode="source-bbox",
    )

    saved = Image.open(output_path).convert("RGB")
    assert saved.getpixel((0, 0)) == (200, 10, 10)
    assert crop_info["bbox_region_source"] == "scale500_candidate_crop"
    assert crop_info["source_crop_path"] == str(candidate_dir / "crop.png")
