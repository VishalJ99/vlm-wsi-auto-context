from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_detector_pipeline as pipeline  # noqa: E402


def test_odd_one_out_effective_flags_require_ok_parse() -> None:
    malformed = {
        "parse_status": "wrong_types",
        "flagged_candidate_orders": [1, 3],
        "effective_flagged_candidate_orders": [1, 3],
    }
    assert pipeline._effective_odd_flagged_orders(malformed) == set()

    ok = {
        "parse_status": "ok_single_flag_recovered",
        "flagged_candidate_orders": [1, 3],
    }
    assert pipeline._effective_odd_flagged_orders(ok) == {1, 3}

    errored = {
        "parse_status": "ok",
        "flagged_candidate_orders": [2],
        "error": "RuntimeError('boom')",
    }
    assert pipeline._effective_odd_flagged_orders(errored) == set()


def test_odd_flag_fields_preserve_raw_but_clear_effective_on_non_ok() -> None:
    task = {
        "patches": [
            {"id": 1, "candidate_order": 10},
            {"id": 2, "candidate_order": 20},
        ]
    }
    fields = pipeline._odd_flag_fields(task, {"flagged_artifacts": [2]}, "patch_id_mismatch")

    assert fields["flagged_artifacts"] == [2]
    assert fields["flagged_candidate_orders"] == [20]
    assert fields["filter_eligible"] is False
    assert fields["effective_flagged_artifacts"] == []
    assert fields["effective_flagged_candidate_orders"] == []
    assert fields["filter_skip_reason"] == "parse_status:patch_id_mismatch"


def test_reuse_existing_requires_matching_cache_fingerprint(tmp_path: Path) -> None:
    args = argparse.Namespace(
        backend="openrouter",
        model="google/gemini-3-flash-preview",
        temperature=0.0,
        reuse_existing=True,
    )
    result_path = tmp_path / "stage.json"
    pipeline._write_json(result_path, {"ok": True})

    assert not pipeline._reuse_allowed(result_path, "stage", args, {"input": "a"})

    pipeline._write_cache_sidecar(result_path, "stage", args, {"input": "a"})
    assert pipeline._reuse_allowed(result_path, "stage", args, {"input": "a"})
    assert not pipeline._reuse_allowed(result_path, "stage", args, {"input": "b"})


def test_stage2b_selection_exposes_missed_tissue_alias() -> None:
    args = argparse.Namespace(stage2b_trigger_source="first", force_stage3_redetect=False)
    selected = pipeline._stage2b_trigger_selection(
        {
            "first_non_minor_detection_failure": True,
            "first_parsed_response": {"answer": "yes", "justification": "Possible missed tissue."},
        },
        args,
    )

    assert selected["stage2b_trigger_non_minor_detection_failure"] is True
    assert selected["stage2b_missed_tissue_trigger"] is True
    assert selected["stage3_redetect_triggered"] is True
