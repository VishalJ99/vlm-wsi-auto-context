#!/usr/bin/env python3
"""Compare original Stage 6 disagreement labels with a no-enumeration rerun."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
from collections import Counter
from pathlib import Path
from typing import Any

from stage1_detection_review_pilot import _repo_git_commit, _timestamp


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DISAGREEMENTS = (
    REPO_ROOT
    / "runs/stage1_detector_pilot_v1/stage1_detection_review_v1"
    / "stage6_all100_gemini31pro_preview_hires_explain_v1/comparison"
    / "stage6_gemini3flash_vs_gemini31pro_disagreements.csv"
)
DEFAULT_NO_ENUM_SUMMARY = (
    REPO_ROOT
    / "runs/stage1_detector_pilot_v1/stage1_detection_review_v1"
    / "stage6_gemini31pro_no_enum_disagreement_rerun_v1/high_thinking"
    / "summary/stage6_crop_tissue_artifact_high_thinking.csv"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "runs/stage1_detector_pilot_v1/stage1_detection_review_v1"
    / "stage6_gemini31pro_no_enum_disagreement_rerun_v1/comparison"
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _key(row: dict[str, str]) -> tuple[int, int, str]:
    return int(row["case_index"]), int(row["candidate_order"]), row["candidate_id"]


def compare(args: argparse.Namespace) -> dict[str, Any]:
    source_rows = _read_csv(args.source_disagreements)
    no_enum_rows = _read_csv(args.no_enum_summary)
    no_enum_by_key = {_key(row): row for row in no_enum_rows}
    if len(no_enum_by_key) != len(no_enum_rows):
        raise SystemExit("Duplicate join keys in no-enumeration summary")
    missing = sorted(set(_key(row) for row in source_rows) - set(no_enum_by_key))
    if missing:
        raise SystemExit(f"No-enumeration summary is missing source disagreement rows: {missing[:5]}")

    rows: list[dict[str, Any]] = []
    for source in source_rows:
        no_enum = no_enum_by_key[_key(source)]
        rows.append(
            {
                "case_index": source["case_index"],
                "case_display": source["case_display"],
                "candidate_order": source["candidate_order"],
                "candidate_id": source["candidate_id"],
                "flash_original_decision": source["flash_decision"],
                "pro_original_with_enum_decision": source["pro_decision"],
                "pro_no_enum_decision": no_enum["tissue_focus_decision"],
                "no_enum_matches_flash": str(no_enum["tissue_focus_decision"] == source["flash_decision"]).lower(),
                "no_enum_matches_original_pro": str(no_enum["tissue_focus_decision"] == source["pro_decision"]).lower(),
                "source_selected_overlay_path": source["selected_overlay_path"],
                "no_enum_selected_overlay_path": no_enum["selected_overlay_path"],
                "crop_path": no_enum["crop_path"],
                "no_enum_raw_response": no_enum["raw_response"],
            }
        )

    output_csv = args.output_root / "stage6_original_vs_no_enum_pro_disagreement_rerun.csv"
    _write_csv(output_csv, rows, list(rows[0]))
    summary = {
        "created_at": _timestamp(),
        "ticket": "PER-207",
        "git_commit": _repo_git_commit(),
        "source_disagreement_csv": str(args.source_disagreements.resolve()),
        "no_enum_summary_csv": str(args.no_enum_summary.resolve()),
        "output_root": str(args.output_root.resolve()),
        "output_csv": str(output_csv.resolve()),
        "rows": len(rows),
        "no_enum_decision_counts": dict(Counter(row["pro_no_enum_decision"] for row in rows)),
        "no_enum_matches_flash_count": sum(row["no_enum_matches_flash"] == "true" for row in rows),
        "no_enum_matches_original_pro_count": sum(row["no_enum_matches_original_pro"] == "true" for row in rows),
        "transition_counts": dict(
            Counter(
                (
                    f"{row['flash_original_decision']}/"
                    f"{row['pro_original_with_enum_decision']}->"
                    f"{row['pro_no_enum_decision']}"
                )
                for row in rows
            )
        ),
        "stayed_original_pro_no_cases": [
            [row["case_index"], row["candidate_order"], row["candidate_id"]]
            for row in rows
            if row["no_enum_matches_original_pro"] == "true"
        ],
    }
    summary_path = args.output_root / "stage6_original_vs_no_enum_pro_disagreement_rerun_summary.json"
    _write_json(summary_path, summary)
    _write_reproduction(args, output_csv, summary_path, summary)
    return summary


def _write_reproduction(args: argparse.Namespace, output_csv: Path, summary_path: Path, summary: dict[str, Any]) -> None:
    command = " ".join(
        shlex.quote(part)
        for part in [
            "python",
            "scripts/stage6_compare_no_enum_rerun.py",
            "--source-disagreements",
            str(args.source_disagreements.resolve()),
            "--no-enum-summary",
            str(args.no_enum_summary.resolve()),
            "--output-root",
            str(args.output_root.resolve()),
        ]
    )
    text = f"""\
Stage 6 original vs no-enumeration Pro rerun comparison
======================================================

Created: {summary['created_at']}
Ticket: PER-207
Git commit: {summary['git_commit']}

Objective:
Join the original Gemini 3 Flash vs Gemini 3.1 Pro disagreement set with the
Gemini 3.1 Pro no-enumeration rerun, preserving per-crop label transitions.

Inputs:
- Source disagreement CSV: {args.source_disagreements.resolve()}
- No-enumeration Pro summary CSV: {args.no_enum_summary.resolve()}

Command:
{command}

Outputs:
- Comparison CSV: {output_csv.resolve()}
- Summary JSON: {summary_path.resolve()}

Counts:
- Rows: {summary['rows']}
- No-enumeration decision counts: {summary['no_enum_decision_counts']}
- No-enumeration matches original Flash: {summary['no_enum_matches_flash_count']}
- No-enumeration matches original Pro: {summary['no_enum_matches_original_pro_count']}
- Transition counts: {summary['transition_counts']}
"""
    (args.output_root / "reproduction.txt").write_text(text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-disagreements", type=Path, default=DEFAULT_SOURCE_DISAGREEMENTS)
    parser.add_argument("--no-enum-summary", type=Path, default=DEFAULT_NO_ENUM_SUMMARY)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = compare(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
