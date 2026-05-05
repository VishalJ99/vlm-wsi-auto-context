"""Helpers for thresholding calibration reviewer precision/recall outputs."""

from __future__ import annotations

import re
from typing import Any, Optional


def parse_percentage_value(value: Any) -> Optional[float]:
    """Parse reviewer percentages from values like '95%', 95, or 0.95."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
    else:
        text = str(value).strip()
        if not text:
            return None
        match = re.search(r"-?\d+(?:\.\d+)?", text)
        if not match:
            return None
        numeric = float(match.group(0))
    if numeric > 1.0:
        numeric /= 100.0
    if numeric < 0.0 or numeric > 1.0:
        return None
    return numeric


def _parse_metric_section(section: Any) -> Optional[float]:
    if isinstance(section, dict):
        for key in ("percentage", "percent", "score", "value"):
            parsed = parse_percentage_value(section.get(key))
            if parsed is not None:
                return parsed
        return None
    return parse_percentage_value(section)


def build_qc_result(
    parsed_json: Optional[dict],
    precision_threshold: float,
    recall_threshold: float,
) -> dict[str, Any]:
    """Return precision/recall pass booleans plus an ANDed overall pass."""
    thresholds = {
        "precision_threshold": float(precision_threshold),
        "recall_threshold": float(recall_threshold),
        "comparison": "gt",
    }
    qc: dict[str, Any] = {
        "precision": None,
        "recall": None,
        "precision_pass": None,
        "recall_pass": None,
        "overall_pass": None,
        "thresholds": thresholds,
        "reason": None,
    }
    if not isinstance(parsed_json, dict):
        qc["reason"] = "missing_parsed_json"
        return qc

    precision = _parse_metric_section(parsed_json.get("precision"))
    recall = _parse_metric_section(parsed_json.get("recall"))
    qc["precision"] = precision
    qc["recall"] = recall

    if precision is None or recall is None:
        qc["reason"] = "missing_precision_or_recall_percentage"
        return qc

    precision_pass = precision > thresholds["precision_threshold"]
    recall_pass = recall > thresholds["recall_threshold"]
    qc["precision_pass"] = precision_pass
    qc["recall_pass"] = recall_pass
    qc["overall_pass"] = bool(precision_pass and recall_pass)
    qc["reason"] = "thresholded"
    return qc
