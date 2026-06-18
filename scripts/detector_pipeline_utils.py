#!/usr/bin/env python3
"""Stable adapter surface for the integrated detector pipeline.

The detector runner was assembled from several pilot scripts. Keep those pilot
imports behind this module so the runner depends on one pipeline-facing API.
"""

from __future__ import annotations

from stage1_detection_review_pilot import (
    _chat_with_images as chat_with_images,
    _extract_json_payload as extract_json_payload,
    _font as font,
    _load_raw_orientation_bboxes as load_raw_orientation_bboxes,
    _normalised_detection_items as normalised_detection_items,
    _repo_git_commit as repo_git_commit,
    _safe_slug as safe_slug,
    _timestamp as timestamp,
)
from stage1_review_trigger_router import (
    _chat_text as chat_text,
    _parse_router_response as parse_router_response,
)
from stage4_crop_prompt_packet import (
    _normalised_yxyx_to_level0 as normalised_yxyx_to_level0,
    _pad_level0_bbox as pad_level0_bbox,
    _read_padded_crop as read_padded_crop,
)
from stage6_crop_tp_fp_review import _parse_tissue_yes_no as parse_tissue_yes_no
from stage6_odd_one_out_artifact_review import _parse_response as parse_odd_one_out_response
from stage7_post_stage3_crop_redetect_pipeline import (
    _crop_pixel_bbox_to_wsi_norm as crop_pixel_bbox_to_wsi_norm,
    _draw_boxes_overlay as draw_boxes_overlay,
    _draw_crop_detection_overlay as draw_crop_detection_overlay,
    _draw_odd_sheet as draw_odd_sheet,
    _expand_yxyx as expand_yxyx,
    _merge_yxyx_boxes as merge_yxyx_boxes,
    _norm_to_image_bbox as norm_to_image_bbox,
    _save_vlm_jpeg as save_vlm_jpeg,
    _write_csv as write_csv,
    _write_json as write_json,
    _write_jsonl as write_jsonl,
    _yxyx_overlap_metrics as yxyx_overlap_metrics,
)
