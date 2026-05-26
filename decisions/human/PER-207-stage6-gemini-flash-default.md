# PER-207 Stage 6 Gemini 3 Flash Default

Ticket: PER-207
Date: 2026-05-26
Author: Human

## Decision

Use Gemini 3 Flash high-thinking as the current Stage 6 crop-level tissue
detection default.

Some noise is acceptable for now. If Stage 6 noise becomes a practical
bottleneck for detector-label curation, foreground-segmentation selection, or
distillation, revisit the stage as a focused improvement task.

## Rationale

The current objective is a usable high-recall tissue detector and crop-level
filtering path, not perfect crop classification.

Gemini 3.1 Pro Preview looked stricter on the all-100 Stage 6 comparison, but a
follow-up no-enumeration rerun showed that much of the original Pro-vs-Flash
delta was caused by the numeric candidate-order label drawn inside the red bbox.
After removing the number from the image, Pro matched the original Flash label
on most of the original disagreement crops.

Given that result, Pro should not replace Flash as the default Stage 6 tissue
detector based on the numbered-overlay comparison.

## Evidence

- Current Gemini 3 Flash Stage 6 all-crop PDF:
  `/data2/vj724/vlm-wsi-auto-context/runs/stage1_detector_pilot_v1/stage1_detection_review_v1/stage6_all100_contains_explain_hires_prompt_v1/high_thinking/visuals/stage6_crop_tissue_artifact_high_thinking.pdf`
- Current Gemini 3 Flash final all-stage detector PDF:
  `/data2/vj724/vlm-wsi-auto-context/runs/stage1_detector_pilot_v1/stage1_detection_review_v1/stage6_all100_contains_explain_hires_prompt_v1/final_packet/visuals/stage1_to_stage6_final_detector_all100.pdf`
- Gemini 3.1 Pro no-enumeration rerun:
  `/data2/vj724/vlm-wsi-auto-context/runs/stage1_detector_pilot_v1/stage1_detection_review_v1/stage6_gemini31pro_no_enum_disagreement_rerun_v1`

## Consequences

- Treat `google/gemini-3-flash-preview` with `reasoning.effort=high` as the
  active Stage 6 model for the pilot detector-label workflow.
- Do not spend additional effort replacing Stage 6 with Gemini 3.1 Pro unless
  a separate all-crop rectangle-only Pro run is explicitly needed.
- Future Stage 6 image inputs should avoid drawing numeric enumeration inside
  the bbox; use rectangle-only overlays when regenerating crop inputs.
