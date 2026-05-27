# PER-207 Stage 6 Containment-Aware Merge

Ticket: PER-207
Date: 2026-05-27
Author: Human

## Decision

Use containment-aware duplicate merging for Stage 6 final detector packets.

After filtering Stage 6 crop classifications to tissue-positive candidates,
merge two boxes when either:

- standard IoU is greater than `0.40`, or
- intersection area divided by the smaller box area is at least `0.80`.

Then expand final merged boxes by `10%`, matching the existing final-box margin.

## Rationale

The standard IoU-only merge missed visually duplicate boxes when one candidate
was mostly contained inside a larger neighboring candidate. In case `49/100`
(`EVG | patient_017 | ANONPATH00004`), each duplicate pair had standard IoU
below `0.40` but overlap-over-smaller-box above `0.80`.

This is a deterministic postprocessing fix and does not change the Stage 6
model, prompt, or crop classifications.

## Evidence

- Current containment-merge final all-stage detector PDF:
  `/data2/vj724/vlm-wsi-auto-context/runs/stage1_detector_pilot_v1/stage1_detection_review_v1/stage6_all100_contains_explain_hires_prompt_v1/final_packet_containment_merge_v1/visuals/stage1_to_stage6_final_detector_all100.pdf`
- Summary:
  `/data2/vj724/vlm-wsi-auto-context/runs/stage1_detector_pilot_v1/stage1_detection_review_v1/stage6_all100_contains_explain_hires_prompt_v1/final_packet_containment_merge_v1/summary/stage1_to_stage6_final_summary.json`
- Case `49/100` changed from `6` Stage 6 yes boxes to `3` final boxes.
- Across the current Gemini 3 Flash high-thinking packet, final boxes changed
  from `445` to `437` via `8` containment merge events and `0` standard-IoU
  merge events.

## Consequences

- Treat containment-aware merging as the active Stage 6 final-packet
  postprocessing rule.
- Preserve old IoU-only final packets as historical comparison artifacts.
- Future final-packet summaries should report total merge events split by
  standard-IoU and containment-overlap triggers.
