# PER-207 Skip Stage 7 Agentic BBox Refinement

Date: 2026-05-25

## Decision

Skip the Stage 7 agentic bbox-refinement loop as an active pilot pipeline step.

## Rationale

The detector-oracle pilot needs a robust high-recall tissue detector and a
medium-power crop false-positive filter. A per-crop agentic loop that asks the
VLM to expand or contract boxes across up to three iterations is too expensive
and operationally heavy for this phase.

The acceptable finalization policy is deterministic:

- keep Stage 6 `yes` tissue-focused candidates,
- drop Stage 6 `no` artifact/noise candidates,
- merge retained boxes with standard IoU `>0.40`,
- expand final boxes by 10% to preserve margin.

## Consequences

- `prompts/stage1_detector_oracle/stage7_crop_bbox_adjustment.txt` is retained
  only as a parked experimental prompt.
- The pilot-100 final detector packet documents Stage 1 through Stage 6 IO plus
  deterministic final boxes:
  `/data2/vj724/vlm-wsi-auto-context/runs/stage1_detector_pilot_v1/stage1_detection_review_v1/stage6_all100_final_detector_v1/final_packet/visuals/stage1_to_stage6_final_detector_all100.pdf`.
- Revisit Stage 7 only if manual review shows deterministic margin expansion
  is failing in a way that materially affects downstream segmentation or
  detector-distillation labels.
