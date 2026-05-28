# PER-207 Post-Stage-3 Crop-Redetect Pipeline

Date: 2026-05-28
Ticket: PER-207
Status: pending human review

## Decision

After Stage 3, postprocess detector boxes with the current merge logic
(`IoU > 0.40` or overlap-over-smaller-box `>= 0.80`) and expand the merged
boxes by `15%`. Reread those higher-resolution WSI crops and rerun the same
Stage 1 high-recall detector prompt on each crop.

Map crop-relative detections back to WSI coordinates, merge again with the same
logic, and reread classification crops with `10%` padding. Run the current
Stage 6 tissue-containment classifier. After classification, run the PER-237 v2
odd-one-out comparative artifact filter only for cases with more than one
remaining tissue-positive crop, then remove flagged crops from the final box
set.

## Rationale

Recent visual review showed that Stage 1 redetection on higher-resolution crops
after Stage 3 can produce more granular detections and can split large boxes
that contain substantial white space. The PER-237 comparative task adds a second
artifact-removal signal for residual tissue-artifact patches that the binary
classification task does not reliably remove.

## Operational Consequences

- The pipeline increases VLM calls because Stage 3 boxes become crop-level
  redetection tasks before classification.
- The final artifact review should keep the `>1` crop gate because the
  odd-one-out task has no meaningful comparison set for single-crop cases.
- Large comparison sets can require higher output-token limits; the smoke runner
  keeps `--rerun-incomplete` support so truncated or schema-invalid JSON rows can
  be repaired without rerunning parser-clean cases.

## Evidence

- Smoke output root:
  `/data2/vj724/vlm-wsi-auto-context/runs/stage1_detector_pilot_v1/stage1_detection_review_v1/stage7_post_stage3_crop_redetect_oddoneout_smoke_v1/`.
- The 10-case smoke run produced `38` post-Stage-3 boxes, `38` crop-redetect
  tasks, `116` crop-redetect detections, `90` classification tasks, `70`
  classification-positive crops, `11` odd-one-out flagged crops, and `59` final
  boxes.
- Parser status was `10/10 ok` for the final odd-one-out JSONL after targeted
  rerun-incomplete repair.
- Review PDF:
  `/data2/vj724/vlm-wsi-auto-context/runs/stage1_detector_pilot_v1/stage1_detection_review_v1/stage7_post_stage3_crop_redetect_oddoneout_smoke_v1/visuals/stage7_post_stage3_crop_redetect_oddoneout_smoke.pdf`.
