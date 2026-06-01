# PER-248 Coverage-Aware Detector Recall

Status: pending human review

## Context

The scale-500 YOLO distillation uses VLM detector outputs as weak labels. Some
weak-label boxes split one contiguous tissue component into multiple smaller
boxes, while the trained YOLO model sometimes predicts one broader box that
contains those smaller boxes.

Under strict object-detection matching (`IoU >= 0.50`), a contained weak-label
box can be counted as missed when the YOLO prediction is much larger than the GT
box. This is useful for measuring localization tightness, but it overstates
tissue recall failures for the downstream review-loop use case.

## Decision

For PER-248 detector distillation readouts, report strict IoU metrics as
localization/tightness metrics and additionally report a recall-first coverage
metric:

`IoU >= 0.50 OR intersection / GT_area >= 0.90`

Use the coverage-aware metric when deciding whether the distilled detector is
safe enough as a high-recall tissue-candidate detector.

## Evidence

On the scale-500 YOLO11n stain-jitter test split:

- Strict IoU>=0.50 recall at `conf=0.05`: `201/213 = 94.4%`
- Coverage-aware recall at `conf=0.05`: `211/213 = 99.1%`
- Coverage-aware recall at `conf=0.01`: `213/213 = 100%`

The corrected coverage audit is:

`/data2/vj724/vlm-wsi-auto-context/runs/detector_distillation/yolo_scale500_per248_v1/gt_coverage_audit_yolo11n_stainjitter_conf005_v1/gt_coverage_audit.pdf`

The qualitative subagent audit found that most strict-IoU misses were weak-label
granularity or loose-box localization artifacts rather than true tissue misses.

## Consequences

- Use strict IoU to track box tightness and downstream crop precision.
- Use coverage-aware recall to track whether real tissue is available to the
  review loop.
- Future detector reports should avoid labeling strict-IoU unmatched boxes as
  "missed tissue" unless they also fail the coverage-aware criterion.
