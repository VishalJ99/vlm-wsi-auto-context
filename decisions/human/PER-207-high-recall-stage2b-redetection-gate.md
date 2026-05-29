# PER-207 High-Recall Stage 2b Redetection Gate

Date: 2026-05-29
Status: recorded; not yet implemented

## Decision

For the next Stage 2b to Stage 3 routing revision, favor high recall. If Stage
2a reports any suspected missed tissue, Stage 2b should trigger Stage 3
feedback redetection rather than deciding whether the miss is non-minor.

The current `scripts/run_detector_pipeline.py` implementation is unchanged for
now. It still uses the existing non-minor detection-failure router unless
`--force-stage3-redetect` is set.

## Rationale

The downstream pipeline now has Stage 6 tissue/artifact classification and Stage
7 comparative artifact removal. That makes it safer to over-include suspected
tissue at the Stage 2b gate and let downstream filters remove false positives,
rather than risk missing faint true tissue by requiring a non-minor failure
threshold before redetection.

## Evidence

In the reviewed pilot-100 PDF
`/data2/vj724/vlm-wsi-auto-context/test-pipeline-integrated-skip-stage4/visuals/final_detections_pilot100.pdf`,
SV40 patient 046 slide 001 and SV40 patient 004 slide 001 missed actual tissue.
In both cases Stage 2a noticed the miss, but Stage 2b adjudication classified
the miss as insufficiently non-minor, so Stage 3 feedback redetection did not
run.

## Consequence

The next router prompt or implementation should separate the routing question
from severity adjudication. The operational question for Stage 3 should be:
did Stage 2a identify any plausible missed tissue that should be redetected?
Severity can remain a review field, but it should not block redetection while
the pipeline is being tuned for high recall.
