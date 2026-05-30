# PER-207 High-Recall Stage 2b Redetection Gate

Date: 2026-05-29
Status: implemented in `scripts/run_detector_pipeline.py`

## Decision

For the Stage 2b to Stage 3 routing revision, favor high recall. If the first
Stage 2b router pass says the Stage 2a review describes a missed tissue
candidate, trigger Stage 3 feedback redetection.

`scripts/run_detector_pipeline.py` now defaults to
`--stage2b-trigger-source first`, which skips the older second-pass adjudicator.
The previous two-pass non-minor-failure gate remains available only for
reproduction with `--stage2b-trigger-source adjudicated`.

## Rationale

The second Stage 2b adjudication pass was originally introduced to reduce false
positive feedback redetections from the first router pass. That conservative
gate became too expensive for recall once Stage 2a was reliably identifying
plausible missed tissue.

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

The follow-up first-pass-trigger comparison packet is the evidence for removing
the second pass from the default route:
`/data2/vj724/vlm-wsi-auto-context/test-pipeline-stage2b-firstpass-trigger-skip-stage4/visuals/stage2b_firstpass_trigger_downstream_comparison.pdf`.
It reprocessed the 11 first-pass-positive cases with Stage 3 forced for that
subset, skipped Stage 4 crop redetection, and showed the downstream final boxes
produced by the new default trigger setting.

## Consequence

The router now separates the routing question from severity adjudication in the
pipeline default. The operational question for Stage 3 is: did the first Stage
2b pass identify any plausible missed tissue that should be redetected? Severity
can remain a review field, but it should not block redetection while the
pipeline is being tuned for high recall.
