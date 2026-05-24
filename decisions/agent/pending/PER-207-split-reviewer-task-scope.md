# PER-207: Split Stage 1 Reviewer Into Single-Purpose Subchecks

Status: pending human review

Date: 2026-05-24

## Context

The Stage 1 detector oracle is being optimized for WSI thumbnail tissue-region
detection before detector distillation and downstream segmentation distillation.
Recent pilot review exposed edge cases where a combined reviewer/refiner task
was too broad:

- It could grade existing boxes as acceptable while not considering missed
  candidate tissue outside the boxes.
- It could inconsistently mark narrow tissue-like structures with internal white
  space as too loose.
- It could leave genuinely loose boxes marked as acceptable.
- It could remove boxes it classified as noise, which is useful, but this should
  happen in a task explicitly scoped to filtering rather than mixed with global
  coverage and refinement.

## Evidence

- Case `070/100 | SV40 | patient_004 | ANONPATH00527 |
  sv40_patient_004_slide_001.svs` shows early same-model self-correction:
  Gemini 3 Flash reviewed a missed tissue region and feedback-conditioned
  redetection recovered it.
- Raw `rot0` zero-box cases `30,45,50,100` show early coverage-to-redetection
  recovery: all four were flagged as missed detections and all four produced
  detections on the feedback pass.
- Case `003/100 | JONES | patient_026 | ANONPATH00599 |
  jones_patient_026_slide_001.svs` shows prompt/task-scope sensitivity: the
  per-bbox reviewer marked all boxes as `ok` / `signal`, but a point-blank
  missed-candidate probe returned `missed_potential_tissue_candidates=true` with
  high confidence.

Detailed evidence and artifact paths are recorded in
`docs/per207_detector_oracle_findings.md`.

## Proposed Decision

Use a KISS reviewer decomposition instead of a single combined
review/refine/filter prompt:

1. Initial detector:
   Find broad potential tissue-like foreground regions. Favor recall.
2. Coverage reviewer:
   Ask only whether visible potential tissue candidates are missed.
3. Geometry reviewer/refiner:
   Ask only whether bbox corners need gross refinement because tissue is cut off
   or the box is genuinely over-broad.
4. Medium-power crop QC:
   After geometry is stable, classify each crop as tissue-like material versus
   non-tissue/artifact and filter before sampler promotion.

## Consequences

- More VLM calls may be required on difficult cases, but each call has a simpler
  visual objective.
- The prompt contract becomes easier to evaluate because each call has one main
  failure mode.
- Feedback redetection should be triggered by coverage/geometry failures, not by
  a mixed reviewer schema whose negative answer may only mean "existing boxes
  look ok".
- Crop QC should not run on clearly imprecise boxes unless the goal is explicitly
  exploratory, because a loose box containing both tissue and artifact can
  confuse the tissue-vs-artifact decision.

## Review Needed

Approve, revise, or discard this proposed task-scope split before making it the
stable Stage 1 detector-oracle contract.
