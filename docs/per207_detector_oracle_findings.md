# PER-207 Detector Oracle Findings

Date: 2026-05-24

Ticket: PER-207

This note records the current evidence from the Stage 1 WSI detector-oracle
pilot. These are early signals, not final performance claims. Manual review and
larger ablations are still required before using the outputs as training labels
or replacing four-orientation TTA.

## Finding 1: Early Signs Of VLM Self-Correction

Gemini 3 Flash shows early evidence that it can recover missed tissue-like
regions when a second detector call is conditioned on reviewer feedback.

### Case 070: Same-Model Review And Redetection

Case:
`070/100 | SV40 | patient_004 | ANONPATH00527 | sv40_patient_004_slide_001.svs`

Evidence:
- Commit `1124a84` added the first feedback-redetection experiment.
- Commit `0cf6e4b` added the faithful per-model detect-review-redetect loop
  comparison.
- Commit `85d5f48` documented the Gemini 3 self-correction setup caveat.
- Report PDF:
  `/data2/vj724/vlm-wsi-auto-context/runs/stage1_detector_pilot_v1/stage1_detection_review_v1/feedback_redetect/070_sv40_patient_004_slide_001/feedback_redetect_report.pdf`
- Model-comparison PDF:
  `/data2/vj724/vlm-wsi-auto-context/runs/stage1_detector_pilot_v1/stage1_detection_review_v1/feedback_redetect/070_sv40_patient_004_slide_001/comparison/case070_feedback_redetect_model_comparison.pdf`
- Full-loop raw-output comparison PDF:
  `/data2/vj724/vlm-wsi-auto-context/runs/stage1_detector_pilot_v1/stage1_detection_review_v1/model_loops/070_sv40_patient_004_slide_001/comparison/case070_full_model_loop_prompt_v1_raw_comparison.pdf`

Observed signal:
- The initial detector missed faint central tissue.
- The reviewer flagged the missed tissue.
- The feedback-conditioned redetection recovered the missed central tissue and
  retained a right-hand tissue detection.
- The later faithful full-loop comparison also showed Gemini 3 Flash detecting,
  reviewing, and redetecting from its own outputs, although the recovered output
  still contained a small duplicate/fragment bbox and needs manual acceptance.

Limits:
- The original proof used the existing Stage 1 output and blind reviewer
  feedback from the pilot, while the later model loop used a fresh single direct
  first-pass call. These are related but not identical experimental setups.
- This is a case-level proof of concept, not a calibrated self-correction rate.

### Raw Rot0 Zero-Box Cases: Coverage Review To Redetection

Cases:
- `030/100 | SV40 | patient_046`
- `045/100 | SV40 | patient_030`
- `050/100 | SV40 | patient_017`
- `100/100 | SV40 | patient_001`

Evidence:
- Commit `f2fea17` added raw-orientation coverage review and feedback
  redetection scripts.
- Coverage-review output root:
  `/data2/vj724/vlm-wsi-auto-context/runs/stage1_detector_pilot_v1/stage1_detection_review_v1/raw_rot0_coverage_review_missing_cases_v1`
- Coverage-review reproduction:
  `/data2/vj724/vlm-wsi-auto-context/runs/stage1_detector_pilot_v1/stage1_detection_review_v1/raw_rot0_coverage_review_missing_cases_v1/reproduction.txt`
- Feedback-redetection output root:
  `/data2/vj724/vlm-wsi-auto-context/runs/stage1_detector_pilot_v1/stage1_detection_review_v1/raw_rot0_feedback_redetect_missing_cases_v1`
- Feedback-redetection PDF:
  `/data2/vj724/vlm-wsi-auto-context/runs/stage1_detector_pilot_v1/stage1_detection_review_v1/raw_rot0_feedback_redetect_missing_cases_v1/visuals/feedback_redetect_report.pdf`
- Feedback-redetection reproduction:
  `/data2/vj724/vlm-wsi-auto-context/runs/stage1_detector_pilot_v1/stage1_detection_review_v1/raw_rot0_feedback_redetect_missing_cases_v1/reproduction.txt`

Observed signal:
- Coverage reviewer flagged all four zero-box raw `rot0` cases as missed
  detections with high confidence.
- Feedback redetection returned boxes for all four cases: `4/4` with detections,
  `8` total detections, `0` errors.

Limits:
- These are deliberately selected zero-box failures, not a random sample.
- The recovered boxes still need visual acceptance before being treated as
  labels or as evidence to remove TTA.

### Pilot-100 Feedback Packet

Evidence:
- Commit `d79cf83` added the pilot-wide raw overlay feedback packet runner.
- PDF:
  `/data2/vj724/vlm-wsi-auto-context/runs/stage1_detector_pilot_v1/stage1_detection_review_v1/raw_overlay_feedback_packet_rot0_v1/visuals/raw_overlay_feedback_packet.pdf`
- Reproduction:
  `/data2/vj724/vlm-wsi-auto-context/runs/stage1_detector_pilot_v1/stage1_detection_review_v1/raw_overlay_feedback_packet_rot0_v1/reproduction.txt`

Observed signal:
- `100` pilot cases processed.
- `96` raw `rot0` cases had parseable bboxes; `4` were zero-box cases.
- `592` bbox review rows.
- First-pass reviewer errors: `0`.
- Second-pass calls: `26/100`.
- Second-pass errors: `0`.
- Second-pass detections returned: `209`.

Interpretation:
The pilot packet is useful as a manual-review artifact for judging whether a
light reviewer/refiner loop is plausible. It is not yet a performance result.

## Finding 2: Prompt Scope Matters

Gemini 3 Flash can see some failures when asked directly, but the combined
per-bbox reviewer task can miss them if the relevant question is outside the
task scope. The unit of task scope matters.

### Jones Patient 026: Per-Bbox Reviewer Missed Coverage Failure

Case:
`003/100 | JONES | patient_026 | ANONPATH00599 | jones_patient_026_slide_001.svs`

Earlier multi-output reviewer behavior:
- In the pilot-100 raw overlay packet, the reviewer graded the six raw `rot0`
  boxes as `ok` / `signal`.
- It did not trigger a second pass.
- This was expected from the prompt design: that pass explicitly reviewed
  per-bbox tightness and signal/noise, and did not ask for missed tissue outside
  the boxes.

Point-blank probe evidence:
- Commit `30a1ec2` added `scripts/stage1_edge_case_missed_candidate_probe.py`.
- PDF:
  `/data2/vj724/vlm-wsi-auto-context/runs/stage1_detector_pilot_v1/stage1_detection_review_v1/edge_case_probes/missed_candidate_point_blank_v1/visuals/missed_candidate_point_blank.pdf`
- Reproduction:
  `/data2/vj724/vlm-wsi-auto-context/runs/stage1_detector_pilot_v1/stage1_detection_review_v1/edge_case_probes/missed_candidate_point_blank_v1/reproduction.txt`

Observed signal:
- The point-blank prompt asked only whether the overlay missed visible potential
  tissue candidates.
- Gemini 3 Flash answered `missed_potential_tissue_candidates=true` with
  `confidence=high`.
- It identified fragmented tissue-like clusters to the right of the main
  vertical strips in all three specimen groups.

Interpretation:
This is not a visual floor failure for Gemini 3 Flash on this case. The model
can detect the missed-candidate failure when the task is scoped directly to
coverage. The failure is in the wrapper: the prior task asked for per-box review
and not for global coverage.

## Current Methodological Takeaway

Use simple, separable reviewer calls rather than asking a cheap Flash model to
perform multiple coupled judgments in one response.

Recommended KISS decomposition:

1. Detection call:
   Detect broad potential regions of tissue-like foreground on the thumbnail.
   Favor recall over perfect box tightness.
2. Coverage review:
   Ask only whether visible potential tissue candidates were missed.
3. Geometry review:
   If coverage is acceptable, ask only whether any bbox corners need gross
   refinement because tissue is cut off or the box is genuinely over-broad.
4. Crop QC:
   After geometry is stable, read medium-power crops and ask whether each crop
   is genuine tissue-like material or non-tissue/artifact.

This keeps each VLM call closer to a single visual task and should reduce the
failure mode where a model answers the requested per-bbox question while
silently omitting a separate coverage question.

## Open Questions

- What is the measured coverage-review sensitivity/specificity over the full
  pilot 100 after manual adjudication?
- Does single-orientation detection plus coverage/geometry review match or beat
  four-orientation TTA after manual acceptance?
- How often does feedback redetection improve boxes versus over-delete or
  over-split acceptable detections?
- Which failures require a stronger model rather than prompt decomposition?
