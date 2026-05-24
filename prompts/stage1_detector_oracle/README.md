# Stage 1 Detector Oracle Prompts

PER-207 prompt registry for the thumbnail detector-oracle pipeline.

Current pipeline stages:

1. `stage1_high_recall_potential_tissue_candidates.txt`:
   Gemini 3 Flash low-thinking detector prompt for high-recall potential
   tissue-like foreground candidates.
2. `stage2a_missed_or_overcoverage_review.txt`:
   Gemini 3 Flash high-thinking thumbnail reviewer prompt for missed
   potential tissue-like objects or global/near-full overcoverage.
3. `stage2b_nonminor_detection_failure_binary.txt`:
   Gemini 3 Flash text router prompt for deciding whether the Stage 2a
   free-text review reports a non-minor detection error. This is intentionally
   binary and does not mention downstream triggering.
   `stage2b_nonminor_detection_failure_json.txt` is the JSON-output variant
   with yes/no plus justification.
   `stage2b_nonminor_detection_failure_adjudicate_json.txt` is the second-pass
   JSON adjudicator over the original review plus the first-pass Stage 2b
   answer/reason.
   `stage2b_review_trigger_router.txt` is the older simple trigger-framed
   prompt.
   `stage2b_review_trigger_router_v2_conservative.txt` is the stricter
   calibration variant that ignores tiny/faint/questionable fragments when the
   reviewer says the main tissue-like regions were localized.
4. `stage2c_feedback_redetect_with_review.txt`:
   Current second-pass redetection prompt that consumes source thumbnail,
   first-pass overlay, bbox geometry, and reviewer feedback.
5. `stage3_refinement_minimal_wrapper.txt`:
   Minimal Stage 3 wrapper for rerunning the unchanged high-recall Stage 1
   detector task from the source thumbnail plus raw Stage 1 overlay and raw
   Stage 2a reviewer feedback. This is the first refinement/redetection
   experiment after Stage 2b flags a non-minor detection failure.

Legacy structured reviewer prompts are retained for reproducibility:

- `legacy_zero_box_coverage_review.txt`
- `legacy_bbox_geometry_review.txt`

Later crop-level prompts for split review, false-positive/true-positive
classification, and iterative crop-level bbox refinement should be added here
when those stages are finalized.
