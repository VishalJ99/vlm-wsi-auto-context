# PER-207 Stage 3 Redetect Additive Retention

Status: pending human review

## Context

In the scale-500 detector run, case
`anon_c92082c4_8ecc_4a83_b6cf_cdfad0026815` showed that Stage 2a identified
missed tissue and Stage 3 feedback redetection ran. The previous integration
treated a successful Stage 3 redetection response as a replacement source for
downstream boxes.

That replacement policy can reduce recall: Stage 3 is prompted from feedback
about missed tissue, so its output is not guaranteed to restate every good Stage
1 box. A downstream replacement can therefore remove boxes that were already
acceptable before redetect.

## Decision

Treat Stage 3 feedback redetection as additive. Retain all Stage 1 source boxes.
For each Stage 3 box:

- if `IoU >= 0.75`, merge it into the existing box;
- else if `intersection / area(stage3_box) >= 0.80`, merge it into the
  existing box;
- else add it as a new box.

The active source for successful Stage 3 cases becomes
`stage1_plus_stage3_feedback_redetection`.

## Evidence

For `anon_c92082c4_8ecc_4a83_b6cf_cdfad0026815`, the original scale-500 saved
Stage 2a `raw_response` was truncated mid-sentence at 249 visible characters.
Its usage reported `completion_tokens=1196` with `reasoning_tokens=1148` under
`stage2a_max_tokens=1200`, leaving almost no visible-response budget. This made
the Stage 3 prompt operate on incomplete feedback.

The rerun comparison for this case should use a higher Stage 2a max-token budget
and the additive Stage 3 rule, then compare final boxes to the original
scale-500 output.

## Consequences

- Stage 3 can improve recall by adding missed tissue candidates.
- Stage 3 can no longer remove Stage 1 candidates by omission.
- Already-covered Stage 3 boxes are merged into the existing active box, so the
  policy can modestly expand a retained Stage 1 box without creating an obvious
  duplicate crop.
- Existing scale-500 outputs remain historical artifacts until rerun.
