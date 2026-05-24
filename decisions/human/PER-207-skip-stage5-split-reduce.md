# PER-207 Skip Stage 5 Split/Reduce Reviewer

Ticket: PER-207
Date: 2026-05-24
Author: Human

## Decision

Skip the Stage 5 split/reduce reviewer as an active pipeline step.

Do not scale the crop-level question "does this selected detection contain
multiple instances and can it be split/reduced further?" across the pilot set
for now. Keep the existing Stage 5 prompt, runner, and PDFs as experimental
evidence only.

The next active crop-level stage is true-positive/false-positive crop
classification on the higher-resolution Stage 4 rereads.

## Rationale

The split/reduce task is ill posed for this detector-oracle pipeline. A bbox can
cover more than one visible tissue-like instance, but still be practically
atomic for the intended purpose if it focuses on one usable tissue candidate and
the overlap/redundancy is acceptable. Conversely, deciding whether a box should
be split can require subjective reasoning about instance identity, overlap,
and downstream handling rather than a simple object-detection judgement.

This pushes the task toward larger-model reasoning without enough return on
investment. The pipeline needs a decent high-recall tissue detector and
crop-level filtering, not perfect atomic instance boxes.

## Evidence

The Stage 5 subset run produced reviewable low- and high-thinking PDFs over 36
higher-resolution selected-candidate overlays:

- Low-thinking PDF:
  `/data2/vj724/vlm-wsi-auto-context/runs/stage1_detector_pilot_v1/stage1_detection_review_v1/stage5_reduce_review_v1/low_thinking/visuals/stage5_reduce_review_low_thinking.pdf`
- High-thinking PDF:
  `/data2/vj724/vlm-wsi-auto-context/runs/stage1_detector_pilot_v1/stage1_detection_review_v1/stage5_reduce_review_v1/high_thinking/visuals/stage5_reduce_review_high_thinking.pdf`

Both settings returned the same aggregate counts (`yes=29`, `no=7`,
`unknown=0`), but the task framing did not justify another active reviewer loop.

## Consequences

- Stage 5 split/reduce is marked skipped in the prompt registry and logbook.
- Existing Stage 5 artifacts remain useful as evidence of why the task was not
  promoted.
- The active pipeline moves from Stage 4 higher-resolution crop export directly
  to crop true-positive/false-positive classification.
- If detector-distillation labels later need stricter instance atomicity, this
  should be revisited as a separate larger-model experiment with explicit
  acceptance criteria.
