# PER-188 Reviewer QC Policy

Ticket: PER-188
Date: 2026-05-05
Author: Human

## Decision

Use `google/gemini-3-flash-preview` through OpenRouter as the default foreground
segmentation reviewer model, with OpenRouter reasoning effort `high` and
`prompts/calibration_reviewer.txt`.

Persist thresholded calibration booleans for each bbox review:

- `precision_pass = precision > precision_threshold`
- `recall_pass = recall > recall_threshold`
- `overall_pass = precision_pass and recall_pass`

The default thresholds are `precision_threshold = 0.9` and
`recall_threshold = 0.9`, exposed as command-line args so they can be tuned as
manual good/bad labels accumulate.

## Rationale

Foreground review needs sortable, machine-readable per-bbox labels for triage
and later distillation. Keeping precision and recall as separate booleans
preserves whether a failure is over-inclusion or under-inclusion, while the
ANDed `overall_pass` gives a simple default pass/fail label.

## Consequences

- Reviewer outputs should include raw precision/recall values, threshold args,
  `precision_pass`, `recall_pass`, and `overall_pass`.
- Case-level decisions should aggregate bbox-level booleans rather than hiding
  precision and recall failures behind one score.
- Future patch-classifier distillation should treat the thresholds as tunable
  dataset policy, not a hard-coded model property.
