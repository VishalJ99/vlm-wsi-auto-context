# PER-188 Reviewer Prompt Policy

Ticket: PER-188
Date: 2026-05-04
Author: Human

## Decision

Use `prompts/calibration_reviewer.txt` or `prompts/subjective_reviewer.txt` for foreground segmentation review.

Do not use `prompts/objective_reviewer.txt` as the foreground-review default. It is a legacy/worse version of the calibration reviewer and should only be used when explicitly comparing against old runs.

## Rationale

Reviewer outputs will feed foreground segmentation quality assessment and later patch-classifier distillation labels. Prompt choice therefore changes downstream quality signals and should be stable.

## Consequences

- Default reviewer commands should use `prompts/calibration_reviewer.txt`.
- Use `prompts/subjective_reviewer.txt` when narrative expert judgment is more important than numeric calibration.
- Prior objective-prompt outputs should be treated as comparison artifacts, not the preferred review result.
