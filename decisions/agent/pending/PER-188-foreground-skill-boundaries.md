# PER-188 Foreground Skill Boundaries

Ticket: PER-188
Date: 2026-04-30

## Decision

Keep foreground segmentation and mask review in one Codex skill for now. Model review is currently a prompt over a crop plus mask/overlay, so a separate reviewer skill would duplicate the foreground workflow rather than create a clearer boundary.

Represent foreground/background work as three routes:

1. TRIDENT baseline with TRIDENT IO.
2. This repo's VLM auto-context pipeline with Stage 3/6/7 IO.
3. A future distilled patch-classifier route that is not yet implemented.

## Rationale

The repo already has reviewer prompts and batch review entry points, and the same operator needs to choose when TRIDENT is good enough versus when to escalate to VLM. Keeping review in the same skill preserves that decision point.

The distilled route should not be presented as available until the repo has a dataset exporter, training script, and Stage6-compatible distilled inference runner. Current repo artifacts can supply VLM teacher labels, but they do not constitute a trainer.

## Consequences

- The skill includes review guidance instead of creating a separate review-only skill.
- TRIDENT outputs need an adapter into this repo's Stage3-style reviewer layout.
- Future distillation work should add explicit training and inference scripts before the skill advertises distilled execution as runnable.
