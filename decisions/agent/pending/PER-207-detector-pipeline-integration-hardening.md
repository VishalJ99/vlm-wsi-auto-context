# PER-207 Detector Pipeline Integration Hardening

## Status

Pending human review.

## Context

The integrated detector pipeline was built from several pilot scripts. A
read-only review found it was coherent but still had incremental-integration
seams: Stage 7 could delete boxes after a non-ok parser status,
`--reuse-existing` could mix stale stage artifacts with current semantics,
fallback paths could make degraded outputs look normal, Stage 2b naming still
reflected the older non-minor gate, and the runner imported private helpers from
pilot scripts.

The user asked to address these without changing prompts or the VLM
input/output contracts of the stages.

## Decision

Keep prompts and stage VLM IO unchanged, but harden the orchestration layer.

- Apply Stage 7 comparative removals only when the odd-one-out parser status is
  `ok` or another `ok*` recovery status. Non-ok parses preserve all Stage 6
  tissue-positive boxes and mark the case degraded.
- Make `--reuse-existing` require a matching sidecar cache fingerprint that
  includes the pipeline version, model/backend, prompt hashes, relevant
  thresholds/skip flags, and stage inputs.
- Keep serialized Stage 2b `non_minor_detection_failure` fields for
  compatibility, but add code-level aliasing toward the current missed-tissue
  trigger interpretation.
- Route the runner through `scripts/detector_pipeline_utils.py` as a stable
  adapter surface over pilot-script helpers.

## Consequences

- Legacy reusable artifacts without cache sidecars are rerun instead of reused.
- A malformed Stage 7 response can no longer remove true tissue boxes by parser
  accident.
- Existing PDFs, JSONL tables, and downstream scripts that read the old Stage
  2b fields continue to work.
- Cases with fallback behavior are more visibly marked in final JSON, summary
  JSON, and final overlay titles.
