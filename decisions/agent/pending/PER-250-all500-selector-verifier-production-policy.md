# PER-250 all500 selector/verifier production policy

Date: 2026-06-04
Ticket: PER-250

## Context

The 25-case foreground pilot used verifier-revised bbox selections from a 50-case artifact-dominance experiment. Scaling to all selected scale500 WSIs requires selection coverage for all 500 detector outputs before exporting bboxes into the foreground pipeline.

## Decision

For the all500 production run, use:

- Baseline selector: one `google/gemini-3.1-pro-preview` OpenRouter call per case, reasoning effort `high`, temperature `0.0`, max tokens `16000`.
- Artifact verifier: one `google/gemini-3-flash-preview` OpenRouter call per case, reasoning effort `low`, temperature `0.0`, max tokens `2500`.
- Foreground export selection policy: strict `verifier` using `verifier_selected_box_ids` from the verifier summary JSONL.

Do not rerun the direct prompt-update arm for all500. It was useful as an experiment, but the pilot evidence favored the split baseline-plus-verifier prompt path and the direct arm would add 500 extra Pro-high calls without being used for the foreground export.

## Consequences

- The all500 GPU foreground array depends on a complete verifier summary at `runs/detector_pipeline_scale500_v1/analysis/artifact_redundancy_probe_all500_prohigh_flashlow_v1/summary/results.jsonl`.
- If any verifier row fails to parse or produces no selected IDs, the strict exporter will fail for that case instead of silently falling back to all detector boxes.
- The direct comparison columns in the verifier summary are populated as baseline stubs when running `--verifier-only`; they are not a direct-prompt experiment result.
