# PER-207 SV40 Stage 7 Odd-One-Out Filter Skip

Date: 2026-05-30
Status: adopted operational convention

## Decision

Process SV40 worklists separately with `--skip-odd-one-out-filter` (or the
alias `--skip-stage7-filter`) for detector-pipeline runs.

## Rationale

Stage 7 comparative thumbnail filtering assumes the remaining tissue-positive
crops are broadly homogeneous: one visibly different crop among the set is more
likely to be an artifact outlier. SV40 slides can contain control tissue that is
real tissue but visually different from the target tissue. That breaks the
homogeneous-tissue assumption and can make valid control tissue look like the
odd one out.

The SV40 exception should be handled at the run/worklist level because the skip
flag is a run-level option. Split SV40 cases into their own manifest or output
root when mixing stains, so non-SV40 runs can still use the default Stage 7
artifact filter.

## Consequence

For SV40 runs with the skip flag, final detections are the Stage 6
tissue-positive boxes after the earlier merge/crop/classification stages. The
Stage 7 odd-one-out prompt is not read or applied, and no candidate is removed
only because it differs from the other tissue-positive thumbnail crops.
