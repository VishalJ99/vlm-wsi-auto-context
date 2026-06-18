# PER-250 Stage 7 Fill Holes Without Closing

Ticket: PER-250
Date: 2026-06-03

## Decision

For the selector-seeded foreground pipeline, default Stage 7 morphology to
size-limited binary hole filling enabled and binary closing disabled. The
wrapper should emit `--stage7-skip-close` by default and should not emit
`--stage7-skip-fill-holes` unless explicitly requested. Hole filling should use
`--stage7-max-hole-size 1` by default, meaning only enclosed background
components with area up to one patch-grid cell are filled. Unrestricted
`binary_fill_holes` remains available with `--stage7-max-hole-size 0`. Closing
remains available through `--stage7-close`.

## Rationale

In the green-ink case `anon_18576685_8921_446a_a027_e1e330187f18`, the original
close-on/fill-off Stage 7 output changed tissue patches `434 -> 493` and merged
components `3 -> 1`. The fill-holes/no-close variant preserved separation
(`3 -> 2`) while keeping reviewer QC passing at the corrected 1024-px reviewer
resolution: precision `0.95`, recall `0.99`, overall pass `true`.

The VLM morphology-gate probe also selected `none` for the native bridge-risk
mask and selected `fill_holes` for a synthetic internal-hole perturbation,
while rejecting closing because it merged the two cores.

## Consequences

The current close radius is still `close_kernel=3`, but it is inactive unless
`--stage7-close` is passed. The one-cell hole cap preserves the useful repair
for isolated dropped interior patches while avoiding broad hole filling across
the space between adjacent large tissue components. Future work can add a VLM
or deterministic gate for border-gap cases, but the safe default for this
PER-250 path is no closing plus one-patch enclosed-hole filling.
