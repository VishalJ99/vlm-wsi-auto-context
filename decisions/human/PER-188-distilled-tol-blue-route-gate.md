# PER-188 Distilled Tol Blue Route Gate

Ticket: PER-188
Date: 2026-05-05

## Decision

Once a runnable distilled foreground/background patch classifier exists, use it
as the first foreground segmentation mode and gate it with reviewer QC on Tol
Blue, the current hard OOD stain type for this project.

The operational router has two primary modes:

1. Run the distilled patch classifier, export or reuse reviewer-compatible mask
   inputs, and require reviewer pass evidence on the reviewed tissue-core bboxes.
2. If the distilled route fails reviewer QC, is uncertain, or cannot be reviewed
   at adequate resolution, fall back to the heavier repo VLM foreground route
   with `--stage6-icl-k 1` and `--stage2-force-read-l0`.

If the distilled route remains high quality on Tol Blue, proceed to downstream
linear-probe testing on Tol Blue WSIs and deprioritize foreground-segmentation
fix work unless new reviewer evidence shows a real quality problem.

## Rationale

Tol Blue is the noisiest and most out-of-distribution stain currently being used
for foreground segmentation validation. If a lightweight distilled classifier can
produce reviewer-passing masks there, the foreground mask quality problem is
probably not the limiting issue for UNI/CONCH embedding and linear-probe work.

The repo VLM route is still the quality fallback because ICL k=1 plus
level-0 bbox reads is slower and more expensive, but gives the method the best
available foreground/background reasoning path when the fast route fails.

## Consequences

- The foreground skill should describe a distilled-first, reviewer-gated route
  once the required distilled scripts exist.
- The skill must not present distilled execution as runnable until the repo has
  a dataset exporter, trainer, and Stage6-compatible inference runner.
- Reviewer QC should happen before treating downstream linear-probe performance
  as evidence that foreground segmentation is good enough.
- TRIDENT remains useful as a cheap external baseline and review target, but it
  is not the primary router for this distilled-vs-VLM policy.
