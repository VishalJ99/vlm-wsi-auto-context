# PER-237 Odd-One-Out Artifact Filter Gates

Date: 2026-05-28
Ticket: PER-237
Status: pending human review

## Decision

Use the Stage 6 odd-one-out artifact filter only when a case has more than one
crop. The default runner gate is `--min-crops 2`.

Do not use the odd-one-out consensus result as the sole artifact-removal signal
when the candidate set mixes patient biopsy tissue with control tissue or other
legitimate tissue types that do not share the same low-level visual signature.
Control tissue should be routed or handled separately.

## Rationale

The prompt works by first deriving a majority low-level visual consensus, then
flagging crops that contain no region matching that consensus. With one crop,
there is no meaningful odd-one-out comparison and the model can only compare the
crop to an implicit prior.

The pilot-100 run also exposed a real edge case: control tissue can be legitimate
tissue while still breaking the shared-consensus assumption. In the EVG/SV40
patient 040 examples, the model separated pale core-like biopsy tissue from
brown control-like tissue because they did not share the same low-level
appearance. That is useful evidence that the prompt can detect non-consensus
material, but it is not the same thing as proving those crops are artifacts.

## Operational Consequences

- Future all-manifest odd-one-out runs should use the default `--min-crops 2`
  gate, or an equivalent upstream gate, before paid VLM calls.
- Historical PER-237 pilot-100 reproduction should pass `--min-crops 1` because
  that run intentionally measured all 100 pilot cases before this gate was
  recognized.
- Cases with known or suspected control tissue need a separate control-tissue
  route before odd-one-out artifact removal is treated as actionable.

## Evidence

- Pilot-100 v2 Flash run:
  `/data2/vj724/vlm-wsi-auto-context/runs/stage1_detector_pilot_v1/stage1_detection_review_v1/stage6_odd_one_out_artifact_review_v2_flash_pilot100/`.
- Dry-run gate check on the pilot manifest: `--min-crops 2` keeps `87` cases
  and skips `13` single-crop cases; `--min-crops 1` keeps all `100`.
- Visual examples reviewed on 2026-05-28: EVG patient 040, SV40 patient 040,
  and H&E patient 001 PDF pages with red flagged-crop outlines.
