# PER-188 TRIDENT Contour Review Resolution

Ticket: PER-188
Date: 2026-05-05
Author: Human

## Decision

Treat paths under `trident_output_*/contours/*.jpg` and
`trident_output_*/contours_geojson/*.geojson` as foreground segmentation review
requests for the TRIDENT route.

For contour JPG inputs, infer the sibling GeoJSON path. Resolve the source WSI
from the anonymous `anon_<uuid>.svs` slide ID using the local WSI manifest,
especially `/data2/vj724/wsi-agents/all_svs_fpaths.csv`, then export
Stage3-compatible reviewer inputs before running the VLM reviewer.

## Rationale

TRIDENT contour JPGs are QC thumbnails, not enough to review segmentation
quality at useful resolution. The reviewer needs WSI pixels, a rasterized mask,
and an overlay at matched crop resolution. The anonymous slide ID plus the WSI
manifest provides the missing source WSI path.

## Consequences

- Agents should not ask for a WSI path immediately when a TRIDENT contour path
  is supplied; first try manifest-based resolution.
- `scripts/export_trident_reviewer_inputs.py` is the canonical bridge from
  TRIDENT contours to reviewer inputs.
- If the anonymous slide ID is missing or maps to multiple/nonexistent WSIs, ask
  the user for the source WSI or manifest.
