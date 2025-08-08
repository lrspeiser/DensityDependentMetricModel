---
name: Paper - Promote tidal_band to primary + README update
about: Tracking issue for Step 2
labels: paper, tracking, tidal_band
---

## Summary
Promote `tidal_band` to the primary model, ingest results, update figures, and edit README.

## Goals
- Promote `tidal_band` to primary model in the paper context.
- Ingest latest run results and summaries.
- Update figures (PNGs) and JSON summaries.
- Edit README to reflect model promotion and updated analysis flow.
- Ensure CI runs on the working branch and publishes PNG/JSON artifacts.

## Deliverables
- Updated README with `tidal_band` as primary.
- Figures and JSON summaries committed or uploaded as CI artifacts.
- CI workflow triggers on this branch and uploads PNG/JSON artifacts.
- Descriptive commits, e.g.,
  - "paper: promote tidal_band to primary model; ingest results; update figures; edit README."

## Acceptance Criteria
- CI passes on branch `paper/tidal-band-primary-README` and uploads any produced PNG/JSON artifacts.
- README updated with guidance about `tidal_band` being primary.
- Artifacts from recent runs are discoverable via CI artifact tab or committed under `results/`.
- Commit history contains descriptive messages as above.

## Links
- Tracking doc: docs/tracking/paper-tidal-band-primary-README.md
