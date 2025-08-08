# Tracking: Promote tidal_band to primary model and README update

This tracking document outlines the scope for Step 2 related to promoting `tidal_band` to the primary model, ingesting results, updating figures, and editing the README.

References: Step 1 goals and decisions (summarize below if needed).

## Goals
- Promote `tidal_band` to primary model in the paper context.
- Ingest latest run results and summaries.
- Update figures (PNGs) and JSON summaries in repo artifacts.
- Edit README to reflect model promotion and updated analysis flow.
- Ensure CI runs on the working branch and publishes PNG/JSON artifacts.

## Deliverables
- Updated README with `tidal_band` as primary.
- Figures and JSON summaries committed or uploaded as CI artifacts.
- CI workflow that triggers on this branch and uploads PNG/JSON artifacts.
- Descriptive commits, e.g.,
  - "paper: promote tidal_band to primary model; ingest results; update figures; edit README."

## Acceptance Criteria
- CI passes on branch `paper/tidal-band-primary-README` and uploads any produced PNG/JSON artifacts.
- README in root/docs updated with clear guidance about `tidal_band` being primary.
- Artifacts from recent runs are discoverable via CI artifact tab or committed paths under `results/`.
- Commit history contains descriptive messages as above.

## Notes
- GitHub CLI is not configured locally here; open an issue manually via GitHub UI with the above sections or provide me a PAT/enable GH CLI to automate.
