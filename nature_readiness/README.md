# Nature Readiness (non-destructive scaffolding)

Purpose
- Provide an isolated, review-first CLI and module skeleton to run "Nature Physics readiness" checks without touching existing code or manuscript files.
- Everything here is additive. It does not overwrite any existing modules, docs, or figures.
- By default, all commands run in preview (dry-run) mode and write outputs under results/nature_readiness/.

What’s included (stubs; safe to review)
- CLI with subcommands: theory, solar, lensing, dynamics, clusters, cosmology, bayes, all
- Minimal configs under nature_readiness/configs/
- Stub modules under nature_readiness/* for each domain (placeholders only)

Non-destructive defaults
- Dry-run/preview is on by default. You can pass --no-dry-run later to actually execute routines after review.
- Outputs go to results/nature_readiness/ by default.

Data and web services
- If any step ever uses a web service or API keys, the code will include a comment pointing to nature_readiness/data/README_DATA.md which explains setup. Keys are not inlined in code or logs.

Quick start (preview only)
- Show help: python -m nature_readiness.cli --help
- Preview theory checks: python -m nature_readiness.cli theory --all
- Preview an end-to-end bundle: python -m nature_readiness.cli all --fast

Notes
- This is a scaffold for review. You can adjust scope, names, and content before enabling any non-dry runs.

