#!/usr/bin/env python3
"""
Nature Readiness CLI (preview-first, non-destructive)
- Provides subcommands to preview what will run for theory, solar-system, lensing,
  galaxy dynamics, clusters, cosmology, and Bayesian validation.
- Defaults to dry-run preview; no existing files or code paths are modified.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any, Dict

DEFAULT_CONFIG = Path("nature_readiness/configs/defaults.yaml")
DEFAULT_OUTPUT = Path("results/nature_readiness")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="nature_readiness",
        description="Preview and run Nature Physics readiness checks (isolated package)",
    )
    p.add_argument("command", choices=[
        "theory", "solar", "lensing", "dynamics", "clusters", "cosmology", "bayes", "all"
    ], help="Which group of checks to run")
    p.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to YAML config (optional)")
    p.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output directory (default: results/nature_readiness)")
    p.add_argument("--dry-run", dest="dry_run", action="store_true", default=True, help="Preview only (default)")
    p.add_argument("--no-dry-run", dest="dry_run", action="store_false", help="Execute (after review)")

    # Flags per command (kept light; real logic lives in modules this CLI will call later)
    p.add_argument("--all", action="store_true", help="Run all checks in the selected group where applicable")
    p.add_argument("--strong", action="store_true", help="Lensing: include strong lensing preview")
    p.add_argument("--weak", action="store_true", help="Lensing: include weak lensing preview")
    p.add_argument("--sparc", action="store_true", help="Dynamics: include SPARC subset preview")
    p.add_argument("--compare", nargs="*", default=None, help="Dynamics: models to compare (e.g., lcdm mond ddmm)")
    p.add_argument("--bullet", action="store_true", help="Clusters: include Bullet-like preview")
    p.add_argument("--compressed", action="store_true", help="Cosmology: compressed BAO/CMB preview")
    p.add_argument("--evidence", action="store_true", help="Bayes: evidence preview")
    p.add_argument("--kfold", type=int, default=None, help="Bayes: K-fold value (preview)")
    p.add_argument("--ppc", action="store_true", help="Bayes: posterior predictive checks preview")
    p.add_argument("--sbc", action="store_true", help="Bayes: simulation-based calibration preview")
    p.add_argument("--fast", action="store_true", help="All: fast editorial bundle preview")
    return p


def _preview_plan(args: argparse.Namespace) -> Dict[str, Any]:
    """Construct a dry-run plan from args (no side effects)."""
    plan: Dict[str, Any] = {
        "command": args.command,
        "config": str(args.config),
        "output": str(args.output),
        "dry_run": bool(args.dry_run),
        "options": {},
    }
    # Record flags generically
    for k, v in vars(args).items():
        if k not in ("command", "config", "output", "dry_run"):
            plan["options"][k] = v
    return plan


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    # Always print a preview plan. If not dry-run, this will still print before execution.
    plan = _preview_plan(args)
    print(json.dumps({"preview": plan}, indent=2))

    if args.dry_run:
        # Non-destructive: exit before doing anything.
        print("[nature_readiness] Dry-run mode: no actions were executed. Review the plan above.")
        return 0

    # Execution hooks will be wired later, after review.
    print("[nature_readiness] Execution mode was requested, but this scaffold intentionally contains stubs only.\n"
          "Please review and enable execution in the respective modules after sign-off.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

