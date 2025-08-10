#!/usr/bin/env python3
"""
CLI for generating PPC plots from existing SPARC artifacts.

Examples:
- Single-galaxy envelope (NGC 3198):
  python scripts/ppc_plots.py residual-envelope \
    --json images/sparc_env_fit_ngc3198.json \
    --out images/ppc_ngc3198_envelope.png

- Stacked residual histogram across latest ER fits:
  python scripts/ppc_plots.py stacked-hist \
    --json-glob "images/sparc_env_fit_*.json" \
    --out images/ed_sparc_residual_hist.png
"""
from __future__ import annotations
from pathlib import Path
import argparse
import glob
import sys

# Ensure repo root on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.ppc import (
    load_fit_json,
    try_load_posterior_npz,
    ppc_residual_envelopes,
    plot_residual_envelopes,
    aggregate_sparc_residuals,
    plot_residual_hist,
)
from data_loaders.sparc_loader import load_rotmod


def cmd_residual_envelope(args) -> None:
    json_path = Path(args.json)
    meta = load_fit_json(json_path)
    rotmod_path = meta.get("file_rotmod")
    if not rotmod_path:
        raise SystemExit(f"file_rotmod missing in {json_path}")
    data = load_rotmod(rotmod_path)
    R = data["R_kpc"]; Vobs = data["Vobs_kms"]; eV = data["eVobs_kms"]
    Vgas = data["Vgas_kms"]; Vdisk = data["Vdisk_kms"]; Vbul = data["Vbul_kms"]

    params = meta.get("params", {})

    # Optional posterior
    stem = json_path.with_suffix("")
    samples, weights, names = try_load_posterior_npz(stem)

    env = ppc_residual_envelopes(samples, weights, params, R, Vobs, eV, Vgas, Vdisk, Vbul,
                                 quantiles=(0.16, 0.5, 0.84), max_draws=args.max_draws,
                                 sanity=meta.get("sanity", {}))
    out_png = Path(args.out) if args.out else Path("images") / f"ppc_{meta.get('galaxy_id','galaxy').lower()}_envelope.png"
    title = f"{meta.get('galaxy_id', 'Galaxy')}: PPC residuals"
    plot_residual_envelopes(R, Vobs, eV, env, out_png, title=title, show_residuals=not args.no_residual_panel)
    print(f"Saved: {out_png}")


def cmd_stacked_hist(args) -> None:
    paths = []
    for pat in args.json_glob:
        paths.extend([Path(p) for p in glob.glob(pat)])
    if not paths:
        raise SystemExit("No JSON files matched. Provide --json-glob patterns.")
    res = aggregate_sparc_residuals(paths, standardize=not args.raw)
    out_png = Path(args.out) if args.out else Path("images") / "ed_sparc_residual_hist.png"
    title = "Stacked standardized residuals (SPARC)" if not args.raw else "Stacked raw residuals (km/s)"
    plot_residual_hist(res, out_png, title=title)
    print(f"Saved: {out_png}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_env = sub.add_parser("residual-envelope", help="Make single-galaxy PPC residual envelope plot")
    ap_env.add_argument("--json", required=True, help="Path to ER fit JSON (mode=fit) with file_rotmod and params")
    ap_env.add_argument("--out", default=None, help="Output PNG path")
    ap_env.add_argument("--max-draws", type=int, default=500)
    ap_env.add_argument("--no-residual-panel", action="store_true", help="Only plot V(R) with PPC band, no residual panel")
    ap_env.set_defaults(func=cmd_residual_envelope)

    ap_stack = sub.add_parser("stacked-hist", help="Make stacked residual histogram from many ER fit JSONs")
    ap_stack.add_argument("--json-glob", nargs="+", required=True, help="Glob(s) to ER fit JSONs (e.g., images/sparc_env_fit_*.json)")
    ap_stack.add_argument("--out", default=None, help="Output PNG path")
    ap_stack.add_argument("--raw", action="store_true", help="Use raw residuals instead of standardized by sigma")
    ap_stack.set_defaults(func=cmd_stacked_hist)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

