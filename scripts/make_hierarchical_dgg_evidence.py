#!/usr/bin/env python3
"""
make_hierarchical_dgg_evidence.py

Compute per-galaxy DGG (RAR-plateau with hierarchical a0 prior) log evidence by
integrating the per-galaxy a0 likelihood grid against the hierarchical prior
N(ln a0; mu, sigma). Then compute ΔlogZ vs GR using χ²_GR from the SPARC
summary, save a CSV, and plot a histogram.

Inputs
- --results-root: path to results/next_steps/<run_name>
- --images-root: optional path to images/next_steps/<run_name>
- --mu --sigma: override hierarchical prior parameters (ln a0). If not provided,
  this tool will try results_root/hierarchical_a0_posterior_summary.json (p50) or
  results_root/hierarchical_a0_summary.json.
- --gr-summary: optional explicit path to SPARC summary CSV (defaults to
  results_root/sparc_a0_summary.csv)

Outputs
- results_root/hierarchical_dgg_evidence.csv with columns:
  galaxy, logZ_DGG, logL_GR, delta_logZ_DGG_vs_GR
- images_root/delta_logZ_dgg_vs_gr.png (histogram)
- results_root/hierarchical_dgg_evidence_summary.json (summary stats)

Notes
- The absolute Gaussian-likelihood normalization cancels in ΔlogZ when both
  Z_DGG and Z_GR are computed with the same per-galaxy noise model.
- This script does not require dynesty; it uses the precomputed grids from
  scripts/next_steps_from_run.py.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import math
import sys


def _load_mu_sigma(results_root: Path, mu_arg: float | None, sigma_arg: float | None) -> tuple[float, float]:
    if mu_arg is not None and sigma_arg is not None:
        return float(mu_arg), float(sigma_arg)
    # Try posterior summary first
    p = results_root / 'hierarchical_a0_posterior_summary.json'
    if p.exists():
        j = json.loads(p.read_text(encoding='utf-8'))
        mu = float(j['mu_ln_a0']['p50'])
        sigma = float(j['sigma_ln_a0']['p50'])
        return mu, sigma
    # Fallback to MLE summary
    p = results_root / 'hierarchical_a0_summary.json'
    if p.exists():
        j = json.loads(p.read_text(encoding='utf-8'))
        mu = float(j['mu'])
        sigma = float(j['sigma'])
        return mu, sigma
    raise FileNotFoundError('Could not determine (mu, sigma); pass --mu/--sigma or run hierarchical step.')


def _read_gr_summary(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not path.exists():
        return out
    with path.open('r', encoding='utf-8') as f:
        header = f.readline().strip().split(',')
        for line in f:
            if not line.strip():
                continue
            parts = [p.strip() for p in line.strip().split(',')]
            if len(parts) < 5:
                continue
            try:
                gal = parts[0]
                a0_best = float(parts[1]) if parts[1].lower() != 'nan' else float('nan')
                chi2_rar = float(parts[2])
                chi2_gr = float(parts[3])
                dof = int(parts[4]) if parts[4].isdigit() else None
                out[gal] = {'a0_best': a0_best, 'chi2_rar': chi2_rar, 'chi2_gr': chi2_gr, 'dof': dof}
            except Exception:
                continue
    return out


def _log_normal_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    # Normal in ln a0 space: pdf over ln a0 is N(mu, sigma).
    # We integrate over ln a0 directly; no Jacobian needed beyond d(ln a0) which we handle by grid spacing.
    s = max(float(sigma), 1e-9)
    z = (x - float(mu)) / s
    return (1.0/(s*np.sqrt(2.0*np.pi))) * np.exp(-0.5 * z*z)


def _logsumexp(arr: np.ndarray) -> float:
    m = float(np.nanmax(arr))
    return float(m + np.log(np.sum(np.exp(arr - m))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--results-root', required=True, help='results/next_steps/<run_name> directory')
    ap.add_argument('--images-root', default=None, help='images/next_steps/<run_name> directory (optional)')
    ap.add_argument('--mu', type=float, default=None, help='Override mu (ln a0) for hierarchical prior')
    ap.add_argument('--sigma', type=float, default=None, help='Override sigma (ln a0) for hierarchical prior')
    ap.add_argument('--gr-summary', default=None, help='Path to sparc_a0_summary.csv (defaults inside results root)')
    ap.add_argument('--grid-subdir', default='sparc_a0_grids', help='Subdir under results root containing per-galaxy grids')
    args = ap.parse_args()

    results_root = Path(args.results_root)
    images_root = Path(args.images_root) if args.images_root else (results_root.parents[1] / 'images' / 'next_steps' / results_root.name)

    mu, sigma = _load_mu_sigma(results_root, args.mu, args.sigma)

    # Load GR chi2 summary
    gr_csv = Path(args.gr_summary) if args.gr_summary else (results_root / 'sparc_a0_summary.csv')
    gr_map = _read_gr_summary(gr_csv)

    grids = sorted((results_root / args.grid_subdir).glob('*.csv'))
    rows = []
    deltas = []

    for g in grids:
        gid = g.stem.replace('_', ' ')
        # Read grid (log10_a0, chi2)
        ln_a0 = []
        chi2 = []
        with g.open('r', encoding='utf-8') as f:
            f.readline()
            for line in f:
                a, v = line.strip().split(',')
                ln_a0.append(float(a) * math.log(10.0))
                chi2.append(float(v))
        if len(ln_a0) < 3:
            continue
        ln_a0 = np.asarray(ln_a0, float)
        chi2 = np.asarray(chi2, float)
        # Per-galaxy likelihood in ln a0 grid: L = exp(-0.5*chi2)
        ll = -0.5 * chi2
        # Prior density over ln a0
        pr = _log_normal_pdf(ln_a0, mu, sigma)
        # Grid spacing dln a0 (assume near-uniform; use median spacing)
        diffs = np.diff(ln_a0)
        dln = float(np.nanmedian(diffs)) if len(diffs) else 1.0
        # log evidence: log ∫ L(a0) π(a0) d(ln a0) ≈ logsumexp(log(L) + log(π) + log(dln))
        logs = ll + np.log(np.maximum(pr, 1e-300)) + np.log(max(dln, 1e-12))
        logZ_dgg = _logsumexp(logs)
        # GR log-likelihood at baryons-only model
        gr = gr_map.get(gid)
        if gr is None:
            # Try alternative galaxy id normalization
            gid2 = gid.upper()
            gr = gr_map.get(gid2)
        if gr is None:
            # skip if no GR chi2
            continue
        logL_gr = -0.5 * float(gr['chi2_gr'])
        delta = float(logZ_dgg - logL_gr)
        rows.append({'galaxy': gid, 'logZ_DGG': logZ_dgg, 'logL_GR': logL_gr, 'delta_logZ_DGG_vs_GR': delta})
        deltas.append(delta)

    # Write CSV
    out_csv = results_root / 'hierarchical_dgg_evidence.csv'
    with out_csv.open('w', encoding='utf-8') as f:
        f.write('galaxy,logZ_DGG,logL_GR,delta_logZ_DGG_vs_GR\n')
        for r in rows:
            f.write(f"{r['galaxy']},{r['logZ_DGG']:.6f},{r['logL_GR']:.6f},{r['delta_logZ_DGG_vs_GR']:.6f}\n")

    # Summary stats
    deltas_arr = np.asarray(deltas, float)
    summary = {
        'N': int(len(deltas_arr)),
        'mu_ln_a0': float(mu),
        'sigma_ln_a0': float(sigma),
        'delta_logZ_stats': {
            'mean': float(np.nanmean(deltas_arr)) if len(deltas_arr) else float('nan'),
            'median': float(np.nanmedian(deltas_arr)) if len(deltas_arr) else float('nan'),
            'p16': float(np.nanpercentile(deltas_arr, 16)) if len(deltas_arr) else float('nan'),
            'p84': float(np.nanpercentile(deltas_arr, 84)) if len(deltas_arr) else float('nan'),
        }
    }
    (results_root / 'hierarchical_dgg_evidence_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

    # Plot histogram
    try:
        import matplotlib.pyplot as plt
        images_root.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(7.0, 4.6))
        nb = min(50, max(10, int(np.sqrt(max(len(deltas_arr), 1)))))
        plt.hist(deltas_arr, bins=nb, color='#3b82f6', alpha=0.85, edgecolor='white')
        plt.axvline(0.0, color='k', ls=':', lw=1.2)
        plt.xlabel('Δlog Z (DGG − GR)')
        plt.ylabel('Number of galaxies')
        plt.title(f'DGG vs GR evidence across sample (N={len(deltas_arr)})')
        plt.grid(alpha=0.2)
        out_png = images_root / 'delta_logZ_dgg_vs_gr.png'
        plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()
    except Exception as e:
        print(f"Warning: could not render histogram ({e})", file=sys.stderr)

    print(f"Saved: {out_csv}")


if __name__ == '__main__':
    main()

