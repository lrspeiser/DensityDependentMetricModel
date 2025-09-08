#!/usr/bin/env python3
"""
build_a0_posterior_from_grids.py

Combine per-galaxy a0 chi^2 grids (written by next_steps_from_run.py under
results/next_steps/<run_name>/sparc_a0_grids/*.csv) into a single posterior over a0.
Saves an NPZ with samples and param_names=['a0_m_s2'] for downstream use.

This script uses only local files; no web or API keys are required.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np


def load_grids(grids_dir: Path):
    grids = []
    for p in sorted(grids_dir.glob('*.csv')):
        try:
            xs = []
            c2 = []
            with p.open('r', encoding='utf-8') as f:
                header = f.readline()
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) < 2:
                        continue
                    xs.append(float(parts[0]))  # log10_a0
                    c2.append(float(parts[1]))
            if len(xs) >= 4:
                grids.append((np.asarray(xs, float), np.asarray(c2, float)))
        except Exception:
            continue
    return grids


def combine_to_posterior(grids, n_grid: int = 400):
    # Determine common ln a0 range from overlap across galaxies
    ln_mins = []
    ln_maxs = []
    for xs, _ in grids:
        ln_mins.append(np.min(xs))
        ln_maxs.append(np.max(xs))
    if not ln_mins or not ln_maxs:
        raise RuntimeError('No grids found with sufficient points')
    ln_lo = max(ln_mins); ln_hi = min(ln_maxs)
    if ln_hi <= ln_lo:
        raise RuntimeError('No overlap in ln a0 across grids')
    ln_a0 = np.linspace(ln_lo, ln_hi, int(max(50, n_grid)))

    # Sum log-likelihoods over galaxies
    loglike = np.zeros_like(ln_a0)
    for xs, c2 in grids:
        # Interpolate chi2 onto ln_a0; mask non-finite
        c2i = np.interp(ln_a0, xs, c2, left=np.nan, right=np.nan)
        mask = np.isfinite(c2i)
        # -0.5 chi2, else -inf
        lgi = -0.5 * c2i
        lgi[~mask] = -np.inf
        # Add (log prior for ln a0 is constant for log-uniform a0 prior)
        loglike += lgi

    # Stabilize and convert to weights
    m = np.nanmax(loglike)
    w = np.exp(loglike - m)
    w = np.where(np.isfinite(w), w, 0.0)
    if np.sum(w) <= 0:
        raise RuntimeError('Combined posterior has zero support; check grids')
    w = w / np.sum(w)
    return ln_a0, w


def resample_from_posterior(ln_a0, w, n_samples: int = 5000):
    # Draw samples from the discrete posterior
    idx = np.random.default_rng().choice(np.arange(len(ln_a0)), size=int(n_samples), replace=True, p=w)
    ln_samp = ln_a0[idx]
    a0_samp = 10.0 ** ln_samp
    return a0_samp


def main():
    ap = argparse.ArgumentParser(description='Build a0 posterior NPZ from SPARC per-galaxy grids')
    ap.add_argument('--grids-dir', required=True, help='Path to sparc_a0_grids directory')
    ap.add_argument('--out-npz', required=True, help='Path to write NPZ (e.g., results/.../posterior_samples.npz)')
    ap.add_argument('--n-samples', type=int, default=5000)
    args = ap.parse_args()

    grids = load_grids(Path(args.grids_dir))
    if not grids:
        raise SystemExit('No usable grids found')
    ln_a0, w = combine_to_posterior(grids)
    a0_samp = resample_from_posterior(ln_a0, w, n_samples=int(args.n_samples))
    # Save NPZ in the format expected by downstream tools
    samples = a0_samp.reshape(-1, 1)
    param_names = np.array(['a0_m_s2'])
    np.savez(args.out_npz, samples=samples, param_names=param_names)
    print(f'Wrote posterior samples to {args.out_npz} with {samples.shape[0]} samples')


if __name__ == '__main__':
    main()

