#!/usr/bin/env python3
"""
Plot rar_plateau (acceleration-based RAR bridge) vs GR for the Milky Way.

- Reads best-fit parameters from a rar_plateau run directory (npz file saved by the dynesty GPU runner).
- Uses the comprehensive CuPy backend to compute GR (baryons-only) and rar_plateau curves on a smooth grid.
- Saves a two-panel PNG: velocities and (RAR - GR) delta.

Usage:
  python scripts/plot_rar_plateau_mw_comparison.py --run_dir runs/rar_plateau_mw_full --out images/rar_plateau_analysis/rar_plateau_mw_comparison.png

If --run_dir is omitted or does not contain results, it falls back to runs/rar_plateau_mw_dryrun.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import sys

# Add repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import cupy as cp  # type: ignore
from core.density_metric_cupy import (
    v_baryon_comprehensive_kms_cupy,
    v_total_kms_cupy,
)


def _load_best_from_npz(npz_path: Path) -> dict:
    data = np.load(str(npz_path))
    # param names
    names = None
    if "param_names" in data:
        names = [str(x) for x in data["param_names"]]
    elif "names" in data:
        names = [str(x) for x in data["names"]]
    # best params vector
    best = None
    for key in ("best_params", "best", "x_best"):
        if key in data:
            best = np.asarray(data[key])
            break
    if best is None and "samples" in data and "logl" in data:
        # Fallback: take sample with max logl
        logl = np.asarray(data["logl"])  # (Ns,)
        idx = int(np.argmax(logl))
        samples = np.asarray(data["samples"])  # (Ns, ndim)
        best = samples[idx]
    if best is None or names is None:
        raise RuntimeError("Could not locate best_params and/or param_names in npz")
    if best.ndim == 0:
        best = best.reshape(1)
    return dict(zip(names, best.tolist()))


def _default_baryon_params() -> dict:
    # Mirror runner fixed_params for consistency
    return {
        'M_disk_thin_solar': 4.0e10,
        'M_disk_thick_solar': 1.5e10,
        'M_bulge_solar': 1.2e10,
        'M_gas_solar': 3.0e10,
        'R_d_thin_kpc': 2.6,
        'R_d_thick_kpc': 4.5,
        'R_d_gas_kpc': 7.0,
        'a_bulge_kpc': 0.7,
        'h_z_thin_kpc': 0.3,
        'h_z_thick_kpc': 0.9,
        'h_z_gas_kpc': 0.15,
        'include_disk_thin': True,
        'include_disk_thick': True,
        'include_bulge': True,
        'include_gas': True,
    }


def build_params_for_model(a0_params: dict) -> dict:
    p = _default_baryon_params()
    # Add rar_plateau keys
    p.update({
        'a0_m_s2': float(a0_params.get('a0_m_s2', 1.2e-10)),
        'allow_experimental': True,
    })
    # Optional gates (kept off by default)
    for k in ('zeta_env', 'rho_c_solar_kpc3', 'gamma_exp'):
        if k in a0_params:
            p[k] = float(a0_params[k])
    return p


def make_plot(params: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Radii (1..30 kpc log grid)
    R = np.logspace(0.0, np.log10(30.0), 240).astype(np.float32)
    R_cp = cp.asarray(R, dtype=cp.float32)

    # GR (baryons-only)
    v_gr = cp.asnumpy(v_baryon_comprehensive_kms_cupy(R_cp, params))

    # rar_plateau (experimental)
    v_rar = cp.asnumpy(v_total_kms_cupy(R_cp, dict(params), xi_type='rar_plateau'))

    # Build figure
    fig = plt.figure(figsize=(12, 9))

    # Panel 1: velocities
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(R, v_gr, 'b--', lw=2.5, alpha=0.85, label='GR (baryons only)')
    ax1.plot(R, v_rar, 'r-', lw=2.8, alpha=0.95, label='RAR-Plateau')
    ax1.axvline(8.5, color='orange', ls='--', alpha=0.5, lw=2)
    ax1.text(8.5, max(v_rar.max(), v_gr.max())*0.95, '☉', ha='center', va='top', color='orange')
    ax1.set_ylabel('Circular velocity (km/s)')
    ax1.set_xlim(0, 31)
    ax1.set_ylim(0, max(v_rar.max(), v_gr.max())*1.10)
    ax1.set_title('Milky Way: RAR-Plateau vs GR')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')

    # Panel 2: delta vs GR
    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    ax2.plot(R, v_rar - v_gr, 'r-', lw=2.5, alpha=0.95, label='RAR-Plateau - GR')
    ax2.axhline(0.0, color='k', lw=0.8)
    ax2.set_xlabel('Radius (kpc)')
    ax2.set_ylabel('Δv (km/s)')
    ax2.set_xlim(0, 31)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best')

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_dir', type=str, default='runs/rar_plateau_mw_full', help='Path to rar_plateau run directory')
    ap.add_argument('--out', type=str, default='images/rar_plateau_analysis/rar_plateau_mw_comparison.png', help='Output PNG path')
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_path = Path(args.out)

    npz = run_dir / 'stellar_fit_cupy_rar_plateau_results.npz'
    if not npz.exists():
        # Fallback to dryrun
        alt = REPO_ROOT / 'runs' / 'rar_plateau_mw_dryrun' / 'stellar_fit_cupy_rar_plateau_results.npz'
        if alt.exists():
            npz = alt
        else:
            raise FileNotFoundError(f"No results npz found in {run_dir} or fallback {alt}")

    best = _load_best_from_npz(npz)
    params = build_params_for_model(best)

    make_plot(params, out_path)


if __name__ == '__main__':
    main()

