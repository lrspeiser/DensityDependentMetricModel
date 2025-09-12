#!/usr/bin/env python3
"""
Fit ER-on-SPARC model parameters to a single Rotmod/SPARC galaxy file by minimizing chi^2.
CPU-only, fast. No CuPy or dynesty required. If SciPy is available, uses scipy.optimize;
otherwise falls back to a quick random search + local refinement.

Usage examples:
  # Fit and plot NGC 3198
  python tools/fit_sparc_er.py \
    --file external_data/Rotmod_LTG/NGC3198_rotmod.dat \
    --name "NGC 3198" \
    --out images/sparc_ngc3198_fit.png

  # Fit and plot M33 (NGC 598)
  python tools/fit_sparc_er.py \
    --file external_data/Rotmod_LTG/NGC0598_rotmod.dat \
    --name "M33" \
    --out images/sparc_m33_fit.png

Parameters being fit (with defaults and bounds):
  - lambda_max ∈ [0, 6] (init 4.0)
  - R0 (kpc) ∈ [2, 40] (init 15.0)
  - sigma_lnR ∈ [0.2, 1.5] (init 0.7)
  - w_min ∈ [0.0, 0.2] (init 0.02)
  - ups_disk ∈ [0.1, 1.0] (init 0.5)
  - ups_bul ∈ [0.1, 1.0] (init 0.7)

Outputs:
  - Prints best-fit parameters and chi^2/dof
  - Saves plot with constrained vs extrapolation ER curve
  - Writes a small JSON next to the plot with parameters
"""
from __future__ import annotations
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import matplotlib.pyplot as plt

# Ensure repo root on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_loaders.sparc_loader import load_rotmod
from models.er_sparc import v_bar_from_components, xi_log_normal_R

try:
    from scipy import optimize as _opt  # type: ignore
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


def chi2_for_params(R, Vobs, eV, Vgas, Vdisk, Vbul, p):
    lambda_max, R0, sigma_lnR, w_min, ups_d, ups_b = p
    vbar = v_bar_from_components(R, Vgas, Vdisk, Vbul, ups_d, ups_b)
    xi = xi_log_normal_R(R, lambda_max, R0, sigma_lnR, w_min)
    vmod = np.sqrt(np.clip(xi, 0.0, None)) * vbar
    w = 1.0 / np.clip(eV, 1e-3, None)
    r = (Vobs - vmod) * w
    return float(np.sum(r*r))


def fit_params(R, Vobs, eV, Vgas, Vdisk, Vbul, init, bounds):
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    x0 = np.clip(np.array(init, dtype=float), lo, hi)

    def obj(x):
        x = np.clip(x, lo, hi)
        return chi2_for_params(R, Vobs, eV, Vgas, Vdisk, Vbul, x)

    if _HAS_SCIPY:
        res = _opt.minimize(obj, x0, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-4})
        x_best = np.clip(res.x, lo, hi)
        return x_best, obj(x_best)
    # Fallback: random search + local refine (few iterations)
    rng = np.random.default_rng(42)
    best_x = x0.copy()
    best_f = obj(best_x)
    for _ in range(2000):
        cand = lo + rng.random(size=lo.shape) * (hi - lo)
        f = obj(cand)
        if f < best_f:
            best_x, best_f = cand, f
    # Local small-step refine
    step = 0.05 * (hi - lo)
    for _ in range(200):
        cand = np.clip(best_x + rng.normal(scale=step), lo, hi)
        f = obj(cand)
        if f < best_f:
            best_x, best_f = cand, f
        step *= 0.98
    return best_x, best_f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--out", default=None, help="Output plot path (png)")
    # Initial values
    ap.add_argument("--lambda_max", type=float, default=4.0)
    ap.add_argument("--R0", type=float, default=15.0)
    ap.add_argument("--sigma_lnR", type=float, default=0.7)
    ap.add_argument("--w_min", type=float, default=0.02)
    ap.add_argument("--ups_disk", type=float, default=0.5)
    ap.add_argument("--ups_bul", type=float, default=0.7)
    args = ap.parse_args()

    data = load_rotmod(args.file)
    R = data["R_kpc"]; Vobs = data["Vobs_kms"]; eV = data["eVobs_kms"]
    Vgas = data["Vgas_kms"]; Vdisk = data["Vdisk_kms"]; Vbul = data["Vbul_kms"]

    init = [args.lambda_max, args.R0, args.sigma_lnR, args.w_min, args.ups_disk, args.ups_bul]
    bounds = [
        (0.0, 6.0),    # lambda_max
        (2.0, 40.0),   # R0
        (0.2, 1.5),    # sigma_lnR
        (0.0, 0.2),    # w_min
        (0.1, 1.0),    # ups_disk
        (0.1, 1.0),    # ups_bul
    ]

    x_best, chi2_best = fit_params(R, Vobs, eV, Vgas, Vdisk, Vbul, init, bounds)
    dof = max(1, R.size - len(x_best))
    print("Best-fit:")
    print({
        'lambda_max': float(x_best[0]),
        'R0_kpc': float(x_best[1]),
        'sigma_lnR': float(x_best[2]),
        'w_min': float(x_best[3]),
        'ups_disk': float(x_best[4]),
        'ups_bul': float(x_best[5]),
        'chi2': float(chi2_best),
        'chi2_dof': float(chi2_best/dof),
    })

    # Plot
    name = args.name or Path(args.file).stem.replace("_rotmod", "")
    out_path = Path(args.out) if args.out else Path("images")/f"sparc_fit_{name.lower().replace(' ','_')}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    vbar = v_bar_from_components(R, Vgas, Vdisk, Vbul, x_best[4], x_best[5])
    xi = xi_log_normal_R(R, x_best[0], x_best[1], x_best[2], x_best[3])
    vmod = np.sqrt(np.clip(xi, 0.0, None)) * vbar

    plt.figure(figsize=(10,7))
    plt.errorbar(R, Vobs, yerr=eV, fmt='o', color='k', ms=4, lw=1, alpha=0.8, label='Observed (SPARC)')
    plt.plot(R, vbar, 'b--', lw=2, label='GR (baryons)')

    # Extrapolation split on a grid
    R_grid = np.linspace(max(1e-3, R.min()), max(R.max()*1.2, R.max()+5), 400)
    vbar_g = np.interp(R_grid, R, vbar)
    xi_g = xi_log_normal_R(R_grid, x_best[0], x_best[1], x_best[2], x_best[3])
    ver_g = np.sqrt(np.clip(xi_g, 0, None)) * vbar_g
    R_data_max = float(np.max(R))
    m_in = R_grid <= R_data_max
    m_out = ~m_in
    if np.any(m_in):
        plt.plot(R_grid[m_in], ver_g[m_in], 'r-', lw=2.5, label='RAR Plateau — constrained')
    if np.any(m_out):
        plt.plot(R_grid[m_out], ver_g[m_out], color='#FF8C00', ls='--', lw=2.5, label='RAR Plateau — extrapolation')
        plt.axvspan(R_data_max, R_grid.max(), color='#FFA500', alpha=0.08)
    plt.axvline(R_data_max, color='k', ls=':', alpha=0.6, label=f"Max data R ≈ {R_data_max:.1f} kpc")

    plt.xlabel('Radius R (kpc)')
    plt.ylabel('Circular speed v (km/s)')
    plt.title(f'{name}: SPARC RAR Plateau fit (chi2/dof={chi2_best/dof:.2f})')
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=False)
    plt.xlim(0, max(R.max()*1.2, R.max()+5))
    ymax = max(np.nanmax(Vobs+eV), np.nanmax(vbar)*1.2)
    plt.ylim(0, max(300, float(ymax)+40))
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")

    # Write parameters JSON
    meta = {
        'galaxy': name,
        'file': str(Path(args.file)),
        'params': {
            'lambda_max': float(x_best[0]),
            'R0_kpc': float(x_best[1]),
            'sigma_lnR': float(x_best[2]),
            'w_min': float(x_best[3]),
            'ups_disk': float(x_best[4]),
            'ups_bul': float(x_best[5]),
        },
        'chi2': float(chi2_best),
        'chi2_dof': float(chi2_best/dof),
    }
    with open(out_path.with_suffix('.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    print(f"Saved: {out_path.with_suffix('.json')}")


if __name__ == '__main__':
    main()
