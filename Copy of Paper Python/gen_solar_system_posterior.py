#!/usr/bin/env python3
"""
Generate Solar-System screening figure and table from ER posterior samples.

- Reads one or more ER evidence NPZ/JSON pairs for a galaxy.
- Computes |xi-1|(rho,T) over a preset Solar-System density grid.
- Produces a figure with posterior median and 68% bands for two variants:
  (i) exponential S_rho; (ii) power-law S_rho with w_min cap.
- Writes a small markdown table with median [p16,p84] at key reference points.

Usage examples:
  python tools/gen_solar_system_posterior.py \
    --exp-npz images/sparc_env_fit_NGC3198_exp_hi.npz \
    --pow-npz images/sparc_env_fit_NGC3198_power_wmin005_hi.npz \
    --out-png images/solar_system_posterior_ngc3198.png \
    --out-md docs/solar/solar_posterior_ngc3198.md

Notes:
- Assumes NPZ sidecars were saved by tools/fit_sparc_er_env.py in evidence mode.
- Uses posterior samples + logwt as importance weights.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np
import json
import matplotlib.pyplot as plt

# Ensure repo root on path for `models` imports when run as a script
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from utils.plot_style import apply_paper_style
except Exception:
    def apply_paper_style():
        pass

# xi models
from models.er_env import xi_env as xi_power
try:
    from models.er_env import xi_env_exp as xi_exp
except Exception:
    xi_exp = None

# Default Solar-System density grid and reference points (Msun/kpc^3)
# Rough order-of-magnitude placeholders; adjust as needed for final doc.
SS_POINTS = [
    ("Earth lab", 1.0e21),
    ("Earth crust", 1.0e19),
    ("Earth orbit", 1.0e19),
    ("Sun photosphere", 1.0e20),
    ("Jupiter atm", 1.0e18),
    ("Saturn atm", 5.0e17),
    ("Neptune atm", 1.0e17),
]

# High-T regime assumption: T >> T0; we will set a representative T high tail value
DEFAULT_T_HIGH = 300.0


def load_npz(npz_path: Path):
    d = np.load(npz_path, allow_pickle=True)
    samples = d["samples"]
    logwt = d["logwt"]
    param_names = list(map(str, d["param_names"]))
    return samples, logwt, param_names


def weighted_quantile(x: np.ndarray, w: np.ndarray, qs=(0.16, 0.5, 0.84)):
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    s = np.argsort(x)
    x, w = x[s], w[s]
    cw = np.cumsum(w)
    if cw[-1] <= 0:
        return [np.nan for _ in qs]
    cw = cw / cw[-1]
    return [np.interp(q, cw, x) for q in qs]


def draw_xi_band(npz_path: Path, variant: str, rho_grid: np.ndarray, T_high: float):
    samples, logwt, names = load_npz(npz_path)
    wt = np.exp(logwt - np.max(logwt))
    # Param order from fit tool: [log10_rho_c, gamma_exp, lambda_max, lnT0, sigma_lnT, w_min, ups_d, ups_b]
    idx = {n: i for i, n in enumerate(names)}
    lrc_i, gam_i, lam_i, lnT0_i, sig_i, wmin_i = idx.get('log10_rho_c',0), idx.get('gamma_exp',1), idx.get('lambda_max',2), idx.get('lnT0',3), idx.get('sigma_lnT',4), idx.get('w_min',5)
    # Compute xi-1 for each sample across rho_grid
    vals = []
    for s in samples:
        log10_rho_c = float(s[lrc_i]); rho_c = 10 ** log10_rho_c
        gamma_exp = float(s[gam_i]); lam = float(s[lam_i])
        T0 = float(np.exp(s[lnT0_i])); sig = float(s[sig_i]); w_min = float(s[wmin_i])
        if variant == 'exp':
            if xi_exp is None:
                raise RuntimeError("Exponential variant not available")
            xi = xi_exp(rho_grid, np.full_like(rho_grid, T_high), lam, rho_c, gamma_exp, T0, sig, w_min)
        else:
            xi = xi_power(rho_grid, np.full_like(rho_grid, T_high), lam, rho_c, gamma_exp, T0, sig, w_min)
        vals.append(np.clip(xi - 1.0, 0.0, None))
    vals = np.array(vals)  # [Nsamp, Nrho]
    # weight per sample; replicate across rho
    W = wt[:, None] / np.sum(wt)
    # Compute weighted quantiles per rho
    qs = []
    for j in range(vals.shape[1]):
        q16, q50, q84 = weighted_quantile(vals[:, j], W[:, 0], qs=(0.16, 0.5, 0.84))
        qs.append((q16, q50, q84))
    qs = np.array(qs)
    return qs  # shape [Nrho, 3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp-npz', type=str, required=True, help='NPZ from exp-screen evidence run')
    ap.add_argument('--pow-npz', type=str, required=True, help='NPZ from power-screen (w_min cap) evidence run')
    ap.add_argument('--out-png', type=str, required=True)
    ap.add_argument('--out-md', type=str, required=True)
    ap.add_argument('--T-high', type=float, default=DEFAULT_T_HIGH, help='Representative high-T value for Solar-System regime')
    args = ap.parse_args()

    rho_grid = np.logspace(15, 22, 200)  # Msun/kpc^3

    exp_qs = draw_xi_band(Path(args.exp_npz), 'exp', rho_grid, args.T_high)
    pow_qs = draw_xi_band(Path(args.pow_npz), 'power', rho_grid, args.T_high)

    apply_paper_style()
    plt.figure(figsize=(8,6))
    # Plot median and band for both
    plt.fill_between(rho_grid, exp_qs[:,0], exp_qs[:,2], color='#ffcccc', alpha=0.5, label='exp 68%')
    plt.plot(rho_grid, exp_qs[:,1], color='red', lw=2, label='exp median')
    plt.fill_between(rho_grid, pow_qs[:,0], pow_qs[:,2], color='#cce5ff', alpha=0.5, label='power 68%')
    plt.plot(rho_grid, pow_qs[:,1], color='blue', lw=2, label='power median')
    # Cassini/LLR/inverse-square simple bands (illustrative constants)
    cassini = 2.3e-5
    invsq = 1e-12
    plt.axhspan(0, cassini, color='gray', alpha=0.15, label='Cassini |γ-1|')
    plt.axhspan(0, invsq, color='green', alpha=0.10, label='Inverse-square ~1e-12')
    plt.xscale('log'); plt.yscale('log')
    plt.xlabel('Density ρ [Msun/kpc^3] (Solar-System range)')
    plt.ylabel('|ξ−1|')
    plt.title('Solar-System screening bands from galaxy posteriors')
    plt.legend(frameon=False, loc='lower left')
    plt.tight_layout()
    Path(args.out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out_png, dpi=150)

    # Small table at reference points
    rows = []
    for label, rho in SS_POINTS:
        # nearest index
        j = int(np.clip(np.argmin(np.abs(rho_grid - rho)), 0, len(rho_grid)-1))
        e16,e50,e84 = exp_qs[j,0],exp_qs[j,1],exp_qs[j,2]
        p16,p50,p84 = pow_qs[j,0],pow_qs[j,1],pow_qs[j,2]
        rows.append((label, rho, (e50,e16,e84), (p50,p16,p84)))

    with open(args.out_md, 'w', encoding='utf-8') as f:
        lines = []
        lines.append('# Solar-System posterior screening (exp vs power)')
        lines.append('')
        lines.append('Posterior-derived |ξ−1| using the same (T0, σ_lnT, w_min) posterior as galaxy fits. T is taken in the high-T tail.')
        lines.append('')
        lines.append('| Env | ρ [Msun/kpc^3] | exp median [16,84] | power median [16,84] |')
        lines.append('|---|---:|---:|---:|')
        for label, rho, expv, powv in rows:
            lines.append(f"| {label} | {rho:.3e} | {expv[0]:.3e} [{expv[1]:.3e},{expv[2]:.3e}] | {powv[0]:.3e} [{powv[1]:.3e},{powv[2]:.3e}] |")
        f.write('\n'.join(lines))

if __name__ == '__main__':
    main()

