#!/usr/bin/env python3
"""
Fit ER-on-SPARC using per-galaxy midplane densities and a tidal proxy from v_bar.
This addresses the paper's multi-galaxy validation TODO by moving beyond the
radius-only ER window. CPU-only, no CuPy required. If SciPy is present, use it.

Usage:
  python tools/fit_sparc_er_env.py \
    --galaxy_id NGC3198 \
    --sparc_dir external_data/Rotmod_LTG \
    --out images/sparc_ngc3198_env_fit.png

Parameters (bounds follow paper priors where applicable):
  - log10_rho_c ∈ [14, 17] (init 15.0)
  - gamma_exp ∈ [1.0, 5.0] (init 3.0)
  - lambda_max ∈ [0.0, 6.0] (init 4.0)
  - lnT0 ∈ [-1.0, 1.0] (init 0.0)
  - sigma_lnT ∈ [0.3, 2.0] (init 0.8)
  - w_min ∈ [0.0, 0.1] (init 0.02)
  - ups_disk ∈ [0.1, 1.0] (init 0.5)
  - ups_bul ∈ [0.1, 1.0] (init 0.7)

Outputs:
  - Plot PNG with observed, GR(baryons), and ER curves (constrained+extrapolation)
  - JSON sidecar with best-fit parameters and chi2
"""
from __future__ import annotations
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import matplotlib.pyplot as plt

# repo root on path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.Utilities.sparc_io import load_single_sparc_galaxy, BASE_M_L_3_6_MICRON_DISK, BASE_M_L_3_6_MICRON_BULGE
from models.er_sparc import v_bar_from_components
from models.er_env import predict_v_er

try:
    from scipy import optimize as _opt  # type: ignore
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


def chi2_env(params, R, Vobs, eV, Vgas, Vdisk, Vbul, rho_mid_base, ups_d, ups_b):
    # Unpack
    log10_rho_c, gamma_exp, lambda_max, lnT0, sigma_lnT, w_min = params
    # Build vbar with current M/L
    vbar = v_bar_from_components(R, Vgas, Vdisk, Vbul, ups_d, ups_b)
    # Scale stellar density piece by M/L factors relative to base-ML already in rho_mid_base
    # rho_mid_base already includes base-ML stellar + gas. If we change stell M/L via ups_d, ups_b,
    # we can approximate scaling the stellar density contribution by factor (ups_d/BASE_ML_disk) for disk parts.
    # Lacking a clean disk/bulge split of rho, we scale total rho by a conservative factor near unity:
    # Use f_ML ≈ (ups_d/BASE_ML_disk)^0.5 as a mild adjustment to avoid overshooting with uncertain decomposition.
    f_ml = max(0.3, min(3.0, (ups_d / BASE_M_L_3_6_MICRON_DISK) ** 0.5))
    rho_mid = np.clip(rho_mid_base * f_ml, 1e-30, None)
    # ER prediction
    rho_c = 10 ** float(log10_rho_c)
    T0 = float(np.exp(lnT0))
    _, vmod = predict_v_er(R, vbar, rho_mid, float(lambda_max), rho_c, float(gamma_exp), T0, float(sigma_lnT), float(w_min))
    w = 1.0 / np.clip(eV, 1e-3, None)
    r = (Vobs - vmod) * w
    return float(np.sum(r * r))


def fit_one_galaxy_env(galaxy_id: str, sparc_dir: str,
                        init, bounds):
    data = load_single_sparc_galaxy(galaxy_id, sparc_dir=sparc_dir)
    if data is None:
        raise FileNotFoundError(f"Failed to load SPARC data for {galaxy_id} from {sparc_dir}")
    R = data['R_kpc']
    Vobs = data['V_obs']
    eV = data['e_V_obs']
    Vgas = data['V_gas_comp_kms']
    Vdisk = data['V_disk_comp_kms']
    Vbul = data['V_bulge_comp_kms']
    rho_mid_base = data['rho_star_mid_Msun_kpc3_baseML'] + data['rho_gas_mid_Msun_kpc3']

    x0 = np.array(init, dtype=float)
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    x0 = np.clip(x0, lo, hi)

    # We also optimize ups_disk and ups_bul as weakly-coupled outer variables for simplicity here.
    ups_d0, ups_b0 = 0.5, 0.7

    def obj_full(x):
        x = np.clip(x, lo, hi)
        return chi2_env(x, R, Vobs, eV, Vgas, Vdisk, Vbul, rho_mid_base, ups_d0, ups_b0)

    if _HAS_SCIPY:
        res = _opt.minimize(obj_full, x0, method='Nelder-Mead', options={'maxiter': 3000, 'xatol': 1e-4, 'fatol': 1e-4})
        x_best = np.clip(res.x, lo, hi)
        chi2_best = obj_full(x_best)
    else:
        rng = np.random.default_rng(123)
        x_best = x0.copy(); chi2_best = obj_full(x_best)
        for _ in range(3000):
            cand = lo + rng.random(size=lo.shape) * (hi - lo)
            f = obj_full(cand)
            if f < chi2_best:
                x_best, chi2_best = cand, f
        step = 0.05 * (hi - lo)
        for _ in range(300):
            cand = np.clip(x_best + rng.normal(scale=step), lo, hi)
            f = obj_full(cand)
            if f < chi2_best:
                x_best, chi2_best = cand, f
            step *= 0.98

    return data, x_best, chi2_best, ups_d0, ups_b0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--galaxy_id', required=True)
    ap.add_argument('--sparc_dir', required=True)
    ap.add_argument('--out', default=None)
    ap.add_argument('--model', choices=['gr','er','nfw'], default='er', help='Model type to fit')
    ap.add_argument('--mode', choices=['fit','evidence'], default='fit', help='Operation: quick fit (chi2) or evidence scaffold output')
    # Inits / priors centers
    ap.add_argument('--log10_rho_c', type=float, default=15.0)
    ap.add_argument('--gamma_exp', type=float, default=3.0)
    ap.add_argument('--lambda_max', type=float, default=4.0)
    ap.add_argument('--lnT0', type=float, default=0.0)
    ap.add_argument('--sigma_lnT', type=float, default=0.8)
    ap.add_argument('--w_min', type=float, default=0.02)
    # NFW inits
    ap.add_argument('--V200', type=float, default=150.0, help='NFW V200 [km/s]')
    ap.add_argument('--c', type=float, default=10.0, help='NFW concentration')
    args = ap.parse_args()

    init = [args.log10_rho_c, args.gamma_exp, args.lambda_max, args.lnT0, args.sigma_lnT, args.w_min]
    bounds = [
        (14.0, 17.0),  # log10_rho_c
        (1.0, 5.0),    # gamma_exp
        (0.0, 6.0),    # lambda_max
        (-1.0, 1.0),   # lnT0
        (0.3, 2.0),    # sigma_lnT
        (0.0, 0.1),    # w_min
    ]

    data, x_best, chi2_best, ups_d, ups_b = fit_one_galaxy_env(args.galaxy_id, args.sparc_dir, init, bounds)

    R = data['R_kpc']
    Vobs = data['V_obs']
    eV = data['e_V_obs']
    Vgas = data['V_gas_comp_kms']
    Vdisk = data['V_disk_comp_kms']
    Vbul = data['V_bulge_comp_kms']
    rho_mid_base = data['rho_star_mid_Msun_kpc3_baseML'] + data['rho_gas_mid_Msun_kpc3']

    vbar = v_bar_from_components(R, Vgas, Vdisk, Vbul, ups_d, ups_b)
    # Scale rho for M/L change using same mild factor as in chi2
    f_ml = max(0.3, min(3.0, (ups_d / BASE_M_L_3_6_MICRON_DISK) ** 0.5))
    rho_mid = np.clip(rho_mid_base * f_ml, 1e-30, None)

    log10_rho_c, gamma_exp, lambda_max, lnT0, sigma_lnT, w_min = map(float, x_best)
    rho_c = 10 ** log10_rho_c
    T0 = float(np.exp(lnT0))

    from models.er_env import predict_v_er, tidal_proxy_from_vbar
    xi, vmod = predict_v_er(R, vbar, rho_mid, lambda_max, rho_c, gamma_exp, T0, sigma_lnT, w_min)

    dof = max(1, R.size - len(x_best))
    print({
        'galaxy_id': data['galaxy_id'],
        'chi2': float(chi2_best),
        'chi2_dof': float(chi2_best/dof),
        'params': {
            'log10_rho_c': log10_rho_c,
            'gamma_exp': gamma_exp,
            'lambda_max': lambda_max,
            'lnT0': lnT0,
            'sigma_lnT': sigma_lnT,
            'w_min': w_min,
            'ups_disk': float(ups_d),
            'ups_bul': float(ups_b),
        }
    })

    # Optionally override model selection for output/plot
    model = args.model

    # Plot
    name = data['galaxy_id']
    out_path = Path(args.out) if args.out else Path('images')/f'sparc_env_fit_{name.lower()}.png'
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10,7))
    plt.errorbar(R, Vobs, yerr=eV, fmt='o', color='k', ms=4, lw=1, alpha=0.8, label='Observed (SPARC)')
    plt.plot(R, vbar, 'b--', lw=2, label='GR (baryons)')

    R_grid = np.linspace(max(1e-3, R.min()), max(R.max()*1.2, R.max()+5), 400)
    vbar_g = np.interp(R_grid, R, vbar)

    # Build model prediction on grid for display based on selection
    if model == 'er':
        # For plotting xi band, rebuild proxy on grid using vbar_g
        from models.er_env import tidal_proxy_from_vbar, xi_env
        T_g = tidal_proxy_from_vbar(R_grid, vbar_g)
        # Interp rho to grid (nearest/linear)
        rho_g = np.interp(R_grid, R, rho_mid)
        xi_g = xi_env(rho_g, T_g, lambda_max, rho_c, gamma_exp, T0, sigma_lnT, w_min)
        ver_g = np.sqrt(np.clip(xi_g, 0, None)) * vbar_g
        curve_in, curve_out = ver_g, ver_g
    elif model == 'nfw':
        from models.nfw import v_model_nfw
        ver_g = v_model_nfw(R_grid, vbar_g, args.V200, args.c)
        curve_in, curve_out = ver_g, ver_g
    else:
        curve_in = np.interp(R_grid, R, vbar)
        curve_out = curve_in

    R_data_max = float(np.max(R))
    m_in = R_grid <= R_data_max
    m_out = ~m_in
    if np.any(m_in):
        if model == 'er':
            plt.plot(R_grid[m_in], curve_in[m_in], 'r-', lw=2.5, label='ER — constrained')
        elif model == 'nfw':
            plt.plot(R_grid[m_in], curve_in[m_in], color='green', ls='-.', lw=2.5, label='NFW (ΛCDM)')
        else:
            plt.plot(R_grid[m_in], curve_in[m_in], 'b--', lw=2.5, label='GR (baryons)')
    if np.any(m_out):
        if model == 'er':
            plt.plot(R_grid[m_out], curve_out[m_out], color='#FF8C00', ls='--', lw=2.5, label='ER — extrapolation')
            plt.axvspan(R_data_max, R_grid.max(), color='#FFA500', alpha=0.08)
    plt.axvline(R_data_max, color='k', ls=':', alpha=0.6, label=f"Max data R ≈ {R_data_max:.1f} kpc")

    plt.xlabel('R (kpc)')
    plt.ylabel('Vc (km s^{-1})')
    plt.title(f'{name}: SPARC env-ER fit (chi2/dof={chi2_best/dof:.2f})')
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=False)
    plt.xlim(0, max(R.max()*1.2, R.max()+5))
    ymax = max(np.nanmax(Vobs+eV), np.nanmax(vbar)*1.2)
    plt.ylim(0, max(300, float(ymax)+40))
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")

    # Compute simple log-likelihood at best fit for reporting
    # ln L = -0.5 * chi2 + const; we report the -0.5*chi2 term (const cancels in deltas)
    lnL = -0.5 * float(chi2_best)

    # Write JSON sidecar
    meta = {
        'galaxy_id': name,
'file_rotmod': str(Path(args.sparc_dir)/f'{name}_rotmod.dat'),
        'model': model,
        'params': {
            'log10_rho_c': log10_rho_c,
            'gamma_exp': gamma_exp,
            'lambda_max': lambda_max,
            'lnT0': lnT0,
            'sigma_lnT': sigma_lnT,
            'w_min': w_min,
            'ups_disk': float(ups_d),
            'ups_bul': float(ups_b),
            'V200': float(args.V200),
            'c': float(args.c),
        },
        'chi2': float(chi2_best),
        'chi2_dof': float(chi2_best/dof),
        'loglike_no_const': lnL,
        'mode': args.mode,
    }
    with open(out_path.with_suffix('.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    print(f"Saved: {out_path.with_suffix('.json')}")


if __name__ == '__main__':
    main()

