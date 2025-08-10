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
from models.er_env import xi_env, tidal_proxy_from_vbar

try:
    from scipy import optimize as _opt  # type: ignore
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

# Optional dynesty for evidence mode (non-invasive; does not touch original runners)
try:
    import dynesty  # type: ignore
    _HAS_DYNASTY = True
except Exception:
    _HAS_DYNASTY = False


def _compute_tidal_proxy(R, vbar, mode: str = "curvature", norm: str = "robust"):
    R = np.asarray(R, dtype=float)
    vb = np.asarray(vbar, dtype=float)
    R_safe = np.clip(R, 1e-6, None)
    if mode == "curvature":
        g = (np.maximum(vb, 0.0) ** 2) / R_safe
        from models.er_env import finite_diff
        dgdR = finite_diff(g, R_safe)
        T_raw = np.abs(dgdR)
    elif mode == "shear":
        Omega = np.maximum(vb, 0.0) / R_safe
        from models.er_env import finite_diff
        dO_dR = finite_diff(Omega, R_safe)
        T_raw = np.abs(dO_dR)
    elif mode == "epicyclic":
        Omega = np.maximum(vb, 0.0) / R_safe
        from models.er_env import finite_diff
        dO_dR = finite_diff(Omega, R_safe)
        kappa2 = 2.0 * Omega * (2.0 * Omega + R_safe * dO_dR)
        T_raw = np.sqrt(np.clip(np.abs(kappa2), 0.0, None))
    else:
        T_raw = tidal_proxy_from_vbar(R, vb)
    # normalization
    if norm == "robust":
        med = float(np.nanmedian(T_raw))
        mad = float(np.nanmedian(np.abs(T_raw - med)))
        scale = med if med > 0 else (np.nanmax(T_raw) if np.nanmax(T_raw) > 0 else 1.0)
        Tn = T_raw / scale
    else:
        med = float(np.nanmedian(T_raw))
        scale = med if med > 0 else (np.nanmax(T_raw) if np.nanmax(T_raw) > 0 else 1.0)
        Tn = T_raw / scale
    return np.clip(Tn, 1e-12, None)


def chi2_env(params, R, Vobs, eV, Vgas, Vdisk, Vbul, rho_mid_base,
             tidal_mode: str, tidal_norm: str,
             sigma_floor: float,
             fit_ml: tuple[bool, bool], ml_priors: tuple[float, float], ml_sigmas: tuple[float, float]):
    # Unpack
    log10_rho_c, gamma_exp, lambda_max, lnT0, sigma_lnT, w_min, ups_d, ups_b = params
    # Build vbar with current M/L
    vbar = v_bar_from_components(R, Vgas, Vdisk, Vbul, ups_d, ups_b)
    # Scale rho_mid for M/L change (approximate; mild factor)
    f_ml = max(0.3, min(3.0, (ups_d / BASE_M_L_3_6_MICRON_DISK) ** 0.5))
    rho_mid = np.clip(rho_mid_base * f_ml, 1e-30, None)
    # ER prediction
    rho_c = 10 ** float(log10_rho_c)
    T0 = float(np.exp(lnT0))
    T = _compute_tidal_proxy(R, vbar, mode=tidal_mode, norm=tidal_norm)
    xi = xi_env(rho_mid, T, float(lambda_max), rho_c, float(gamma_exp), T0, float(sigma_lnT), float(w_min))
    vmod = np.sqrt(np.clip(xi, 0.0, None)) * np.maximum(vbar, 0.0)
    # errors with sigma floor
    e_eff = np.sqrt(np.asarray(eV, dtype=float)**2 + float(max(0.0, sigma_floor))**2)
    e_eff = np.where(e_eff > 0, e_eff, 1.0)
    r = (Vobs - vmod) / e_eff
    chi2 = float(np.sum(r * r))
    # M/L priors if requested
    pen = 0.0
    if fit_ml[0]:
        mu, sig = ml_priors[0], max(ml_sigmas[0], 1e-6)
        pen += ((ups_d - mu) / sig) ** 2
    if fit_ml[1]:
        mu, sig = ml_priors[1], max(ml_sigmas[1], 1e-6)
        pen += ((ups_b - mu) / sig) ** 2
    return chi2 + pen


def fit_one_galaxy_env(galaxy_id: str, sparc_dir: str,
                        init, bounds,
                        tidal_mode: str, tidal_norm: str,
                        sigma_floor: float,
                        fit_ml_flags: tuple[bool, bool],
                        ml_priors: tuple[float, float],
                        ml_sigmas: tuple[float, float]):
    data = load_single_sparc_galaxy(
        galaxy_id,
        sparc_dir=sparc_dir,
        assume_gas_hz_kpc=float(args.hz_gas),
        assume_stellar_hz_kpc=float(args.hz_star),
        assume_hz_alpha=float(args.hz_alpha),
    )
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

    # Now optimize including ups_d and ups_b as parameters (at the end of vector)
    def obj_full(x):
        x = np.clip(x, lo, hi)
        return chi2_env(x, R, Vobs, eV, Vgas, Vdisk, Vbul, rho_mid_base,
                        tidal_mode, tidal_norm, sigma_floor,
                        fit_ml_flags, ml_priors, ml_sigmas)

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

    return data, x_best, chi2_best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--galaxy_id', required=True)
    ap.add_argument('--sparc_dir', required=True)
    ap.add_argument('--out', default=None)
    ap.add_argument('--model', choices=['gr','er','nfw'], default='er', help='Model type to fit')
    ap.add_argument('--mode', choices=['fit','evidence'], default='fit', help='Operation: quick fit (chi2) or evidence scaffold output')
    # Likelihood controls
    ap.add_argument('--sigma-floor', type=float, default=0.0, help='Velocity error floor (km/s) added in quadrature')
    # Gas truncation controls (forwarded to loader via env for now)
    ap.add_argument('--gas-truncation', choices=['RHI','KRD'], default=None, help='Truncation scheme for gas disk')
    ap.add_argument('--gas-krd', type=float, default=3.0, help='k for Rmax = k Rd when using KRD truncation')
    ap.add_argument('--gas-profile', choices=['RHI','Vgas'], default='RHI', help='Gas profile model: RHI-truncated exponential (preferred) or Vgas-shaped fallback')
    # Tidal proxy controls
    ap.add_argument('--T-proxy', choices=['curvature','shear','epicyclic'], default='curvature')
    ap.add_argument('--tidal-norm', choices=['robust','simple'], default='robust')
    # Screening family
    ap.add_argument('--rho-screen', choices=['power','exp'], default='power', help='S_rho screening family: power (1/[1+(rho/rho_c)^gamma]) or exp (exp(-(rho/rho_c)^gamma))')
    # MasterSheet priors on distance/inclination (logged for parity)
    ap.add_argument('--use-master-priors', action='store_true', help='Record and (optionally) apply MasterSheet D/i priors for parity across models')
    # Vertical profile assumptions (recorded in JSON for reproducibility)
    ap.add_argument('--hz-star', type=float, default=0.30, help='Assumed stellar scale height h_z,* (kpc)')
    ap.add_argument('--hz-gas', type=float, default=0.10, help='Assumed gas scale height h_z,gas (kpc)')
    ap.add_argument('--hz-alpha', type=float, default=0.0, help='Optional linear flaring coefficient alpha (dimensionless)')
    # Dynesty evidence controls (non-invasive)
    ap.add_argument('--nlive', type=int, default=1000, help='Dynesty nlive (evidence mode)')
    ap.add_argument('--maxcall', type=int, default=200000, help='Dynesty maxcall (evidence mode)')
    ap.add_argument('--dlogz-target', type=float, default=0.01, help='Dynesty target dlogz (evidence mode)')
    ap.add_argument('--seed', type=int, default=42, help='Random seed for sampler (evidence mode)')
    # Inits / priors centers
    ap.add_argument('--log10_rho_c', type=float, default=15.0)
    ap.add_argument('--gamma_exp', type=float, default=3.0)
    ap.add_argument('--lambda_max', type=float, default=4.0)
    ap.add_argument('--prior-lambda-max', type=float, default=6.0, help='Upper prior bound for lambda_max')
    ap.add_argument('--lnT0', type=float, default=0.0)
    ap.add_argument('--sigma_lnT', type=float, default=0.8)
    ap.add_argument('--w_min', type=float, default=0.02)
    ap.add_argument('--prior-wmin-max', type=float, default=0.1, help='Upper prior bound for w_min')
    # M/L fitting
    ap.add_argument('--fit-ml', nargs='*', choices=['disk','bulge'], default=[], help='Fit stellar M/L for selected components')
    ap.add_argument('--ml-prior-disk', type=float, default=0.5)
    ap.add_argument('--ml-prior-disk-sigma', type=float, default=0.1)
    ap.add_argument('--ml-prior-bulge', type=float, default=0.7)
    ap.add_argument('--ml-prior-bulge-sigma', type=float, default=0.1)
    # NFW inits
    ap.add_argument('--V200', type=float, default=150.0, help='NFW V200 [km/s]')
    ap.add_argument('--c', type=float, default=10.0, help='NFW concentration')
    args = ap.parse_args()

    # Set gas truncation envs for loader if flags provided
    import os as _os
    if args.gas_truncation is not None:
        _os.environ['SPARC_GAS_RMAX_MODE'] = args.gas_truncation.upper()
        if args.gas_truncation.upper() == 'KRD':
            _os.environ['SPARC_GAS_KRD'] = str(args.gas_krd)
    # Gas profile override
    if args.gas_profile.upper() == 'VGAS':
        _os.environ['SPARC_GAS_FORCE_VGAS'] = '1'
    else:
        _os.environ.pop('SPARC_GAS_FORCE_VGAS', None)
    # Parameter vector includes ups_d and ups_b at the end
    # Initialize ups based on whether we fit them
    init_ups_d = args.ml_prior_disk
    init_ups_b = args.ml_prior_bulge
    init = [args.log10_rho_c, args.gamma_exp, args.lambda_max, args.lnT0, args.sigma_lnT, args.w_min, init_ups_d, init_ups_b]
    bounds = [
        (14.0, 17.0),  # log10_rho_c
        (1.0, 5.0),    # gamma_exp
        (0.0, float(args.prior_lambda_max)),    # lambda_max upper bound adjustable
        (-1.0, 1.0),   # lnT0
        (0.3, 2.0),    # sigma_lnT
        (0.0, float(args.prior_wmin_max)),    # w_min upper bound adjustable
        (0.3, 0.8),    # ups_disk (SPARC-style clamp)
        (0.5, 1.0),    # ups_bulge (SPARC-style clamp)
    ]

    fit_ml_flags = (('disk' in args.fit_ml), ('bulge' in args.fit_ml))
    ml_priors = (args.ml_prior_disk, args.ml_prior_bulge)
    ml_sigmas = (args.ml_prior_disk_sigma, args.ml_prior_bulge_sigma)

    data, x_best, chi2_best = fit_one_galaxy_env(
        args.galaxy_id, args.sparc_dir,
        init, bounds,
        args.T_proxy, args.tidal_norm,
        args.sigma_floor,
        fit_ml_flags,
        ml_priors,
        ml_sigmas,
    )

    R = data['R_kpc']
    Vobs = data['V_obs']
    eV = data['e_V_obs']
    Vgas = data['V_gas_comp_kms']
    Vdisk = data['V_disk_comp_kms']
    Vbul = data['V_bulge_comp_kms']
    rho_mid_base = data['rho_star_mid_Msun_kpc3_baseML'] + data['rho_gas_mid_Msun_kpc3']

    # Unpack best params (including M/L)
    log10_rho_c, gamma_exp, lambda_max, lnT0, sigma_lnT, w_min, ups_d, ups_b = map(float, x_best)
    vbar = v_bar_from_components(R, Vgas, Vdisk, Vbul, ups_d, ups_b)
    f_ml = max(0.3, min(3.0, (ups_d / BASE_M_L_3_6_MICRON_DISK) ** 0.5))
    rho_mid = np.clip(rho_mid_base * f_ml, 1e-30, None)
    rho_c = 10 ** log10_rho_c
    T0 = float(np.exp(lnT0))

    # Build T with selected proxy and normalization
    T = _compute_tidal_proxy(R, vbar, mode=args.T_proxy, norm=args.tidal_norm)
    xi = (xi_env if args.rho_screen == 'power' else __import__('models.er_env', fromlist=['xi_env_exp']).xi_env_exp)(rho_mid, T, lambda_max, rho_c, gamma_exp, T0, sigma_lnT, w_min)
    vmod = np.sqrt(np.clip(xi, 0.0, None)) * np.maximum(vbar, 0.0)
    # Sanity diagnostics: xi stats and W(T) peak location
    from models.er_env import W_log_normal
    W_vals = W_log_normal(T, T0, sigma_lnT, w_min)
    xi_stats = {
        'xi_min': float(np.nanmin(xi)),
        'xi_med': float(np.nanmedian(xi)),
        'xi_max': float(np.nanmax(xi)),
    }
    R_peak_W = float(R[np.nanargmax(W_vals)]) if np.all(np.isfinite(W_vals)) and (W_vals.size > 0) else float('nan')

    dof = max(1, R.size - len(x_best))
    print({
        'galaxy_id': data['galaxy_id'],
        'chi2': float(chi2_best),
        'chi2_dof': float(chi2_best/dof),
'params': {
            'log10_rho_c': log10_rho_c,
            'rho_screen': args.rho_screen,
            'gamma_exp': gamma_exp,
            'lambda_max': lambda_max,
            'lnT0': lnT0,
            'sigma_lnT': sigma_lnT,
            'w_min': w_min,
            'ups_disk': float(ups_d),
            'ups_bul': float(ups_b),
            'sigma_floor': float(args.sigma_floor),
            'T_proxy': args.T_proxy,
            'tidal_norm': args.tidal_norm,
            'hz_star_kpc': float(args.hz_star),
            'hz_gas_kpc': float(args.hz_gas),
            'hz_alpha': float(args.hz_alpha),
        },
        'priors': {
            'use_master_priors': bool(args.use_master_priors),
            'distance_Mpc': float(data.get('distance_Mpc', np.nan)),
            'e_distance_Mpc': float(data.get('e_distance_Mpc', np.nan)),
            'incl_deg': float(data.get('incl_deg', np.nan)),
            'e_incl_deg': float(data.get('e_incl_deg', np.nan)),
        },
'gas': {
            'profile_mode': data.get('gas_profile_mode', None),
            'Rd_kpc': float(data.get('gas_Rd_kpc', np.nan)),
            'Sigma0': float(data.get('gas_Sigma0', np.nan)),
            'Rmax_kpc': float(data.get('gas_Rmax_kpc', np.nan)),
            'hz_star_kpc': float(data.get('assumed_hz_stellar_kpc', np.nan)),
            'hz_gas_kpc': float(data.get('assumed_hz_gas_kpc', np.nan)),
            'hz_alpha': float(data.get('assumed_hz_alpha', np.nan)),
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

    # Evidence mode (optional, non-invasive)
    logZ = None; logZ_err = None
    if args.mode == 'evidence' and _HAS_DYNASTY:
        # Build loglike and prior transform for dynesty on current data/problem
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        ndim = lo.size
        def prior_transform(u):
            u = np.asarray(u)
            return lo + u * (hi - lo)
        def loglike(theta):
            theta = np.asarray(theta, dtype=float)
            c2 = chi2_env(theta, R, Vobs, eV, Vgas, Vdisk, Vbul, rho_mid_base,
                          args.T_proxy, args.tidal_norm, args.sigma_floor,
                          fit_ml_flags, ml_priors, ml_sigmas)
            return -0.5 * float(c2)
        rng = np.random.default_rng(int(args.seed))
        dsampler = dynesty.DynamicNestedSampler(loglike, prior_transform, ndim=ndim, bound='multi', sample='rslice', rstate=rng)
        dsampler.run_nested(maxcall=int(args.maxcall), dlogz_init=float(args.dlogz_target), nlive_init=int(args.nlive))
        res = dsampler.results
        logZ = float(res.logz[-1])
        logZ_err = float(res.logzerr[-1]) if hasattr(res, 'logzerr') else None

    # Compute simple log-likelihood at best fit for reporting (also used if evidence not available)
    # ln L = -0.5 * chi2 + const; we report the -0.5*chi2 term (const cancels in deltas)
    lnL = -0.5 * float(chi2_best)

    # Write JSON sidecar
    meta = {
        'galaxy_id': name,
'file_rotmod': str(Path(args.sparc_dir)/f'{name}_rotmod.dat'),
        'model': model,
        'mask': 'none',
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
        'priors': {
            'use_master_priors': bool(args.use_master_priors),
            'distance_Mpc': float(data.get('distance_Mpc', np.nan)),
            'e_distance_Mpc': float(data.get('e_distance_Mpc', np.nan)),
            'incl_deg': float(data.get('incl_deg', np.nan)),
            'e_incl_deg': float(data.get('e_incl_deg', np.nan)),
        },
        'gas': {
            'profile_mode': data.get('gas_profile_mode', None),
            'Rd_kpc': float(data.get('gas_Rd_kpc', np.nan)),
            'Sigma0': float(data.get('gas_Sigma0', np.nan)),
            'Rmax_kpc': float(data.get('gas_Rmax_kpc', np.nan)),
            'mass_mismatch': float(data.get('gas_mass_mismatch', np.nan)),
            'penalty_mass': float(data.get('gas_penalty_mass', 0.0)),
        },
'sanity': {
            'xi_min': xi_stats['xi_min'],
            'xi_med': xi_stats['xi_med'],
            'xi_max': xi_stats['xi_max'],
            'R_peak_W': R_peak_W,
            'sigma_floor': float(args.sigma_floor),
            'T_proxy': args.T_proxy,
            'tidal_norm': args.tidal_norm,
            'hz_star_kpc': float(args.hz_star),
            'hz_gas_kpc': float(args.hz_gas),
            'hz_alpha': float(args.hz_alpha),
            'prior_edge_hits': {
                'log10_rho_c': (log10_rho_c in [bounds[0][0], bounds[0][1]]),
                'gamma_exp': (gamma_exp in [bounds[1][0], bounds[1][1]]),
                'lambda_max': (lambda_max in [bounds[2][0], bounds[2][1]]),
                'lnT0': (lnT0 in [bounds[3][0], bounds[3][1]]),
                'sigma_lnT': (sigma_lnT in [bounds[4][0], bounds[4][1]]),
                'w_min': (w_min in [bounds[5][0], bounds[5][1]]),
                'ups_disk': (ups_d in [bounds[6][0], bounds[6][1]]),
                'ups_bul': (ups_b in [bounds[7][0], bounds[7][1]]),
            },
        },
        'chi2': float(chi2_best),
        'chi2_dof': float(chi2_best/dof),
        'loglike_no_const': lnL,
'mode': args.mode,
'evidence': {
            'logZ': logZ,
            'logZ_err': logZ_err,
            'nlive': int(args.nlive),
            'maxcall': int(args.maxcall),
            'dlogz_target': float(args.dlogz_target),
            'seed': int(args.seed),
        },
    }
    with open(out_path.with_suffix('.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    print(f"Saved: {out_path.with_suffix('.json')}")


if __name__ == '__main__':
    main()

