#!/usr/bin/env python3
"""
analysis_unified_gate.py — tariff-only comparison harness for the unified gate + energy tariff.

- Overlays μ(z) from unified gate + tariff against Pantheon+
- Computes χ² and reduced χ² for the overlay
- Derives H_eff(z) from z(r) and, if a BAO CSV is provided, fits r_d and reports χ²/dof
- Writes plots under tariff/images/ and a JSON summary under tariff/results/
"""
from __future__ import annotations

import json
import os
import numpy as np
import matplotlib.pyplot as plt

# Robust imports for package or script execution
try:
    from .data_ingest import load_pantheon, load_bao_csv
except Exception:
    from data_ingest import load_pantheon, load_bao_csv
try:
    from .unified_gate_scaffold import GateParams, integrate_tau, calibrate_kappa_to_cmb, uniform_void_sampler
except Exception:
    from unified_gate_scaffold import GateParams, integrate_tau, calibrate_kappa_to_cmb, uniform_void_sampler

IMAGES_DIR = os.path.join(os.path.dirname(__file__), 'images')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

C_KM_S = 299_792.458

# ---------- Core builders ----------

def build_z_of_r(params: GateParams, y_value: float, rho_gamma_evcm3: float, D_max_Mpc: float = 8000.0) -> tuple[np.ndarray,np.ndarray]:
    """Return (r grid in Mpc, z(r)) monotonically increasing.

    Tariff-only path: 1+z = exp(tau(r)).
    """
    r = np.linspace(0.0, D_max_Mpc, 1601)
    z_vals = np.zeros_like(r)
    for i, rr in enumerate(r):
        sampler = uniform_void_sampler(y_value, rho_gamma_evcm3, rr, n_steps=2000)
        tau, _ = integrate_tau(sampler, params, with_psi=False)
        z_vals[i] = np.expm1(tau)
    z_vals = np.maximum.accumulate(z_vals)
    return r, z_vals


def r_of_z(z_grid: np.ndarray, r: np.ndarray, z_vals: np.ndarray) -> np.ndarray:
    return np.interp(z_grid, z_vals, r)


def mu_from_unified_gate(z_grid: np.ndarray, params: GateParams, y_value: float, rho_gamma_evcm3: float, mode: str = 'tariff_only') -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute mu(z) for overlay.
    - mode 'tariff_only': 1+z = exp(tau), D_L=(1+z) r(z)
    - mode 'frw_overlay': 1+z_obs = (1+z_frw) * exp(tau(z_frw)), r = ∫ c/H_FRW dz_frw
    Returns: (mu(z_grid), r(z_grid) [Mpc], r_grid_full [Mpc], z_of_r_full, tau_of_r_full)
    """
    if mode == 'frw_overlay':
        # Build a FRW z_frw grid and accumulate tau along comoving distance
        z_frw = np.linspace(0.0, 2.5, 3001)
        # FRW H(z) (flat LCDM; shape-only helper consistent with analysis_baselines)
        H0, Om = 67.4, 0.315
        E = np.sqrt(Om*(1+z_frw)**3 + (1-Om))
        H_frw = H0 * E
        # comoving distance r(z_frw)
        r = np.cumsum((C_KM_S / np.maximum(H_frw, 1e-9)) * np.gradient(z_frw))
        r[0] = 0.0
        # integrate tau along r using uniform sampler up to each r
        z_obs = np.zeros_like(z_frw)
        tau_arr = np.zeros_like(z_frw)
        for i in range(len(z_frw)):
            rr = float(r[i])
            sampler = uniform_void_sampler(y_value, rho_gamma_evcm3, rr, n_steps=1000)
            tau, _ = integrate_tau(sampler, params, with_psi=False)
            tau_arr[i] = tau
            z_obs[i] = (1.0 + z_frw[i]) * np.exp(tau) - 1.0
        # Invert z_obs→r by monotone interpolation
        z_mon = np.maximum.accumulate(z_obs)
        # Build output on requested z_grid
        r_mpc = np.interp(z_grid, z_mon, r)
        d_pc = (1.0 + z_grid) * r_mpc * 1.0e6
        mu = np.full_like(z_grid, np.nan)
        mask = d_pc > 0
        mu[mask] = 5.0 * np.log10(d_pc[mask]) - 5.0
        return mu, r_mpc, r, z_mon, tau_arr
    else:
        r, z_vals = build_z_of_r(params, y_value, rho_gamma_evcm3)
        r_mpc = r_of_z(z_grid, r, z_vals)
        # Luminosity distance in the tariff-only track (flat): D_L(z) = (1+z) * r(z)
        d_pc = (1.0 + z_grid) * r_mpc * 1.0e6
        mu = np.full_like(z_grid, np.nan)
        mask = d_pc > 0
        mu[mask] = 5.0 * np.log10(d_pc[mask]) - 5.0
        tau_arr = np.log1p(z_vals)
        return mu, r_mpc, r, z_vals, tau_arr

# ---------- Hubble overlay and chi-squared ----------

def overlay_hubble_unified(pantheon_path: str, params: GateParams, y_value: float = 0.1, rho_gamma_evcm3: float = 0.26, mode: str = 'tariff_only', sigma_int_mag: float = 0.0):
    z_data, mu_data, mu_err = load_pantheon(pantheon_path)
    # Effective per-SN uncertainty with optional intrinsic scatter
    mu_err_eff = np.sqrt(mu_err**2 + float(sigma_int_mag)**2)
    # Try to load an external covariance matrix (if present)
    try:
        from .data_ingest import try_load_pantheon_cov
    except Exception:
        from data_ingest import try_load_pantheon_cov
    C_full = try_load_pantheon_cov(pantheon_path)
    z_grid = np.logspace(-3.0, np.log10(max(z_data.max(), 1e-3)), 600)

    mu_model, _, r_grid, z_of_r, tau_of_r = mu_from_unified_gate(z_grid, params, y_value, rho_gamma_evcm3, mode=mode)
    # Interpolate model onto data z for chi-squared
    mu_model_at_data = np.interp(z_data, z_grid, mu_model)
    valid = np.isfinite(mu_model_at_data) & np.isfinite(mu_data) & np.isfinite(mu_err_eff) & (mu_err_eff > 0)
    dof_raw = max(int(np.count_nonzero(valid) - 1), 1)

    if C_full is not None and C_full.shape[0] == len(mu_data):
        # Use full covariance (add intrinsic scatter on diagonal if requested)
        C = np.array(C_full, dtype=float)
        if sigma_int_mag > 0:
            C = C.copy()
            idx = np.where(valid)[0]
            C[idx[:,None], idx[None,:]] += np.eye(len(idx)) * (sigma_int_mag**2)
        # Build selector for valid entries
        idx = np.where(valid)[0]
        y = mu_data[idx]
        m = mu_model_at_data[idx]
        one = np.ones_like(y)
        C_sel = C[np.ix_(idx, idx)]
        # Invert covariance robustly
        try:
            C_inv = np.linalg.inv(C_sel)
        except np.linalg.LinAlgError:
            C_inv = np.linalg.pinv(C_sel, rcond=1e-10)
        # Raw chi2
        dy = y - m
        chi2_raw = float(dy @ C_inv @ dy)
        red_chi2_raw = chi2_raw / dof_raw
        # Anchored chi2: minimize wrt delta_mu
        num = one @ (C_inv @ dy)
        den = one @ (C_inv @ one)
        delta_mu = float(num / den)
        dy_anch = y - (m + delta_mu)
        chi2_anch = float(dy_anch @ C_inv @ dy_anch)
        red_chi2_anch = chi2_anch / dof_raw
        sn_fit_method = 'anchored_cov'
    else:
        # Diagonal-only with intrinsic scatter
        w = 1.0 / (mu_err_eff[valid]**2)
        # Raw (un-anchored) chi2
        chi2_raw = float(np.sum(((mu_data[valid] - mu_model_at_data[valid]) / mu_err_eff[valid])**2))
        red_chi2_raw = chi2_raw / dof_raw
        # Float a constant magnitude offset (anchor MB or H0)
        delta_mu = float(np.sum(w * (mu_data[valid] - mu_model_at_data[valid])) / np.sum(w))
        mu_model_at_data_anch = mu_model_at_data + delta_mu
        chi2_anch = float(np.sum(((mu_data[valid] - mu_model_at_data_anch[valid]) / mu_err_eff[valid])**2))
        red_chi2_anch = chi2_anch / dof_raw
        sn_fit_method = 'anchored_diag'

    # Plot with anchored overlay for visual comparison
    plt.figure(figsize=(10,6))
    plt.errorbar(z_data, mu_data, yerr=mu_err, fmt='.', color='gray', alpha=0.45, label='Pantheon+SH0ES')
    plt.plot(z_grid, mu_model + delta_mu, '-', color='crimson', lw=2.0, label='Unified Gate + Tariff (anchored)')
    plt.xscale('log'); plt.xlabel('z'); plt.ylabel('μ')
    plt.grid(alpha=0.3); plt.legend()
    out = os.path.join(IMAGES_DIR, 'unified_gate_hubble_overlay.png')
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
    print(f"Saved {out}")

    # LOS environment correlation: S(r) = tau(r) / kappa (units: Mpc)
    kappa = float(params.kappa_per_Mpc) if hasattr(params, 'kappa_per_Mpc') else 1.0
    S_r = np.asarray(tau_of_r, float) / max(kappa, 1e-30)
    # Interpolate S at SN positions using r(z)
    r_at_data = np.interp(z_data, z_of_r, r_grid)
    S_at_data = np.interp(r_at_data, r_grid, S_r)
    # Residuals after anchoring (valid subset)
    mu_model_at_data = np.interp(z_data, z_grid, mu_model)
    resid = (mu_data - (mu_model_at_data + delta_mu))[valid]
    Sv = S_at_data[valid]
    wv = 1.0 / (mu_err_eff[valid]**2)
    # Weighted linear regression resid = a + b*S
    X = np.vstack([np.ones_like(Sv), Sv]).T
    XtWX = X.T @ (wv[:, None] * X)
    XtWy = X.T @ (wv * resid)
    try:
        beta = np.linalg.solve(XtWX, XtWy)
        cov_beta = np.linalg.inv(XtWX)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(XtWX) @ XtWy
        cov_beta = np.linalg.pinv(XtWX)
    a0, b1 = float(beta[0]), float(beta[1])
    resid_fit = resid - (a0 + b1 * Sv)
    dof_reg = max(int(np.count_nonzero(valid) - 2), 1)
    chi2_w = float(np.sum(wv * resid_fit**2))
    sigma2 = chi2_w / dof_reg
    cov_beta = cov_beta * sigma2
    b1_se = float(np.sqrt(max(cov_beta[1, 1], 0.0)))
    t_stat = float(b1 / b1_se) if b1_se > 0 else float('inf')
    # Weighted R^2
    ybar_w = float(np.sum(wv * resid) / np.sum(wv))
    ss_tot = float(np.sum(wv * (resid - ybar_w) ** 2))
    ss_res = float(np.sum(wv * resid_fit ** 2))
    r2_w = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float('nan')
    # Plot residuals vs S
    plt.figure(figsize=(8,6))
    plt.scatter(Sv, resid, s=8, alpha=0.6, label='SN residuals (anchored)')
    xs = np.linspace(np.nanmin(Sv), np.nanmax(Sv), 200)
    plt.plot(xs, a0 + b1 * xs, 'r-', label=f'fit: resid = {a0:.3f} + {b1:.3e} S')
    plt.xlabel('S (Mpc)')
    plt.ylabel('Δμ (mag)')
    plt.grid(alpha=0.3)
    plt.legend()
    out_resid = os.path.join(IMAGES_DIR, 'unified_gate_sn_residuals_vs_S.png')
    plt.tight_layout(); plt.savefig(out_resid, dpi=150); plt.close()

    return {
        'hubble_plot': out,
        'sn_fit_method': sn_fit_method,
        'sigma_int_mag': float(sigma_int_mag),
        'anchor_delta_mu_mag': delta_mu,
        'chi2_raw': chi2_raw,
        'red_chi2_raw': red_chi2_raw,
        'chi2': chi2_anch,
        'red_chi2': red_chi2_anch,
        'dof': dof_raw,
        'los_env_correlation': {
            'plot': out_resid,
            'slope_mag_per_Mpc': b1,
            'slope_stderr': b1_se,
            't_stat': t_stat,
            'dof': dof_reg,
            'r2_weighted': r2_w
        }
    }, (r_grid, z_of_r, tau_of_r)

# ---------- H_eff and BAO shape overlays ----------

def heff_from_z_of_r(r: np.ndarray, z_of_r: np.ndarray, mode: str = 'tariff_only') -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (z grid, H_eff(z) [km/s/Mpc], r(z) on that grid).

    - 'tariff_only': H_eff = c * dz/dr, D_M = r(z), D_H = c/H_eff.
    - 'frw_overlay': use FRW mapping: z_obs=(1+z_frw)exp(tau)-1 and r(z_obs)=∫ c/H_frw dz_frw.
    """
    if mode == 'frw_overlay':
        z_frw = np.linspace(0.0, 2.5, 3001)
        H0, Om = 67.4, 0.315
        E = np.sqrt(Om*(1+z_frw)**3 + (1-Om))
        H_frw = H0 * E
        r_frw = np.cumsum((C_KM_S / np.maximum(H_frw, 1e-9)) * np.gradient(z_frw)); r_frw[0]=0.0
        # Build z_obs via tau(r)
        # Approximate tau as function of r using the existing z(r) tariff-only builder at small kappa
        r_tariff, z_tariff = r, z_of_r
        # Interpolate tau(r) = ln(1+z_tariff)
        tau_of_r = np.log1p(z_tariff)
        # Map each z_frw to r_frw then tau, then z_obs
        tau_frw = np.interp(r_frw, r_tariff, tau_of_r)
        z_obs = (1.0 + z_frw) * np.exp(tau_frw) - 1.0
        z_grid = np.maximum.accumulate(z_obs)
        r_of_z_grid = np.interp(z_grid, z_obs, r_frw)
        dr_dz = np.gradient(r_of_z_grid, z_grid)
        dz_dr = 1.0 / np.clip(dr_dz, 1e-30, np.inf)
        H_eff = C_KM_S * dz_dr
        return z_grid, H_eff, r_of_z_grid
    else:
        # tariff-only
        z_grid = np.linspace(max(1e-6, float(z_of_r[1])), float(z_of_r[-1]), 2000)
        r_of_z_grid = np.interp(z_grid, z_of_r, r)
        dr_dz = np.gradient(r_of_z_grid, z_grid)
        dz_dr = 1.0 / np.clip(dr_dz, 1e-30, np.inf)
        H_eff = C_KM_S * dz_dr
        return z_grid, H_eff, r_of_z_grid


def bao_shape_overlay(z_of_r: np.ndarray, r: np.ndarray, bao_csv_path: str | None, mode: str = 'tariff_only') -> dict:
    z_grid, H_eff, r_of_z_grid = heff_from_z_of_r(r, z_of_r, mode=mode)
    # Correct BAO proxies
    DM = r_of_z_grid
    DH = C_KM_S / H_eff

    plt.figure(figsize=(10,8))
    plt.subplot(2,1,1)
    plt.title('Unified Gate — H_eff(z) and BAO proxies (corrected)')
    plt.plot(z_grid, H_eff, lw=2, label='H_eff(z) = c dz/dr')
    plt.ylabel('H_eff [km/s/Mpc]'); plt.grid(alpha=0.3); plt.legend()

    plt.subplot(2,1,2)
    plt.plot(z_grid, DM, lw=2, label='D_M(z) = r(z)')
    plt.plot(z_grid, DH, lw=2, label='D_H(z) = c/H_eff')
    plt.xlabel('z'); plt.ylabel('Distance [Mpc]')
    plt.grid(alpha=0.3); plt.legend()

    out = os.path.join(IMAGES_DIR, 'unified_gate_bao_proxies.png')
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
    print(f"Saved {out}")

    metrics = {
        'bao_plot': out,
        'H_eff_summary': {
            'z': z_grid.tolist(),
            'H_eff': H_eff.tolist(),
            'D_M': DM.tolist(),
            'D_H': DH.tolist(),
        },
    }

    if bao_csv_path is not None and os.path.exists(bao_csv_path):
        try:
            bao = load_bao_csv(bao_csv_path)
            z_b = bao['z']
            # Try DM/DH first
            if 'D_M_over_rd' in bao and 'D_H_over_rd' in bao:
                DM_b = bao['D_M_over_rd']; DH_b = bao['D_H_over_rd']
                eDM = bao.get('D_M_err', np.ones_like(DM_b)*0.05)
                eDH = bao.get('D_H_err', np.ones_like(DH_b)*0.05)
                DM_m = np.interp(z_b, z_grid, DM)
                DH_m = np.interp(z_b, z_grid, DH)
                A = np.concatenate([DM_m/eDM, DH_m/eDH])
                y = np.concatenate([DM_b/eDM, DH_b/eDH])
                inv_rd = (A @ y) / (A @ A + 1e-300)
                rd_best = 1.0 / inv_rd
                chi2 = np.sum(((DM_m/rd_best - DM_b)/eDM)**2 + ((DH_m/rd_best - DH_b)/eDH)**2)
                dof = max(1, 2*len(z_b) - 1)
                metrics.update({'rd_best_Mpc': float(rd_best), 'bao_chi2': float(chi2), 'bao_red_chi2': float(chi2/dof), 'bao_dof': int(dof)})
            elif 'DV_over_rd' in bao:
                DV_b = bao['DV_over_rd']
                eDV = bao.get('DV_err', np.ones_like(DV_b)*0.05)
                DM_m = np.interp(z_b, z_grid, DM)
                DH_m = np.interp(z_b, z_grid, DH)
                DV_m = ( (z_b**2) * (DM_m**2) * DH_m ) ** (1.0/3.0)
                A = DV_m / eDV
                y = DV_b / eDV
                inv_rd = (A @ y) / (A @ A + 1e-300)
                rd_best = 1.0 / inv_rd
                chi2 = np.sum(((DV_m/rd_best - DV_b)/eDV)**2)
                dof = max(1, len(z_b) - 1)
                metrics.update({'rd_best_Mpc': float(rd_best), 'bao_chi2': float(chi2), 'bao_red_chi2': float(chi2/dof), 'bao_dof': int(dof)})
        except Exception as e:
            print(f"[WARN] BAO overlay failed: {e}")
    return metrics

# ---------- Orchestrator ----------

def _scan_kappa_mb(pantheon_path: str, params: GateParams, y_value: float, rho_gamma_evcm3: float, mode: str, sigma_int_mag: float, factors: np.ndarray) -> dict:
    """Scan kappa_per_Mpc over multiplicative factors; anchor MB at each step; return best-fit summary."""
    base_k = float(params.kappa_per_Mpc)
    grid = []
    best = {'chi2': float('inf')}
    for f in factors:
        k_try = base_k * float(f)
        p_try = GateParams(eta=params.eta, p=params.p, q=params.q, rho_star_evcm3=params.rho_star_evcm3, kappa_per_Mpc=k_try, sigma=params.sigma, enable_backreaction=params.enable_backreaction)
        hub, _ = overlay_hubble_unified(pantheon_path, p_try, y_value, rho_gamma_evcm3, mode=mode, sigma_int_mag=sigma_int_mag)
        entry = {'kappa_per_Mpc': k_try, 'chi2': hub.get('chi2', float('nan')), 'red_chi2': hub.get('red_chi2', float('nan')), 'delta_mu': hub.get('anchor_delta_mu_mag', float('nan'))}
        grid.append(entry)
        if np.isfinite(entry['chi2']) and entry['chi2'] < best['chi2']:
            best = dict(entry)
    return {'base_kappa_per_Mpc': base_k, 'factors': factors.tolist(), 'grid': grid, 'best': best}


def analyze_unified_gate(pantheon_path: str, bao_csv_path: str | None, params: GateParams, y_value: float = 0.1, rho_gamma_evcm3: float = 0.26, mode: str = 'tariff_only', sigma_int_mag: float = 0.0, do_kappa_scan: bool = True):
    overlay_metrics, (r_grid, z_of_r, tau_of_r) = overlay_hubble_unified(pantheon_path, params, y_value, rho_gamma_evcm3, mode=mode, sigma_int_mag=sigma_int_mag)
    bao_metrics = bao_shape_overlay(z_of_r, r_grid, bao_csv_path, mode=mode)
    # Consistency checks/invariants
    z_arr = np.asarray(z_of_r, float)
    r_arr = np.asarray(r_grid, float)
    alpha_est = np.gradient(np.log1p(z_arr), r_arr)
    H_via_dzdr = C_KM_S * np.gradient(z_arr, r_arr)
    H_via_alpha = C_KM_S * (1.0 + z_arr) * alpha_est
    valid = np.isfinite(H_via_dzdr) & np.isfinite(H_via_alpha) & (H_via_dzdr > 0) & (H_via_alpha > 0)
    rel_diff = np.zeros_like(H_via_dzdr)
    rel_diff[valid] = np.abs(H_via_dzdr[valid] - H_via_alpha[valid]) / np.maximum(H_via_dzdr[valid], 1e-30)
    heff_identity_rms = float(np.sqrt(np.mean(rel_diff[valid]**2))) if np.any(valid) else float('nan')
    try:
        z_data, _, _ = load_pantheon(pantheon_path)
        r_at_data = np.interp(z_data, z_of_r, r_grid)
        DL_from_duality = (1.0 + z_data) * r_at_data
        mu_from_duality = 5.0 * np.log10(DL_from_duality * 1.0e6) - 5.0
        z_grid_overlay = np.logspace(-3.0, np.log10(max(z_data.max(), 1e-3)), 600)
        mu_model, _, _, _, _ = mu_from_unified_gate(z_grid_overlay, params, y_value, rho_gamma_evcm3, mode=mode)
        mu_model_at_data = np.interp(z_data, z_grid_overlay, mu_model)
        dd_rms = float(np.sqrt(np.mean((mu_from_duality - mu_model_at_data)**2)))
        distance_duality_ok = dd_rms < 1e-6
    except Exception:
        distance_duality_ok = False
        dd_rms = float('nan')
    consistency = {
        'heff_identity_rms': heff_identity_rms,
        'distance_duality_ok': distance_duality_ok,
        'distance_duality_mu_rms': dd_rms,
        'tolman_p': None,
        'time_dilation_p': None,
    }
    summary = {
        'params': vars(params),
        'hubble': overlay_metrics,
        'bao': bao_metrics,
        'consistency': consistency,
    }
    if do_kappa_scan:
        factors = np.logspace(-0.3, 0.3,  nine := 9)  # ~0.50x to ~2.0x in 9 steps
        scan = _scan_kappa_mb(pantheon_path, params, y_value, rho_gamma_evcm3, mode, sigma_int_mag, factors)
        summary['hubble']['kappa_scan'] = scan
    H_via_dzdr = C_KM_S * np.gradient(z_arr, r_arr)
    H_via_alpha = C_KM_S * (1.0 + z_arr) * alpha_est
    # Compute RMS relative difference over valid range
    valid = np.isfinite(H_via_dzdr) & np.isfinite(H_via_alpha) & (H_via_dzdr > 0) & (H_via_alpha > 0)
    rel_diff = np.zeros_like(H_via_dzdr)
    rel_diff[valid] = np.abs(H_via_dzdr[valid] - H_via_alpha[valid]) / np.maximum(H_via_dzdr[valid], 1e-30)
    heff_identity_rms = float(np.sqrt(np.mean(rel_diff[valid]**2))) if np.any(valid) else float('nan')

    # 2) distance duality check at Pantheon z grid used in overlay
    try:
        z_data, _, _ = load_pantheon(pantheon_path)
        # r(z) from our z(r)
        r_at_data = np.interp(z_data, z_of_r, r_grid)
        DL_from_duality = (1.0 + z_data) * r_at_data  # Mpc
        mu_from_duality = 5.0 * np.log10(DL_from_duality * 1.0e6) - 5.0
        # Compare to model mu at data
        z_grid_overlay = np.logspace(-3.0, np.log10(max(z_data.max(), 1e-3)), 600)
        mu_model, _, _, _ = mu_from_unified_gate(z_grid_overlay, params, y_value, rho_gamma_evcm3, mode=mode)
        mu_model_at_data = np.interp(z_data, z_grid_overlay, mu_model)
        dd_rms = float(np.sqrt(np.mean((mu_from_duality - mu_model_at_data)**2)))
        distance_duality_ok = dd_rms < 1e-6  # should be numerically identical after our fix
    except Exception:
        distance_duality_ok = False
        dd_rms = float('nan')

    consistency = {
        'heff_identity_rms': heff_identity_rms,
        'distance_duality_ok': distance_duality_ok,
        'distance_duality_mu_rms': dd_rms,
        'tolman_p': None,
        'time_dilation_p': None,
    }

    summary = {
        'params': vars(params),
        'hubble': overlay_metrics,
        'bao': bao_metrics,
        'consistency': consistency,
    }
    # Annotate BAO fit availability
    if bao_csv_path is None or not os.path.exists(bao_csv_path):
        summary['bao']['bao_fit_available'] = False
        summary['bao']['note'] = 'Provide tariff/data/bao_compilation.csv with columns z and D_M_over_rd/D_H_over_rd or DV_over_rd to enable shape-only fit.'
    else:
        summary['bao']['bao_fit_available'] = True
    out_json = os.path.join(RESULTS_DIR, 'unified_gate_metrics.json')
    with open(out_json, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {out_json}")
    return out_json


if __name__ == '__main__':
    from pathlib import Path
    REPO_ROOT = Path(__file__).resolve().parents[1]
    pantheon = str(REPO_ROOT / 'external_data' / 'pantheon' / 'Pantheon+SH0ES.dat')
    bao_csv_path = REPO_ROOT / 'tariff' / 'data' / 'bao_compilation.csv'
    # Small-overlay default: q=0, small kappa; FRW+tariff overlay mode
    params = GateParams(eta=3.0, p=1.5, q=0.0, rho_star_evcm3=0.26, kappa_per_Mpc=1e-5)
    analyze_unified_gate(pantheon, str(bao_csv_path) if bao_csv_path.exists() else None, params, mode='frw_overlay', sigma_int_mag=0.1)
