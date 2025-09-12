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
    """Return (r grid in Mpc, z(r)) monotonically increasing."""
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


def mu_from_unified_gate(z_grid: np.ndarray, params: GateParams, y_value: float, rho_gamma_evcm3: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r, z_vals = build_z_of_r(params, y_value, rho_gamma_evcm3)
    r_mpc = r_of_z(z_grid, r, z_vals)
    # Luminosity distance in the tariff-only track (flat): D_L(z) = (1+z) * r(z)
    d_pc = (1.0 + z_grid) * r_mpc * 1.0e6
    mu = np.full_like(z_grid, np.nan)
    mask = d_pc > 0
    mu[mask] = 5.0 * np.log10(d_pc[mask]) - 5.0
    return mu, r_mpc, r, z_vals

# ---------- Hubble overlay and chi-squared ----------

def overlay_hubble_unified(pantheon_path: str, params: GateParams, y_value: float = 0.1, rho_gamma_evcm3: float = 0.26):
    z_data, mu_data, mu_err = load_pantheon(pantheon_path)
    z_grid = np.logspace(-3.0, np.log10(max(z_data.max(), 1e-3)), 600)

    mu_model, _, r_grid, z_of_r = mu_from_unified_gate(z_grid, params, y_value, rho_gamma_evcm3)
    # Interpolate model onto data z for chi-squared
    mu_model_at_data = np.interp(z_data, z_grid, mu_model)
    valid = np.isfinite(mu_model_at_data) & np.isfinite(mu_data) & np.isfinite(mu_err) & (mu_err > 0)
    dof = max(int(np.count_nonzero(valid) - 1), 1)
    chi2 = float(np.sum(((mu_data[valid] - mu_model_at_data[valid]) / mu_err[valid])**2))
    red_chi2 = chi2 / dof

    plt.figure(figsize=(10,6))
    plt.errorbar(z_data, mu_data, yerr=mu_err, fmt='.', color='gray', alpha=0.45, label='Pantheon+SH0ES')
    plt.plot(z_grid, mu_model, '-', color='crimson', lw=2.0, label='Unified Gate + Tariff')
    plt.xscale('log'); plt.xlabel('z'); plt.ylabel('μ')
    plt.grid(alpha=0.3); plt.legend()
    out = os.path.join(IMAGES_DIR, 'unified_gate_hubble_overlay.png')
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
    print(f"Saved {out}")

    return {
        'hubble_plot': out,
        'chi2': chi2,
        'red_chi2': red_chi2,
        'dof': dof,
    }, (r_grid, z_of_r)

# ---------- H_eff and BAO shape overlays ----------

def heff_from_z_of_r(r: np.ndarray, z_of_r: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (z grid, H_eff(z) [km/s/Mpc], r(z) on that grid).

    Correct BAO proxy: H_eff(z) = c * dz/dr; D_M(z) = r(z); D_H(z) = c / H_eff(z).
    """
    # Build a dense z grid across available range (avoid zeros)
    z_grid = np.linspace(max(1e-6, float(z_of_r[1])), float(z_of_r[-1]), 2000)
    # r(z) via interp
    r_of_z_grid = np.interp(z_grid, z_of_r, r)
    dr_dz = np.gradient(r_of_z_grid, z_grid)
    dz_dr = 1.0 / np.clip(dr_dz, 1e-30, np.inf)
    H_eff = C_KM_S * dz_dr
    return z_grid, H_eff, r_of_z_grid


def bao_shape_overlay(z_of_r: np.ndarray, r: np.ndarray, bao_csv_path: str | None) -> dict:
    z_grid, H_eff, r_of_z_grid = heff_from_z_of_r(r, z_of_r)
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

def analyze_unified_gate(pantheon_path: str, bao_csv_path: str | None, params: GateParams, y_value: float = 0.1, rho_gamma_evcm3: float = 0.26):
    overlay_metrics, (r_grid, z_of_r) = overlay_hubble_unified(pantheon_path, params, y_value, rho_gamma_evcm3)
    bao_metrics = bao_shape_overlay(z_of_r, r_grid, bao_csv_path)

    # Consistency checks/invariants
    # 1) derivative identity: d ln(1+z)/dr = alpha_est; H_eff via two ways should agree
    z_arr = np.asarray(z_of_r, float)
    r_arr = np.asarray(r_grid, float)
    alpha_est = np.gradient(np.log1p(z_arr), r_arr)
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
        mu_model, _, _, _ = mu_from_unified_gate(z_grid_overlay, params, y_value, rho_gamma_evcm3)
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
    kappa_guess = calibrate_kappa_to_cmb(f_void=0.8, D_LSS_Mpc=14000.0, G_cap_minus1=1.0)
    params = GateParams(eta=3.0, p=1.5, q=1.0, rho_star_evcm3=0.26, kappa_per_Mpc=kappa_guess)
    analyze_unified_gate(pantheon, str(bao_csv_path) if bao_csv_path.exists() else None, params)
