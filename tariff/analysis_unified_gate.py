#!/usr/bin/env python3
"""
analysis_unified_gate.py — tariff-only comparison harness for the unified gate + energy tariff.

- Overlays μ(z) from unified gate + tariff against Pantheon+
- Uses unified_gate_scaffold for G, tariff integration, and κ calibration
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt

from .data_ingest import load_pantheon
from .unified_gate_scaffold import GateParams, gate_G, integrate_tau, calibrate_kappa_to_cmb, uniform_void_sampler

IMAGES_DIR = os.path.join(os.path.dirname(__file__), 'images')
os.makedirs(IMAGES_DIR, exist_ok=True)

C_KM_S = 299_792.458


def build_z_of_r(params: GateParams, y_value: float, rho_gamma_evcm3: float, D_max_Mpc: float = 6000.0) -> tuple[np.ndarray,np.ndarray]:
    # Build a monotone z(r) lookup for a uniform path sampler (demonstration)
    r = np.linspace(0.0, D_max_Mpc, 1201)
    z_vals = np.zeros_like(r)
    for i, rr in enumerate(r):
        sampler = uniform_void_sampler(y_value, rho_gamma_evcm3, rr, n_steps=2000)
        tau, _ = integrate_tau(sampler, params, with_psi=False)
        z_vals[i] = np.expm1(tau)
    # enforce strict monotonicity
    z_vals = np.maximum.accumulate(z_vals)
    return r, z_vals


def r_of_z(z_grid: np.ndarray, r: np.ndarray, z_vals: np.ndarray) -> np.ndarray:
    # monotone inverse by interpolation
    return np.interp(z_grid, z_vals, r)


def mu_from_unified_gate(z_grid: np.ndarray, params: GateParams, y_value: float, rho_gamma_evcm3: float) -> tuple[np.ndarray, np.ndarray]:
    r, z_vals = build_z_of_r(params, y_value, rho_gamma_evcm3)
    r_mpc = r_of_z(z_grid, r, z_vals)
    d_pc = r_mpc * 1.0e6
    mu = np.full_like(z_grid, np.nan)
    mask = d_pc > 0
    mu[mask] = 5.0 * np.log10(d_pc[mask]) - 5.0
    return mu, r_mpc


def overlay_hubble_unified(pantheon_path: str, params: GateParams, y_value: float = 0.1, rho_gamma_evcm3: float = 0.26):
    z_data, mu_data, mu_err = load_pantheon(pantheon_path)
    z_grid = np.logspace(-3.0, np.log10(max(z_data.max(), 1e-3)), 400)

    mu_model, r_mpc = mu_from_unified_gate(z_grid, params, y_value, rho_gamma_evcm3)

    plt.figure(figsize=(10,6))
    plt.errorbar(z_data, mu_data, yerr=mu_err, fmt='.', color='gray', alpha=0.45, label='Pantheon+SH0ES')
    plt.plot(z_grid, mu_model, '-', color='crimson', lw=2.0, label='Unified Gate + Tariff')
    plt.xscale('log'); plt.xlabel('z'); plt.ylabel('μ')
    plt.grid(alpha=0.3); plt.legend()
    out = os.path.join(IMAGES_DIR, 'unified_gate_hubble_overlay.png')
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
    print(f"Saved {out}")


if __name__ == '__main__':
    pantheon = os.path.join('external_data','pantheon','Pantheon+SH0ES.dat')
    # calibrate kappa for demonstration (assume nearly saturated G-1 ~ 1 and f_void ~ 0.8)
    kappa_guess = calibrate_kappa_to_cmb(f_void=0.8, D_LSS_Mpc=14000.0, G_cap_minus1=1.0)
    params = GateParams(eta=3.0, p=1.5, q=1.0, rho_star_evcm3=0.26, kappa_per_Mpc=kappa_guess)
    overlay_hubble_unified(pantheon, params)
