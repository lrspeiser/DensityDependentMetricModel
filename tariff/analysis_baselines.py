#!/usr/bin/env python3
"""
analysis_baselines.py — tariff-only baseline checks for GR and data sanity.

Produces plots to tariff/images/ and prints key baseline metrics.

Checks:
- Hubble diagram linear baseline (small-z): μ_lin(z; H0) vs Pantheon+ (chi^2)
- CMB spectral shape: fit Planck T' to provided spectrum (FIRAS-like)
- Tolman exponent p from SB data via μ-like fit
- SN time-dilation exponent p_t from timescale/stretch vs (1+z)
- FRW helper (flat ΛCDM) for D_M(z), D_H(z) shape-only overlays
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
from math import log10

from .data_ingest import (
    load_pantheon,
    load_cmb_spectrum_csv,
    load_bao_csv,
    load_tolman_csv,
    load_sntd_csv,
)

IMAGES_DIR = os.path.join(os.path.dirname(__file__), 'images')
os.makedirs(IMAGES_DIR, exist_ok=True)

C_KM_S = 299_792.458

# ---------- Hubble diagram baseline ----------

def mu_linear(z: np.ndarray, H0: float) -> np.ndarray:
    d_mpc = (C_KM_S / float(H0)) * z
    d_pc = d_mpc * 1.0e6
    out = np.full_like(z, np.nan, dtype=float)
    mask = d_pc > 0
    out[mask] = 5.0 * np.log10(d_pc[mask]) - 5.0
    return out


def baseline_hubble(pantheon_path: str, H0_planck: float = 67.4, H0_shoes: float = 73.0) -> None:
    z, mu, mu_err = load_pantheon(pantheon_path)
    z_plot = np.logspace(-3.0, np.log10(max(z.max(), 1e-3)), 300)
    mu_pl = mu_linear(z_plot, H0_planck)
    mu_sh = mu_linear(z_plot, H0_shoes)

    plt.figure(figsize=(10,6))
    plt.errorbar(z, mu, yerr=mu_err, fmt='.', color='gray', alpha=0.4, label='Pantheon+SH0ES')
    plt.plot(z_plot, mu_pl, '--', label=f'Linear Hubble (H0={H0_planck})')
    plt.plot(z_plot, mu_sh, ':', label=f'Linear Hubble (H0={H0_shoes})')
    plt.xscale('log'); plt.xlabel('z'); plt.ylabel('μ')
    plt.grid(alpha=0.3); plt.legend()
    out = os.path.join(IMAGES_DIR, 'baseline_hubble.png')
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
    print(f"Saved {out}")

# ---------- CMB spectral shape ----------

def planck_I_nu(nu_Hz: np.ndarray, T: float) -> np.ndarray:
    h = 6.62607015e-34
    kB = 1.380649e-23
    c = 299_792_458.0
    x = (h * nu_Hz) / (kB * T)
    ex = np.exp(np.clip(x, 1e-12, 700))
    return (2.0 * h * nu_Hz**3) / (c**2 * (ex - 1.0))


def fit_cmb_temperature(cmb_csv_path: str) -> None:
    nu, I = load_cmb_spectrum_csv(cmb_csv_path)
    y = I / (I.max() + 1e-300)
    Ts = np.linspace(1.0, 5.0, 4001)
    res = []
    for T in Ts:
        m = planck_I_nu(nu, T)
        m = m / (m.max() + 1e-300)
        res.append(np.sqrt(np.mean((y - m)**2)))
    i = int(np.argmin(res))
    T_best = float(Ts[i]); rms = float(res[i])
    print(f"CMB fit: T'={T_best:.4f} K, rms_frac_resid={rms:.3e}")
    plt.figure(figsize=(10,6))
    plt.title("CMB spectral shape baseline")
    plt.plot(nu/1e9, y, label='Data (normalized)')
    m = planck_I_nu(nu, T_best)
    plt.plot(nu/1e9, m/(m.max()+1e-300), label=f"Planck(T'={T_best:.3f}K)")
    plt.xlabel('Frequency [GHz]'); plt.ylabel('Normalized intensity')
    plt.grid(alpha=0.3); plt.legend()
    out = os.path.join(IMAGES_DIR, 'baseline_cmb_spectrum.png')
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
    print(f"Saved {out}")

# ---------- Tolman & SN time-dilation ----------

def fit_tolman_exponent(sb_csv_path: str) -> None:
    try:
        z, SB, SB_err = load_tolman_csv(sb_csv_path)
    except FileNotFoundError:
        print(f"Tolman CSV not found: {sb_csv_path}"); return
    # Fit S ∝ (1+z)^-p via linear regression in log space
    x = np.log1p(z)
    y = np.log(SB)
    w = 1.0 / np.clip(SB_err, 1e-12, np.inf)
    A = np.vstack([np.ones_like(x), -x]).T
    sol, *_ = np.linalg.lstsq(A * w[:,None], y * w, rcond=None)
    lnC, p = sol[0], sol[1]
    print(f"Tolman exponent p ≈ {p:.3f}")
    # Plot
    zp = np.linspace(z.min(), z.max(), 200)
    Sp = np.exp(lnC) * (1+zp)**(-p)
    plt.figure(figsize=(10,6))
    plt.errorbar(z, SB, yerr=SB_err, fmt='.', alpha=0.5, label='Data')
    plt.plot(zp, Sp, label=f'Fit p={p:.3f}')
    plt.xlabel('z'); plt.ylabel('Surface brightness (arb.)')
    plt.grid(alpha=0.3); plt.legend()
    out = os.path.join(IMAGES_DIR, 'baseline_tolman.png')
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
    print(f"Saved {out}")


def fit_sntd_exponent(sntd_csv_path: str) -> None:
    try:
        z, t, terr = load_sntd_csv(sntd_csv_path)
    except FileNotFoundError:
        print(f"SN time-dilation CSV not found: {sntd_csv_path}"); return
    x = np.log1p(z)
    y = np.log(t)
    w = 1.0 / np.clip(terr, 1e-12, np.inf)
    A = np.vstack([np.ones_like(x), x]).T
    sol, *_ = np.linalg.lstsq(A * w[:,None], y * w, rcond=None)
    lnA, p_t = sol[0], sol[1]
    print(f"SN time-dilation exponent p_t ≈ {p_t:.3f}")
    # Plot
    zp = np.linspace(z.min(), z.max(), 200)
    tp = np.exp(lnA) * (1+zp)**(p_t)
    plt.figure(figsize=(10,6))
    plt.errorbar(z, t, yerr=terr, fmt='.', alpha=0.5, label='Data')
    plt.plot(zp, tp, label=f'Fit p_t={p_t:.3f}')
    plt.xlabel('z'); plt.ylabel('Timescale/stretch (arb.)')
    plt.grid(alpha=0.3); plt.legend()
    out = os.path.join(IMAGES_DIR, 'baseline_sntd.png')
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
    print(f"Saved {out}")

# ---------- Simple FRW helper for BAO shape-only overlays ----------

def frw_shape_DM_DH(z: np.ndarray, H0: float = 67.4, Omega_m: float = 0.315, c_km_s: float = C_KM_S):
    # flat ΛCDM comoving radial distance and H(z); numeric integral for DM
    z = np.asarray(z, float)
    def E(zz):
        return np.sqrt(Omega_m*(1+zz)**3 + (1-Omega_m))
    # trapezoid for ∫ dz/E(z)
    zz = np.linspace(0.0, max(1e-3, z.max()), 4096)
    integ = np.cumsum(np.r_[0.0, 0.5*(1.0/E(zz[:-1]) + 1.0/E(zz[1:])) * np.diff(zz)])
    DM = (c_km_s/H0) * np.interp(z, zz, integ)
    DH = (c_km_s/H0) * 1.0/np.clip(E(z), 1e-12, np.inf)
    return DM, DH

if __name__ == '__main__':
    # Example minimal baseline run (adjust paths as needed)
    pantheon = os.path.join('external_data','pantheon','Pantheon+SH0ES.dat')
    baseline_hubble(pantheon)
    # fit_cmb_temperature(os.path.join('tariff','data','cmb_firas_like.csv'))
    # fit_tolman_exponent(os.path.join('tariff','data','tolman_sb.csv'))
    # fit_sntd_exponent(os.path.join('tariff','data','sn_timedilation.csv'))
