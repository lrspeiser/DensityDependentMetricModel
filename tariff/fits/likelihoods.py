# fits/likelihoods.py
# Turn model predictions into χ² for CMB ℓ_A, SNe (with κ), BAO proxies, and RAR.

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from models.plateaus_model import (
    PlateausBackground, PlateausParams, Cosmology,
    distance_modulus_mu, delta_mu_kappa,
    rar_predict_gobs
)

# ------------
# CMB ℓ_A
# ------------
def chi2_cmb_ellA(bg: PlateausBackground, ellA_obs: float, sigma: float) -> Dict:
    ellA_th = bg.ell_A()
    chi2 = ((ellA_th - ellA_obs) / sigma) ** 2
    return {"ellA_th": float(ellA_th), "chi2": float(chi2)}


# ------------
# SNe with (optional) full covariance and analytic M-nuisance elimination ("anchored_fullcov")
# ------------
def chi2_sne(
    bg: PlateausBackground,
    df: pd.DataFrame,
    cov: np.ndarray | None,
    kappa_per_Mpc: float,
    method: str = "anchored_fullcov"
) -> Dict:
    z = df["z"].values.astype(float)
    DL = np.array([bg.D_L(zi) for zi in z])  # Mpc
    mu_th = distance_modulus_mu(DL)

    # κ-dimming
    chi = np.array([bg.comoving_distance(zi) for zi in z])  # Mpc
    mu_th = mu_th + delta_mu_kappa(chi, kappa_per_Mpc)

    # Observed μ
    if "mu" in df.columns:
        mu_obs = df["mu"].values.astype(float)
    elif "mB" in df.columns:
        # If given apparent magnitude mB only, treat M as nuisance (same math below still valid).
        mu_obs = df["mB"].values.astype(float)
    else:
        raise ValueError("SNe table must contain 'mu' or 'mB'.")

    if cov is None:
        # Diagonal-only errors
        sigma = df["sigma_mu"].values.astype(float)
        Cinv = np.diag(1.0 / np.clip(sigma, 1e-12, None) ** 2)
    else:
        Cinv = np.linalg.inv(cov)

    # Analytic elimination of a constant offset (absolute magnitude M or anchor)
    # r = μ_obs - μ_th - Δ (Δ ≡ nuisance).  χ² = rᵀC⁻¹r - (1ᵀC⁻¹r)² / (1ᵀC⁻¹1)
    ones = np.ones_like(mu_th)
    r = mu_obs - mu_th
    Cinv_r = Cinv @ r
    Cinv_1 = Cinv @ ones
    denom = float(ones @ Cinv_1)
    delta_hat = float(ones @ Cinv_r) / denom
    r_hat = r - delta_hat * ones
    chi2 = float(r_hat @ (Cinv @ r_hat))

    return {
        "chi2": chi2,
        "N": len(z),
        "delta_M_best": delta_hat,
        "mu_th_first5": mu_th[:5].tolist(),
    }


# ------------
# BAO observables
# ------------
def rd_drag(bg: PlateausBackground) -> float:
    # Eisenstein & Hu (1998) z_d approximation; treat ω_m = ω_b (no CDM).
    h = bg.cosmo.H0 / 100.0
    wb = bg.cosmo.omega_b
    wm = bg.cosmo.omega_b
    b1 = 0.313 * wm ** -0.419 * (1.0 + 0.607 * wm ** 0.674)
    b2 = 0.238 * wm ** 0.223
    zd = 1291.0 * wm ** 0.251 / (1.0 + 0.659 * wm ** 0.828) * (1.0 + b1 * wb ** b2)
    return bg.sound_horizon(zd)


def bao_theory(bg: PlateausBackground, z: float) -> Dict[str, float]:
    DM = bg.D_M(z)
    DH = C_LIGHT / bg.cosmo.H0 / bg.E(1.0/(1.0+z))  # Hubble distance at z
    DL = bg.D_L(z)
    DV = ( (DM**2) * z * DH ) ** (1.0/3.0)
    rd = rd_drag(bg)
    return {
        "DM_over_rd": DM / rd,
        "DH_over_rd": DH / rd,
        "DV_over_rd": DV / rd,
    }


def chi2_bao(bg: PlateausBackground, df: pd.DataFrame) -> Dict:
    resid2_sum = 0.0
    for _, row in df.iterrows():
        th = bao_theory(bg, float(row["z"]))[row["observable"]]
        obs = float(row["value"])
        sig = float(row["sigma"])
        resid2_sum += ((th - obs) / sig) ** 2
    return {"chi2": float(resid2_sum), "N": int(len(df))}


# ------------
# RAR
# ------------
def chi2_rar(df: pd.DataFrame, a0_m_s2: float, form: str) -> Dict:
    gbar = df["gbar_m_s2"].values.astype(float)
    gobs = df["gobs_m_s2"].values.astype(float)
    if "sigma_gobs_m_s2" in df.columns:
        sig = df["sigma_gobs_m_s2"].values.astype(float)
    else:
        # fall back to fractional 0.1 dex scatter ≈ 25% if not provided
        sig = 0.25 * np.maximum(1e-30, gobs)

    from models.plateaus_model import rar_predict_gobs
    gpred = rar_predict_gobs(gbar, a0_m_s2, form)
    chi2 = float(np.sum(((gpred - gobs) / np.clip(sig, 1e-30, None)) ** 2))
    return {"chi2": chi2, "N": int(len(df))}