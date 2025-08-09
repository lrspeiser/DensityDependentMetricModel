"""
Core ER environment functions: S_rho(ρ), W(T), and xi(ρ, T), plus simple proxies
for per-galaxy analyses when full 3D tidal tensors are not available.

This module matches the paper's bounded ER form:
  xi(ρ, T) = 1 + λ_max * S_rho(ρ) * W(T), with 0 ≤ S_rho, W ≤ 1.

- S_rho(ρ) = 1 / (1 + (ρ/ρ_c)^γ_exp)
- W(T)     = max(w_min, exp(-(ln(T/T0))^2 / (2 σ_lnT^2)))

It also provides a simple tidal proxy based on the radial gradient of the
baryonic acceleration g_bar(R) = v_bar(R)^2 / R:
  T_proxy(R) = | d/dR g_bar(R) | normalized by its median over data radii.

These proxies allow per-galaxy ER analyses using SPARC Rotmod component curves
and midplane densities derived from HI and 3.6μm surface brightness files.
"""
from __future__ import annotations
from typing import Tuple
import numpy as np


def S_rho_powerlaw(rho: np.ndarray, rho_c: float, gamma_exp: float) -> np.ndarray:
    rho = np.asarray(rho, dtype=float)
    rho_c = max(float(rho_c), 1e-30)
    gamma = max(float(gamma_exp), 1e-9)
    with np.errstate(over='ignore', invalid='ignore'):
        x = (rho / rho_c) ** gamma
    return 1.0 / (1.0 + np.clip(x, 0.0, np.inf))


def W_log_normal(T: np.ndarray, T0: float, sigma_lnT: float, w_min: float) -> np.ndarray:
    T = np.asarray(T, dtype=float)
    T0 = max(float(T0), 1e-12)
    sig = max(float(sigma_lnT), 1e-6)
    wmin = float(np.clip(w_min, 0.0, 1.0))
    with np.errstate(divide='ignore', invalid='ignore'):
        ln = np.log(np.clip(T / T0, 1e-300, np.inf))
    W = np.exp(-0.5 * (ln / sig) ** 2)
    return np.maximum(np.clip(W, 0.0, 1.0), wmin)


def xi_env(rho: np.ndarray,
           T: np.ndarray,
           lambda_max: float,
           rho_c: float,
           gamma_exp: float,
           T0: float,
           sigma_lnT: float,
           w_min: float) -> np.ndarray:
    rho = np.asarray(rho, dtype=float)
    T = np.asarray(T, dtype=float)
    if rho.shape != T.shape:
        raise ValueError(f"rho and T must have the same shape, got {rho.shape} vs {T.shape}")
    lam = float(max(lambda_max, 0.0))
    S = S_rho_powerlaw(rho, rho_c, gamma_exp)
    W = W_log_normal(T, T0, sigma_lnT, w_min)
    xi = 1.0 + lam * (S * W)
    return np.clip(xi, 1.0, 1.0 + lam)


def finite_diff(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """First derivative dy/dx with robust edge handling (central, forward/backward)."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if y.size != x.size or y.size < 2:
        return np.zeros_like(y)
    dydx = np.empty_like(y)
    # central diffs
    dx = x[2:] - x[:-2]
    dx = np.where(dx == 0, 1e-30, dx)
    dydx[1:-1] = (y[2:] - y[:-2]) / dx
    # edges
    dx0 = x[1] - x[0]
    dxe = x[-1] - x[-2]
    dydx[0] = (y[1] - y[0]) / (dx0 if dx0 != 0 else 1e-30)
    dydx[-1] = (y[-1] - y[-2]) / (dxe if dxe != 0 else 1e-30)
    return dydx


def tidal_proxy_from_vbar(R_kpc: np.ndarray, vbar_kms: np.ndarray) -> np.ndarray:
    """
    Construct a simple tidal indicator proxy T(R) from the baryonic acceleration
    g_bar(R) = v_bar(R)^2 / R. Use the magnitude of its radial gradient:
        T_raw = | d/dR g_bar(R) |
    and normalize by the median value over the sampled radii to make it dimensionless
    and O(1). If median is 0, fallback to max or 1.
    """
    R = np.asarray(R_kpc, dtype=float)
    vb = np.asarray(vbar_kms, dtype=float)
    R_safe = np.clip(R, 1e-6, None)
    g_bar = (np.maximum(vb, 0.0)**2) / R_safe
    dg_dR = finite_diff(g_bar, R_safe)
    T_raw = np.abs(dg_dR)
    med = float(np.nanmedian(T_raw))
    if not np.isfinite(med) or med <= 0:
        mx = float(np.nanmax(T_raw))
        scale = mx if (np.isfinite(mx) and mx > 0) else 1.0
    else:
        scale = med
    return T_raw / scale


def predict_v_er(R_kpc: np.ndarray,
                 vbar_kms: np.ndarray,
                 rho_mid_Msun_kpc3: np.ndarray,
                 lambda_max: float,
                 rho_c: float,
                 gamma_exp: float,
                 T0: float,
                 sigma_lnT: float,
                 w_min: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given R, baryonic circular speed vbar(R), and a midplane density profile ρ_mid(R),
    build a tidal proxy T(R), compute xi(ρ,T), and return (xi, v_model=√xi vbar).
    """
    T = tidal_proxy_from_vbar(R_kpc, vbar_kms)
    xi = xi_env(rho_mid_Msun_kpc3, T, lambda_max, rho_c, gamma_exp, T0, sigma_lnT, w_min)
    v_model = np.sqrt(np.clip(xi, 0.0, None)) * np.maximum(vbar_kms, 0.0)
    return xi, v_model

