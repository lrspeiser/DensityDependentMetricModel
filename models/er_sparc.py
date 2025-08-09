"""
ER-on-SPARC model utilities.

We build v_bar(R) from component rotation curves provided by SPARC/Rotmod:
- Vdisk, Vbulge at mass-to-light ratios Υ_d, Υ_b (default 0.5 and 0.7 typical 3.6μm priors)
  Note: velocities scale as sqrt(Υ). If curves are tabulated for Υ=1, we multiply by sqrt(Υ).
- Vgas used as-is (gas mass fixed by H I).

Then apply a bounded ER enhancement as a function of radius as a proxy for
low-density/tidal band, using a log-normal window in R and a floor w_min:

xi(R) = 1 + lambda_max * W(R),  W(R) = max(w_min, exp(-(ln(R/R0))^2/(2*sigma_lnR^2)))

This is a pragmatic approximation when full 3D densities/tides are not available.
"""
from __future__ import annotations
from typing import Dict, Tuple
import numpy as np


def v_bar_from_components(R_kpc: np.ndarray, Vgas: np.ndarray, Vdisk: np.ndarray, Vbul: np.ndarray,
                          ups_disk: float = 0.5, ups_bul: float = 0.7) -> np.ndarray:
    sd = float(max(ups_disk, 0.0)) ** 0.5
    sb = float(max(ups_bul, 0.0)) ** 0.5
    v2 = (Vgas)**2 + (sd * Vdisk)**2 + (sb * Vbul)**2
    return np.sqrt(np.clip(v2, 0.0, None))


def xi_log_normal_R(R_kpc: np.ndarray, lambda_max: float = 3.0, R0_kpc: float = 15.0,
                    sigma_lnR: float = 0.6, w_min: float = 0.0) -> np.ndarray:
    R = np.asarray(R_kpc, dtype=float)
    lm = float(max(lambda_max, 0.0))
    sig = max(float(sigma_lnR), 1e-3)
    wmin = float(np.clip(w_min, 0.0, 1.0))
    with np.errstate(divide='ignore'):
        ln = np.log(np.clip(R / max(R0_kpc, 1e-3), 1e-9, None))
    W = np.exp(-0.5 * (ln / sig) ** 2)
    W = np.maximum(W, wmin)
    return 1.0 + lm * np.clip(W, 0.0, 1.0)


def v_er_from_components(R_kpc: np.ndarray, Vgas: np.ndarray, Vdisk: np.ndarray, Vbul: np.ndarray,
                         ups_disk: float, ups_bul: float,
                         lambda_max: float, R0_kpc: float, sigma_lnR: float, w_min: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    vbar = v_bar_from_components(R_kpc, Vgas, Vdisk, Vbul, ups_disk, ups_bul)
    xi = xi_log_normal_R(R_kpc, lambda_max, R0_kpc, sigma_lnR, w_min)
    ver = np.sqrt(np.clip(xi, 0.0, None)) * vbar
    return vbar, xi, ver
