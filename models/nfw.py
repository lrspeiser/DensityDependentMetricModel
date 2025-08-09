#!/usr/bin/env python3
"""
Simple NFW halo utilities for SPARC fits.

This is a minimal implementation intended for baseline ΛCDM comparisons in SPARC rotation curves.
It provides:
- v_nfw(R, V200, c): circular speed from an NFW halo
- v_model_gr(R, vbar): GR (baryons-only)
- v_model_nfw(R, vbar, V200, c): total speed from baryons + NFW halo

Units: R in kpc, velocities in km/s.
Assumes a fiducial H0 to convert V200 to R200 via R200 = V200 / (10 H0) in physical units.
For our purposes we use H0 = 70 km/s/Mpc.

Note: This is sufficient for comparative fits; a more careful treatment can expose M200-c relations.
"""
from __future__ import annotations
import numpy as np

H0 = 70.0  # km/s/Mpc
KM_S_PER_MPC = 3.0856775814913673e19  # not used directly, only scaling

# Convert V200 to R200 using V200 = sqrt(G M200/R200) and definition of 200 rho_crit
# For simplicity and to avoid G and rho_crit bookkeeping, we adopt the commonly used
# approximation R200 [kpc] ≈ V200 / (10 H0) with H0 in km/s/Mpc and converted to kpc units.
# 1 Mpc = 1000 kpc, so R200[kpc] ≈ V200 / (10 * H0) * 1000.

def r200_from_v200_kpc(V200: float, h0: float = H0) -> float:
    return float(V200 / (10.0 * h0) * 1000.0)


def v_nfw(R_kpc: np.ndarray, V200: float, c: float, h0: float = H0) -> np.ndarray:
    """
    NFW halo circular velocity profile.
    R_kpc: radii [kpc]
    V200: circular speed at R200 [km/s]
    c: concentration
    Returns v_halo [km/s] at each R.
    """
    R = np.asarray(R_kpc, dtype=float)
    R200 = r200_from_v200_kpc(V200, h0)
    rs = float(R200 / max(1e-6, c))
    x = np.clip(R / max(1e-12, rs), 1e-8, None)
    # Enclosed mass factor f(x) = ln(1+x) - x/(1+x)
    def f(z):
        return np.log1p(z) - z/(1.0+z)
    norm = f(c)
    # v^2 = V200^2 * (1/x) * f(x) / f(c)
    vx2 = (V200**2) * (f(x) / np.clip(x, 1e-12, None)) / max(norm, 1e-12)
    return np.sqrt(np.clip(vx2, 0.0, None))


def v_model_gr(vbar: np.ndarray) -> np.ndarray:
    return np.asarray(vbar, dtype=float)


def v_model_nfw(R_kpc: np.ndarray, vbar: np.ndarray, V200: float, c: float) -> np.ndarray:
    v_h = v_nfw(R_kpc, V200, c)
    vbar = np.asarray(vbar, dtype=float)
    return np.sqrt(np.clip(vbar**2 + v_h**2, 0.0, None))
