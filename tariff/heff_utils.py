#!/usr/bin/env python3
"""
heff_utils.py — Helpers for H_eff, D_M, D_H from monotone z(r) with identity checks.

Functions:
- heff_from_z_of_r(r: np.ndarray, z_of_r: np.ndarray) -> dict(z, H_eff, D_M, D_H)
- heff_identity_check(r: np.ndarray, z_of_r: np.ndarray) -> float (RMS of identity)
"""
from __future__ import annotations

import numpy as np
from typing import Dict

C_KM_S = 299_792.458


def heff_from_z_of_r(r: np.ndarray, z_of_r: np.ndarray) -> Dict[str, np.ndarray]:
    r = np.asarray(r, float)
    z = np.asarray(z_of_r, float)
    # Build a strictly monotone z grid (clip and accumulate to enforce monotonicity)
    z_mono = np.maximum.accumulate(np.clip(z, 0.0, None))
    # Define uniform z grid over available range (avoid duplicate endpoints)
    z_grid = np.linspace(float(z_mono[1]), float(z_mono[-1]), 2000) if len(z_mono) > 2 else z_mono
    # r(z) via interpolation
    r_of_z = np.interp(z_grid, z_mono, r)
    # H_eff(z) = c dz/dr = c / (dr/dz)
    dr_dz = np.gradient(r_of_z, z_grid)
    dz_dr = 1.0 / np.clip(dr_dz, 1e-30, np.inf)
    H_eff = C_KM_S * dz_dr
    D_M = r_of_z
    D_H = C_KM_S / np.clip(H_eff, 1e-30, np.inf)
    return {'z': z_grid, 'H_eff': H_eff, 'D_M': D_M, 'D_H': D_H}


def heff_identity_check(r: np.ndarray, z_of_r: np.ndarray) -> float:
    r = np.asarray(r, float)
    z = np.asarray(z_of_r, float)
    # Work on native grid for identity RMS
    dr = np.gradient(r)
    dz = np.gradient(z)
    dz_dr = dz / np.clip(dr, 1e-30, np.inf)
    lhs = C_KM_S * dz_dr
    rhs = C_KM_S * (1.0 + z) * np.gradient(np.log1p(z), r)
    valid = np.isfinite(lhs) & np.isfinite(rhs) & (lhs > 0) & (rhs > 0)
    rel = np.zeros_like(lhs)
    rel[valid] = np.abs(lhs[valid] - rhs[valid]) / np.maximum(lhs[valid], 1e-30)
    return float(np.sqrt(np.mean(rel[valid]**2))) if np.any(valid) else float('nan')
