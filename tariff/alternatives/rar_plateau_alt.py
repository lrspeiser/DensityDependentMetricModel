#!/usr/bin/env python3
"""
rar_plateau_alt.py — Self-contained SVT-inspired alternative: quasi-static 'dielectric' plateau model.

This does NOT modify the main pipeline. It provides:
- A late-time, quasi-static modified Poisson solver in 1D spherical symmetry that yields a plateau boost.
- A smooth 'gate' S(a) that deactivates the modification at early times (for narrative completeness).
- Convenience functions to build synthetic g_bar(r), solve for g_obs(r), and map to a RAR-like curve.

Physics model (quasi-static):
    div[(1 + chi(g)) grad Phi] = 4 pi G_N rho_b
with
    chi(g) = (a_p / g) / (1 + (g / g_t)^m)
so that for g << g_t: g_obs ≈ g_N + a_p (a plateau), and for g >> g_t: g_obs ≈ g_N.

This is a toy solver for demonstration/figure generation only.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Tuple

G_N = 4.3009172700363e-6  # [kpc (km/s)^2 / Msun] convenient astro units

@dataclass
class PlateauParams:
    a_p: float = 1.2e-10      # [m s^-2] plateau acceleration ~ RAR floor
    g_t: float = 1.2e-10      # [m s^-2] turnover scale ~ a0
    m: float = 3.0            # sharpness of transition


def chi_of_g(g_si: np.ndarray, p: PlateauParams) -> np.ndarray:
    g = np.maximum(g_si, 1e-30)
    return (p.a_p / g) / (1.0 + (g / p.g_t) ** p.m)


def gate_S_of_a(a: float, rho_star_evcm3: float = 0.26, n: float = 4.0) -> float:
    """Early-time gate S(a) ~ [1 + (rho_bg/rho_*)^n]^-1. Here rho_bg ∝ a^-4 (radiation proxy).
    This is illustrative; not used in the static solver, but kept for completeness.
    """
    a = float(a)
    a = np.clip(a, 1e-6, 1.0)
    rho_bg = (1.0 / a) ** 4
    return 1.0 / (1.0 + (rho_bg / 1.0) ** n)


def synthetic_gbar_profile(r_kpc: np.ndarray, M_baryon_Msun: float = 5e10, r_scale_kpc: float = 3.0) -> np.ndarray:
    """Simple spherical mass model -> g_N(r) = G_N M(<r)/r^2. Hernquist-like cumulative mass.
    Return g_bar in SI [m s^-2].
    """
    M = M_baryon_Msun
    rs = r_scale_kpc
    # Hernquist cumulative mass: M(<r) = M * r^2 / (r+rs)^2
    r = np.maximum(r_kpc, 1e-6)
    Mcum = M * (r**2) / (r + rs) ** 2
    # g_N in astro units: (km/s)^2 per kpc; convert to m/s^2: 1 (km/s)^2/kpc = (1e6 m^2/s^2) / (3.08567758e19 m) = 3.2407793e-14 m/s^2
    gN_astro = G_N * Mcum / (r**2)
    gN_si = gN_astro * 3.2407793e-14
    return gN_si


def solve_plateau_gobs(g_bar_si: np.ndarray, p: PlateauParams) -> np.ndarray:
    """Local algebraic closure for the toy dielectric: g_obs ≈ g_bar + a_p / (1 + (g_obs/g_t)^m).
    Solve per-point with a fixed-point iteration (robust for demo)."""
    g_bar = np.asarray(g_bar_si, float)
    g = np.copy(g_bar)
    for _ in range(200):
        denom = 1.0 + (np.maximum(g, 1e-30) / p.g_t) ** p.m
        g_new = g_bar + p.a_p / denom
        if np.allclose(g_new, g, rtol=1e-6, atol=1e-14):
            break
        g = g_new
    return g


def build_rar_curve(r_kpc: np.ndarray, M_baryon_Msun: float, p: PlateauParams) -> Tuple[np.ndarray, np.ndarray]:
    g_bar = synthetic_gbar_profile(r_kpc, M_baryon_Msun=M_baryon_Msun)
    g_obs = solve_plateau_gobs(g_bar, p)
    return g_bar, g_obs

if __name__ == "__main__":
    # quick sanity run
    r = np.logspace(-1, 2.5, 200)
    g_bar, g_obs = build_rar_curve(r, 5e10, PlateauParams())
    print(g_bar[:3], g_obs[:3])
