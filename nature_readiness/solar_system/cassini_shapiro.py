"""Cassini Shapiro delay (non-interactive, reproducible)
Computes the standard Shapiro time delay for a Solar conjunction and shows how an
amplitude rescaling ε≡ξ−1 would project if γ=1, clarifying the GM degeneracy.
No web/API usage here. For any ephemeris data, see nature_readiness/data/README_DATA.md.

This module is code-only; figures/CSV are produced by callers in scripts or notebooks.
"""
from __future__ import annotations
from typing import Dict, Any
import math

C = 299_792_458.0  # m/s
G = 6.67430e-11    # m^3 kg^-1 s^-2
M_SUN = 1.98847e30 # kg
AU = 1.495978707e11


def shapiro_one_way_seconds(r_E_m: float, r_R_m: float, R_m: float, GM_m3s2: float, gamma: float = 1.0) -> float:
    """One-way Shapiro delay in seconds for a static spherical field (PPN, isotropic coords).
    Δt = (1+γ) GM/c^3 * ln((r_E + r_R + R)/(r_E + r_R − R)).
    """
    numer = r_E_m + r_R_m + R_m
    denom = r_E_m + r_R_m - R_m
    if denom <= 0:
        raise ValueError("Invalid geometry: denominator <= 0 in Shapiro formula")
    return (1.0 + float(gamma)) * GM_m3s2 / (C**3) * math.log(numer / denom)


def simulate_cassini_shapiro() -> Dict[str, Any]:
    """Compute two-way (uplink+downlink) Shapiro delay at conjunction.
    Returns baseline GR delay and an apparent-γ shift if an ε-amplitude rescaling
    is present in light propagation but not in GM (non-degenerate case).
    """
    # Rough conjunction geometry (order-of-magnitude): Earth ~1 AU, Saturn ~9.5 AU,
    # impact parameter ~ a few solar radii; we approximate R ≈ r_E + r_R (small angle).
    r_E = 1.0 * AU
    r_R = 9.5 * AU
    # Small-angle approximation: R ≈ r_E + r_R − δ with δ ≪ r_E+r_R to avoid singularity
    delta = 0.01 * AU
    R = r_E + r_R - delta

    GM = G * M_SUN

    # Baseline GR (γ=1), two-way delay (seconds)
    one_way = shapiro_one_way_seconds(r_E, r_R, R, GM, gamma=1.0)
    two_way_gr = 2.0 * one_way

    # Apparent-γ mapping if light path sees (1+ε) GM but ephemerides use GM (non-degenerate case)
    # Then γ_app ≈ 1 + ε when fitting Cassini with fixed GM.
    eps_example = 2.3e-5  # Cassini-level
    two_way_eps = 2.0 * shapiro_one_way_seconds(r_E, r_R, R, GM * (1.0 + eps_example), gamma=1.0)
    gamma_apparent = (two_way_eps / two_way_gr) - 1.0  # ≈ ε

    return {
        "status": "ok",
        "two_way_shapiro_gr_s": two_way_gr,
        "two_way_shapiro_eps_s": two_way_eps,
        "gamma_apparent_for_eps": gamma_apparent,
        "note": (
            "If GM used by ephemerides already includes the same (1+ε), the apparent γ remains 1 (degenerate-amplitude case). "
            "Otherwise Cassini would bound |ε| at the ~|γ−1| level."
        ),
    }


if __name__ == "__main__":
    out = simulate_cassini_shapiro()
    print({k: (round(v, 6) if isinstance(v, float) else v) for k, v in out.items()})

