#!/usr/bin/env python3
"""
energy_coupled_gate.py — Optional Sakharov-style "energy → gravity" coupling
scaffold for the tariff explorer. This module stays confined to /tariff and does
NOT touch core project code.

Formulas
- Energy functional (coarse-grained, minimal version): E_ell ≈ u_gamma (CMB+EBL proxy).
- Gate-to-gate coupling (dimensionless):
    a0_eff(g) = a0 * [ 1 + zeta_energy * (E_ell / E0) * H(g/a0) ] ,
    H(y) = 1 / (1 + y^beta_energy).
- Insert a0_eff into the RAR-plateau xi:
    xi(g) = min[ 1/2 + sqrt(1/4 + a0_eff/g), D_max ].

Notes
- Units for u_gamma and E0 default to eV/cm^3 so their ratio is unitless; no SI
  conversion is required inside this module as long as both are in the same units.
- The caller supplies g in m/s^2 and a0 in m/s^2.
- Safeguards handle g<=0 by returning D_max (strict plateau limit).
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class EnergyCouplingParams:
    enabled: bool = False
    zeta_energy: float = 1.0        # dimensionless coupling strength
    beta_energy: float = 2.0        # shape parameter for H(y)
    u_gamma_evcm3: float = 0.26     # CMB energy density today (eV/cm^3)
    E0_evcm3: float = 0.26          # reference energy density (eV/cm^3)


def H_gate(y: float, beta: float) -> float:
    """H(y) = 1 / (1 + y^beta), clipped to [0, 1]."""
    if not math.isfinite(y) or y <= 0.0:
        return 1.0
    if not math.isfinite(beta) or beta <= 0.0:
        beta = 2.0
    val = 1.0 / (1.0 + (y ** beta))
    # numerical guard
    return float(min(max(val, 0.0), 1.0))


def a0_eff_energy_coupled(g_bar_m_s2: float,
                          a0_m_s2: float,
                          params: EnergyCouplingParams) -> float:
    """Compute a0_eff with energy coupling at a local g_bar.

    a0_eff = a0 * [1 + zeta * (u_gamma/E0) * H(g_bar/a0)]
    """
    if (not params.enabled) or (not math.isfinite(g_bar_m_s2)) or (g_bar_m_s2 < 0.0):
        return float(a0_m_s2)
    # dimensionless energy ratio (same units)
    try:
        E_ratio = float(params.u_gamma_evcm3) / float(params.E0_evcm3) if float(params.E0_evcm3) != 0.0 else 0.0
    except Exception:
        E_ratio = 0.0
    y = (g_bar_m_s2 / float(a0_m_s2)) if float(a0_m_s2) > 0.0 else 1e9
    H = H_gate(y, float(params.beta_energy))
    return float(a0_m_s2) * (1.0 + float(params.zeta_energy) * E_ratio * H)


def xi_rar_plateau_energy_coupled(g_bar_m_s2: float,
                                  a0_m_s2: float,
                                  D_max: float,
                                  params: EnergyCouplingParams) -> float:
    """RAR-plateau xi with energy-coupled a0_eff. Falls back to standard xi if disabled."""
    if not math.isfinite(g_bar_m_s2) or g_bar_m_s2 <= 0.0:
        return float(D_max)
    if (not params.enabled):
        # Standard xi: 0.5 + sqrt(0.25 + a0/g)
        val = 0.5 + math.sqrt(0.25 + float(a0_m_s2) / max(g_bar_m_s2, 1e-300))
        return float(min(val, D_max))
    a0_eff = a0_eff_energy_coupled(g_bar_m_s2, a0_m_s2, params)
    val = 0.5 + math.sqrt(0.25 + a0_eff / max(g_bar_m_s2, 1e-300))
    return float(min(val, D_max))

