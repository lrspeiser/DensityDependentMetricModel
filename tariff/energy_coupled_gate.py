#!/usr/bin/env python3
"""
energy_coupled_gate.py — Optional Sakharov-style "energy → gravity" coupling
scaffold for the tariff explorer. This module stays confined to /tariff and does
NOT touch core project code.

Formulas (Option A: energy → gravity via ξ with a0_eff)
- Energy functional (coarse-grained, minimal version): E_ell ≈ u_gamma (CMB+EBL proxy).
- Gate-to-gate coupling (dimensionless):
    a0_eff(g) = a0 * [ 1 + zeta_energy * (E_ell / E0) * H(g/a0) ] ,
    H(y) = 1 / (1 + y^beta_energy).
- Insert a0_eff into the RAR-plateau ξ:
    ξ(g) = min[ 1/2 + sqrt(1/4 + a0_eff/g), D_max ].

Tariff link
- Use the same gate in the photon tariff: d ln E / dℓ = -κ [ξ - 1] f_env(…).
- This reciprocity implements “one gate, two observables.” See tariff/README.md (Model box).

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


def xi_energy_to_gravity(
    g_bar_m_s2: float,
    rho_gamma_evcm3: float,
    a0_m_s2: float = 1.2e-10,
    D_max: float = 30.0,
    params: EnergyCouplingParams | None = None,
) -> float:
    """RAR-compatible energy→gravity mapping (Option A) used by the tariff path when
    energy coupling is enabled.

    Equations (see tariff/README.md “Model box”):
      y = g_bar / a0
      H(y; β) = 1 / (1 + y^β)
      a0_eff = a0 [ 1 + ζ (ρ_γ / E0) H(y; β) ]
      ξ(g_bar, ρ_γ) = min[ 1/2 + sqrt(1/4 + a0_eff/g_bar), D_max ]

    Units:
      - g_bar and a0: m/s^2
      - ρ_γ and E0: eV/cm^3 (ratio is unitless)

    Solar-System screening:
      For y ≫ 1 and β ≳ 2, H → y^{-β} and ξ−1 ≈ 0.5 a0/g (well within Cassini at 1–30 AU).
    """
    # Parameter defaults
    zeta = float(params.zeta_energy) if params is not None else 1.0
    beta = float(params.beta_energy) if params is not None else 2.0
    E0 = float(params.E0_evcm3) if (params is not None and hasattr(params, 'E0_evcm3') and params.E0_evcm3) else 0.26
    a0 = float(a0_m_s2)

    # Guards and deep-void plateau
    g = float(g_bar_m_s2)
    if not math.isfinite(g) or g <= 0.0:
        return float(D_max)

    # H-gate and energy ratio
    y = g / max(a0, 1e-300)
    H = H_gate(y, beta)
    try:
        e_ratio = float(rho_gamma_evcm3) / float(E0) if float(E0) != 0.0 else 0.0
    except Exception:
        e_ratio = 0.0

    # a0_eff and ξ
    a0_eff = a0 * (1.0 + zeta * e_ratio * H)
    val = 0.5 + math.sqrt(0.25 + a0_eff / max(g, 1e-300))
    return float(min(val, D_max))

