#!/usr/bin/env python3
"""
unified_gate_scaffold.py — tariff-only computation scaffold for the Unified Gate Law.

This module implements:
- GateParams: parameters (η, p, q, rho_star, kappa, sigma, enable_backreaction)
- gate_G(y, rho_gamma, params, psi=None): unified gate G(y, rho_gamma)
- tariff_dlnE_dell(y, rho_gamma, params, psi=None): -kappa * (G - 1)
- integrate_tau(path_sampler, params, with_psi=False): integrate τ = ∫ (G-1) kappa dl, optional ψ back-reaction
- calibrate_kappa_to_cmb(f_void, D_LSS_Mpc, G_cap_minus1=1.0): pick kappa so τ ≈ ln(1100)

Notes:
- Option A vs B: The tariff path can be driven either by Option A (RAR-compatible ξ with energy-coupled a0_eff; see tariff/energy_coupled_gate.py) or by this scaffold’s Option B multiplicative gate G(y, ρ_γ). The analyzer frequently uses GateParams (Option B); the CLI tariff model uses Option A when --energy-coupled is enabled. See tariff/README.md for the Model box and Solar-System safety.
- Remains confined to tariff/ and can be imported by energy_tariff_model.py if desired.
- Path sampler is any generator yielding tuples (y, rho_gamma, dl_Mpc) along LOS.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generator, Iterable, Optional, Tuple
import math

@dataclass
class GateParams:
    eta: float = 3.0           # amplitude of enhancement (dimensionless)
    p: float = 1.5             # RAR gate exponent
    q: float = 0.0             # photon-energy gate exponent (default to 0: drop ργ handle by default)
    rho_star_evcm3: float = 0.26  # reference photon energy density (eV/cm^3), ~CMB today
    kappa_per_Mpc: float = 1e-5   # tariff coupling κ in 1/Mpc (small overlay)
    sigma: float = 0.0            # tiny back-reaction strength in f(psi)
    enable_backreaction: bool = False


def _inv_smooth_pow(x: float, expo: float) -> float:
    x = max(float(x), 0.0)
    e = max(float(expo), 0.0)
    return 1.0 / (1.0 + (x ** e)) if e > 0.0 else 1.0


def gate_G(y: float, rho_gamma_evcm3: float, params: GateParams, psi: Optional[float] = None) -> float:
    """Unified gate: G = 1 + η (1+y^p)^-1 (1+(ρ_γ/ρ_*)^q)^-1 [× f(ψ)]."""
    y = max(float(y), 0.0)
    rg = max(float(rho_gamma_evcm3), 0.0)
    y_gate = _inv_smooth_pow(y, params.p)
    e_gate = _inv_smooth_pow(rg / max(params.rho_star_evcm3, 1e-30), params.q)
    G = 1.0 + float(params.eta) * y_gate * e_gate
    if params.enable_backreaction and params.sigma != 0.0 and psi is not None:
        # f(psi) ≈ exp(σ ψ) (stable for small σ)
        G = 1.0 + (G - 1.0) * math.exp(params.sigma * float(psi))
    return max(G, 1.0)


def tariff_dlnE_dell(y: float, rho_gamma_evcm3: float, params: GateParams, psi: Optional[float] = None) -> float:
    """Tariff rate: d ln E / dℓ = -κ [G - 1]."""
    G = gate_G(y, rho_gamma_evcm3, params, psi=psi)
    return -float(params.kappa_per_Mpc) * max(G - 1.0, 0.0)


def integrate_tau(
    path_sampler: Iterable[Tuple[float, float, float]],
    params: GateParams,
    with_psi: bool = False,
    gamma_backreaction: float = 0.0,
) -> Tuple[float, Optional[float]]:
    """
    Integrate τ = ∫ κ [G - 1] dl along the path; optionally accumulate ψ via dψ/dℓ = γ [G - 1].
    path_sampler yields (y, rho_gamma_evcm3, dl_Mpc).
    Returns (tau, psi_final or None).
    """
    tau = 0.0
    psi = 0.0 if (with_psi and params.enable_backreaction and gamma_backreaction != 0.0) else None
    for y, rho_g, dl in path_sampler:
        G = gate_G(y, rho_g, params, psi=psi)
        tau += float(params.kappa_per_Mpc) * max(G - 1.0, 0.0) * float(dl)
        if psi is not None:
            psi += float(gamma_backreaction) * max(G - 1.0, 0.0) * float(dl)
    return float(tau), (float(psi) if psi is not None else None)


def calibrate_kappa_to_cmb(
    f_void: float,
    D_LSS_Mpc: float,
    G_cap_minus1: float = 1.0,
    tau_target: float = math.log(1100.0),
) -> float:
    """
    Back-of-envelope κ calibration to reach τ ≈ ln(1100) along a void-weighted LOS.
    Assumes near-saturated gate along a fraction f_void of the distance to LSS.
    """
    f = min(max(float(f_void), 0.0), 1.0)
    D = max(float(D_LSS_Mpc), 1.0)
    cap = max(float(G_cap_minus1), 1e-9)
    return float(tau_target) / (cap * f * D)


# Example LOS sampler for quick tests

def uniform_void_sampler(y_value: float, rho_gamma_evcm3: float, D_Mpc: float, n_steps: int = 1000):
    dl = float(D_Mpc) / max(int(n_steps), 1)
    for _ in range(max(int(n_steps), 1)):
        yield float(y_value), float(rho_gamma_evcm3), dl
