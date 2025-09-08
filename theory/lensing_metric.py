#!/usr/bin/env python3
"""
Metric-based lensing utilities from Φ+Ψ under the weak-field relativistic mapping.

This module computes ΔΣ(R) and α(R) from the Weyl potential Φ_W = (Φ+Ψ)/2,
then provides θ_E using a last-crossing rule.

References
- docs/paper_appendix_relativistic.md for the underlying subclass and mapping.
- We assume spherical symmetry for these helpers. The orchestrator applies a
  monotone-envelope stabilization for Σ̄(R) before solving for θ_E.
"""
from __future__ import annotations
import numpy as np
from typing import Tuple

from .relativistic import (
    weak_field_potentials,
    mean_surface_density_from_phiW,
    phi_env_from_xi,
    G_SI,
    C_SI,
)


def delta_sigma_from_phiW(
    R_m: np.ndarray,
    phiW: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Σ̄(R) and ΔΣ(R) from Φ_W on a radial grid R_m (meters).

    Returns (Sigma_bar, DeltaSigma) in kg/m^2.
    """
    _, Sigma_bar = mean_surface_density_from_phiW(R_m, phiW)
    # Σ(R) via Abel trick is noisy; for manuscript we approximate ΔΣ ≈ Σ̄ − Σ using
    # a smoothed derivative. For stability, use local slope to estimate Σ.
    # d ln Σ̄ / d ln R ~ slope; set Σ ≈ Σ̄ * (1 − slope) as a crude proxy.
    eps = 1e-12
    lnR = np.log(np.maximum(R_m, eps))
    lnS = np.log(np.maximum(Sigma_bar, eps))
    slope = np.gradient(lnS) / np.gradient(lnR)
    Sigma = np.clip(Sigma_bar * (1.0 - slope), 0.0, np.inf)
    DeltaSigma = np.clip(Sigma_bar - Sigma, 0.0, np.inf)
    return Sigma_bar, DeltaSigma


def deflection_angle_from_phiW(
    R_m: np.ndarray,
    Menc_kg: np.ndarray,
) -> np.ndarray:
    """Compute α(R) from enclosed mass as α = 4 G M(<R)/(c^2 R).

    This uses the GR-like expression with the effective enclosed mass implied by
    Φ_W. Units: α in radians.
    """
    R = np.asarray(R_m, float)
    M = np.asarray(Menc_kg, float)
    alpha = np.zeros_like(R)
    m = R > 0
    alpha[m] = 4.0 * G_SI * M[m] / (C_SI**2 * R[m])
    return alpha

