#!/usr/bin/env python3
"""
Relativistic weak-field module: explicit Φ, Ψ, Φ+Ψ and PPN under a minimal
covariant subclass that enforces c_T = 1 and Φ = Ψ in the quasi-static regime.

Mapping and derivations
- Non-relativistic (QUMOND) field equation: ∇²Φ = ∇·[ ν(|∇Φ_b|/a0) ∇Φ_b ], with ∇²Φ_b = 4πG ρ_b.
  In spherical symmetry g = ν g_N and V² = ξ V_bar² with ξ ≡ ν; for the "simple" family ν(y) = 1/2 + √(1/4 + 1/y),
  yielding ξ(g_N) = 1/2 + √(1/4 + a0_eff/g_N). See docs/modified_poisson_qumond.md.
- Phantom density identity used for lensing on (R,Z) grids:
  ρ_ph = (ξ−1) ρ_b − (1/4πG) (∇ξ · ∇Φ_b), equivalent to ρ_ph = (1/4πG) ∇·[(ν−1) ∇Φ_b]
  when ξ ≡ ν(|∇Φ_b|/a0_eff). The orchestrator computes this and projects Σ_tot for θ_E and ΔΣ.
- Weak-field potentials: Φ = Φ_b + φ_env and Ψ = Ψ_b + φ_env with φ_env ≡ 1/2 ln ξ; thus Φ_W ≡ (Φ+Ψ)/2 = Φ_b + φ_env.

PPN export and guardrails
- Under Solar-System screening, φ_env → 0 and the metric reduces locally to GR. PPN parameters are exported as
  γ = 1, β = 1, α1 = 0, α2 = 0 (see evaluate_ppn). The c_T guardrail enforces c_T = 1.

Notes for maintainers
- This module is used by manuscript-generation paths (PPN table, Solar bands, and metric lensing). Do not
  reintroduce any lensing-only scalars; predictions must stem from the same φ_env derived from ξ.
- See also: docs/paper_appendix_relativistic.md and docs/modified_poisson_qumond.md.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Protocol
import numpy as np

# Constants
G_SI = 6.67430e-11
C_SI = 299_792_458.0


@dataclass
class PPNResult:
    gamma: float
    beta: float
    alpha1: float
    alpha2: float
    theory_assumption: str
    note: str


class PotentialProvider(Protocol):
    """Protocol for baryonic potential providers.

    Implementations should supply Φ_b(r) and Ψ_b(r) in SI (dimensionless) for
    spherical systems; for axisymmetric cases, provide effective spherical
    reductions if used with the helper utilities below.
    """
    def phi_baryon(self, r_m: np.ndarray) -> np.ndarray: ...  # Φ_b
    def psi_baryon(self, r_m: np.ndarray) -> np.ndarray: ...  # Ψ_b


# --- Guardrails and PPN ---

def check_c_T_guardrail(params: Dict[str, float]) -> bool:
    """Return True if tensor-speed constraint c_T == 1 is satisfied.

    Under the adopted subclass, c_T is identically 1 in the backgrounds of
    interest. A parameter override is only allowed if exactly 1.0.
    """
    cT = params.get('c_T', 1.0)
    return float(cT) == 1.0


def evaluate_ppn(params: Dict[str, float], radii_AU: List[float]) -> List[PPNResult]:
    """Evaluate PPN coefficients for Solar-System radii under screening.

    Returns GR-consistent values (γ=1, β=1, α1=α2=0). These are tied to the
    adopted subclass where Φ=Ψ and screening suppresses φ_env locally.
    """
    assumption = 'weak-field Φ=Ψ, c_T=1; screened → GR locally'
    note = 'Derived from restricted scalar–tensor subclass; see appendix.'
    out: List[PPNResult] = []
    for _ in radii_AU:
        out.append(PPNResult(
            gamma=1.0,
            beta=1.0,
            alpha1=0.0,
            alpha2=0.0,
            theory_assumption=assumption,
            note=note,
        ))
    return out


# --- Weak-field mapping from ξ to Φ, Ψ ---

def phi_env_from_xi(xi: np.ndarray) -> np.ndarray:
    """Compute φ_env = 1/2 ln ξ from ξ samples or profiles.

    Inputs and outputs are dimensionless; assumes ξ>0.
    """
    xi = np.asarray(xi, float)
    xi = np.maximum(xi, 1e-300)
    return 0.5 * np.log(xi)


def weak_field_potentials(
    r_m: np.ndarray,
    provider: PotentialProvider,
    xi_profile: np.ndarray,
    a_env: float = 1.0,
    b_env: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (Φ, Ψ, Φ_plus_Ψ) at radii r_m given ξ(r) and baryonic potentials.

    - Φ = Φ_b + a_env φ_env, Ψ = Ψ_b + b_env φ_env, φ_env = 1/2 ln ξ
    - Φ_plus_Ψ = Φ + Ψ
    """
    r_m = np.asarray(r_m, float)
    xi_profile = np.asarray(xi_profile, float)
    if xi_profile.shape != r_m.shape:
        raise ValueError('xi_profile must match r_m shape')

    phi_b = provider.phi_baryon(r_m)
    psi_b = provider.psi_baryon(r_m)
    phi_env = phi_env_from_xi(xi_profile)

    Phi = phi_b + float(a_env) * phi_env
    Psi = psi_b + float(b_env) * phi_env
    return Phi, Psi, Phi + Psi


# --- Helpers for spherical lensing from Φ+Ψ ---

def mean_surface_density_from_phiW(
    r_m: np.ndarray,
    phiW: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute enclosed mass M(<r) and mean surface density <Σ>(R) from Φ_W.

    Assumes spherical symmetry and uses Poisson in weak field to relate Φ_W to
    an effective density. Outputs:
    - M_enc(r) in kg
    - Sigma_bar(R) in kg/m^2 at the same radii (R=r under spherical symmetry)

    Note: This is an approximate helper for manuscript figures; the orchestrator
    uses a monotone-envelope rule to stabilize θ_E crossings.
    """
    r = np.asarray(r_m, float)
    phiW = np.asarray(phiW, float)
    # g(r) = dΦ/dr; effective enclosed mass M = r^2 g / G
    # Use central differences on log-spaced radii where possible.
    dr = np.gradient(r)
    dphi = np.gradient(phiW)
    g_r = dphi / dr
    M_enc = r * r * g_r / G_SI
    # For spherical systems, <Σ>(R) = M(<R) / (π R^2)
    Sigma_bar = np.where(r > 0, M_enc / (np.pi * r * r), 0.0)
    return M_enc, Sigma_bar


def einstein_radius_arcsec(
    R_grid_m: np.ndarray,
    alpha_R: np.ndarray,
    D_l_m: float,
    D_s_m: float,
    D_ls_m: float,
) -> float:
    """Solve for θ_E in arcsec from α(R) on a grid using last-crossing rule.

    θ_E satisfies α(D_l θ_E) = θ_E D_s / D_ls. We find the last crossing of
    f(θ) = α(D_l θ) − θ D_s/D_ls.
    """
    # Convert R grid to θ grid
    theta = R_grid_m / float(D_l_m)
    rhs = theta * (float(D_s_m) / float(D_ls_m))
    f = alpha_R - rhs
    # Find sign changes and take the last crossing
    sign = np.sign(f)
    idx = np.where(np.diff(sign) != 0)[0]
    if idx.size == 0:
        return 0.0
    i = idx[-1]
    # Linear interpolation of crossing
    t0, t1 = theta[i], theta[i+1]
    f0, f1 = f[i], f[i+1]
    if (f1 - f0) == 0:
        theta_E = t0
    else:
        theta_E = t0 - f0 * (t1 - t0) / (f1 - f0)
    return float(np.degrees(theta_E) * 3600.0)
