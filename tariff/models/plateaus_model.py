# models/plateaus_model.py
# Core cosmology + observables for the three-regime "plateaus" model.

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Tuple, Literal
import numpy as np

C_LIGHT = 299792.458  # km/s
LN10 = np.log(10.0)


# -----------------------------
# Parameters
# -----------------------------
@dataclass
class Cosmology:
    H0: float = 70.0                 # km/s/Mpc
    Tcmb: float = 2.7255             # K
    Neff: float = 3.046              # Effective relativistic dof
    omega_b: float = 0.02237         # Ω_b h^2 (baryon density)
    flat: bool = False               # if False, Ω_k0 = 1 - (Ω_b+Ω_r) at a=1 (Λ=0)


@dataclass
class PlateausParams:
    # Regime 1: Early-time temporary boost to G
    G_eff_boost: float = 1.30        # multiplier at very early times
    a_trans: float = 1.0e-4          # transition scale factor (boost fades near this a)
    n_trans: float = 4.0             # transition steepness (higher = sharper)
    z_star_override: float | None = None  # optional fixed z_* (e.g. 1089.9). None = use HS96 approx.

    # Regime 2: MOND/RAR gate for galaxies
    a0_m_s2: float = 1.2e-10         # MOND scale [m/s^2]
    nu_form: Literal["simple", "standard"] = "standard"

    # Regime 3: Photon energy-loss dimming
    kappa_per_Mpc: float = 0.0       # LOS opacity per Mpc (dimensionless per Mpc)


# -----------------------------
# Helpers: radiation densities today
# -----------------------------
def omega_gamma_h2(Tcmb: float = 2.7255) -> float:
    # Ω_γ h^2 ≈ 2.4728e-5 (Tcmb/2.7255)^4
    return 2.4728e-5 * (Tcmb / 2.7255) ** 4


def omega_nu_h2(Neff: float, Tcmb: float = 2.7255) -> float:
    # Ω_ν h^2 = Ω_γ h^2 * 0.22713 * N_eff
    return omega_gamma_h2(Tcmb) * 0.22713 * Neff


# -----------------------------
# Early-time G gate (Regime 1)
# -----------------------------
def g_gate(a: float, boost: float, a_trans: float, n_trans: float) -> float:
    """
    Smooth gate: g(a)=1+(boost-1)*S(a), S(a) = 1 / (1 + (a/a_trans)^n_trans).
    Early (a << a_trans): g -> boost. Late (a >> a_trans): g -> 1.
    """
    S = 1.0 / (1.0 + (a / a_trans) ** n_trans)
    return 1.0 + (boost - 1.0) * S


# -----------------------------
# Expansion history with variable G
# -----------------------------
class PlateausBackground:
    def __init__(self, cosmo: Cosmology, params: PlateausParams):
        self.cosmo = cosmo
        self.params = params

        h = cosmo.H0 / 100.0
        self.Om_b = cosmo.omega_b / h**2
        self.Om_r = (omega_gamma_h2(cosmo.Tcmb) + omega_nu_h2(cosmo.Neff, cosmo.Tcmb)) / h**2

        # No Λ, no CDM. Close the budget with curvature unless flat=True.
        if cosmo.flat:
            self.Om_k = 0.0
        else:
            self.Om_k = 1.0 - (self.Om_b + self.Om_r)

    def E(self, a: float) -> float:
        """Dimensionless H/H0 with variable G(a) on (b + r)."""
        g = g_gate(a, self.params.G_eff_boost, self.params.a_trans, self.params.n_trans)
        return np.sqrt(
            g * (self.Om_b * a**-3 + self.Om_r * a**-4) + self.Om_k * a**-2
        )

    # Comoving distance χ and transverse comoving D_M (curvature-aware)
    def _integrand_chi(self, a: np.ndarray) -> np.ndarray:
        return 1.0 / (a**2 * self.E(a))

    def comoving_distance(self, z: float) -> float:
        a = 1.0 / (1.0 + z)
        a_grid = np.geomspace(a, 1.0, 4096)
        chi = (C_LIGHT / self.cosmo.H0) * np.trapz(self._integrand_chi(a_grid), a_grid)  # [Mpc]
        return chi

    def D_M(self, z: float) -> float:
        chi = self.comoving_distance(z)
        Ok = self.Om_k
        if abs(Ok) < 1e-12:
            return chi
        x = np.sqrt(abs(Ok)) * self.cosmo.H0 * chi / C_LIGHT
        if Ok > 0:
            return (C_LIGHT / self.cosmo.H0) * np.sinh(x) / np.sqrt(Ok)
        else:
            return (C_LIGHT / self.cosmo.H0) * np.sin(x) / np.sqrt(-Ok)

    def D_L(self, z: float) -> float:
        return (1.0 + z) * self.D_M(z)

    # Sound speed and horizons
    def _R(self, a: float) -> float:
        # R(a) = 3ρ_b/(4ρ_γ) = (3/4)*(Ω_b/Ω_γ)*a
        h = self.cosmo.H0 / 100.0
        Ogamma = omega_gamma_h2(self.cosmo.Tcmb) / h**2
        return 0.75 * (self.Om_b / Ogamma) * a

    def _cs_over_a2E(self, a: np.ndarray) -> np.ndarray:
        R = self._R(a)
        cs_c = 1.0 / np.sqrt(3.0 * (1.0 + R))    # c_s / c
        return cs_c / (a**2 * self.E(a))

    def sound_horizon(self, z: float) -> float:
        """r_s(z) = (c/H0) ∫_0^{a(z)} [c_s/(a^2 E(a))] da."""
        a = 1.0 / (1.0 + z)
        a_grid = np.geomspace(1e-8, a, 8192)
        rs = (C_LIGHT / self.cosmo.H0) * np.trapz(self._cs_over_a2E(a_grid), a_grid)
        return rs

    # z_star (last scattering) — Hu & Sugiyama 1996 approx, with ω_m ≡ ω_b here
    def z_star(self) -> float:
        if self.params.z_star_override is not None:
            return self.params.z_star_override
        h = self.cosmo.H0 / 100.0
        wb = self.cosmo.omega_b
        wm = self.cosmo.omega_b  # CDM-free model; use baryons as "matter" in HS96
        g1 = 0.0783 * wb**-0.238 / (1.0 + 39.5 * wb**0.763)
        g2 = 0.560 / (1.0 + 21.1 * wb**1.81)
        return 1048.0 * (1.0 + 0.00124 * wb**-0.738) * (1.0 + g1 * wm**g2)

    def ell_A(self) -> float:
        zstar = self.z_star()
        DM = self.D_M(zstar)
        rs = self.sound_horizon(zstar)
        return np.pi * DM / rs


# -----------------------------
# Regime 3: SN Ia with κ-dimming
# -----------------------------
def distance_modulus_mu(DL_Mpc: np.ndarray) -> np.ndarray:
    # μ = 5 log10(DL/Mpc) + 25
    return 5.0 * np.log10(np.clip(DL_Mpc, 1e-30, None)) + 25.0


def delta_mu_kappa(chi_Mpc: np.ndarray, kappa_per_Mpc: float) -> np.ndarray:
    # Flux attenuation F -> F * exp(-τ), τ = κ χ; DL^eff = DL * e^{τ/2}; μ shift = (5/2) log10(e) τ
    return (2.5 / LN10) * kappa_per_Mpc * chi_Mpc


# -----------------------------
# Regime 2: MOND/RAR gate
# -----------------------------
def nu_interpolating(y: np.ndarray, form: Literal["simple", "standard"]) -> np.ndarray:
    """
    ν(y) where y = g_bar / a0. Predict g_tot = ν(y) * g_bar.
    - "simple":  ν = 0.5 + sqrt(0.25 + 1/y)
    - "standard": ν = 0.5 + 0.5*sqrt(1 + 4/y)
    """
    y = np.clip(y, 1e-30, None)
    if form == "simple":
        return 0.5 + np.sqrt(0.25 + 1.0 / y)
    else:  # "standard"
        return 0.5 + 0.5 * np.sqrt(1.0 + 4.0 / y)


def rar_predict_gobs(gbar: np.ndarray, a0_m_s2: float, form: Literal["simple", "standard"]) -> np.ndarray:
    y = gbar / a0_m_s2
    return nu_interpolating(y, form) * gbar