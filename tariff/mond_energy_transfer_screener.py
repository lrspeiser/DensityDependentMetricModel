# mond_energy_transfer_screener.py
# Add a temporary early-universe G boost and scan it to match ℓ_A ≈ 301.
# Then (optionally) evaluate SNe, BAO, and RAR under the same parameter set.

from __future__ import annotations
import argparse, json
import numpy as np
import pandas as pd

from models.plateaus_model import Cosmology, PlateausParams, PlateausBackground
from data_io.loaders import (
    load_cmb_ellA_constraint, load_sne_table, maybe_load_covariance,
    load_bao_table, load_rar_table
)
from fits.likelihoods import chi2_cmb_ellA, chi2_sne, chi2_bao, chi2_rar


def scan_G_boost_for_ellA(
    cosmo: Cosmology,
    base_params: PlateausParams,
    ellA_obs: float,
    sigma_ellA: float,
    boosts: np.ndarray
) -> dict:
    rows = []
    for b in boosts:
        params = PlateausParams(**{**base_params.__dict__, "G_eff_boost": float(b)})
        bg = PlateausBackground(cosmo, params)
        out = chi2_cmb_ellA(bg, ellA_obs, sigma_ellA)
        rows.append({"G_eff_boost": float(b), **out})
    rows = sorted(rows, key=lambda r: r["chi2"])
    best = rows[0]
    return {"grid": rows, "best": best}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--H0", type=float, default=70.0)
    ap.add_argument("--omega_b", type=float, default=0.02237)
    ap.add_argument("--Neff", type=float, default=3.046)
    ap.add_argument("--Tcmb", type=float, default=2.7255)
    ap.add_argument("--flat", action="store_true", help="enforce Ω_k=0 at a=1 (no Λ)")

    ap.add_argument("--a_trans", type=float, default=1.0e-4)
    ap.add_argument("--n_trans", type=float, default=4.0)
    ap.add_argument("--G_eff_boost_init", type=float, default=1.30)
    ap.add_argument("--G_eff_boost_max", type=float, default=1.50)
    ap.add_argument("--G_eff_boost_steps", type=int, default=21)
    ap.add_argument("--z_star_override", type=float, default=None)

    ap.add_argument("--a0_m_s2", type=float, default=1.2e-10)
    ap.add_argument("--nu_form", type=str, default="standard", choices=["standard", "simple"])
    ap.add_argument("--kappa_per_Mpc", type=float, default=0.0)

    ap.add_argument("--sne_csv", type=str, default="")
    ap.add_argument("--bao_csv", type=str, default="")
    ap.add_argument("--rar_csv", type=str, default="")
    ap.add_argument("--sn_fit_method", type=str, default="anchored_fullcov")
    ap.add_argument("--ellA_value", type=float, default=301.0)
    ap.add_argument("--ellA_sigma", type=float, default=0.1)

    ap.add_argument("--json_out", type=str, default="")
    args = ap.parse_args()

    cosmo = Cosmology(H0=args.H0, omega_b=args.omega_b, Neff=args.Neff, Tcmb=args.Tcmb, flat=args.flat)
    params = PlateausParams(
        G_eff_boost=args.G_eff_boost_init,
        a_trans=args.a_trans, n_trans=args.n_trans,
        z_star_override=args.z_star_override,
        a0_m_s2=args.a0_m_s2, nu_form=args.nu_form,
        kappa_per_Mpc=args.kappa_per_Mpc
    )

    ellA_obs, ellA_sig = load_cmb_ellA_constraint(args.ellA_value, args.ellA_sigma)
    boosts = np.linspace(args.G_eff_boost_init, args.G_eff_boost_max, args.G_eff_boost_steps)

    # --- Regime 1: find boost that matches ℓ_A
    scan = scan_G_boost_for_ellA(cosmo, params, ellA_obs, ellA_sig, boosts)
    best_params = PlateausParams(**{**params.__dict__, "G_eff_boost": scan["best"]["ellA_th"] * 0.0 + scan["best"]["G_eff_boost"]})
    bg_best = PlateausBackground(cosmo, best_params)

    result = {"ellA_scan": scan, "best_params": best_params.__dict__}

    # --- Optional: Regime 3 SNe
    if args.sne_csv:
        sne = load_sne_table(args.sne_csv)
        cov = maybe_load_covariance(args.sne_csv)
        sn_out = chi2_sne(bg_best, sne, cov, best_params.kappa_per_Mpc, method=args.sn_fit_method)
        result["sne"] = sn_out

    # --- Optional: BAO
    if args.bao_csv:
        bao = load_bao_table(args.bao_csv)
        bao_out = chi2_bao(bg_best, bao)
        result["bao"] = bao_out

    # --- Optional: Regime 2 RAR
    if args.rar_csv:
        rar = load_rar_table(args.rar_csv)
        rar_out = chi2_rar(rar, best_params.a0_m_s2, best_params.nu_form)
        result["rar"] = rar_out

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(result, f, indent=2, default=float)
    else:
        print(json.dumps(result, indent=2, default=float))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Quick MOND+Energy-Transfer screener for SN distances and CMB acoustic scale.

- Background: flat FRW in GR with (Ω_b, Ω_cdm=0 by default, Ω_Λ, Ω_r).
- Traditional MOND is reserved for galaxy dynamics (not used in background).
- Energy transfer (photon energy non-conservation) along the line of sight:
    tau(z) = ∫ κ0 (1+z')^n [c/H(z')] dz'      (comoving-Mpc based)
    Δμ(z)  = 1.086 * tau(z)

Outputs:
- Luminosity distance d_L(z) and shifted distance modulus μ_th(z)+Δμ(z)
- CMB acoustic scale ℓ_A ≈ π D_A(z_*) / r_s(z_*)
- Sound horizon r_d ≡ r_s(z_drag)
- Optional coarse parameter scan

Author: You
"""

from dataclasses import dataclass
from math import sqrt, log, pi
import numpy as np

# ------------------------- Physical constants -------------------------
c_km_s = 299792.458         # speed of light km/s
T_CMB = 2.7255              # K
# Ω_γ h^2 ≈ 2.469e-5 (T/2.7255)^4
def omega_gamma_h2(T=T_CMB):
    return 2.469e-5*(T/2.7255)**4

# ------------------------- Utilities: safe integrators -----------------
def simpson_integral(func, a, b, n=4096):
    """Simple Simpson integration over [a,b] with n even subintervals."""
    if n % 2 == 1:
        n += 1
    x = np.linspace(a, b, n+1)
    y = func(x)
    S = y[0] + y[-1] + 4.0*np.sum(y[1:-1:2]) + 2.0*np.sum(y[2:-2:2])
    return (b - a) * S / (3.0 * n)

# ------------------------- Cosmology container ------------------------
@dataclass
class Cosmo:
    H0: float = 70.0             # km/s/Mpc
    Omega_b: float = 0.05
    Omega_cdm: float = 0.0       # default 0 for MOND-style/no-CDM background
    Omega_L: float = 0.7
    N_eff: float = 3.046
    Tcmb: float = T_CMB

    def __post_init__(self):
        h = self.H0/100.0
        Ogam_h2 = omega_gamma_h2(self.Tcmb)
        self.Omega_gamma = Ogam_h2 / (h*h)
        # massless neutrino radiation
        self.Omega_r = self.Omega_gamma*(1.0 + 0.2271*self.N_eff)
        # close flatness if not exact:
        self.Omega_m = self.Omega_b + self.Omega_cdm
        self.Omega_k = 1.0 - (self.Omega_m + self.Omega_L + self.Omega_r)

    # E(z) = H(z)/H0
    def Ez(self, z):
        zp1 = 1.0 + z
        return np.sqrt(self.Omega_r*zp1**4 + self.Omega_m*zp1**3 + self.Omega_k*zp1**2 + self.Omega_L)

    def H(self, z):
        return self.H0*self.Ez(z)

    # Comoving line-of-sight distance χ(z) [Mpc]
    def chi(self, z):
        f = lambda zz: c_km_s/self.H(zz)
        return simpson_integral(f, 0.0, z, n=4096)

    # Angular diameter distance [Mpc]; assumes flatness (good enough here)
    def D_A(self, z):
        return self.chi(z)/(1.0+z)

    # Luminosity distance [Mpc]
    def D_L(self, z):
        return (1.0+z)**2 * self.D_A(z)

    # Distance modulus (without energy transfer)
    def mu_th(self, z):
        dL = self.D_L(z)  # Mpc
        return 5.0*np.log10(dL*1e6/10.0)  # convert Mpc -> pc, then μ = 5 log10(d/10pc)

    # Recombination redshift z_* (Hu & Sugiyama / Dodelson form)
    def z_star(self):
        h = self.H0/100.0
        Omh2 = (self.Omega_b + self.Omega_cdm)*h*h
        Obh2 = self.Omega_b*h*h
        g1 = 0.0783*Obh2**(-0.238)/(1.0 + 39.5*Obh2**0.763)
        g2 = 0.560/(1.0 + 21.1*Obh2**1.81)
        return 1048.0*(1.0 + 0.00124*Obh2**(-0.738))*(1.0 + g1*Omh2**g2)

    # Drag epoch z_d (Eisenstein & Hu 1998)
    def z_drag(self):
        h = self.H0/100.0
        Omh2 = (self.Omega_b + self.Omega_cdm)*h*h
        Obh2 = self.Omega_b*h*h
        b1 = 0.313*(Omh2)**(-0.419)*(1.0 + 0.607*(Omh2)**0.674)
        b2 = 0.238*(Omh2)**(0.223)
        return 1291.0*(Omh2**0.251)/(1.0 + 0.659*(Omh2)**0.828) * (1.0 + b1*(Obh2**b2))

    # Sound speed in the photon-baryon fluid
    def c_s(self, z):
        R = 3.0*self.Omega_b/(4.0*self.Omega_gamma) * 1.0/(1.0+z)  # R(z)=3ρ_b/4ρ_γ ∝ (1+z)^-1
        return c_km_s/np.sqrt(3.0*(1.0 + R))

    # Sound horizon r_s(z) [Mpc]
    def r_s(self, z):
        f = lambda zz: self.c_s(zz)/self.H(zz)
        return simpson_integral(f, z, 1.0e5, n=16384)

    # Acoustic scale ℓ_A ≈ π D_A(z*) / r_s(z*)
    def ell_A(self):
        zstar = self.z_star()
        return pi*self.D_A(zstar)/self.r_s(zstar)

    # Shift parameter R ≡ sqrt(Ω_m H0^2) D_A(z*) / c (often quoted)
    def shift_R(self):
        zstar = self.z_star()
        return sqrt(self.Omega_m)*(self.H0/c_km_s)*self.D_A(zstar)

# ----------------------- Energy-transfer model ------------------------
@dataclass
class EnergyTransfer:
    kappa0_per_Mpc: float = 0.0   # base κ0 in 1/Mpc (comoving)
    n_power: float = 0.0          # κ(z) = κ0 (1+z)^n

    def tau(self, z, cosmo: Cosmo):
        """Optical depth τ(z) from z=0 to z via comoving line element c/H(z)."""
        if self.kappa0_per_Mpc == 0.0:
            return 0.0
        def integrand(zz):
            return self.kappa0_per_Mpc*((1.0+zz)**self.n_power)*(c_km_s/cosmo.H(zz))
        return simpson_integral(integrand, 0.0, z, n=4096)

    def delta_mu(self, z, cosmo: Cosmo):
        """Magnitude shift from energy transfer: Δμ = 1.086 τ."""
        return 1.086*self.tau(z, cosmo)

# ----------------------- SN χ² (optional) -----------------------------
def chi2_sne(z, mu_obs, sigma_mu, cosmo: Cosmo, etrans: EnergyTransfer, Mcal=0.0):
    """
    χ² for SN Hubble diagram. Mcal is a global nuisance offset (e.g. absolute magnitude calibration).
    """
    z = np.asarray(z); mu_obs = np.asarray(mu_obs); sigma_mu = np.asarray(sigma_mu)
    mu_th = np.array([cosmo.mu_th(zi) for zi in z]) + np.array([etrans.delta_mu(zi, cosmo) for zi in z]) + Mcal
    return np.sum(((mu_obs - mu_th)/sigma_mu)**2)

# ----------------------- Demo / scan ----------------------------------
def demo():
    # Baseline: MOND-like background (no CDM), flat, reasonable Ω_b and Λ, N_eff standard
    cos = Cosmo(H0=70.0, Omega_b=0.05, Omega_cdm=0.0, Omega_L=0.7, N_eff=3.046)

    # Try a small energy transfer: κ0 ~ few × 1e-6 /Mpc, n ~ 0..1
    et = EnergyTransfer(kappa0_per_Mpc=5.0e-6, n_power=0.0)

    # Report some key numbers
    z_star = cos.z_star()
    z_d = cos.z_drag()
    rs_star = cos.r_s(z_star)
    rd = cos.r_s(z_d)
    ellA = cos.ell_A()

    print("=== Background (no CDM) ===")
    print(f"H0 = {cos.H0:.2f} km/s/Mpc,  Ω_b = {cos.Omega_b:.3f}, Ω_cdm = {cos.Omega_cdm:.3f}, Ω_Λ = {cos.Omega_L:.3f}, Ω_r = {cos.Omega_r:.3e}")
    print(f"z_* (recombination) ≈ {z_star:.1f},   z_drag ≈ {z_d:.1f}")
    print(f"Sound horizon r_s(z_*) ≈ {rs_star:.3f} Mpc,  r_d ≡ r_s(z_drag) ≈ {rd:.3f} Mpc")
    print(f"Acoustic scale ℓ_A ≈ {ellA:.1f}")

    # Example SN dimming from energy transfer
    for ztest in [0.1, 0.5, 1.0, 1.5]:
        dmu = et.delta_mu(ztest, cos)
        print(f"Δμ_energy(z={ztest:.1f}) = {dmu:.3f} mag")

    # Optional coarse κ scan to see effect on SN and ℓ_A without CDM
    print("\n=== Coarse κ scan (ℓ_A and Δμ at z=1) ===")
    for kappa0 in [0.0, 3e-6, 5e-6, 1e-5, 2e-5]:
        et_scan = EnergyTransfer(kappa0_per_Mpc=kappa0, n_power=0.0)
        ellA_scan = cos.ell_A()  # ℓ_A unaffected by κ in this simple model (transfer acts after emission)
        dmu1 = et_scan.delta_mu(1.0, cos)
        print(f"kappa0 = {kappa0:8.2e}  ->  ℓ_A = {ellA_scan:6.1f},   Δμ(z=1) = {dmu1:+.3f} mag")

if __name__ == "__main__":
    demo()
