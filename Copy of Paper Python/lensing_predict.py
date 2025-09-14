#!/usr/bin/env python3
"""
Lensing prediction tool for density-aware TFR with disformal weak-field coupling.

Two modes:
- gr:    standard GR lensing by baryons (no DM)
- tfr:   adds an environmental lensing contribution via phi_env = 0.5*ln(xi)
         and deflection alpha_env(R) = (a_env + b_env) * grad_perp ∫ phi_env dz

Implements a spherical Hernquist baryon model and a flexible screened power-law
ansatz for phi_env(r). Computes Einstein radius given (z_l, z_s) and uncertainties.

This is a first-pass, testable phenomenology that resolves the conformal issue by
allowing a disformal-type coupling (a_env != b_env) so that photons do couple to
phi_env gradients while Solar-System screening (phi_env ~ 0 at high density/temperature)
keeps local constraints intact.
"""
import argparse
import math
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
from scipy import integrate
from scipy.optimize import brentq

# Cosmology (flat LCDM) for angular diameter distances: simple, adequate for first pass
H0 = 70.0  # km/s/Mpc
Omega_m = 0.3
c_kms = 299792.458
G_SI = 6.67430e-11
M_sun_kg = 1.98847e30
kpc_m = 3.085677581491367e19
arcsec = (1/206265.0)


def E_z(z):
    return math.sqrt(Omega_m*(1+z)**3 + (1 - Omega_m))


def D_c(z):
    # Comoving distance in Mpc
    integrand = lambda zp: c_kms / (H0 * E_z(zp))
    Dc, _ = integrate.quad(integrand, 0.0, z, epsabs=0, epsrel=1e-6)
    return Dc  # Mpc


def D_a(z1, z2=None):
    if z2 is None:
        # observer to z1
        return D_c(z1) / (1+z1)
    else:
        Dc1 = D_c(z1)
        Dc2 = D_c(z2)
        if Dc2 <= Dc1:
            return 0.0
        return (Dc2 - Dc1) / (1+z2)


@dataclass
class Hernquist:
    M_star: float  # Msun
    a_kpc: float   # kpc (Hernquist scale length, ~ R_e/1.8153)

    def M3d(self, r_kpc: float) -> float:
        x = r_kpc / self.a_kpc
        return self.M_star * (x**2) / (1 + x)**2

    def M2d(self, R_kpc: float) -> float:
        # Projected mass within R for Hernquist (analytic)
        # Use expression from Hernquist 1990
        x = R_kpc / self.a_kpc
        if x < 1:
            f = (math.acos(x) / math.sqrt(1 - x*x))
        elif x > 1:
            f = (math.acosh(x) / math.sqrt(x*x - 1))
        else:  # x == 1
            f = 1.0
        return self.M_star * ( (x*x - 1) * f + 1 )


@dataclass
class PhiEnv:
    A_env: float  # amplitude of xi-1 at r0 (dimensionless ~ <= 1)
    p: float      # radial fall-off index (>0)
    r0_kpc: float # reference radius (kpc)

    def phi(self, r_kpc: float) -> float:
        # phi_env = 0.5 * ln(1 + A_env (r/r0)^-p)
        if r_kpc <= 0:
            r_kpc = 1e-6
        val = 0.5 * math.log(1.0 + self.A_env * (r_kpc / self.r0_kpc)**(-self.p))
        return val


def alpha_gr_hernquist(R_kpc: float, lens: Hernquist) -> float:
    """Deflection angle in radians (GR) for spherical lens with projected mass M2d.
    alpha = 4GM(<R)/(c^2 R)
    """
    M_proj = lens.M2d(R_kpc) * M_sun_kg
    R_m = R_kpc * kpc_m
    alpha = 4 * G_SI * M_proj / ( (c_kms*1000)**2 * R_m )
    return alpha


# Note: alpha_env_numeric below computes the environmental deflection; this helper was redundant and removed.


def alpha_env_numeric(R_kpc: float, phi_env: PhiEnv, a_plus_b: float) -> float:
    # Correct unit handling: I(R) has units of kpc (phi is dimensionless, dz in kpc). dI/dR is dimensionless.
    # Deflection angle is 2 * grad_perp ∫ Φ_W dz; for our env piece with coefficient (a+b)/2 inside Φ_W,
    # alpha_env = (a+b) * dI/dR. Since I is in kpc and R in kpc, dI/dR is dimensionless. So alpha_env is dimensionless.
    zmax = 2000.0
    Nz = 2000
    z = np.linspace(-zmax, zmax, Nz)

    def I_of_R(Rval):
        r = np.sqrt(Rval*Rval + z*z)
        phi_vals = 0.5*np.log(1.0 + phi_env.A_env * (r/phi_env.r0_kpc)**(-phi_env.p))
        return np.trapz(phi_vals, z)

    dR = max(1e-3, 1e-3*R_kpc)
    I_plus = I_of_R(R_kpc + dR)
    I_minus = I_of_R(max(R_kpc - dR, 1e-6))
    dIdR = (I_plus - I_minus) / (2*dR)
    return a_plus_b * dIdR


def einstein_radius_arcsec(lens: Hernquist, phi_env: PhiEnv, z_l: float, z_s: float,
                           mode: str = "tfr", a_env: float = 1.0, b_env: float = 1.0) -> float:
    Dl = D_a(z_l)
    Ds = D_a(z_s)
    Dls = D_a(z_l, z_s)
    if Dl <= 0 or Ds <= 0 or Dls <= 0:
        return 0.0

    def lens_eq(R_kpc):
        theta = (R_kpc / (Dl*1e3)) / arcsec  # arcsec, for reference
        # Total deflection in radians
        alpha_b = alpha_gr_hernquist(R_kpc, lens)
        if mode == "gr":
            alpha_tot = alpha_b
        else:
            alpha_e = alpha_env_numeric(R_kpc, phi_env, a_env + b_env)
            alpha_tot = alpha_b + alpha_e
        # Einstein condition: alpha(R) = R * Ds / (Dl*Dls)
        return alpha_tot - R_kpc / (Dl*1e3) * Ds / Dls

    # Bracket search for root in R_kpc
    Rmin, Rmax = 0.01, 100.0
    fmin = lens_eq(Rmin)
    fmax = lens_eq(Rmax)
    if fmin*fmax > 0:
        # Try expand
        for fac in [200, 500, 1000]:
            fmax = lens_eq(Rmax*fac)
            if fmin*fmax <= 0:
                Rmax *= fac
                break
        else:
            return 0.0

    R_E = brentq(lens_eq, Rmin, Rmax, maxiter=200)
    theta_E_rad = R_E / (Dl*1e3)  # kpc / (kpc per arcsec^-1)
    theta_E_arcsec = theta_E_rad / arcsec
    return theta_E_arcsec


def mc_theta_E(n: int, lens: Hernquist, dM: float, da: float,
               penv: PhiEnv, dA: float, dp: float, z_l: float, z_s: float,
               mode: str, a_env: float, b_env: float) -> Tuple[float, float, float]:
    samples = []
    rng = np.random.default_rng(42)
    for _ in range(n):
        M = rng.normal(lens.M_star, dM)
        a = rng.normal(lens.a_kpc, da)
        A = max(0.0, rng.normal(penv.A_env, dA))
        p = max(0.1, rng.normal(penv.p, dp))
        l = Hernquist(M, a)
        pe = PhiEnv(A, p, penv.r0_kpc)
        th = einstein_radius_arcsec(l, pe, z_l, z_s, mode=mode, a_env=a_env, b_env=b_env)
        if th > 0:
            samples.append(th)
    if not samples:
        return 0.0, 0.0, 0.0
    arr = np.array(samples)
    return float(np.median(arr)), float(np.percentile(arr, 16)), float(np.percentile(arr, 84))


def main():
    ap = argparse.ArgumentParser(description="Predict Einstein radius with TFR-lensing")
    ap.add_argument("--mode", choices=["gr", "tfr"], default="tfr")
    ap.add_argument("--z_l", type=float, required=True)
    ap.add_argument("--z_s", type=float, required=True)
    ap.add_argument("--M_star", type=float, required=True, help="Stellar mass Msun")
    ap.add_argument("--a_kpc", type=float, required=True, help="Hernquist scale (kpc)")
    ap.add_argument("--A_env", type=float, default=0.2, help="Amplitude of xi-1 at r0")
    ap.add_argument("--p_env", type=float, default=1.2, help="Radial falloff index")
    ap.add_argument("--r0_kpc", type=float, default=5.0, help="Reference radius for env")
    ap.add_argument("--a_env", type=float, default=1.0)
    ap.add_argument("--b_env", type=float, default=1.0)
    ap.add_argument("--mc", type=int, default=2000, help="MC samples for uncertainty")
    ap.add_argument("--dM", type=float, default=0.1, help="1-sigma on log10 M_star if --logM")
    ap.add_argument("--dM_abs", type=float, default=None, help="Absolute sigma on M_star (Msun)")
    ap.add_argument("--da", type=float, default=0.2, help="sigma on a_kpc (kpc)")
    ap.add_argument("--dA", type=float, default=0.1)
    ap.add_argument("--dp", type=float, default=0.2)
    ap.add_argument("--logM", action="store_true", help="Interpret M_star as log10(M/Msun)")
    ap.add_argument("--worked_example", action="store_true", help="Run a SLACS-like example")

    args = ap.parse_args()

    if args.worked_example:
        # A simple SLACS-like case
        z_l, z_s = 0.2, 0.6
        Re_kpc = 5.0
        a_kpc = Re_kpc / 1.8153
        logM = 11.2
        M = 10**logM
        lens = Hernquist(M, a_kpc)
        penv = PhiEnv(A_env=0.3, p=1.1, r0_kpc=5.0)
        th_gr = einstein_radius_arcsec(lens, penv, z_l, z_s, mode="gr")
        th_tfr = einstein_radius_arcsec(lens, penv, z_l, z_s, mode="tfr", a_env=1.0, b_env=1.0)
        med, p16, p84 = mc_theta_E(1000, lens, dM=0.2*M, da=0.5, penv=penv, dA=0.1, dp=0.2,
                                   z_l=z_l, z_s=z_s, mode="tfr", a_env=1.0, b_env=1.0)
        print(f"Worked example (z_l={z_l}, z_s={z_s}, Re~{Re_kpc} kpc, logM={logM}):")
        print(f"  theta_E_GR   = {th_gr:.2f} arcsec")
        print(f"  theta_E_TFR  = {th_tfr:.2f} arcsec")
        print(f"  theta_E_TFR (MC median [16,84]) = {med:.2f} [{p16:.2f}, {p84:.2f}] arcsec")
        return

    z_l, z_s = args.z_l, args.z_s
    if args.logM:
        M = 10**args.M_star
        dM_abs = (math.log(10) * M * args.dM) if args.dM_abs is None else args.dM_abs
    else:
        M = args.M_star
        dM_abs = args.dM_abs if args.dM_abs is not None else 0.2*M

    lens = Hernquist(M_star=M, a_kpc=args.a_kpc)
    penv = PhiEnv(A_env=args.A_env, p=args.p_env, r0_kpc=args.r0_kpc)

    th = einstein_radius_arcsec(lens, penv, z_l, z_s, mode=args.mode, a_env=args.a_env, b_env=args.b_env)
    print(f"theta_E_{args.mode} = {th:.3f} arcsec")

    if args.mc > 0:
        med, p16, p84 = mc_theta_E(args.mc, lens, dM=dM_abs, da=args.da,
                                   penv=penv, dA=args.dA, dp=args.dp,
                                   z_l=z_l, z_s=z_s, mode=args.mode,
                                   a_env=args.a_env, b_env=args.b_env)
        print(f"theta_E_{args.mode} (median [16,84]) = {med:.3f} [{p16:.3f}, {p84:.3f}] arcsec")


if __name__ == "__main__":
    main()
