#!/usr/bin/env python3
"""
Plot the time history (from early times to today) comparing the three-regime plateaus model (with early-time
G boost) against a ΛCDM baseline. Produces a multi-panel figure with:
 - G_eff(a)/G for the plateaus model
 - H(a)/H0 for both models
 - Sound horizon r_s(a)
 - Angular scale proxy ℓ(a)=π D_M(a)/r_s(a) with a marker at a=a_* (last scattering)

Usage:
  python tariff/scripts/plot_plateaus_history.py \
      --H0 70 --omega_b 0.02237 --Neff 3.046 --flat \
      --G_eff_boost 1.35 --a_trans 3e-4 --n_trans 6 \
      --out tariff/images/plateaus_history.png

If you pass --scan-json results from scan_plateaus_ellA.py, the best parameters are used.
"""
from __future__ import annotations

import argparse, json, os
import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.plateaus_model import Cosmology, PlateausParams, PlateausBackground, omega_gamma_h2, omega_nu_h2

C = 299792.458


def lcdm_background(H0: float, Omega_m: float, Omega_L: float, Tcmb: float = 2.7255, Neff: float = 3.046):
    h = H0/100.0
    Omega_r = (omega_gamma_h2(Tcmb) + omega_nu_h2(Neff, Tcmb))/h**2
    Omega_k = 1.0 - (Omega_m + Omega_L + Omega_r)
    def E(a):
        return np.sqrt(Omega_r*a**-4 + Omega_m*a**-3 + Omega_k*a**-2 + Omega_L)
    def DM(z):
        a = 1.0/(1.0+z)
        aa = np.geomspace(a, 1.0, 4096)
        chi = (C/H0) * np.trapz(1.0/(aa**2*E(aa)), aa)
        return chi if abs(Omega_k)<1e-12 else chi  # flat baseline intended
    def R(a):
        Omega_gamma = omega_gamma_h2(Tcmb)/h**2
        return 0.75*(Omega_m*0.0 + (Omega_m*0.0 + 0.0) + 0.0)  # not used
    def cs_over_a2E(a):
        # R(a) = 3ρ_b/(4ρ_γ); use Ω_b from ω_b ≈ assume Ω_b≈0.049 for baseline visuals (approx)
        Omega_b = 0.049
        Omega_gamma = omega_gamma_h2(Tcmb)/h**2
        R = 0.75*(Omega_b/Omega_gamma)*a
        cs_c = 1.0/np.sqrt(3.0*(1.0+R))
        return cs_c/(a**2*E(a))
    def rs(z):
        a = 1.0/(1.0+z)
        aa = np.geomspace(1e-8, a, 8192)
        return (C/H0)*np.trapz(cs_over_a2E(aa), aa)
    return E, DM, rs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--H0", type=float, default=70.0)
    ap.add_argument("--omega_b", type=float, default=0.02237)
    ap.add_argument("--Neff", type=float, default=3.046)
    ap.add_argument("--Tcmb", type=float, default=2.7255)
    ap.add_argument("--flat", action="store_true")

    ap.add_argument("--G_eff_boost", type=float, default=1.35)
    ap.add_argument("--a_trans", type=float, default=3e-4)
    ap.add_argument("--n_trans", type=float, default=6.0)
    ap.add_argument("--scan-json", type=str, default="")
    ap.add_argument("--out", type=str, default=os.path.join("tariff","images","plateaus_history.png"))
    args = ap.parse_args()

    if args.scan_json and os.path.exists(args.scan_json):
        with open(args.scan_json, "r") as f:
            scan = json.load(f)
        b = float(scan["best"]["G_eff_boost"]) if "best" in scan else float(scan.get("G_eff_boost", args.G_eff_boost))
        a_t = float(scan["best"]["a_trans"]) if "best" in scan else args.a_trans
        n_t = float(scan["best"]["n_trans"]) if "best" in scan else args.n_trans
        args.G_eff_boost, args.a_trans, args.n_trans = b, a_t, n_t

    cosmo = Cosmology(H0=args.H0, omega_b=args.omega_b, Neff=args.Neff, Tcmb=args.Tcmb, flat=args.flat)
    params = PlateausParams(G_eff_boost=args.G_eff_boost, a_trans=args.a_trans, n_trans=args.n_trans)
    bg = PlateausBackground(cosmo, params)

    # Baseline LCDM comparison (flat): Planck-like
    H0_lcdm, Om_lcdm, OL_lcdm = 67.4, 0.315, 0.685
    E_LCDM, DM_LCDM, RS_LCDM = lcdm_background(H0_lcdm, Om_lcdm, OL_lcdm, Tcmb=args.Tcmb, Neff=args.Neff)

    # Grids in a
    a = np.geomspace(1e-8, 1.0, 4096)
    E_plateaus = bg.E(a)
    # Plateaus G(a)/G
    def g_gate(a_val):
        S = 1.0/(1.0 + (a_val/args.a_trans)**args.n_trans)
        return 1.0 + (args.G_eff_boost - 1.0)*S
    Grel = g_gate(a)

    # H(a)/H0 for both
    H_plateaus = args.H0*E_plateaus
    H_LCDM = H0_lcdm*E_LCDM(a)

    # Sound horizon cumulative vs a for both
    aa_rs = np.geomspace(1e-8, 1.0, 2000)
    def _rs_cumulative_plateaus(aa):
        # Compute running integral r_s(a) = (c/H0) ∫_0^{a} c_s/(a'^2 E(a')) da'
        h = args.H0/100.0
        Ogamma = omega_gamma_h2(args.Tcmb)/h**2
        Ob = cosmo.omega_b/(h**2)
        def integrand(t):
            R = 0.75*(Ob/Ogamma)*t
            cs_c = 1.0/np.sqrt(3.0*(1.0+R))
            return cs_c/(t**2*bg.E(t))
        vals = []
        for i in range(len(aa)):
            tgrid = aa[:i+1]
            y = integrand(tgrid)
            vals.append((C/args.H0)*np.trapz(y, tgrid))
        return np.array(vals)

    def _rs_cumulative_lcdm(aa):
        # Compute running r_s(a) in ΛCDM baseline
        h0 = H0_lcdm/100.0
        Ogamma = omega_gamma_h2(args.Tcmb)/h0**2
        Ob = 0.049
        def integrand(t):
            R = 0.75*(Ob/Ogamma)*t
            cs_c = 1.0/np.sqrt(3.0*(1.0+R))
            return cs_c/(t**2*E_LCDM(t))
        vals = []
        for i in range(len(aa)):
            tgrid = aa[:i+1]
            y = integrand(tgrid)
            vals.append((C/H0_lcdm)*np.trapz(y, tgrid))
        return np.array(vals)

    rs_plateaus = _rs_cumulative_plateaus(aa_rs)
    rs_lcdm = _rs_cumulative_lcdm(aa_rs)

    # ℓ(a)=π D_M(a)/r_s(a) proxy
    def ell_of_a(a_val, rs_func, DM_func, H0):
        z = 1.0/a_val - 1.0
        DM = np.array([DM_func(zz) for zz in z])
        return np.pi*DM/rs_func

    # Last scattering
    z_star = bg.z_star()
    a_star = 1.0/(1.0+z_star)
    ellA_plateaus = bg.ell_A()

    # Plot
    fig, axs = plt.subplots(2, 2, figsize=(12, 9))

    # G(a)/G
    axs[0,0].loglog(a, Grel, label='Plateaus G_eff/G')
    axs[0,0].axvline(a_star, color='k', ls='--', alpha=0.6, label='a* (recombination)')
    axs[0,0].set_xlabel('a')
    axs[0,0].set_ylabel('G_eff/G')
    axs[0,0].grid(alpha=0.3, which='both')
    axs[0,0].legend()

    # H(a)
    axs[0,1].loglog(a, H_plateaus, label='Plateaus H(a)')
    axs[0,1].loglog(a, H_LCDM, label='ΛCDM H(a)')
    axs[0,1].axvline(a_star, color='k', ls='--', alpha=0.6)
    axs[0,1].set_xlabel('a')
    axs[0,1].set_ylabel('H(a) [km/s/Mpc]')
    axs[0,1].grid(alpha=0.3, which='both')
    axs[0,1].legend()

    # r_s(a)
    axs[1,0].loglog(aa_rs, rs_plateaus, label='Plateaus r_s(a)')
    axs[1,0].loglog(aa_rs, rs_lcdm, label='ΛCDM r_s(a)')
    axs[1,0].axvline(a_star, color='k', ls='--', alpha=0.6)
    axs[1,0].set_xlabel('a')
    axs[1,0].set_ylabel('r_s [Mpc] (cumulative)')
    axs[1,0].grid(alpha=0.3, which='both')
    axs[1,0].legend()

    # ℓ_A annotation
    axs[1,1].axis('off')
    txt = f"Plateaus parameters\nG_boost={args.G_eff_boost:.3f}, a_trans={args.a_trans:.2e}, n_trans={args.n_trans:.1f}\n" \
          f"z*={z_star:.1f},  ℓ_A(plateaus)≈{ellA_plateaus:.1f}\n" \
          f"Baseline ΛCDM: H0={H0_lcdm}, Ωm={Om_lcdm}, ΩΛ={OL_lcdm}"
    axs[1,1].text(0.02, 0.8, txt, fontsize=11)

    out = args.out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print("Saved", out)


if __name__ == "__main__":
    main()