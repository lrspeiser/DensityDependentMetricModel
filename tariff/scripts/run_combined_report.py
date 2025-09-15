#!/usr/bin/env python3
"""
Run a combined report for the three-regime plateaus model:
- Locks an ℓ_A-viable early-time G window (you pass parameters found from the scan)
- Computes SNe (anchored full-cov if available) with κ dimming
- Computes BAO long-form χ² and overlays
- Computes RAR χ²
- Writes JSON summary and a multi-panel figure

Example:
  python tariff/scripts/run_combined_report.py \
    --H0 70 --omega_b 0.02237 --Neff 3.046 --Omega_k 0.0 --flat \
    --G_eff_boost 0.92 --a_trans 1.2e-3 --n_trans 8 \
    --kappa_per_Mpc 9.2e-6 \
    --sne_csv path/to/sne.csv \
    --bao_csv path/to/bao_longform.csv \
    --rar_csv path/to/rar.csv \
    --ellA_value 301.0 --ellA_sigma 0.1 \
    --json_out tariff/results/combined_report.json \
    --fig_out tariff/images/regime_summary.png
"""
from __future__ import annotations

import argparse, json, os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.plateaus_model import Cosmology, PlateausParams, PlateausBackground
from data_io.loaders import load_cmb_ellA_constraint, load_sne_table, maybe_load_covariance, load_bao_table, load_rar_table
from fits.likelihoods import chi2_cmb_ellA, chi2_sne, chi2_bao, chi2_rar, rd_drag, bao_theory


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--H0", type=float, default=70.0)
    ap.add_argument("--omega_b", type=float, default=0.02237)
    ap.add_argument("--Neff", type=float, default=3.046)
    ap.add_argument("--Tcmb", type=float, default=2.7255)
    ap.add_argument("--flat", action="store_true")
    ap.add_argument("--Omega_k", type=float, default=None)

    ap.add_argument("--G_eff_boost", type=float, default=1.0)
    ap.add_argument("--a_trans", type=float, default=1e-4)
    ap.add_argument("--n_trans", type=float, default=4)
    ap.add_argument("--kappa_per_Mpc", type=float, default=0.0)

    ap.add_argument("--sne_csv", type=str, default="")
    ap.add_argument("--bao_csv", type=str, default="")
    ap.add_argument("--rar_csv", type=str, default="")
    ap.add_argument("--ellA_value", type=float, default=301.0)
    ap.add_argument("--ellA_sigma", type=float, default=0.1)

    ap.add_argument("--json_out", type=str, default=os.path.join("tariff","results","combined_report.json"))
    ap.add_argument("--fig_out", type=str, default=os.path.join("tariff","images","regime_summary.png"))
    args = ap.parse_args()

    cosmo = Cosmology(H0=args.H0, omega_b=args.omega_b, Neff=args.Neff, Tcmb=args.Tcmb, flat=args.flat, Omega_k_override=args.Omega_k)
    params = PlateausParams(G_eff_boost=args.G_eff_boost, a_trans=args.a_trans, n_trans=args.n_trans, kappa_per_Mpc=args.kappa_per_Mpc)
    bg = PlateausBackground(cosmo, params)

    # ℓ_A
    ellA_obs, ellA_sig = load_cmb_ellA_constraint(args.ellA_value, args.ellA_sigma)
    ellA_out = chi2_cmb_ellA(bg, ellA_obs, ellA_sig)
    zstar = bg.z_star()
    rs = bg.sound_horizon(zstar)
    DM = bg.D_M(zstar)
    DA = DM/(1.0+zstar)

    result = {
        "cosmo_params": {"H0": args.H0, "omega_b": args.omega_b, "Neff": args.Neff, "Omega_k": args.Omega_k},
        "gate_params": {"B_early": args.G_eff_boost, "a_trans": args.a_trans, "n_trans": args.n_trans, "kappa_per_Mpc": args.kappa_per_Mpc},
        "derived": {"z_star": float(zstar), "ellA_th": float(ellA_out["ellA_th"]), "r_s_Mpc": float(rs), "D_A_star_Mpc": float(DA)}
    }

    # SNe
    if args.sne_csv:
        sne = load_sne_table(args.sne_csv)
        cov = maybe_load_covariance(args.sne_csv)
        sn = chi2_sne(bg, sne, cov, args.kappa_per_Mpc, method="anchored_fullcov")
        result["likelihoods"] = result.get("likelihoods", {})
        result["likelihoods"]["SNe"] = sn

    # BAO
    if args.bao_csv:
        bao = load_bao_table(args.bao_csv)
        bao_out = chi2_bao(bg, bao)
        result["likelihoods"] = result.get("likelihoods", {})
        result["likelihoods"]["BAO"] = bao_out

    # RAR
    if args.rar_csv:
        rar = load_rar_table(args.rar_csv)
        rar_out = chi2_rar(rar, params.a0_m_s2, params.nu_form)
        result["likelihoods"] = result.get("likelihoods", {})
        result["likelihoods"]["RAR"] = rar_out

    # Write JSON
    os.makedirs(os.path.dirname(args.json_out), exist_ok=True)
    with open(args.json_out, "w") as f:
        json.dump(result, f, indent=2)
    print("Wrote", args.json_out)

    # Summary figure
    fig, axs = plt.subplots(2, 2, figsize=(12, 9))
    # Panel 1: G(a)/G
    a = np.geomspace(1e-8, 1.0, 2000)
    S = 1.0/(1.0 + (a/args.a_trans)**args.n_trans)
    Grel = 1.0 + (args.G_eff_boost-1.0)*S
    axs[0,0].loglog(a, Grel)
    axs[0,0].set_title("G_eff(a)/G")
    axs[0,0].grid(alpha=0.3, which='both')

    # Panel 2: H(a)
    axs[0,1].loglog(a, args.H0*bg.E(a))
    axs[0,1].set_title("H(a)")
    axs[0,1].grid(alpha=0.3, which='both')

    # Panel 3: BAO overlay if available
    if args.bao_csv:
        zs = sorted(set(bao["z"].astype(float).tolist()))
        dm = [bg.D_M(z) for z in zs]
        dh = [ (1.0 / (bg.E(1.0/(1.0+z)))) * (299792.458/args.H0) for z in zs]
        rd = rd_drag(bg)
        axs[1,0].plot(zs, np.array(dm)/rd, label="DM/rd (th)")
        axs[1,0].plot(zs, np.array(dh)/rd, label="DH/rd (th)")
        # plot points
        for _, row in bao.iterrows():
            if row["observable"] == "DM_over_rd":
                axs[1,0].errorbar([row["z"]],[row["value"]], yerr=[row["sigma"]], fmt='o', color='k')
            elif row["observable"] == "DH_over_rd":
                axs[1,0].errorbar([row["z"]],[row["value"]], yerr=[row["sigma"]], fmt='s', color='gray')
        axs[1,0].set_title("BAO: (DM,DH)/rd")
        axs[1,0].set_xlabel("z"); axs[1,0].grid(alpha=0.3)
    else:
        axs[1,0].axis('off')

    # Panel 4: SNe residuals if available
    if args.sne_csv:
        z = sne["z"].astype(float).values
        mu_th = np.array([5.0*np.log10(bg.D_L(zi))+25.0 for zi in z])
        chi = np.array([bg.comoving_distance(zi) for zi in z])
        mu_th = mu_th + (2.5/np.log(10.0))*args.kappa_per_Mpc*chi
        if "mu" in sne.columns:
            mu_obs = sne["mu"].astype(float).values
        else:
            mu_obs = sne["mB"].astype(float).values
        # anchor shift from fit if available
        delta = result.get("likelihoods",{}).get("SNe",{}).get("delta_M_best",0.0)
        resid = mu_obs - (mu_th + delta)
        axs[1,1].scatter(z, resid, s=6, alpha=0.6)
        axs[1,1].axhline(0, color='k', lw=1)
        axs[1,1].set_title("SNe residuals (anchored, with κ)")
        axs[1,1].set_xlabel("z"); axs[1,1].set_ylabel("Δμ [mag]")
        axs[1,1].grid(alpha=0.3)
    else:
        axs[1,1].axis('off')

    os.makedirs(os.path.dirname(args.fig_out), exist_ok=True)
    plt.tight_layout(); plt.savefig(args.fig_out, dpi=150)
    print("Saved", args.fig_out)


if __name__ == "__main__":
    main()