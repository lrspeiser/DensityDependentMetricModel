#!/usr/bin/env python3
"""
Batch SLACS-like lensing examples using tools/lensing_predict.py primitives.

- Defines a small catalog of example lenses (z_l, z_s, Re, M* ± sigma, etc.).
- Runs GR-only and TFR-lensing (disformal weak-field) predictions of Einstein radius.
- Outputs a markdown summary table and optional per-lens PNGs of alpha(R) and theta_E posteriors.

This is a lightweight, reproducible pilot to accompany docs/lensing.md and the
paper lensing section.
"""
import argparse
import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

# Allow running directly from repo root without installing as a package
import os, sys
_THIS_DIR = os.path.dirname(__file__)
sys.path.insert(0, _THIS_DIR)

from lensing_predict import (
    Hernquist, PhiEnv, einstein_radius_arcsec, mc_theta_E,
    D_a, alpha_gr_hernquist, alpha_env_numeric
)


@dataclass
class LensEntry:
    name: str
    z_l: float
    z_s: float
    Re_kpc: float           # effective radius (kpc)
    log10_M_star: float     # stellar mass (log10 Msun)
    sigma_log10_M: float    # 1-sigma on log10(M)
    sigma_Re: float         # 1-sigma on Re (kpc)


def default_catalog() -> List[LensEntry]:
    # Simple illustrative sample (numbers approximate, replace with curated values later)
    return [
        LensEntry("SLACS-A", z_l=0.20, z_s=0.60, Re_kpc=5.0, log10_M_star=11.2, sigma_log10_M=0.2, sigma_Re=0.5),
        LensEntry("SLACS-B", z_l=0.25, z_s=0.65, Re_kpc=4.0, log10_M_star=11.0, sigma_log10_M=0.2, sigma_Re=0.4),
        LensEntry("SLACS-C", z_l=0.30, z_s=0.80, Re_kpc=6.0, log10_M_star=11.3, sigma_log10_M=0.2, sigma_Re=0.6),
    ]


def run_one(entry: LensEntry, A_env: float, p_env: float, r0_kpc: float,
            a_env: float, b_env: float, mc: int, out_dir: str,
            make_plot: bool) -> Tuple[float, float, float, float, float, float]:
    a_kpc = entry.Re_kpc / 1.8153
    M = 10 ** entry.log10_M_star
    dM_abs = math.log(10) * M * entry.sigma_log10_M
    lens = Hernquist(M_star=M, a_kpc=a_kpc)
    penv = PhiEnv(A_env=A_env, p=p_env, r0_kpc=r0_kpc)

    th_gr = einstein_radius_arcsec(lens, penv, entry.z_l, entry.z_s, mode="gr")
    med_gr, p16_gr, p84_gr = mc_theta_E(
        mc, lens, dM=dM_abs, da=entry.sigma_Re/1.8153,
        penv=penv, dA=0.0, dp=0.0,
        z_l=entry.z_l, z_s=entry.z_s, mode="gr", a_env=a_env, b_env=b_env
    )

    th_tfr = einstein_radius_arcsec(lens, penv, entry.z_l, entry.z_s, mode="tfr", a_env=a_env, b_env=b_env)
    med_tfr, p16_tfr, p84_tfr = mc_theta_E(
        mc, lens, dM=dM_abs, da=entry.sigma_Re/1.8153,
        penv=penv, dA=0.1*A_env if A_env>0 else 0.0, dp=0.2,
        z_l=entry.z_l, z_s=entry.z_s, mode="tfr", a_env=a_env, b_env=b_env
    )

    if make_plot:
        # Deflection vs R plot (kpc) overlay
        Dl = D_a(entry.z_l)
        R = np.linspace(0.1, 30.0, 400)
        alpha_gr = np.array([alpha_gr_hernquist(r, lens) for r in R])
        alpha_env = np.array([alpha_env_numeric(r, penv, a_env+b_env) for r in R])
        alpha_tfr = alpha_gr + alpha_env
        # Einstein radii in kpc for plotting
        RE_gr_kpc = th_gr * (Dl*1e3) * (1.0/206265.0)
        RE_tfr_kpc = th_tfr * (Dl*1e3) * (1.0/206265.0)

        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        ax.plot(R, alpha_gr, label="alpha_GR (rad)")
        ax.plot(R, alpha_tfr, label="alpha_TFR (rad)")
        ax.axvline(RE_gr_kpc, color="#1f77b4", ls=":", label=f"R_E,GR ~ {th_gr:.2f}\"")
        ax.axvline(RE_tfr_kpc, color="#d62728", ls=":", label=f"R_E,TFR ~ {th_tfr:.2f}\"")
        ax.set_xlabel("R [kpc]")
        ax.set_ylabel("Deflection angle [rad]")
        ax.set_title(f"{entry.name}: Deflection profiles")
        ax.legend()
        fig.tight_layout()
        fig.savefig(f"{out_dir}/lensing_{entry.name}.png", dpi=150)
        plt.close(fig)

    return med_gr, p16_gr, p84_gr, med_tfr, p16_tfr, p84_tfr


def write_markdown(catalog: List[LensEntry], results: List[Tuple[float, float, float, float, float, float]],
                   A_env: float, p_env: float, r0_kpc: float, a_env: float, b_env: float,
                   out_md: str):
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Lensing pilot (SLACS-like)\n\n")
        f.write("Parameters:\n\n")
        f.write(f"- A_env = {A_env}, p_env = {p_env}, r0_kpc = {r0_kpc}, a_env = {a_env}, b_env = {b_env}\n\n")
        f.write("| Lens | z_l | z_s | Re [kpc] | log10 M* | theta_E_GR [\"] (50% [16,84]) | theta_E_TFR [\"] (50% [16,84]) |\n")
        f.write("|------|-----:|-----:|---------:|----------:|----------------------------------:|-------------------------------------:|\n")
        for e, (mg, g16, g84, mt, t16, t84) in zip(catalog, results):
            f.write(f"| {e.name} | {e.z_l:.2f} | {e.z_s:.2f} | {e.Re_kpc:.2f} | {e.log10_M_star:.2f} | {mg:.2f} [{g16:.2f},{g84:.2f}] | {mt:.2f} [{t16:.2f},{t84:.2f}] |\n")
        f.write("\nNotes: GR values are baryons-only. TFR values include the environmental deflection term per docs/lensing.md.\n")


def main():
    ap = argparse.ArgumentParser(description="Run SLACS-like lensing batch and emit markdown table/plots")
    ap.add_argument("--out-md", default="docs/lensing_results.md")
    ap.add_argument("--out-dir", default="images")
    ap.add_argument("--A-env", type=float, default=0.3, dest="A_env")
    ap.add_argument("--p-env", type=float, default=1.1, dest="p_env")
    ap.add_argument("--r0-kpc", type=float, default=5.0, dest="r0_kpc")
    ap.add_argument("--a-env", type=float, default=1.0, dest="a_env")
    ap.add_argument("--b-env", type=float, default=1.0, dest="b_env")
    ap.add_argument("--mc", type=int, default=1000)
    ap.add_argument("--no-plots", action="store_true")

    args = ap.parse_args()

    cat = default_catalog()
    results = []
    for e in cat:
        res = run_one(e, A_env=args.A_env, p_env=args.p_env, r0_kpc=args.r0_kpc,
                      a_env=args.a_env, b_env=args.b_env, mc=args.mc,
                      out_dir=args.out_dir, make_plot=(not args.no_plots))
        results.append(res)

    write_markdown(cat, results, args.A_env, args.p_env, args.r0_kpc, args.a_env, args.b_env, args.out_md)
    print(f"Wrote {args.out_md}. Plots in {args.out_dir} (if enabled).")


if __name__ == "__main__":
    main()

