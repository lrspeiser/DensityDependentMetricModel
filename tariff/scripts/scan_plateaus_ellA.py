#!/usr/bin/env python3
"""
Scan early-time G boost parameters (G_eff_boost, a_trans, n_trans) to match the CMB acoustic scale ℓ_A≈301
in the three-regime plateaus model (no CDM background). Writes results to tariff/results/plateaus_ellA_scan.json.

Usage (examples):
  python tariff/scripts/scan_plateaus_ellA.py --flat --H0 70 --omega_b 0.02237 --Neff 3.046 \
      --ellA 301.0 --sigma 0.1 --json-out tariff/results/plateaus_ellA_scan.json
"""
from __future__ import annotations

import argparse, json, os
import numpy as np
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.plateaus_model import Cosmology, PlateausParams, PlateausBackground
from fits.likelihoods import chi2_cmb_ellA


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--H0", type=float, default=70.0)
    ap.add_argument("--omega_b", type=float, default=0.02237)
    ap.add_argument("--Neff", type=float, default=3.046)
    ap.add_argument("--Tcmb", type=float, default=2.7255)
    ap.add_argument("--flat", action="store_true")
    ap.add_argument("--Omega_k_list", type=str, default="", help="comma-separated Ω_k values; if set, overrides flat/closure")

    ap.add_argument("--ellA", type=float, default=301.0)
    ap.add_argument("--sigma", type=float, default=0.1)

    ap.add_argument("--a_trans_grid", type=str, default="5e-5,8e-5,1.2e-4,1.8e-4,2.7e-4,4e-4,6e-4,9e-4,1.4e-3,2.1e-3,3e-3")
    ap.add_argument("--n_trans_grid", type=str, default="2,3,4,5,6,8,10,12")
    ap.add_argument("--B_min", type=float, default=0.60)
    ap.add_argument("--B_max", type=float, default=1.10)
    ap.add_argument("--B_steps", type=int, default=24)
    ap.add_argument("--omega_b_grid", type=str, default="0.02237")
    ap.add_argument("--Neff_grid", type=str, default="3.046")

    ap.add_argument("--json-out", type=str, default=os.path.join("tariff","results","plateaus_ellA_scan.json"))
    args = ap.parse_args()

    a_trans_grid = [float(x) for x in args.a_trans_grid.split(",") if x.strip()]
    n_trans_grid = [float(x) for x in args.n_trans_grid.split(",") if x.strip()]
    B_vals = np.linspace(args.B_min, args.B_max, int(args.B_steps))
    omega_b_vals = [float(x) for x in args.omega_b_grid.split(",") if x.strip()]
    Neff_vals = [float(x) for x in args.Neff_grid.split(",") if x.strip()]
    Omega_k_vals = [float(x) for x in args.Omega_k_list.split(",") if x.strip()] if args.Omega_k_list else [None]

    rows = []
    best = None
    for Omk in Omega_k_vals:
        for wb in omega_b_vals:
            for ne in Neff_vals:
                cosmo = Cosmology(H0=args.H0, omega_b=wb, Neff=ne, Tcmb=args.Tcmb, flat=args.flat, Omega_k_override=Omk)
                for a_t in a_trans_grid:
                    for n_t in n_trans_grid:
                        for b in B_vals:
                            params = PlateausParams(G_eff_boost=float(b), a_trans=float(a_t), n_trans=float(n_t))
                            bg = PlateausBackground(cosmo, params)
                            out = chi2_cmb_ellA(bg, args.ellA, args.sigma)
                            zstar = bg.z_star()
                            rs = bg.sound_horizon(zstar)
                            DM = bg.D_M(zstar)
                            DA = DM/(1.0+zstar)
                            row = {"G_eff_boost": float(b), "a_trans": float(a_t), "n_trans": float(n_t),
                                   "omega_b": float(wb), "Neff": float(ne),
                                   "Omega_k": (float(Omk) if Omk is not None else None),
                                   "z_star": float(zstar), "r_s_Mpc": float(rs), "D_A_star_Mpc": float(DA),
                                   "ellA_th": float(out["ellA_th"]), "chi2": float(out["chi2"])}
                            rows.append(row)
                            if best is None or row["chi2"] < best["chi2"]:
                                best = dict(row)

    result = {"scan": rows, "best": best, "cosmology": cosmo.__dict__}
    out_path = args.json_out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print("Wrote", out_path)
    print("Best:", json.dumps(best, indent=2))


if __name__ == "__main__":
    main()