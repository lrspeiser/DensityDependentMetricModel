#!/usr/bin/env python3
"""
fit_sparc_baselines.py - Quick GR(baryons) and NFW baseline fits for a SPARC galaxy.

Usage examples:
  python tools/fit_sparc_baselines.py --galaxy NGC3198 --sparc-dir external_data/Rotmod_LTG
  python tools/fit_sparc_baselines.py --galaxy NGC3198 --model nfw --sparc-dir external_data/Rotmod_LTG

Notes:
- Uses rotmod components from SPARC: V_gas, V_disk, V_bulge.
- Assumes rotmod V_disk and V_bulge are computed at base M/L=0.5 (disk) and 0.7 (bulge), consistent with sparc_io.
- GR (baryons-only): v_model = vbar = sqrt(V_gas^2 + V_disk^2 + V_bulge^2).
- NFW: v_model = sqrt(vbar^2 + v_nfw^2), where v_nfw from models/nfw.py.
- Reports best-fit params and chi2/dof.
"""
from __future__ import annotations
import argparse
import numpy as np
import pathlib

import sys
# Ensure project root is on sys.path so 'utils' and 'models' can be imported when running as a script
try:
    import pathlib as _pl
    _ROOT = str(_pl.Path(__file__).resolve().parents[1])
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)
except Exception:
    pass

from utils.Utilities.sparc_io import load_single_sparc_galaxy
from models.nfw import v_model_gr, v_model_nfw

try:
    from scipy.optimize import minimize
except Exception:
    minimize = None


def build_vbar(gal):
    Vg = np.asarray(gal['V_gas_comp_kms'], dtype=float)
    Vd = np.asarray(gal['V_disk_comp_kms'], dtype=float)
    Vb = np.asarray(gal['V_bulge_comp_kms'], dtype=float)
    vbar = np.sqrt(np.clip(Vg**2 + Vd**2 + Vb**2, 0.0, None))
    return vbar


def chi2(v_model, v_obs, e_v, sigma_floor=0.0):
    e = np.asarray(e_v, dtype=float)
    # Apply velocity error floor in quadrature
    if sigma_floor and sigma_floor > 0:
        e = np.sqrt(e**2 + float(sigma_floor)**2)
    e = np.where(e > 0, e, np.maximum(1.0, 0.05*np.maximum(v_obs, 1.0)))
    r = (v_obs - v_model) / e
    return float(np.sum(r*r))


def fit_nfw(R_kpc, vbar, v_obs, e_v, sigma_floor=0.0):
    if minimize is None:
        # Simple coarse grid fallback if scipy not available
        best = (np.inf, 120.0, 10.0)
        for V200 in np.linspace(40, 300, 66):
            for c in np.linspace(2, 25, 47):
                vmod = v_model_nfw(R_kpc, vbar, V200, c)
                c2 = chi2(vmod, v_obs, e_v, sigma_floor=sigma_floor)
                if c2 < best[0]:
                    best = (c2, V200, c)
        chi2_best, V200_best, c_best = best
        dof = max(len(R_kpc) - 2, 1)
        return {"V200": V200_best, "c": c_best, "chi2": chi2_best, "chi2_dof": chi2_best/dof}

    # With scipy: do a local minimize with sensible start
    v2000, c0 = 120.0, 10.0
    x0 = np.array([v2000, c0], dtype=float)

    def objective(x):
        V200, c = float(x[0]), float(x[1])
        if not (10.0 <= V200 <= 400.0 and 1.0 <= c <= 40.0):
            return 1e12 + np.sum(np.square(x))
        vmod = v_model_nfw(R_kpc, vbar, V200, c)
        return chi2(vmod, v_obs, e_v, sigma_floor=sigma_floor)

    res = minimize(objective, x0, method="Nelder-Mead", options={"maxiter": 5000, "xatol": 1e-3, "fatol": 1e-3})
    V200_best, c_best = float(res.x[0]), float(res.x[1])
    vmod = v_model_nfw(R_kpc, vbar, V200_best, c_best)
    chi2_best = chi2(vmod, v_obs, e_v, sigma_floor=sigma_floor)
    dof = max(len(R_kpc) - 2, 1)
    return {"V200": V200_best, "c": c_best, "chi2": chi2_best, "chi2_dof": chi2_best/dof}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--galaxy", required=True, help="SPARC galaxy ID, e.g., NGC3198")
    ap.add_argument("--sparc-dir", default="data/sparc_data", help="Directory containing SPARC .dat files and MasterSheet_SPARC.csv")
    ap.add_argument("--model", choices=["gr", "nfw", "both"], default="both")
    ap.add_argument("--sigma-floor", type=float, default=0.0, help="Velocity error floor (km/s) added in quadrature")
    args = ap.parse_args()

    gal = load_single_sparc_galaxy(args.galaxy, sparc_dir=pathlib.Path(args.sparc_dir))
    if gal is None:
        print(f"Failed to load galaxy {args.galaxy}")
        return 2

    R = np.asarray(gal['R_kpc'], dtype=float)
    v_obs = np.asarray(gal['V_obs'], dtype=float)
    e_v = np.asarray(gal['e_V_obs'], dtype=float)
    vbar = build_vbar(gal)

    if args.model in ("gr", "both"):
        v_gr = v_model_gr(vbar)
        c2 = chi2(v_gr, v_obs, e_v, sigma_floor=args.sigma_floor)
        dof = max(len(R) - 0, 1)
        print(f"GR(baryons-only): chi2={c2:.2f}, chi2/dof={c2/dof:.2f}")

    if args.model in ("nfw", "both"):
        res = fit_nfw(R, vbar, v_obs, e_v, sigma_floor=args.sigma_floor)
        print(f"NFW fit: V200={res['V200']:.2f} km/s, c={res['c']:.2f}, chi2={res['chi2']:.2f}, chi2/dof={res['chi2_dof']:.2f}")


if __name__ == "__main__":
    main()

