#!/usr/bin/env python3
# tools/stack_lensing_gr_vs_rar.py
# Build GR (stars-only) and RAR (stars+phantom via xi) stacked ΔΣ from per-lens
# profile CSVs produced by scripts/next_steps_from_run.py. Writes a comparison
# CSV and an overlay figure with log-safe shading.

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _stack_delta_sigma(profile_dir: Path, *, which: str = "RAR", ngrid: int = 80):
    # Collect per-lens radii and ΔΣ for the requested component
    Rs = []
    DS = []
    for pf in sorted(profile_dir.glob("*_profiles.csv")):
        try:
            df = pd.read_csv(pf)
            R = df["R_kpc"].to_numpy(float)
            if which.upper() == "RAR":
                if "DeltaSigma_tot" in df.columns:
                    D = df["DeltaSigma_tot"].to_numpy(float)
                else:
                    # Reconstruct from mean_tot, Sigma_tot
                    m = df["mean_tot"].to_numpy(float)
                    s = df["Sigma_tot"].to_numpy(float)
                    D = np.maximum(m - s, 0.0)
            else:  # GR stars only
                m = df["mean_star"].to_numpy(float)
                s = df["Sigma_star"].to_numpy(float)
                D = np.maximum(m - s, 0.0)
            if len(R) >= 5 and np.all(np.isfinite(R)) and np.all(np.isfinite(D)):
                Rs.append(R)
                DS.append(D)
        except Exception:
            continue
    if not Rs:
        raise RuntimeError(f"No usable profile CSVs found in {profile_dir}")
    Rmin = max(r.min() for r in Rs)
    Rmax = min(r.max() for r in Rs)
    if not (Rmax > Rmin):
        raise RuntimeError("No overlapping R-range across lenses for stacking")
    Rgrid = np.logspace(np.log10(Rmin), np.log10(Rmax), int(ngrid))
    DSarr = []
    for R, D in zip(Rs, DS):
        DSarr.append(np.interp(Rgrid, R, D))
    DSarr = np.vstack(DSarr)
    mean = np.nanmean(DSarr, axis=0)
    p16 = np.nanpercentile(DSarr, 16, axis=0)
    p84 = np.nanpercentile(DSarr, 84, axis=0)
    return Rgrid, mean, p16, p84, DSarr.shape[0]


def main():
    ap = argparse.ArgumentParser(description="Stack lensing ΔΣ for GR vs RAR from per-lens profiles")
    ap.add_argument("--profiles-dir", required=True, help="Path to results/.../lensing_metric_profiles/")
    ap.add_argument("--out-csv", required=True, help="Output CSV for comparison stack")
    ap.add_argument("--out-png", required=True, help="Output PNG path for overlay figure")
    ap.add_argument("--ngrid", type=int, default=80, help="Number of R grid points for stack")
    args = ap.parse_args()

    profdir = Path(args.profiles_dir)
    Rr, mean_r, p16_r, p84_r, Nr = _stack_delta_sigma(profdir, which="RAR", ngrid=args.ngrid)
    Rg, mean_g, p16_g, p84_g, Ng = _stack_delta_sigma(profdir, which="GR", ngrid=args.ngrid)

    # Ensure grids align (they should by construction); if not, resample GR to RAR grid
    if not np.allclose(Rr, Rg):
        mean_g = np.interp(Rr, Rg, mean_g)
        p16_g  = np.interp(Rr, Rg, p16_g)
        p84_g  = np.interp(Rr, Rg, p84_g)
        Rg = Rr

    outp = Path(args.out_csv)
    outp.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "R_kpc": Rr,
        "DeltaSigma_RAR_mean": mean_r,
        "DeltaSigma_RAR_p16": p16_r,
        "DeltaSigma_RAR_p84": p84_r,
        "DeltaSigma_GR_mean": mean_g,
        "DeltaSigma_GR_p16": p16_g,
        "DeltaSigma_GR_p84": p84_g,
        "N": [min(Nr, Ng)]*len(Rr),
    })
    df.to_csv(outp, index=False)

    # Plot with log-safe floors
    try:
        peak = float(np.nanmax([np.nanmax(mean_r), np.nanmax(p84_r)]))
        eps = max(1e-16, 1e-12 * peak)
    except Exception:
        eps = 1e-16
    mr = np.maximum(mean_r, eps)
    pr16 = np.maximum(p16_r, eps)
    pr84 = np.maximum(p84_r, eps)
    mg = np.maximum(mean_g, eps)
    pg16 = np.maximum(p16_g, eps)
    pg84 = np.maximum(p84_g, eps)

    plt.figure(figsize=(6.8, 4.4))
    plt.loglog(Rr, mr, 'k-', lw=2, label='ΔΣ mean (RAR metric)')
    plt.fill_between(Rr, pr16, pr84, color='gray', alpha=0.3, label='RAR 16–84%')
    plt.loglog(Rr, mg, color='tab:blue', lw=2, label='ΔΣ mean (GR, stars)')
    plt.fill_between(Rr, pg16, pg84, color='lightblue', alpha=0.35, label='GR 16–84%')
    plt.xlabel('R (kpc)'); plt.ylabel('ΔΣ (Msun/kpc^2)')
    plt.title('Stacked ΔΣ — RAR vs GR (per-lens average)')
    plt.grid(alpha=0.3, which='both'); plt.legend(frameon=False)
    outpng = Path(args.out_png)
    outpng.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(outpng, dpi=140); plt.close()
    print(f"Wrote {outp}\nWrote {outpng}")


if __name__ == "__main__":
    main()

