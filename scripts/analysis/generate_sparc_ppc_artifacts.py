#!/usr/bin/env python3
"""
Aggregate SPARC residuals from per-galaxy overlay source CSVs to produce:
- docs/metrics/sparc_fit_quality.json (summary of reduced-chi2 across the sample)
- docs/figures/sparc_ppc_panel.png (histogram of standardized residuals z)
- docs/metrics/sparc_ppc_hist.csv (histogram data)

Inputs expected (created by scripts/next_steps_from_run.py):
  results/next_steps/<run_name>/sparc_overlay_*_source.csv
Each source CSV must include columns: R_kpc, V_obs_kms, e_V_obs_kms, Vbar_kms, V_model_kms.
"""
from __future__ import annotations
import os, glob, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "next_steps"
DOCS = ROOT / "docs"


def find_latest_run() -> Path:
    cands = sorted((p for p in RESULTS.glob("enhanced_*") if p.is_dir()), key=lambda p: p.stat().st_mtime)
    if not cands:
        raise FileNotFoundError(f"No enhanced_* run found under {RESULTS}")
    return cands[-1]


def load_overlays(run_dir: Path):
    files = sorted(run_dir.glob("sparc_overlay_*_source.csv"))
    return files


def main() -> int:
    run_dir = find_latest_run()
    files = load_overlays(run_dir)
    if not files:
        print(f"[warn] No SPARC overlay source CSVs found under {run_dir}")
        return 1

    all_rows = []
    per_gal_stats = []

    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"[warn] Failed to read {f}: {e}")
            continue
        # Column normalization
        rename = {}
        for c in df.columns:
            cl = c.lower()
            if cl in ("v_obs_kms", "vobs_kms", "v_obs"):
                rename[c] = "V_obs"
            elif cl in ("e_v_obs_kms", "evobs_kms", "e_v_obs"):
                rename[c] = "e_V_obs"
            elif cl in ("vbar_kms", "v_bar_kms", "v_bar"):
                rename[c] = "Vbar"
            elif cl in ("v_model_kms", "vmodel_kms", "v_model"):
                rename[c] = "V_model"
            elif cl in ("r_kpc", "r"):
                rename[c] = "R_kpc"
        df = df.rename(columns=rename)
        missing = [k for k in ("V_obs", "e_V_obs", "V_model") if k not in df.columns]
        if missing:
            print(f"[warn] Skipping {f.name}: missing columns {missing}")
            continue
        v_obs = pd.to_numeric(df["V_obs"], errors="coerce")
        v_mod = pd.to_numeric(df["V_model"], errors="coerce")
        e_v = pd.to_numeric(df["e_V_obs"], errors="coerce")
        # Apply headline floor policy for standardization
        SIGMA_FLOOR = 6.0
        FRAC_FLOOR = 0.05
        sigma_eff = np.sqrt(e_v**2 + SIGMA_FLOOR**2 + (FRAC_FLOOR * v_obs)**2)
        z = (v_obs - v_mod) / sigma_eff.replace(0, np.nan)
        z = z.replace([np.inf, -np.inf], np.nan)
        zc = z.dropna()
        if len(zc) == 0:
            continue
        chi2 = float(np.sum(zc.to_numpy() ** 2))
        dof = max(int(len(zc)) - 1, 1)
        chi2_red = chi2 / dof
        per_gal_stats.append(chi2_red)
        all_rows.append(zc)

    if not all_rows:
        print("[warn] No valid residuals aggregated.")
        return 1

    z_all = pd.concat(all_rows, ignore_index=True)

    # Summary JSON
    DOCS.joinpath("metrics").mkdir(parents=True, exist_ok=True)
    fit_json = {
        "n_galaxies": len(per_gal_stats),
        "n_points": int(z_all.shape[0]),
        "chi2_red_median": float(np.median(per_gal_stats)),
        "chi2_red_mean": float(np.mean(per_gal_stats)),
        "chi2_red_p16": float(np.percentile(per_gal_stats, 16)),
        "chi2_red_p84": float(np.percentile(per_gal_stats, 84)),
        "z_raw_p50": float(np.median(z_all)),
        "z_raw_p16": float(np.percentile(z_all, 16)),
        "z_raw_p84": float(np.percentile(z_all, 84)),
        "sigma_floor_kms": 6.0,
        "obs_frac_sigma": 0.05,
        "notes": "Standardized residuals z=(Vobs−Vmodel)/sqrt(σ^2 + σ_floor^2 + (f·V_obs)^2). Histogram clipped to [−5,5] for display.",
    }
    (DOCS/"metrics"/"sparc_fit_quality.json").write_text(json.dumps(fit_json, indent=2))

    # Histogram panel
    DOCS.joinpath("figures").mkdir(parents=True, exist_ok=True)
    z_clip = z_all.clip(-5, 5)
    counts, edges = np.histogram(z_clip, bins=30, density=True)
    x = 0.5 * (edges[:-1] + edges[1:])
    plt.figure(figsize=(6.4, 4.2))
    plt.bar(x, counts, width=(edges[1]-edges[0]), alpha=0.6, color="tab:blue", label="SPARC standardized residuals")
    xs = np.linspace(-5, 5, 400)
    gauss = 1.0/np.sqrt(2*np.pi) * np.exp(-0.5 * xs**2)
    plt.plot(xs, gauss, "k--", lw=1.2, label="N(0,1)")
    plt.xlabel("z = (V_obs − V_model) / σ_V")
    plt.ylabel("Density")
    plt.title("SPARC PPC: standardized residuals")
    plt.legend(frameon=False)
    out_png = DOCS/"figures"/"sparc_ppc_panel.png"
    plt.tight_layout(); plt.savefig(out_png, dpi=140); plt.close()

    # Save histogram data
    hist_df = pd.DataFrame({"bin_lo": edges[:-1], "bin_hi": edges[1:], "density": counts})
    (DOCS/"metrics"/"sparc_ppc_hist.csv").write_text(hist_df.to_csv(index=False))

    print("Generated:")
    print(" -", DOCS/"metrics"/"sparc_fit_quality.json")
    print(" -", out_png)
    print(" -", DOCS/"metrics"/"sparc_ppc_hist.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

