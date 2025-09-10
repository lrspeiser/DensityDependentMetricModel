#!/usr/bin/env python3
# tools/replot_lensing_stack.py
# Re-plot the stacked ΔΣ figure from an existing Source-Data CSV, applying a small
# positive floor for log-scale stability so the 16–84% band does not collapse to 0.
# This is a plotting-only utility; the stack computation itself is in scripts/next_steps_from_run.py.

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser(description="Replot stacked ΔΣ with a log-safe floor from Source-Data CSV")
    ap.add_argument("--source", required=True, help="Path to lensing_metric_stack_source.csv")
    ap.add_argument("--out", required=True, help="Output PNG path for the figure")
    args = ap.parse_args()

    src = Path(args.source)
    if not src.exists():
        raise FileNotFoundError(f"Source CSV not found: {src}")

    df = pd.read_csv(src)
    # Expected columns
    cols_base = ["R_kpc", "DeltaSigma_mean", "DeltaSigma_p16", "DeltaSigma_p84"]
    for c in cols_base:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}' in {src}")

    R = df["R_kpc"].to_numpy(float)
    mean = df["DeltaSigma_mean"].to_numpy(float)
    p16  = df["DeltaSigma_p16"].to_numpy(float)
    p84  = df["DeltaSigma_p84"].to_numpy(float)

    # Optional systematics bands
    mean_sys = df["DeltaSigma_mean_sys"].to_numpy(float) if "DeltaSigma_mean_sys" in df.columns else None

    # Positive floor based on data dynamic range
    try:
        peak = float(np.nanmax([np.nanmax(mean), np.nanmax(p84)]))
        eps = max(1e-16, 1e-12 * peak)
    except Exception:
        eps = 1e-16

    mean_plot = np.maximum(mean, eps)
    p16_plot  = np.maximum(p16,  eps)
    p84_plot  = np.maximum(p84,  eps)

    plt.figure(figsize=(6.8, 4.4))
    plt.loglog(R, mean_plot, 'k-', lw=2, label='ΔΣ (RAR metric) mean')
    plt.fill_between(R, p16_plot, p84_plot, color='gray', alpha=0.3, label='16–84%')
    if mean_sys is not None:
        mean_sys_plot = np.maximum(mean_sys, eps)
        plt.loglog(R, mean_sys_plot, 'r-', lw=1.6, alpha=0.8, label='ΔΣ + systematics')

    plt.xlabel('R (kpc)')
    plt.ylabel('ΔΣ (Msun/kpc^2)')
    plt.title('Stacked ΔΣ from metric predictions (per-lens average)')
    plt.grid(alpha=0.3, which='both')
    plt.legend(frameon=False)
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(outp, dpi=140); plt.close()
    print(f"Wrote {outp}")


if __name__ == "__main__":
    main()

