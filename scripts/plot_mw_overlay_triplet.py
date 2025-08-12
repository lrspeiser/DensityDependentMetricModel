#!/usr/bin/env python3
"""
Milky Way rotation curve overlay: GR vs TFR (tidal_band) vs NFW.

- Loads Gaia DR3 binned data via the same local-only loader as tools/plot_rotation_comparison.py
- Extracts median parameter values from the provided run dirs (GR, TFR, NFW)
- Plots three curves on top of the Gaia medians with 16–84% band

Usage:
  python scripts/plot_mw_overlay_triplet.py \
    --gr runs/gr_20250811_232403 \
    --tfr runs/tidal_band_20250810_102330 \
    --nfw runs/nfw_20250812_082825 \
    --out images/rotation_comparison_triplet.png
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
from typing import Dict, Any, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Repo paths and imports shared with tools/plot_rotation_comparison.py
REPO_ROOT = Path(__file__).resolve().parents[1]
for p in [REPO_ROOT, REPO_ROOT / "core", REPO_ROOT / "runners", REPO_ROOT / "tools"]:
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

# Reuse helpers from tools/plot_rotation_comparison.py when possible
from tools.plot_rotation_comparison import (
    load_gaia_local_or_fail, process_gaia_data,
    bin_by_radius, default_baryon_params, default_tidal_band_params,
    build_param_dict, _load_params_from_run
)

# Backend model bridges (CPU or CuPy)
try:
    from density_metric2 import v_total_kms as _v_total_cpu, v_baryon_total_newtonian_kms as _v_baryon_cpu
    def v_total_kms(R_kpc: np.ndarray, params: Dict[str, Any], xi_type: str) -> np.ndarray:
        return _v_total_cpu(R_kpc, params, xi_type=xi_type)
    def v_baryon_total_newtonian_kms(R_kpc: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        return _v_baryon_cpu(R_kpc, params)
except Exception:
    import cupy as cp  # type: ignore
    from core.density_metric_cupy import v_total_kms_cupy as _v_total_cu
    try:
        from core.density_metric_cupy import to_cupy_array as _to_cu, to_numpy_array as _to_np  # type: ignore
    except Exception:
        _to_cu = lambda x: cp.asarray(x)
        _to_np = lambda x: cp.asnumpy(x)
    def v_total_kms(R_kpc: np.ndarray, params: Dict[str, Any], xi_type: str) -> np.ndarray:
        R_cu = _to_cu(np.asarray(R_kpc, dtype=np.float32))
        v_cu = _v_total_cu(R_cu, params, xi_type=xi_type)
        return _to_np(v_cu)
    def v_baryon_total_newtonian_kms(R_kpc: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        R_cu = _to_cu(np.asarray(R_kpc, dtype=np.float32))
        v_cu = _v_total_cu(R_cu, params, xi_type='gr')
        return _to_np(v_cu)


def _params_from_runs(gr_dir: Optional[str], tfr_dir: Optional[str], nfw_dir: Optional[str]) -> Tuple[dict, dict, dict]:
    bary = default_baryon_params()
    tfr = default_tidal_band_params()
    nfw_params: dict = {}
    # Update from GR
    if gr_dir:
        gp = _load_params_from_run(Path(gr_dir)) or {}
        # Update baryon parameters
        for k in list(bary.keys()):
            if k in gp:
                bary[k] = gp[k]
    # Update from TFR
    if tfr_dir:
        tp = _load_params_from_run(Path(tfr_dir)) or {}
        for k in list(bary.keys()):
            if k in tp:
                bary[k] = tp[k]
        for key in ("rho_c_solar_kpc3", "gamma_exp", "lambda_max", "T0", "sigma_lnT", "wmin"):
            if key in tp:
                tfr[key] = tp[key]
    # Extract NFW params (halo + possibly baryon updates)
    if nfw_dir:
        npz = Path(nfw_dir) / "posterior_samples.npz"
        if npz.exists():
            try:
                data = np.load(str(npz))
                names = None
                if "names" in data:
                    names = [str(n) for n in data["names"]]
                elif "param_names" in data:
                    names = [str(n) for n in data["param_names"]]
                samples = None
                for key in ("samples", "posterior_samples", "xs"):
                    if key in data:
                        samples = np.asarray(data[key]); break
                weights = None
                for key in ("weights", "w"):
                    if key in data:
                        weights = np.asarray(data[key], dtype=float); break
                if samples is not None:
                    if samples.ndim == 1:
                        samples = samples.reshape(-1, 1)
                    N = samples.shape[0]
                    if weights is None or weights.size != N:
                        weights = np.ones(N) / float(N)
                    else:
                        s = float(np.sum(weights)); weights = weights / s if s > 0 else np.ones(N)/float(N)
                    # Weighted median per parameter
                    if names is None or len(names) != samples.shape[1]:
                        names = [f"param_{i}" for i in range(samples.shape[1])]
                    order = np.argsort(samples, axis=0)
                    for j, name in enumerate(names):
                        idx = order[:, j]
                        xs = samples[idx, j]
                        ws = weights[idx]
                        cdf = np.cumsum(ws); cdf /= cdf[-1]
                        p50 = float(np.interp(0.5, cdf, xs))
                        # Try to update baryon keys if present; otherwise assume it's a halo param and store in nfw_params
                        if name in bary:
                            bary[name] = p50
                        else:
                            nfw_params[name] = p50
            except Exception:
                pass
    # Ensure halo is included for NFW curve
    nfw_params = dict(nfw_params)
    nfw_params['include_halo'] = True
    return bary, tfr, nfw_params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gr", required=True, help="Path to GR run directory")
    ap.add_argument("--tfr", required=True, help="Path to TFR (tidal_band) run directory")
    ap.add_argument("--nfw", required=True, help="Path to NFW run directory")
    ap.add_argument("--out", default=str(REPO_ROOT / "images" / "rotation_comparison_triplet.png"))
    args = ap.parse_args()

    # Load Gaia and bin
    df = load_gaia_local_or_fail(REPO_ROOT)
    df = process_gaia_data(df)
    R_kpc = df["R_kpc"].values.astype(np.float64)
    v_obs = df["v_obs"].values.astype(np.float64)
    bins = np.linspace(2.0, 30.0, 29)
    R_centers, v_med, v_lo, v_hi = bin_by_radius(R_kpc, v_obs, bins=bins)
    R_grid = np.linspace(2.0, 30.0, 400).astype(np.float64)

    # Parameters
    bary, tfr, nfw_params = _params_from_runs(args.gr, args.tfr, args.nfw)
    params_tfr = build_param_dict(bary, tfr)

    # Curves
    v_gr = v_baryon_total_newtonian_kms(R_grid, bary)
    v_tfr = v_total_kms(R_grid, params_tfr, xi_type='tidal_band')
    v_nfw = v_total_kms(R_grid, {**bary, **nfw_params}, xi_type='nfw')

    # Plot
    import math
    plt.figure(figsize=(11, 8))
    valid = np.isfinite(v_med)
    plt.plot(R_centers[valid], v_med[valid], color="#4D4D4D", lw=2, label="Gaia: median stellar speed")
    band_valid = np.isfinite(v_lo) & np.isfinite(v_hi)
    plt.fill_between(R_centers[band_valid], v_lo[band_valid], v_hi[band_valid], color="#A6A6A6", alpha=0.25, label="Gaia: 16–84 percentile")

    plt.plot(R_grid, v_gr, "b--", lw=2, label="GR (baryon-only)")
    plt.plot(R_grid, v_nfw, color="#2CA02C", ls="-.", lw=2.5, label="ΛCDM/NFW")
    plt.plot(R_grid, v_tfr, "r-", lw=2.5, label="TFR (tidal_band)")

    plt.xlabel("Galactocentric radius R (kpc)")
    plt.ylabel("Circular speed v (km/s)")
    plt.title("Milky Way: GR vs ΛCDM/NFW vs TFR (matched runs)")
    plt.grid(True, alpha=0.3)
    plt.xlim(2, 30)
    ymax = np.nanmax([np.nanmax(v_med), np.nanmax(v_gr), np.nanmax(v_tfr), np.nanmax(v_nfw)])
    plt.ylim(0, max(300, float(ymax) + 40))
    plt.legend(frameon=False)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
