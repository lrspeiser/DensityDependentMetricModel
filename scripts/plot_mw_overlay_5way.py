#!/usr/bin/env python3
"""
Milky Way rotation curve overlay: GR vs ΛCDM/NFW vs three tidal models (band2, ratio, noisyor).

- Loads Gaia DR3 binned data via the same local-only loader as tools/plot_rotation_comparison.py
- Extracts median parameter values from the provided run dirs (GR, NFW, tidal runs)
- Plots five curves on top of the Gaia medians with 16–84% band

Usage example:
  python scripts/plot_mw_overlay_5way.py \
    --gr runs/gr_20250812_113949 \
    --nfw runs/nfw_20250812_114008 \
    --tidal-band2 runs/tidal_band2_20250814_104418 \
    --tidal-ratio runs/tidal_ratio_20250814_104554 \
    --tidal-noisyor runs/tidal_noisyor_20250814_104602 \
    --out images/rotation_comparison_5way.png
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

# Reuse helpers from tools.plot_rotation_comparison
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


def _merge_params(base: dict, updates: Optional[dict]) -> dict:
    out = dict(base)
    if updates:
        for k, v in updates.items():
            out[k] = v
    return out


def _extract_baryon_and_xi(gr_dir: Optional[str]) -> dict:
    bary = default_baryon_params()
    if gr_dir:
        gp = _load_params_from_run(Path(gr_dir)) or {}
        for k in list(bary.keys()):
            if k in gp:
                bary[k] = gp[k]
    return bary


def _extract_nfw_params(nfw_dir: Optional[str], bary_ref: dict) -> dict:
    params: dict = {}
    if not nfw_dir:
        return params
    npz = Path(nfw_dir) / "posterior_samples.npz"
    if not npz.exists():
        return params
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
        if samples is None:
            return params
        if samples.ndim == 1:
            samples = samples.reshape(-1, 1)
        N = samples.shape[0]
        if weights is None or weights.size != N:
            weights = np.ones(N) / float(N)
        else:
            s = float(np.sum(weights)); weights = weights / s if s > 0 else np.ones(N)/float(N)
        if names is None or len(names) != samples.shape[1]:
            names = [f"param_{i}" for i in range(samples.shape[1])]
        order = np.argsort(samples, axis=0)
        for j, name in enumerate(names):
            idx = order[:, j]
            xs = samples[idx, j]
            ws = weights[idx]
            cdf = np.cumsum(ws); cdf /= cdf[-1]
            p50 = float(np.interp(0.5, cdf, xs))
            if name in bary_ref:
                # Skip baryon updates here; we want halo params separate
                continue
            params[name] = p50
    except Exception:
        pass
    params['include_halo'] = True
    return params


def _extract_params_from_run(run_dir: Optional[str]) -> dict:
    if not run_dir:
        return {}
    return _load_params_from_run(Path(run_dir)) or {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gr", required=True, help="Path to GR run directory")
    ap.add_argument("--nfw", required=True, help="Path to NFW run directory")
    ap.add_argument("--tidal-band2", dest="t_band2", required=True, help="Path to tidal_band2 run directory")
    ap.add_argument("--tidal-ratio", dest="t_ratio", required=True, help="Path to tidal_ratio run directory")
    ap.add_argument("--tidal-noisyor", dest="t_noisyor", required=True, help="Path to tidal_noisyor run directory")
    ap.add_argument("--out", default=str(REPO_ROOT / "images" / "rotation_comparison_5way.png"))
    args = ap.parse_args()

    # Load Gaia and bin
    df = load_gaia_local_or_fail(REPO_ROOT)
    df = process_gaia_data(df)
    R_kpc = df["R_kpc"].values.astype(np.float64)
    v_obs = df["v_obs"].values.astype(np.float64)
    bins = np.linspace(2.0, 30.0, 29)
    R_centers, v_med, v_lo, v_hi = bin_by_radius(R_kpc, v_obs, bins=bins)
    R_grid = np.linspace(2.0, 30.0, 400).astype(np.float64)

    # Base baryon parameters from GR run
    bary = _extract_baryon_and_xi(args.gr)

    # Curves
    v_gr = v_baryon_total_newtonian_kms(R_grid, bary)

    # NFW curve
    nfw_params = _extract_nfw_params(args.nfw, bary)
    v_nfw = v_total_kms(R_grid, {**bary, **nfw_params}, xi_type='nfw')

    # Tidal runs: load their fitted parameters and compute curves per xi type
    t_band2_params = _extract_params_from_run(args.t_band2)
    t_ratio_params = _extract_params_from_run(args.t_ratio)
    t_noisyor_params = _extract_params_from_run(args.t_noisyor)

    # Mark experimental xi allowances explicitly for plotting
    exp_flag = {'allow_experimental': True}
    v_t_band2 = v_total_kms(R_grid, {**bary, **t_band2_params, **exp_flag}, xi_type='tidal_band2')
    v_t_ratio = v_total_kms(R_grid, {**bary, **t_ratio_params, **exp_flag}, xi_type='tidal_ratio')
    v_t_noisyor = v_total_kms(R_grid, {**bary, **t_noisyor_params, **exp_flag}, xi_type='tidal_noisyor')

    # Plot
    plt.figure(figsize=(12, 8))
    valid = np.isfinite(v_med)
    plt.plot(R_centers[valid], v_med[valid], color="#4D4D4D", lw=2, label="Gaia: median stellar speed")
    band_valid = np.isfinite(v_lo) & np.isfinite(v_hi)
    plt.fill_between(R_centers[band_valid], v_lo[band_valid], v_hi[band_valid], color="#A6A6A6", alpha=0.25, label="Gaia: 16–84 percentile")

    plt.plot(R_grid, v_gr, "b--", lw=2, label="GR (baryons only)")
    plt.plot(R_grid, v_nfw, color="#2CA02C", ls="-.", lw=2.5, label="ΛCDM/NFW")

    plt.plot(R_grid, v_t_band2, color="#D62728", lw=2.5, label="Tidal band2")
    plt.plot(R_grid, v_t_ratio, color="#9467BD", lw=2.5, label="Tidal ratio")
    plt.plot(R_grid, v_t_noisyor, color="#FF7F0E", lw=2.5, label="Tidal noisyor")

    plt.xlabel("Galactocentric radius R (kpc)")
    plt.ylabel("Circular speed v (km/s)")
    plt.title("Milky Way Rotation Curve: GR vs NFW vs Three Tidal Models")
    plt.grid(True, alpha=0.3)
    plt.xlim(2, 30)
    ymax = np.nanmax([np.nanmax(v_med), np.nanmax(v_gr), np.nanmax(v_nfw), np.nanmax(v_t_band2), np.nanmax(v_t_ratio), np.nanmax(v_t_noisyor)])
    plt.ylim(0, max(300, float(ymax) + 40))
    plt.legend(frameon=False)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

