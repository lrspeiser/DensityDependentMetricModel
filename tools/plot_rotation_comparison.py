#!/usr/bin/env python3
"""
Plot Milky Way rotation curve comparison:
- Binned Gaia observed median speeds vs radius
- GR (baryon-only) prediction
- DDMM prediction using the 'tidal_band' xi model (with reasonable defaults)

Usage:
  python tools/plot_rotation_comparison.py

Outputs:
  images/rotation_comparison_tidal_band.png

Notes:
- You do NOT need sampling results to generate this figure. We use sensible default
  parameters for both the baryon model and the tidal_band xi(ρ) function. You can
  later update the parameter guesses with best-fit values to refine the curve.
"""
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Allow running from repo root or from this file's directory
REPO_ROOT = Path(__file__).resolve().parents[1]
# Add repo root and common subdirs for flexible imports
for p in [REPO_ROOT, REPO_ROOT / "core", REPO_ROOT / "runners"]:
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from typing import Tuple, Callable, Dict, Any

# Try to import CPU model (density_metric2) first; fallback to CuPy model (core.density_metric_cupy)
# We will provide compatibility wrappers so the rest of the script uses v_total_kms and v_baryon_total_newtonian_kms
XI_FUNCTION_MAP = None
_use_cupy_backend = False

try:
    from density_metric2 import v_total_kms as _v_total_cpu, v_baryon_total_newtonian_kms as _v_baryon_cpu, XI_FUNCTION_MAP as _XI_CPU
    def v_total_kms(R_kpc: np.ndarray, params: Dict[str, Any], xi_type: str) -> np.ndarray:
        return _v_total_cpu(R_kpc, params, xi_type=xi_type)
    def v_baryon_total_newtonian_kms(R_kpc: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        return _v_baryon_cpu(R_kpc, params)
    XI_FUNCTION_MAP = _XI_CPU
except Exception:
    # Fallback to CuPy backend
    try:
        import cupy as cp  # type: ignore
        from core.density_metric_cupy import v_total_kms_cupy as _v_total_cu
        # Optional helpers if available
        try:
            from core.density_metric_cupy import to_cupy_array as _to_cu, to_numpy_array as _to_np  # type: ignore
        except Exception:
            _to_cu = lambda x: cp.asarray(x)
            _to_np = lambda x: cp.asnumpy(x)
        # XI map may not be exposed here; define a minimal map with 'gr' pass-through
        # We won't rely on XI_FUNCTION_MAP contents when using CuPy fallback
        XI_FUNCTION_MAP = {"gr": "gr", "tidal_band": "tidal_band"}
        _use_cupy_backend = True
        def v_total_kms(R_kpc: np.ndarray, params: Dict[str, Any], xi_type: str) -> np.ndarray:
            # Convert numpy -> cupy, call cu backend, convert back
            R_cu = _to_cu(np.asarray(R_kpc, dtype=np.float32))
            v_cu = _v_total_cu(R_cu, params, xi_type=xi_type)
            return _to_np(v_cu)
        def v_baryon_total_newtonian_kms(R_kpc: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
            R_cu = _to_cu(np.asarray(R_kpc, dtype=np.float32))
            v_cu = _v_total_cu(R_cu, params, xi_type='gr')
            return _to_np(v_cu)
    except Exception as e:
        print("ERROR: Could not import either density_metric2 or core.density_metric_cupy.")
        print("Hint: run from the repo root (so REPO_ROOT is correct) or install CuPy if using the GPU backend.")
        raise

# data_io provides Gaia processing; we will do a local-only loader here
try:
    from data_io import process_gaia_data
except Exception:
    try:
        from core.data_io import process_gaia_data  # fallback path seen in CuPy runner
    except Exception as e:
        print("ERROR: Could not import data_io/core.data_io process_gaia_data.")
        raise

# Strict local-only Gaia loader (no remote queries)
import pandas as pd

def load_gaia_local_or_fail(repo_root: Path) -> pd.DataFrame:
    """Load Gaia data from local cached files only.

    Search order:
      1) external_data/gaia_sky_slices/all_sky_gaia.csv
      2) gaia_sky_slices/all_sky_gaia.csv
      3) Merge raw_L*.csv under external_data/gaia_sky_slices
      4) Merge raw_L*.csv under gaia_sky_slices
    If none found, raise with a clear message.
    """
    # 1/2: Pre-merged CSV candidates
    candidates_csv = [
        repo_root / "external_data" / "gaia_sky_slices" / "all_sky_gaia.csv",
        repo_root / "gaia_sky_slices" / "all_sky_gaia.csv",
    ]
    for c in candidates_csv:
        if c.exists():
            print(f"Loading merged Gaia cache: {c}")
            return pd.read_csv(c)

    # 3/4: Merge raw slices if present
    candidates_raw_dirs = [
        repo_root / "external_data" / "gaia_sky_slices",
        repo_root / "gaia_sky_slices",
    ]
    for d in candidates_raw_dirs:
        if d.exists():
            raw_files = sorted(d.glob("raw_L*.csv"))
            if raw_files:
                print(f"Merging {len(raw_files)} raw Gaia slice files in: {d}")
                dfs = []
                for f in raw_files:
                    try:
                        dfs.append(pd.read_csv(f))
                    except Exception as e:
                        print(f"Warning: failed to read {f.name}: {e}")
                if dfs:
                    df_all = pd.concat(dfs, ignore_index=True)
                    # Cache merged CSV for next time
                    out_csv = d / "all_sky_gaia.csv"
                    try:
                        df_all.to_csv(out_csv, index=False)
                        print(f"Cached merged Gaia data to: {out_csv}")
                    except Exception as e:
                        print(f"Warning: failed to write cache {out_csv}: {e}")
                    return df_all

    raise FileNotFoundError(
        "No local Gaia cache/slices found. Expected at external_data/gaia_sky_slices or gaia_sky_slices."
    )


def bin_by_radius(R_kpc: np.ndarray, v_obs: np.ndarray, bins: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bin velocities by radius, computing median and 16/84 percentiles."""
    idx = np.digitize(R_kpc, bins)
    R_centers = 0.5 * (bins[:-1] + bins[1:])
    v_med = np.full(len(R_centers), np.nan)
    v_lo = np.full(len(R_centers), np.nan)
    v_hi = np.full(len(R_centers), np.nan)
    for i in range(1, len(bins)):
        m = (idx == i)
        if np.any(m):
            vv = v_obs[m]
            v_med[i-1] = np.median(vv)
            v_lo[i-1] = np.percentile(vv, 16)
            v_hi[i-1] = np.percentile(vv, 84)
    return R_centers, v_med, v_lo, v_hi


def default_baryon_params() -> dict:
    """Reasonable Milky Way-like defaults (adjust to your preferred baseline)."""
    return {
        'M_disk_thin_solar': 5.0e10,
        'R_d_thin_kpc': 2.6,
        'h_z_thin_kpc': 0.30,
        'M_disk_thick_solar': 1.0e10,
        'R_d_thick_kpc': 3.6,
        'h_z_thick_kpc': 0.90,
        'M_bulge_solar': 1.4e10,
        'a_bulge_kpc': 0.50,
        'M_gas_solar': 1.5e10,
        'R_d_gas_kpc': 7.0,
        'h_z_gas_kpc': 0.15,
        # Component toggles used by model functions
        'include_disk_thin': True,
        'include_disk_thick': True,
        'include_bulge': True,
        'include_gas': True,
    }


def default_tidal_band_params() -> dict:
    """Starting guess for tidal_band xi(ρ) parameters. Tune as needed."""
    return {
        'rho_c_solar_kpc3': 1e16,  # characteristic density
        'gamma_exp': 3.0,          # slope parameter
        'lambda_max': 3.0,         # max enhancement
        'T0': 1e2,                 # characteristic timescale/temperature-like param
        'sigma_lnT': 0.8,          # width
        'wmin': 0.01,              # minimum band weight
    }


def build_param_dict(baryon_params: dict, xi_params: dict) -> dict:
    p = dict(baryon_params)
    p.update(xi_params)
    return p


def main():
    # Output directory
    out_dir = REPO_ROOT / "images"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load and process Gaia data (local-only; no remote queries)
    print("Loading Gaia data from local cache/slices...")
    df = load_gaia_local_or_fail(REPO_ROOT)
    df = process_gaia_data(df)

    R_kpc = df["R_kpc"].values.astype(np.float64)
    v_obs = df["v_obs"].values.astype(np.float64)

    # 2) Bin the data
    bins = np.linspace(2.0, 30.0, 29)
    R_centers, v_med, v_lo, v_hi = bin_by_radius(R_kpc, v_obs, bins=bins)

    # 3) Compute GR (baryon-only) and DDMM curves on a smooth grid
    R_grid = np.linspace(2.0, 30.0, 400).astype(np.float64)

    baryon = default_baryon_params()
    tidal_band = default_tidal_band_params()
    ddmm_params = build_param_dict(baryon, tidal_band)

    print("Computing GR (baryon-only) curve...")
    v_gr = v_baryon_total_newtonian_kms(R_grid, baryon)

    xi_type = "tidal_band"
    if XI_FUNCTION_MAP is not None and xi_type not in XI_FUNCTION_MAP:
        raise RuntimeError(f"xi_type '{xi_type}' not available. Available: {list(XI_FUNCTION_MAP.keys())}")

    print(f"Computing DDMM curve with xi_type='{xi_type}' (backend: {'CuPy' if _use_cupy_backend else 'CPU'})...")
    v_ddmm = v_total_kms(R_grid, ddmm_params, xi_type=xi_type)

    # 4) Plot
    print("Plotting...")
    plt.figure(figsize=(10, 7))

    valid = np.isfinite(v_med)
    plt.plot(R_centers[valid], v_med[valid], color="tab:gray", lw=2, label="Gaia: median stellar speed")
    band_valid = np.isfinite(v_lo) & np.isfinite(v_hi)
    plt.fill_between(R_centers[band_valid], v_lo[band_valid], v_hi[band_valid], color="tab:gray", alpha=0.2, label="Gaia: 16–84 percentile")

    plt.plot(R_grid, v_gr, "b--", lw=2, label="GR (baryon-only)")
    plt.plot(R_grid, v_ddmm, "r-", lw=2, label=f"DDMM ({xi_type})")

    plt.xlabel("Galactocentric radius R (kpc)")
    plt.ylabel("Circular speed v (km/s)")
    plt.title("Milky Way Rotation Curve: Data vs GR vs DDMM (tidal_band)")
    plt.grid(True, alpha=0.3)
    plt.xlim(2, 30)
    ymax = np.nanmax([np.nanmax(v_med), np.nanmax(v_gr), np.nanmax(v_ddmm)])
    plt.ylim(0, max(300, float(ymax) + 40))
    plt.legend()

    out_file = out_dir / "rotation_comparison_tidal_band.png"
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    print(f"Saved: {out_file}")


if __name__ == "__main__":
    main()

