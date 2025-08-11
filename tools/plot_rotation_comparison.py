#!/usr/bin/env python3
"""
Plot Milky Way rotation curve comparison:
- Binned Gaia observed median speeds vs radius
- GR (baryon-only) prediction
- TFR (Tidal Field Relativity) prediction using the 'tidal_band' xi model

Usage examples:
  # Use default, reasonable parameters (no run dirs required)
  python tools/plot_rotation_comparison.py

  # Use fitted parameters from your runs (recommended)
  python tools/plot_rotation_comparison.py \
    --gr_run runs/gr_matched \
    --er_run runs/er_tidal_band

Outputs:
  images/rotation_comparison_tidal_band.png

Notes:
- If --gr_run/--er_run are provided and contain posterior_samples.npz (or
  post_analysis/params_summary.csv), the plotted curves will use the weighted
  median (p50) of each parameter from those runs.
- Otherwise, sensible defaults are used.
"""
import os
import sys
from pathlib import Path
import argparse
import csv
import json
import numpy as np
import matplotlib.pyplot as plt

# Allow running from repo root or from this file's directory
REPO_ROOT = Path(__file__).resolve().parents[1]
# Add repo root and common subdirs for flexible imports
for p in [REPO_ROOT, REPO_ROOT / "core", REPO_ROOT / "runners"]:
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from typing import Tuple, Callable, Dict, Any, Optional

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


def _weighted_quantiles_from_npz(npz_path: Path) -> Optional[Dict[str, float]]:
    try:
        data = np.load(str(npz_path))
        # Prefer explicit param names; support both 'names' and 'param_names'
        names = None
        if "names" in data:
            names = [str(n) for n in data["names"]]
        elif "param_names" in data:
            names = [str(n) for n in data["param_names"]]
        # Samples can be stored under several keys
        samples = None
        for key in ("samples", "posterior_samples", "xs"):
            if key in data:
                samples = np.asarray(data[key])
                break
        # Weights (optional)
        weights = None
        for key in ("weights", "w"):
            if key in data:
                weights = np.asarray(data[key], dtype=float)
                break
        if samples is None:
            return None
        if samples.ndim == 1:
            samples = samples.reshape(-1, 1)
        N = samples.shape[0]
        if weights is None or weights.size != N:
            weights = np.ones(N, dtype=float) / float(N)
        else:
            s = float(np.sum(weights))
            weights = weights / s if s > 0 else np.ones(N, dtype=float) / float(N)
        if names is None or len(names) != samples.shape[1]:
            names = [f"param_{i}" for i in range(samples.shape[1])]
        # Compute weighted median (p50) for each parameter
        order = np.argsort(samples, axis=0)
        params_p50: Dict[str, float] = {}
        for j, name in enumerate(names):
            idx = order[:, j]
            xs = samples[idx, j]
            ws = weights[idx]
            cdf = np.cumsum(ws)
            cdf /= cdf[-1]
            p50 = float(np.interp(0.5, cdf, xs))
            params_p50[name] = p50
        return params_p50
    except Exception:
        return None


def _load_params_from_run(run_dir: Path) -> Optional[Dict[str, float]]:
    # Prefer post_analysis/params_summary.csv (p50 column) if exists
    csv_path = run_dir / "post_analysis" / "params_summary.csv"
    if csv_path.exists():
        try:
            params: Dict[str, float] = {}
            with open(csv_path, newline="", encoding="utf-8") as cf:
                reader = csv.DictReader(cf)
                for row in reader:
                    name = row.get("parameter")
                    p50 = row.get("p50")
                    if name is None or p50 is None:
                        continue
                    try:
                        params[name] = float(p50)
                    except Exception:
                        continue
            if params:
                # If parameters are generic (param_0, param_1, ...), try to remap using npz param_names
                generic = all(k.startswith("param_") for k in params.keys())
                if generic:
                    npz_path = run_dir / "posterior_samples.npz"
                    if npz_path.exists():
                        try:
                            data = np.load(str(npz_path))
                            if "param_names" in data:
                                names = [str(n) for n in data["param_names"]]
                                remapped: Dict[str, float] = {}
                                for i, real_name in enumerate(names):
                                    key = f"param_{i}"
                                    if key in params:
                                        remapped[real_name] = params[key]
                                if remapped:
                                    return remapped
                        except Exception:
                            pass
                return params
        except Exception:
            pass
    # Fallback: posterior_samples.npz
    npz_path = run_dir / "posterior_samples.npz"
    if npz_path.exists():
        return _weighted_quantiles_from_npz(npz_path)
    return None


def _maybe_update_params_from_runs(baryon: dict, xi_params: dict, gr_run: Optional[str], er_run: Optional[str]) -> tuple[dict, dict]:
    b = dict(baryon)
    x = dict(xi_params)
    # Update baryon params from GR run if present
    if gr_run:
        gr_p = _load_params_from_run(Path(gr_run))
        if gr_p:
            b.update({k: v for k, v in gr_p.items() if k in b or k.startswith("M_") or k.endswith("_kpc")})
    # Update xi + baryon params from ER run if present
    if er_run:
        er_p = _load_params_from_run(Path(er_run))
        if er_p:
            # First update baryon keys that match
            for k in list(b.keys()):
                if k in er_p:
                    b[k] = er_p[k]
            # Then update xi params for known keys
            for key in ("rho_c_solar_kpc3", "gamma_exp", "lambda_max", "T0", "sigma_lnT", "wmin"):
                if key in er_p:
                    x[key] = er_p[key]
    return b, x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gr_run", type=str, default=None, help="Path to GR run directory (to read fitted params)")
    ap.add_argument("--er_run", type=str, default=None, help="Path to ER (tidal_band) run directory (to read fitted params)")
    ap.add_argument("--no_split_extrapolation", action="store_true", help="Do not split ER curve; draw as one line")
    ap.add_argument("--no_shade_extrapolation", action="store_true", help="Do not shade extrapolation region")
    args = ap.parse_args()

    # Output directory
    out_dir = REPO_ROOT / "images"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load and process Gaia data (local-only; no remote queries)
    print("Loading Gaia data from local cache/slices...")
    df = load_gaia_local_or_fail(REPO_ROOT)
    df = process_gaia_data(df)

    R_kpc = df["R_kpc"].values.astype(np.float64)
    v_obs = df["v_obs"].values.astype(np.float64)
    R_data_max = float(np.nanmax(R_kpc)) if R_kpc.size else 0.0

    # 2) Bin the data
    bins = np.linspace(2.0, 30.0, 29)
    R_centers, v_med, v_lo, v_hi = bin_by_radius(R_kpc, v_obs, bins=bins)

    # 3) Compute GR (baryon-only) and DDMM curves on a smooth grid
    R_grid = np.linspace(2.0, 30.0, 400).astype(np.float64)

    baryon = default_baryon_params()
    tidal_band = default_tidal_band_params()

    # Optionally override with fitted parameters from runs
    baryon, tidal_band = _maybe_update_params_from_runs(baryon, tidal_band, args.gr_run, args.er_run)

    ddmm_params = build_param_dict(baryon, tidal_band)

    print("Computing GR (baryon-only) curve...")
    v_gr = v_baryon_total_newtonian_kms(R_grid, baryon)

    xi_type = "tidal_band"
    if XI_FUNCTION_MAP is not None and xi_type not in XI_FUNCTION_MAP:
        raise RuntimeError(f"xi_type '{xi_type}' not available. Available: {list(XI_FUNCTION_MAP.keys())}")

    print(f"Computing TFR curve with xi_type='{xi_type}' (backend: {'CuPy' if _use_cupy_backend else 'CPU'})...")
    v_ddmm = v_total_kms(R_grid, ddmm_params, xi_type=xi_type)
    # Report theoretical enhancement cap based on lambda_max, if present
    lam = float(ddmm_params.get('lambda_max', 0.0))
    xi_cap = 1.0 + lam
    print(f"Theoretical TFR enhancement cap xi_max = 1 + lambda_max = {xi_cap:.3f} (cannot reach 100x).")

    # 4) Plot
    print("Plotting...")
    plt.figure(figsize=(11, 8))

    valid = np.isfinite(v_med)
    plt.plot(R_centers[valid], v_med[valid], color="#4D4D4D", lw=2, label="Gaia: median stellar speed")
    band_valid = np.isfinite(v_lo) & np.isfinite(v_hi)
    plt.fill_between(R_centers[band_valid], v_lo[band_valid], v_hi[band_valid], color="#A6A6A6", alpha=0.25, label="Gaia: 16–84 percentile")

    plt.plot(R_grid, v_gr, "b--", lw=2, label="GR (baryon-only)")
    # ER curve: split into data-constrained vs extrapolation beyond Gaia R_max
    if args.no_split_extrapolation:
        plt.plot(R_grid, v_ddmm, "r-", lw=2.5, label=f"TFR ({xi_type})")
    else:
        m_in = R_grid <= R_data_max
        m_out = ~m_in
        if np.any(m_in):
            plt.plot(R_grid[m_in], v_ddmm[m_in], "r-", lw=2.5, label=f"TFR ({xi_type}) — constrained")
        if np.any(m_out):
            plt.plot(R_grid[m_out], v_ddmm[m_out], color="#FF8C00", ls="--", lw=2.5, label=f"TFR ({xi_type}) — extrapolation")

    plt.xlabel("Galactocentric radius R (kpc)")
    plt.ylabel("Circular speed v (km/s)")
    plt.title("Milky Way Rotation Curve: Data vs GR vs TFR (tidal_band)")
    plt.grid(True, alpha=0.3)
    # Mark and optionally shade extrapolation region beyond Gaia data
    if not args.no_split_extrapolation and R_data_max > 0:
        plt.axvline(R_data_max, color="k", ls=":", alpha=0.6, label=f"Max Gaia R ≈ {R_data_max:.1f} kpc")
        if not args.no_shade_extrapolation:
            plt.axvspan(R_data_max, R_grid.max(), color="#FFA500", alpha=0.08)
    plt.xlim(2, 30)
    ymax = np.nanmax([np.nanmax(v_med), np.nanmax(v_gr), np.nanmax(v_ddmm)])
    plt.ylim(0, max(300, float(ymax) + 40))
    plt.legend(frameon=False)

    out_file = out_dir / "rotation_comparison_tidal_band.png"
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    print(f"Saved: {out_file}")
    print(f"Saved: {out_file}")


if __name__ == "__main__":
    main()

