#!/usr/bin/env python3
"""
utils/ppc.py

Posterior Predictive Check (PPC) helpers for SPARC galaxies.
- Load ER/GR model params from existing JSON artifacts
- Recompute model curves on SPARC rotmod radii using existing model utilities
- Compute residual envelopes (median and quantile bands) from posterior samples if available,
  otherwise from a point-estimate (degenerate ensemble) so figures can be produced without new runs.

Expected inputs:
- images/sparc_env_fit_<galaxy>.json (mode=fit) with params and file_rotmod
- Optionally, a posterior NPZ at the same stem with keys: samples (N,K), weights (N,), names (K,)

Outputs:
- Envelope dicts and plotting functions that save PNGs under images/
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, List
import json
import numpy as np
import matplotlib.pyplot as plt
import warnings
import sys

# Ensure repo root on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Reuse existing loaders and model utilities
from data_loaders.sparc_loader import load_rotmod  # type: ignore
from models.er_sparc import (
    v_bar_from_components,
    v_er_from_components,
    xi_log_normal_R,
)

# Optional style helper
try:
    from utils.plot_style import apply_paper_style  # type: ignore
except Exception:
    def apply_paper_style():
        plt.style.use("seaborn-v0_8-whitegrid")


def load_fit_json(json_path: Path) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def try_load_posterior_npz(stem_path: Path):
    """
    Attempt to load a posterior NPZ adjacent to a JSON artifact.
    Recognized names (in priority order):
    - <stem>_posterior.npz
    - <stem>.npz
    Returns (samples, weights, names) or (None, None, None)
    """
    candidates = [stem_path.with_name(stem_path.name + "_posterior.npz"),
                  stem_path.with_suffix(".npz")]
    for p in candidates:
        if p.exists():
            try:
                npz = np.load(p, allow_pickle=True)
                samples = npz["samples"] if "samples" in npz else None
                weights = npz["weights"] if "weights" in npz else None
                names = npz["names"] if "names" in npz else None
                if samples is not None:
                    return samples, weights, names
            except Exception:
                continue
    return None, None, None

def model_curve_from_params(params: dict, R: np.ndarray, Vgas: np.ndarray, Vdisk: np.ndarray, Vbul: np.ndarray, sanity: Optional[dict] = None) -> np.ndarray:
    """
    Build model curve for ER or GR from parameter dict at the SPARC radii.
    For ER: use v_er_from_components with ups_disk, ups_bulge and ER hyperparameters.
    For GR: simply v_bar_from_components.

    Notes:
    - Uses ER-on-SPARC log-normal-in-R proxy. If R0 is missing in params, falls back to
      sanity['R_peak_W'] when provided, else defaults to 15 kpc.
    - sigma width defaults to 0.7 if not provided (or maps from sigma_lnT if present).
    """
    ups_disk = float(params.get("ups_disk", 0.5))
    ups_bul = float(params.get("ups_bul", 0.7))

    # Identify if ER hyperparameters are present
    if any(k in params for k in ("lambda_max", "lnT0", "sigma_lnT", "w_min")):
        lambda_max = float(params.get("lambda_max", 3.0))
        # R0 can be recorded as R0 or R0_kpc; otherwise try sanity.R_peak_W
        if sanity is None:
            sanity = {}
        R0_kpc = float(params.get("R0", params.get("R0_kpc", sanity.get("R_peak_W", 15.0))))
        sigma_lnR = float(params.get("sigma_lnR", params.get("sigma_lnT", 0.7)))
        w_min = float(params.get("w_min", 0.02))
        # v_er_from_components returns (vbar, xi, ver)
        vbar, _, ver = v_er_from_components(R, Vgas, Vdisk, Vbul, ups_disk, ups_bul,
                                            lambda_max, R0_kpc, sigma_lnR, w_min)
        return ver
    else:
        vbar = v_bar_from_components(R, Vgas, Vdisk, Vbul, ups_disk, ups_bul)
        return vbar


def ppc_residual_envelopes(samples: Optional[np.ndarray],
                           weights: Optional[np.ndarray],
                           base_params: dict,
                           R: np.ndarray,
                           Vobs: np.ndarray,
                           eV: np.ndarray,
                           Vgas: np.ndarray,
                           Vdisk: np.ndarray,
                           Vbul: np.ndarray,
                           quantiles=(0.16, 0.5, 0.84),
                           max_draws: int = 500,
                           rng: Optional[np.random.Generator] = None,
                           sanity: Optional[dict] = None) -> dict:
    """
    Compute predictive envelopes for v_model(R) and residuals.
    - If posterior samples are provided: draw up to max_draws with importance weights.
    - Otherwise: use a degenerate ensemble with the point estimate only.
    Returns dict with keys: v_q (QxN), res_q (QxN), q (tuple), and draws_used (int).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Build a function to map a parameter vector to model v(R)
    def params_from_vector(vec: np.ndarray, names: List[str]) -> dict:
        p = dict(base_params)
        for k, v in zip(names, vec):
            # Only update known keys to avoid accidental overrides
            if k in p or k in ("lambda_max", "ups_disk", "ups_bul", "R0", "sigma_lnR", "sigma_lnT", "w_min"):
                p[k] = float(v)
        return p

    V_models = []
    if samples is not None and samples.size > 0:
        names = [str(x) for x in (weights.dtype.names if hasattr(weights, 'dtype') and weights is not None else [])]
        # If names are not available in weights, try to extract from a provided separate names list
        # We will infer possible param names from base_params intersection
        possible_names = list(base_params.keys())
        # Downsample with weights if provided
        idx = np.arange(samples.shape[0])
        if weights is not None and np.ndim(weights) == 1 and len(weights) == len(idx):
            prob = weights / np.sum(weights)
            take = rng.choice(idx, size=min(max_draws, len(idx)), replace=True, p=prob)
        else:
            take = rng.choice(idx, size=min(max_draws, len(idx)), replace=False)
        for j in take:
            vec = samples[j]
            # Map vector to dict using possible_names if dimensions match; else fall back to base_params
            if vec.shape[0] == len(possible_names):
                p = params_from_vector(vec, possible_names)
            else:
                p = base_params
            V_models.append(model_curve_from_params(p, R, Vgas, Vdisk, Vbul, sanity=sanity))
    else:
        # Degenerate ensemble: only the base point estimate
        V_models.append(model_curve_from_params(base_params, R, Vgas, Vdisk, Vbul, sanity=sanity))

    V_models = np.asarray(V_models)  # (M, N)
    q = np.quantile(V_models, quantiles, axis=0)
    # Residuals = v_obs - v_model
    res = Vobs[None, :] - V_models  # (M, N)
    res_q = np.quantile(res, quantiles, axis=0)

    return {
        "v_q": q,
        "res_q": res_q,
        "q": quantiles,
        "draws_used": V_models.shape[0],
    }


def plot_residual_envelopes(R: np.ndarray,
                            Vobs: np.ndarray,
                            eV: np.ndarray,
                            envelopes: dict,
                            out_png: Path,
                            title: str = "Posterior Predictive Residuals",
                            show_residuals: bool = True) -> None:
    apply_paper_style()
    q = envelopes["q"]
    v_q = envelopes["v_q"]  # (Q,N)
    res_q = envelopes["res_q"]  # (Q,N)

    fig, axes = plt.subplots(2 if show_residuals else 1, 1, figsize=(10, 7), sharex=True,
                             gridspec_kw=dict(height_ratios=[2, 1] if show_residuals else [1]))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    ax0 = axes[0]
    # Data
    ax0.errorbar(R, Vobs, yerr=eV, fmt='o', ms=4, color='k', lw=1, alpha=0.8, label='Observed')
    # Median model and 68% band
    ax0.plot(R, v_q[1], 'r-', lw=2, label='Model median')
    ax0.fill_between(R, v_q[0], v_q[2], color='r', alpha=0.15, label='68% PPC band')
    ax0.set_ylabel('Vc (km s$^{-1}$)')
    ax0.grid(True, alpha=0.3)
    ax0.legend(frameon=False)
    ax0.set_title(title)

    if show_residuals and len(axes) > 1:
        ax1 = axes[1]
        ax1.axhline(0.0, color='k', lw=1, ls=':')
        ax1.plot(R, res_q[1], 'b-', lw=1.5, label='Residual median')
        ax1.fill_between(R, res_q[0], res_q[2], color='b', alpha=0.15, label='68% band')
        ax1.errorbar(R, np.zeros_like(R), yerr=eV, fmt='none', ecolor='k', alpha=0.25)
        ax1.set_xlabel('R (kpc)')
        ax1.set_ylabel('Vobs − Vmodel (km s$^{-1}$)')
        ax1.grid(True, alpha=0.3)
        ax1.legend(frameon=False)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)


def aggregate_sparc_residuals(json_paths: List[Path], standardize: bool = True) -> np.ndarray:
    """
    Aggregate residuals across many SPARC ER fit JSONs.
    For each JSON, recompute model curve at observed radii and collect residuals.
    If standardize=True, divide residuals by eV (σ) before stacking.
    Returns a 1D array of stacked residuals.
    """
    all_res = []
    for jp in json_paths:
        try:
            meta = load_fit_json(jp)
            rotmod_path = meta.get("file_rotmod")
            if not rotmod_path:
                continue
            data = load_rotmod(rotmod_path)
            R = data["R_kpc"]; Vobs = data["Vobs_kms"]; eV = data["eVobs_kms"]
            Vgas = data["Vgas_kms"]; Vdisk = data["Vdisk_kms"]; Vbul = data["Vbul_kms"]
            params = meta.get("params", {})
            sanity = meta.get("sanity", {})
            vmod = model_curve_from_params(params, R, Vgas, Vdisk, Vbul, sanity=sanity)
            res = Vobs - vmod
            if standardize:
                with np.errstate(divide='ignore', invalid='ignore'):
                    res = np.where(eV > 0, res / eV, np.nan)
            all_res.append(res)
        except Exception:
            continue
    if not all_res:
        return np.array([])
    return np.concatenate([r[np.isfinite(r)] for r in all_res if r is not None])


essential_colors = {
    "hist": "#2F4F4F",
}


def plot_residual_hist(residuals: np.ndarray, out_png: Path, title: str = "Stacked standardized residuals (SPARC)") -> None:
    apply_paper_style()
    plt.figure(figsize=(8, 5))
    if residuals.size == 0:
        plt.text(0.5, 0.5, 'No residuals found', transform=plt.gca().transAxes, ha='center', va='center')
    else:
        plt.hist(residuals, bins=40, color=essential_colors["hist"], alpha=0.8, density=True)
        mu = float(np.nanmean(residuals)) if residuals.size else 0.0
        sig = float(np.nanstd(residuals)) if residuals.size else 1.0
        plt.axvline(0.0, color='k', ls=':')
        plt.title(f"{title}\nmean={mu:.2f}, std={sig:.2f}, N={residuals.size}")
    plt.xlabel('Standardized residual (sigma)')
    plt.ylabel('Density')
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)

