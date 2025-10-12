"""Model comparison helpers (code-based)
Compute WAIC-like metrics for rotation-curve fits using the a0 grids saved by the
orchestrator. This uses only local data (SPARC rotmods) and does not require web/APIs.

Scope (minimal viable):
- Universal a0: WAIC reduces to -2 * sum log-likelihood at a0_global (no parameter variance).
- Hierarchical (µ,σ): approximate per-galaxy posterior P(ln a0 | data, µ,σ) ∝ exp(-0.5 χ²(ln a0)) × Normal(ln a0; µ,σ),
  then compute per-point lppd and p_waic across parameter samples.

Note: This is a practical, reproducible implementation. For full Bayesian nested sampling,
see runners/dynesty_latest and extend as needed.
"""
from __future__ import annotations
from typing import Dict, Any, List, Tuple
import os
import json
import math
import numpy as np
from pathlib import Path

# Local imports are performed dynamically to avoid hard dependencies

def _load_sparc_galaxy(galaxy_id: str, sparc_dir: Path):
    from utils.Utilities.sparc_io import load_single_sparc_galaxy
    return load_single_sparc_galaxy(galaxy_id, sparc_dir=str(sparc_dir))


def _xi_rar_plateau_numpy(Vbar_kms: np.ndarray, R_kpc: np.ndarray, *, a0_m_s2: float, D_max: float | None = None) -> np.ndarray:
    ACC = 3.240779289e-14
    Vbar_kms = np.asarray(Vbar_kms, float)
    R_kpc = np.asarray(R_kpc, float)
    g_bar = ACC * np.maximum(Vbar_kms, 0.0)**2 / np.maximum(R_kpc, 1e-12)
    D = 0.5 + np.sqrt(0.25 + float(a0_m_s2) / np.maximum(g_bar, 1e-30))
    if D_max is not None and float(D_max) > 1.0:
        D = np.minimum(D, float(D_max))
    return np.maximum(D, 1.0)


def _point_loglike(Vobs: np.ndarray, Vmod: np.ndarray, eV: np.ndarray, sigma_floor: float) -> np.ndarray:
    Vobs = np.asarray(Vobs, float); Vmod = np.asarray(Vmod, float); eV = np.asarray(eV, float)
    sig = np.sqrt(np.maximum(eV, 0.0)**2 + float(sigma_floor)**2)
    res = (Vobs - Vmod) / np.maximum(sig, 1e-12)
    ll = -0.5 * (res**2 + np.log(2.0 * np.pi * sig**2))
    return ll


def compute_waic_rotation_curves(sample: List[str], sparc_dir: Path, grids_dir: Path,
                                 a0_global: float | None, hier_mu: float | None, hier_sigma: float | None,
                                 *, D_max: float | None = None, sigma_floor: float = 5.0,
                                 posterior_samples: int = 64) -> Dict[str, Any]:
    """Compute WAIC-like metrics for universal and hierarchical a0 models.
    Returns a JSON-serializable dict with WAIC_universal and WAIC_hier fields.
    """
    out: Dict[str, Any] = {
        "status": "ok",
        "WAIC_universal": None,
        "WAIC_hier": None,
        "details": {}
    }
    # Universal a0 (if provided)
    if a0_global is not None and math.isfinite(a0_global):
        lppd_sum = 0.0
        details_uni = {}
        for gid in sample:
            dat = _load_sparc_galaxy(gid, sparc_dir)
            if not dat:
                continue
            R = dat['R_kpc']; Vobs = dat['V_obs']; eV = dat['e_V_obs']
            Vbar = np.sqrt(np.maximum(dat['V_gas_comp_kms'],0.0)**2 + np.maximum(dat['V_disk_comp_kms'],0.0)**2 + np.maximum(dat['V_bulge_comp_kms'],0.0)**2)
            xi = _xi_rar_plateau_numpy(Vbar, R, a0_m_s2=float(a0_global), D_max=D_max)
            Vmod = np.sqrt(np.maximum(Vbar,0.0)**2 * xi)
            ll_i = _point_loglike(Vobs, Vmod, eV, sigma_floor)
            lppd_sum += float(np.nansum(ll_i))
            details_uni[gid] = {"npts": int(len(R)), "sum_ll": float(np.nansum(ll_i))}
        waic_uni = -2.0 * lppd_sum
        out["WAIC_universal"] = waic_uni
        out["details"]["universal"] = details_uni

    # Hierarchical WAIC (if µ,σ provided)
    if (hier_mu is not None) and (hier_sigma is not None) and posterior_samples > 0:
        lppd_total = 0.0
        p_waic_total = 0.0
        details_h = {}
        for gid in sample:
            grid_csv = grids_dir / f"{gid.replace(' ','_')}.csv"
            if not grid_csv.exists():
                continue
            # Load chi2(ln a0) grid
            xs = []; c2 = []
            with grid_csv.open('r', encoding='utf-8') as f:
                f.readline()
                for line in f:
                    a, v = line.strip().split(',')
                    xs.append(float(a)); c2.append(float(v))
            ln_a0_grid = np.asarray(xs, float)
            chi2_grid = np.asarray(c2, float)
            # Build posterior weights ∝ exp(-0.5 χ²) × Normal(ln a0; µ,σ)
            prior = np.exp(-0.5*((ln_a0_grid - float(hier_mu))/max(float(hier_sigma),1e-6))**2) / (max(float(hier_sigma),1e-6) * math.sqrt(2.0*math.pi))
            logw = -0.5 * chi2_grid + np.log(np.maximum(prior, 1e-300))
            logw -= np.nanmax(logw)
            w = np.exp(logw)
            if float(np.sum(w)) <= 0:
                continue
            w = w / float(np.sum(w))
            # Draw posterior samples of ln a0
            rng = np.random.default_rng(123)
            idx = rng.choice(np.arange(len(ln_a0_grid)), size=int(posterior_samples), replace=True, p=w)
            ln_a0_samp = ln_a0_grid[idx]
            # Load galaxy data and compute per-point ll for each sample
            dat = _load_sparc_galaxy(gid, sparc_dir)
            if not dat:
                continue
            R = dat['R_kpc']; Vobs = dat['V_obs']; eV = dat['e_V_obs']
            Vbar = np.sqrt(np.maximum(dat['V_gas_comp_kms'],0.0)**2 + np.maximum(dat['V_disk_comp_kms'],0.0)**2 + np.maximum(dat['V_bulge_comp_kms'],0.0)**2)
            ll_samples = []
            for ln_a0 in ln_a0_samp:
                a0 = 10**(float(ln_a0))
                xi = _xi_rar_plateau_numpy(Vbar, R, a0_m_s2=float(a0), D_max=D_max)
                Vmod = np.sqrt(np.maximum(Vbar,0.0)**2 * xi)
                ll_i = _point_loglike(Vobs, Vmod, eV, sigma_floor)
                ll_samples.append(ll_i)
            L = np.vstack(ll_samples)  # shape S x N
            # lppd_i = log mean_s exp(ll_{s,i}); p_waic_i = Var_s(ll_{s,i})
            # stabilize with subtracting max across s for each i
            m = np.nanmax(L, axis=0)
            lppd_i = m + np.log(np.nanmean(np.exp(L - m[None, :]), axis=0))
            p_waic_i = np.nanvar(L, axis=0)
            lppd_total += float(np.nansum(lppd_i))
            p_waic_total += float(np.nansum(p_waic_i))
            details_h[gid] = {"npts": int(len(R)), "sum_lppd": float(np.nansum(lppd_i)), "sum_pwaic": float(np.nansum(p_waic_i))}
        waic_h = -2.0 * (lppd_total - p_waic_total)
        out["WAIC_hier"] = waic_h
        out["details"]["hierarchical"] = details_h

    return out


def write_waic_report(out_path: Path, report: Dict[str, Any]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding='utf-8')

