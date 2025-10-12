#!/usr/bin/env python3
"""
Make an overlaid rotation curve plot for a SPARC galaxy using latest run artifacts.
Overlays:
- Observed V_obs with error bars
- GR (baryons-only) curve
- NFW (ΛCDM) curve (from evidence JSON if params available, or quick fit if requested)
- Tidal (ER/TFR env) curve using density-aware xi_env and stored proxy selection

Examples:
  # NGC 3198 using default artifact locations
  python scripts/plot_sparc_rotation_overlay.py \
    --galaxy-id NGC3198 \
    --sparc-dir external_data/Rotmod_LTG \
    --out images/overlay_ngc3198.png

  # If you want to quickly fit NFW if no params are saved in evidence JSON
  python scripts/plot_sparc_rotation_overlay.py \
    --galaxy-id NGC3198 \
    --sparc-dir external_data/Rotmod_LTG \
    --fit-nfw-if-missing \
    --out images/overlay_ngc3198.png
"""
from __future__ import annotations
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import matplotlib.pyplot as plt

# Ensure repo root on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.Utilities.sparc_io import load_single_sparc_galaxy, BASE_M_L_3_6_MICRON_DISK  # type: ignore
from models.er_sparc import v_bar_from_components  # type: ignore
from models.nfw import v_model_nfw  # type: ignore
from models.er_env import xi_env, tidal_proxy_from_vbar  # type: ignore

try:
    from utils.plot_style import apply_paper_style  # type: ignore
except Exception:
    def apply_paper_style():
        plt.style.use("seaborn-v0_8-whitegrid")

try:
    from scipy import optimize as _opt  # type: ignore
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


def _safe_read_json(p: Path) -> dict | None:
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _compute_tidal_proxy(R, vbar, mode: str = "curvature", norm: str = "robust"):
    R = np.asarray(R, dtype=float)
    vb = np.asarray(vbar, dtype=float)
    R_safe = np.clip(R, 1e-6, None)
    if mode == "curvature":
        g = (np.maximum(vb, 0.0) ** 2) / R_safe
        from models.er_env import finite_diff
        dgdR = finite_diff(g, R_safe)
        T_raw = np.abs(dgdR)
    elif mode == "shear":
        Omega = np.maximum(vb, 0.0) / R_safe
        from models.er_env import finite_diff
        dO_dR = finite_diff(Omega, R_safe)
        T_raw = np.abs(dO_dR)
    elif mode == "epicyclic":
        Omega = np.maximum(vb, 0.0) / R_safe
        from models.er_env import finite_diff
        dO_dR = finite_diff(Omega, R_safe)
        kappa2 = 2.0 * Omega * (2.0 * Omega + R_safe * dO_dR)
        T_raw = np.sqrt(np.clip(np.abs(kappa2), 0.0, None))
    else:
        T_raw = tidal_proxy_from_vbar(R, vb)
    # normalization
    if norm == "robust":
        med = float(np.nanmedian(T_raw))
        scale = med if med > 0 else (np.nanmax(T_raw) if np.nanmax(T_raw) > 0 else 1.0)
        Tn = T_raw / scale
    else:
        med = float(np.nanmedian(T_raw))
        scale = med if med > 0 else (np.nanmax(T_raw) if np.nanmax(T_raw) > 0 else 1.0)
        Tn = T_raw / scale
    return np.clip(Tn, 1e-12, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--galaxy-id', required=True)
    ap.add_argument('--sparc-dir', required=True, help='Directory containing Rotmod_LTG data')
    ap.add_argument('--er-json', default=None, help='Path to ER env fit JSON (defaults to images/sparc_env_fit_<gal>.json)')
    ap.add_argument('--gr-json', default=None, help='Path to GR evidence JSON (defaults to images/sparc_gr_evidence_<gal>.json)')
    ap.add_argument('--nfw-json', default=None, help='Path to NFW evidence JSON (defaults to images/sparc_nfw_evidence_<gal>.json)')
    ap.add_argument('--out', required=True, help='Output PNG path')
    ap.add_argument('--fit-nfw-if-missing', action='store_true', help='Quickly fit NFW (chi2) if no params available in JSON')
    args = ap.parse_args()

    galid = args.galaxy_id

    # Load data via unified loader to access midplane densities used by env fit
    data = load_single_sparc_galaxy(galid, sparc_dir=args.sparc_dir)
    if data is None:
        raise SystemExit(f"Failed to load SPARC galaxy {galid} from {args.sparc_dir}")

    R = np.asarray(data['R_kpc'], dtype=float)
    Vobs = np.asarray(data['V_obs'], dtype=float)
    eV = np.asarray(data['e_V_obs'], dtype=float)
    Vgas = np.asarray(data['V_gas_comp_kms'], dtype=float)
    Vdisk = np.asarray(data['V_disk_comp_kms'], dtype=float)
    Vbul = np.asarray(data['V_bulge_comp_kms'], dtype=float)
    rho_mid_base = np.asarray(data['rho_star_mid_Msun_kpc3_baseML'], dtype=float) + np.asarray(data['rho_gas_mid_Msun_kpc3'], dtype=float)

    # Resolve artifact paths
    images_dir = REPO_ROOT / 'images'
    er_json_path = Path(args.er_json) if args.er_json else images_dir / f'sparc_env_fit_{galid.lower()}.json'
    gr_json_path = Path(args.gr_json) if args.gr_json else images_dir / f'sparc_gr_evidence_{galid.lower()}.json'
    nfw_json_path = Path(args.nfw_json) if args.nfw_json else images_dir / f'sparc_nfw_evidence_{galid.lower()}.json'

    er_meta = _safe_read_json(er_json_path)
    gr_meta = _safe_read_json(gr_json_path)
    nfw_meta = _safe_read_json(nfw_json_path)

    # Adopt ups_disk/bulge from ER fit if available; else defaults
    ups_d = float(er_meta.get('params',{}).get('ups_disk', 0.5)) if er_meta else 0.5
    ups_b = float(er_meta.get('params',{}).get('ups_bul', 0.7)) if er_meta else 0.7

    # Build vbar
    vbar = v_bar_from_components(R, Vgas, Vdisk, Vbul, ups_d, ups_b)

    # GR curve is vbar
    v_gr = np.array(vbar, dtype=float)

    # Tidal (ER env) curve from ER params
    v_er = None
    if er_meta is not None:
        p = er_meta.get('params', {})
        log10_rho_c = float(p.get('log10_rho_c', 15.0))
        gamma_exp = float(p.get('gamma_exp', 3.0))
        lambda_max = float(p.get('lambda_max', 4.0))
        lnT0 = float(p.get('lnT0', 0.0))
        sigma_lnT = float(p.get('sigma_lnT', 0.8))
        w_min = float(p.get('w_min', 0.02))
        T_proxy = p.get('T_proxy', er_meta.get('sanity', {}).get('T_proxy', 'curvature'))
        tidal_norm = p.get('tidal_norm', er_meta.get('sanity', {}).get('tidal_norm', 'robust'))
        rho_screen = p.get('rho_screen', 'power')

        # Adjust rho_mid for M/L change (approximate, consistent with fitter)
        f_ml = max(0.3, min(3.0, (ups_d / BASE_M_L_3_6_MICRON_DISK) ** 0.5))
        rho_mid = np.clip(rho_mid_base * f_ml, 1e-30, None)
        rho_c = 10 ** log10_rho_c
        T = _compute_tidal_proxy(R, vbar, mode=T_proxy, norm=tidal_norm)
        if rho_screen == 'power':
            xi = xi_env(rho_mid, T, lambda_max, rho_c, gamma_exp, np.exp(lnT0), sigma_lnT, w_min)
        else:
            # Import exponential variant lazily
            from models.er_env import xi_env_exp  # type: ignore
            xi = xi_env_exp(rho_mid, T, lambda_max, rho_c, gamma_exp, np.exp(lnT0), sigma_lnT, w_min)
        v_er = np.sqrt(np.clip(xi, 0.0, None)) * np.maximum(vbar, 0.0)

    # NFW curve: try to get params from JSON; else optionally quick-fit
    v_nfw = None
    V200 = None; c = None
    if nfw_meta is not None:
        # Some evidence JSON may not include best-fit params; check common keys
        p = nfw_meta.get('params', {})
        if 'V200' in p and 'c' in p:
            V200 = float(p['V200']); c = float(p['c'])
    if (V200 is None or c is None) and args.fit_nfw_if_missing:
        if not _HAS_SCIPY:
            print("SciPy not available; cannot quick-fit NFW. Skipping NFW curve.")
        else:
            def chi2(theta):
                V200_t, c_t = float(theta[0]), float(theta[1])
                vmod = v_model_nfw(R, vbar, V200_t, c_t)
                e_eff = np.where(eV > 0, eV, 1.0)
                r = (Vobs - vmod) / e_eff
                return float(np.sum(r*r))
            # Reasonable box
            lo = np.array([40.0, 2.0]); hi = np.array([400.0, 40.0])
            x0 = np.array([150.0, 10.0])
            bounds = _opt.Bounds(lo, hi)
            res = _opt.minimize(chi2, x0, method='L-BFGS-B', bounds=bounds)
            V200 = float(np.clip(res.x[0], lo[0], hi[0])); c = float(np.clip(res.x[1], lo[1], hi[1]))
    if V200 is not None and c is not None:
        v_nfw = v_model_nfw(R, vbar, V200, c)

    # Plot
    out_png = Path(args.out)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    apply_paper_style()
    plt.figure(figsize=(10, 7))

    # Observations
    plt.errorbar(R, Vobs, yerr=eV, fmt='o', ms=4, color='k', lw=1, alpha=0.85, label='Observed (SPARC)')

    # Curves
    plt.plot(R, v_gr, 'b--', lw=2.2, label='GR (baryons)')
    if v_nfw is not None:
        plt.plot(R, v_nfw, color='green', ls='-.', lw=2.2, label='NFW (ΛCDM)')
    else:
        plt.text(0.02, 0.96, 'NFW unavailable', transform=plt.gca().transAxes, color='green', fontsize=9, va='top')
    if v_er is not None:
        plt.plot(R, v_er, 'r-', lw=2.4, label='RAR Plateau')
    else:
        plt.text(0.02, 0.90, 'RAR Plateau JSON not found', transform=plt.gca().transAxes, color='red', fontsize=9, va='top')

    plt.xlabel('R (kpc)')
    plt.ylabel('Vc (km s$^{-1}$)')
    plt.title(f'{galid}: Rotation curve with GR, NFW, and RAR Plateau overlays')
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=False)
    plt.xlim(0, max(R.max()*1.2, R.max()+5))
    ymax = max(np.nanmax(Vobs+eV), np.nanmax(v_gr)*1.2)
    plt.ylim(0, max(300, float(ymax)+40))
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f'Saved: {out_png}')


if __name__ == '__main__':
    main()
