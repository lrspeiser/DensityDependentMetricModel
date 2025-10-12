#!/usr/bin/env python3
"""
fit_sparc_composite_evidence.py - Composite (ER/TFR + NFW) evidence for a SPARC galaxy using dynesty.
Model: v_tot^2(R) = xi(R) * vbar(R)^2 + v_halo(R)^2
- xi(R) is ER/TFR radial log-normal window (proxy of tidal-band), see models.er_sparc
- vbar from SPARC components with Υ scaling
- v_halo is NFW with parameters (V200, c)

Outputs JSON with logZ, settings, and priors to images/.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import json
import sys
import numpy as np

# Ensure repo root on path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.Utilities.sparc_io import load_single_sparc_galaxy
from models.er_sparc import v_bar_from_components, xi_log_normal_R
from models.nfw import v_nfw

try:
    import dynesty  # type: ignore
    _HAS_DYNASTY = True
except Exception:
    _HAS_DYNASTY = False


def build_components(gal):
    R = np.asarray(gal['R_kpc'], dtype=float)
    Vobs = np.asarray(gal['V_obs'], dtype=float)
    eV = np.asarray(gal['e_V_obs'], dtype=float)
    Vg = np.asarray(gal['V_gas_comp_kms'], dtype=float)
    Vd = np.asarray(gal['V_disk_comp_kms'], dtype=float)
    Vb = np.asarray(gal['V_bulge_comp_kms'], dtype=float)
    return R, Vobs, eV, Vg, Vd, Vb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--galaxy_id', required=True)
    ap.add_argument('--sparc_dir', required=True)
    ap.add_argument('--sigma-floor', type=float, default=0.0)
    ap.add_argument('--use-master-priors', action='store_true')
    # ER/TFR parameter bounds (uniform)
    ap.add_argument('--lambda_max_min', type=float, default=0.0)
    ap.add_argument('--lambda_max_max', type=float, default=6.0)
    ap.add_argument('--R0_min', type=float, default=2.0)
    ap.add_argument('--R0_max', type=float, default=40.0)
    ap.add_argument('--sigma_lnR_min', type=float, default=0.2)
    ap.add_argument('--sigma_lnR_max', type=float, default=1.5)
    ap.add_argument('--w_min_min', type=float, default=0.0)
    ap.add_argument('--w_min_max', type=float, default=0.2)
    ap.add_argument('--ups_disk_min', type=float, default=0.1)
    ap.add_argument('--ups_disk_max', type=float, default=1.0)
    ap.add_argument('--ups_bul_min', type=float, default=0.1)
    ap.add_argument('--ups_bul_max', type=float, default=1.0)
    # NFW bounds
    ap.add_argument('--V200_min', type=float, default=40.0)
    ap.add_argument('--V200_max', type=float, default=400.0)
    ap.add_argument('--c_min', type=float, default=2.0)
    ap.add_argument('--c_max', type=float, default=40.0)
    # Dynesty knobs
    ap.add_argument('--nlive', type=int, default=1000)
    ap.add_argument('--maxcall', type=int, default=200000)
    ap.add_argument('--dlogz-target', type=float, default=0.01)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    gal = load_single_sparc_galaxy(args.galaxy_id, sparc_dir=args.sparc_dir)
    if gal is None:
        print(f"Failed to load galaxy {args.galaxy_id}")
        sys.exit(2)

    R, Vobs, eV, Vg, Vd, Vb = build_components(gal)

    # Likelihood helpers
    def e_eff():
        e = np.sqrt(eV**2 + float(max(0.0, args.sigma_floor))**2)
        return np.where(e > 0, e, 1.0)

    # Params: [lambda_max, R0, sigma_lnR, w_min, ups_disk, ups_bul, V200, c]
    lo = np.array([
        args.lambda_max_min, args.R0_min, args.sigma_lnR_min,
        args.w_min_min, args.ups_disk_min, args.ups_bul_min,
        args.V200_min, args.c_min
    ], dtype=float)
    hi = np.array([
        args.lambda_max_max, args.R0_max, args.sigma_lnR_max,
        args.w_min_max, args.ups_disk_max, args.ups_bul_max,
        args.V200_max, args.c_max
    ], dtype=float)

    def prior_transform(u):
        u = np.asarray(u, dtype=float)
        return lo + u * (hi - lo)

    def loglike(theta):
        lam, R0, sLR, wmin, ups_d, ups_b, V200, c = map(float, theta)
        vbar = v_bar_from_components(R, Vg, Vd, Vb, ups_d, ups_b)
        xi = xi_log_normal_R(R, lam, R0, sLR, wmin)
        v_tfr = np.sqrt(np.clip(xi, 0.0, None)) * vbar
        v_halo = v_nfw(R, V200, c)
        vmod = np.sqrt(np.clip(v_tfr**2 + v_halo**2, 0.0, None))
        r = (Vobs - vmod) / e_eff()
        return -0.5 * float(np.sum(r*r))

    if not _HAS_DYNASTY:
        print(json.dumps({'galaxy_id': args.galaxy_id, 'error': 'dynesty not available'}))
        sys.exit(0)

    rng = np.random.default_rng(args.seed)
    dsampler = dynesty.DynamicNestedSampler(loglike, prior_transform, ndim=8, bound='multi', sample='rslice', rstate=rng)
    dsampler.run_nested(maxcall=args.maxcall, dlogz_init=args.dlogz_target, nlive_init=args.nlive)
    res = dsampler.results

    out_dir = REPO_ROOT / 'images'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f'sparc_composite_evidence_{args.galaxy_id.lower()}.json'
    meta = {
        'galaxy_id': args.galaxy_id,
        'model': 'er_plus_nfw',
        'sigma_floor': float(args.sigma_floor),
        'priors': {
            'use_master_priors': bool(args.use_master_priors),
            'er': {
                'lambda_max': {'min': float(args.lambda_max_min), 'max': float(args.lambda_max_max)},
                'R0_kpc': {'min': float(args.R0_min), 'max': float(args.R0_max)},
                'sigma_lnR': {'min': float(args.sigma_lnR_min), 'max': float(args.sigma_lnR_max)},
                'w_min': {'min': float(args.w_min_min), 'max': float(args.w_min_max)},
                'ups_disk': {'min': float(args.ups_disk_min), 'max': float(args.ups_disk_max)},
                'ups_bul': {'min': float(args.ups_bul_min), 'max': float(args.ups_bul_max)},
            },
            'nfw': {
                'V200': {'min': float(args.V200_min), 'max': float(args.V200_max)},
                'c': {'min': float(args.c_min), 'max': float(args.c_max)},
            }
        },
        'evidence': {
            'logZ': float(res.logz[-1]),
            'logZ_err': float(res.logzerr[-1]) if hasattr(res, 'logzerr') else None,
            'nlive': int(args.nlive),
            'maxcall': int(args.maxcall),
            'dlogz_target': float(args.dlogz_target),
            'seed': int(args.seed),
        },
    }
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    print(f"Saved: {out_json}")

if __name__ == '__main__':
    main()

