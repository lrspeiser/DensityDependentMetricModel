#!/usr/bin/env python3
# Sandbox lensing predictor using the unified gate registry and a simple
# deprojection to obtain rho(R) near theta_E. This avoids modifying main tools.

from __future__ import annotations
import argparse
import numpy as np
from pathlib import Path
import json

from xi_registry_variants import build_gate

# Minimal Hernquist deprojection for ETG (spherical, n=4 surrogate)
G_CGS = 6.67430e-8
C_LIGHT = 2.99792458e10
KPC_CM = 3.0856775814913673e21
MSUN_CGS = 1.98847e33


def hernquist_rho(M_solar: float, Re_kpc: float, r_kpc: np.ndarray) -> np.ndarray:
    a = Re_kpc / 1.8153
    r = np.maximum(np.asarray(r_kpc, dtype=float), 1e-6)
    rho_kpc = (M_solar / (2.0 * np.pi)) * a / (r * (r + a) ** 3)
    return rho_kpc * (MSUN_CGS / (KPC_CM ** 3))


def einstein_angle_gr(Sigma_crit_cgs: float, R_kpc: np.ndarray, M2D_cgs: np.ndarray) -> float:
    # Very rough scalar surrogate: find R such that M(<R) ~ (c^2/4G) * D * theta_E^2
    # This sandbox focuses on gating; we keep GR mapping symbolic and record xi.
    # Return the R where mean surface density equals Sigma_crit (not exact).
    Sigma_mean = M2D_cgs / (np.pi * (R_kpc * KPC_CM) ** 2)
    idx = np.argmin(np.abs(Sigma_mean - Sigma_crit_cgs))
    return R_kpc[idx]


def main():
    ap = argparse.ArgumentParser(description='Lensing sandbox: gate variants at theta_E')
    ap.add_argument('--gate', default='density', choices=['accel','density','hybrid'])
    ap.add_argument('--a0', type=float, default=1.93e-7)
    ap.add_argument('--rho-c', type=float, default=1e-27)
    ap.add_argument('--gamma', type=float, default=1.0)
    ap.add_argument('--zeta', type=float, default=1.0)
    ap.add_argument('--Dmax', type=float, default=50.0)
    ap.add_argument('--log10Mstar', type=float, required=True)
    ap.add_argument('--Re_kpc', type=float, required=True)
    ap.add_argument('--Sigma_crit_cgs', type=float, required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    Mstar = 10 ** args.log10Mstar
    R = np.linspace(0.1, 50.0, 400)
    rho = hernquist_rho(Mstar, args.Re_kpc, R)

    # Build simple 2D mass by integrating rho in spherical shells (approximate)
    # Here we just use cumulative 3D M(<r) = ∫4πr^2 rho dr ~ Hernquist analytic alternative
    a = args.Re_kpc / 1.8153
    r_cm = R * KPC_CM
    M3D = (Mstar * MSUN_CGS) * (r_cm ** 2) / ((r_cm + a * KPC_CM) ** 2)

    R_E = einstein_angle_gr(args.Sigma_crit_cgs, R, M3D)

    # Gate at theta_E: need gbar and rho at R_E
    # gbar ~ GM(<R)/R^2 (cgs)
    gb = G_CGS * np.interp(R_E, R, M3D) / ((R_E * KPC_CM) ** 2)
    rho_E = np.interp(R_E, R, rho)

    gate = build_gate(args.gate, a0=args.a0, rho_c=args.rho_c, gamma=args.gamma, zeta=args.zeta, Dmax=args.Dmax)
    xi_E = gate(gb, rho_E, R_E)

    out = {
        'gate': args.gate,
        'params': {'a0': args.a0, 'rho_c': args.rho_c, 'gamma': args.gamma, 'zeta': args.zeta, 'Dmax': args.Dmax},
        'Re_kpc': args.Re_kpc,
        'log10Mstar': args.log10Mstar,
        'R_E_kpc': float(R_E),
        'xi_at_R_E': float(np.atleast_1d(xi_E)[0]),
    }
    Path(args.out).write_text(json.dumps(out, indent=2), encoding='utf-8')
    print('Wrote', args.out)


if __name__ == '__main__':
    main()
