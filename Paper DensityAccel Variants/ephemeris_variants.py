#!/usr/bin/env python3
# Sandbox ephemeris variant that uses the same gate registry to compute xi(r)
# for Solar-System ΔG/G checks without touching the main modules.

from __future__ import annotations
import argparse
import numpy as np
from xi_registry_variants import build_gate

G_SI = 6.6743e-11
M_SUN = 1.98847e30
AU_M = 1.495978707e11


def main():
    ap = argparse.ArgumentParser(description="Solar ephemeris ΔG/G with density/accel gate (sandbox)")
    ap.add_argument('--gate', default='density', choices=['accel','density','density-plateau','hybrid'])
    ap.add_argument('--a0', type=float, default=1.93e-10)  # SI for Solar plots if needed
    ap.add_argument('--rho-c', type=float, default=1e-21)  # kg m^-3 (≈1e-24 g cm^-3)
    ap.add_argument('--gamma', type=float, default=1.0)
    ap.add_argument('--zeta', type=float, default=1.0)
    ap.add_argument('--Dmax', type=float, default=50.0)
    ap.add_argument('--n', type=float, default=2.0)
    ap.add_argument('--rho-env', type=float, default=1e-20)  # kg m^-3; simple constant env
    args = ap.parse_args()

    # Radii 1–30 AU
    r = np.linspace(1.0, 30.0, 200) * AU_M
    gN = G_SI * M_SUN / (r**2)  # m s^-2

    # Convert to cgs for the gate if you prefer; here keep SI consistently and
    # pass the same units to gate (accel branch only depends on ratios).
    rho_env = np.full_like(r, args.rho_env)

    gate = build_gate(args.gate, a0=args.a0, rho_c=args.rho_c, gamma=args.gamma,
                      zeta=args.zeta, Dmax=args.Dmax, n=args.n)
    xi = gate(gN, rho_env, r / AU_M)

    dG_over_G = xi - 1.0
    # Print a few sample values
    for R_AU, eps in zip(np.linspace(1,30,6), np.interp(np.linspace(1,30,6), r/AU_M, dG_over_G)):
        print(f"R={R_AU:4.1f} AU  ΔG/G={eps:.3e}")


if __name__ == '__main__':
    main()
