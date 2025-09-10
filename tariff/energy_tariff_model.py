#!/usr/bin/env python3
"""
energy_tariff_model.py — PhotonJourney simulator for the "Energy Tariff" concept
bridging your README's RAR‑plateau ξ(g) and observed redshift z.

Model summary
- Energy loss differential:
    dE/dr = -k * (xi(r) - 1) * E
  with solution:
    1 + z = E_emit / E_obs = exp( k * ∫_0^r (xi(l) - 1) dl ).
- We implement xi(g) exactly as specified in the repo README (Box 1):
    xi(g) = min[ 1/2 + sqrt(1/4 + a0/g), D_max ], with a0 = 1.2e-10 m/s^2 and D_max = 50.
  In deep voids (g << a0), xi → D_max. Inside galaxies (g >> a0), xi → 1.

Provenance
- Formula and parameters follow the repository’s top-level README.md, Box 1 — Exact weak‑field
  formula used in all figures (xi(g) with a finite D_max=50 and a0=1.2e-10 m/s^2). We do not
  modify README.md.

CLI
- By default we calibrate k from an anchor H0 value (Planck-like default 67.4 km/s/Mpc):
    k = H0 / [ c * (D_max - 1) ]
  This ensures the small‑z slope matches the chosen anchor under a uniform void path.
- You can override k with --k-mpc-inv.
- The script simulates non-uniform paths (galaxy → void → galaxy) using a simple piecewise g_bar(r).

Outputs
- Prints sample z at 500 Mpc increments up to --distance-max (default 4000 Mpc).
- Saves a plot comparing the Energy Tariff curve vs linear Hubble lines (Planck & SH0ES).

Caveat
- This is phenomenologically "tired light"-like; large-scale constraints (SN Ia time dilation,
  Tolman surface-brightness test, CMB blackbody, BBN) are stringent and must be addressed in a
  full theory.
"""
from __future__ import annotations

import argparse
import math
from typing import Callable, List

import numpy as np
import matplotlib.pyplot as plt

# Physical constants
C_KM_S = 299_792.458
MPC_TO_M = 3.085677581491367e22

# RAR‑plateau (from README Box 1)
A0 = 1.2e-10   # m/s^2
D_MAX = 50.0   # finite plateau cap


def xi_rar_plateau(g_bar_m_s2: float, a0: float = A0, D_max: float = D_MAX) -> float:
    """RAR‑plateau gate from README (Box 1):
    xi(g) = min[ 1/2 + sqrt(1/4 + a0/g), D_max ].
    In void limit g→0+, xi→D_max. If g<=0, we return D_max.
    """
    if not math.isfinite(g_bar_m_s2) or g_bar_m_s2 <= 0.0:
        return float(D_max)
    val = 0.5 + math.sqrt(0.25 + a0 / max(g_bar_m_s2, 1e-300))
    return float(min(val, D_max))


class PhotonJourney:
    """Simulate a photon’s redshift via the Energy Tariff principle along a path.

    1 + z = exp( k * ∫ (xi(r) - 1) dr ), with r in Mpc and k in 1/Mpc.
    """

    def __init__(self, k_coupling_mpc_inv: float):
        self.k = float(k_coupling_mpc_inv)
        self.a0 = float(A0)
        self.d_max = float(D_MAX)

    @staticmethod
    def piecewise_environment(distance_mpc: float,
                              galaxy_shell_mpc: float = 0.05,
                              g_bar_galaxy: float = 1e-8,
                              g_bar_void: float = 1e-15) -> Callable[[float], float]:
        """Return g_bar(r) in m/s^2 along the path:
        - [0, galaxy_shell_mpc]: host galaxy (high g → xi≈1)
        - (galaxy_shell_mpc, distance_mpc - galaxy_shell_mpc): void (xi saturates to D_MAX)
        - [distance_mpc - galaxy_shell_mpc, distance_mpc]: Milky Way (high g → xi≈1)
        """
        galaxy_shell_mpc = max(float(galaxy_shell_mpc), 0.0)
        distance_mpc = max(float(distance_mpc), 0.0)
        void_len = max(distance_mpc - 2.0 * galaxy_shell_mpc, 0.0)

        def g_bar_at(r_mpc: float) -> float:
            if r_mpc <= galaxy_shell_mpc:
                return g_bar_galaxy
            if r_mpc >= distance_mpc - galaxy_shell_mpc:
                return g_bar_galaxy
            # interior void
            return g_bar_void

        return g_bar_at

    def redshift(self, distance_mpc: float, steps: int = 4000) -> float:
        """Numerically integrate to compute z for a source at distance_mpc.
        Uses simple Riemann sum: ∫(xi-1)dr ≈ Σ (xi(r_i)-1) Δr.
        """
        if distance_mpc <= 0.0:
            return 0.0
        steps = int(max(steps, 10))
        dr = float(distance_mpc) / steps
        g_bar_fn = self.piecewise_environment(distance_mpc)
        accum = 0.0
        # left Riemann sum
        for i in range(steps):
            r = i * dr
            xi = xi_rar_plateau(g_bar_fn(r), self.a0, self.d_max)
            accum += (xi - 1.0) * dr
        expo = self.k * accum
        return float(np.exp(expo) - 1.0)


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Energy Tariff redshift model using README RAR‑plateau xi")
    ap.add_argument("--distance-max", type=float, default=4000.0,
                    help="Max distance (Mpc) to simulate/plot (default 4000)")
    ap.add_argument("--steps", type=int, default=200,
                    help="Number of distance samples for the curve (default 200)")
    ap.add_argument("--k-mpc-inv", type=float, default=None,
                    help="Override k (1/Mpc). If not provided, k is calibrated from --anchor-h0.")
    ap.add_argument("--anchor-h0", type=float, default=67.4,
                    help="H0 (km/s/Mpc) anchor to calibrate k when --k-mpc-inv is not given (default 67.4)")
    args = ap.parse_args(argv)

    # Calibrate or accept k
    if args.k_mpc_inv is None or not math.isfinite(args.k_mpc_inv) or args.k_mpc_inv <= 0:
        # k = H0 / [ c * (D_max - 1) ]  (H0 in km/s/Mpc, c in km/s → k in 1/Mpc)
        k_val = float(args.anchor_h0) / (C_KM_S * (D_MAX - 1.0))
        print(f"Calibrated k from H0={args.anchor_h0} km/s/Mpc => k = {k_val:.9e} 1/Mpc")
        print(f"Per‑Mpc fractional energy loss (H0/c): {(args.anchor_h0/C_KM_S)*100.0:.6f}%/Mpc")
    else:
        k_val = float(args.k_mpc_inv)
        h0_smallz = C_KM_S * k_val * (D_MAX - 1.0)
        print(f"Using user k = {k_val:.9e} 1/Mpc (small‑z slope implies H0 ≈ {h0_smallz:.6f} km/s/Mpc)")

    # Build simulator
    sim = PhotonJourney(k_coupling_mpc_inv=k_val)

    # Distance grid
    dmax = max(float(args.distance_max), 0.0)
    n = int(max(args.steps, 10))
    distances = np.linspace(0.0, dmax, n)

    # Compute z(dist)
    print("Computing redshift curve...")
    z_vals = np.array([sim.redshift(d, steps=4000) for d in distances], dtype=float)

    # Print sample values at 500 Mpc multiples
    for D in range(500, int(dmax)+1, 500):
        zD = sim.redshift(float(D), steps=4000)
        print(f"  z({D:4d} Mpc) = {zD:.6f}")

    # Plot vs linear Hubble laws for comparison
    h0_planck = 67.4
    h0_shoes = 73.0
    z_planck = (h0_planck / C_KM_S) * distances
    z_shoes = (h0_shoes / C_KM_S) * distances

    plt.figure(figsize=(12, 7))
    plt.plot(distances, z_vals, '-', color='crimson', lw=2.5, label='Energy Tariff (RAR‑plateau)')
    plt.plot(distances, z_planck, '--', color='steelblue', lw=1.8, label=f'Linear Hubble (Planck H0={h0_planck})')
    plt.plot(distances, z_shoes, ':', color='seagreen', lw=1.8, label=f'Linear Hubble (SH0ES H0={h0_shoes})')
    plt.title("Predicted redshift from Energy Tariff model (RAR‑plateau ξ)")
    plt.xlabel("Distance (Mpc)")
    plt.ylabel("Redshift z")
    plt.grid(True, ls='--', alpha=0.5)
    plt.legend()
    plt.xlim(0, dmax)
    ymax = max(np.max(z_vals), np.max(z_planck), np.max(z_shoes))
    plt.ylim(0, ymax * 1.1 if ymax > 0 else 1)
    out_png = "energy_tariff_redshift_model.png"
    plt.savefig(out_png, dpi=150)
    print(f"Saved plot: {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

