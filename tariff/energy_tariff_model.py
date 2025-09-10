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
- Optionally, builds a Hubble Diagram (μ vs z) with Pantheon+SH0ES data and computes χ².

Caveat
- This is phenomenologically "tired light"-like; large-scale constraints (SN Ia time dilation,
  Tolman surface-brightness test, CMB blackbody, BBN) are stringent and must be addressed in a
  full theory.
"""
from __future__ import annotations

import argparse
import math
from typing import Callable, List, Tuple

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


# ---------- Data loading (Pantheon+SH0ES) ----------

def load_pantheon_data(filepath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load z (zHD), MU_SH0ES, MU_SH0ES_ERR_DIAG from a Pantheon+SH0ES .dat file.

    Expected header (first line contains column names). We use columns by name:
    - zHD (3rd col)
    - MU_SH0ES (11th col)
    - MU_SH0ES_ERR_DIAG (12th col)
    Lines starting with '#' are ignored. Header lines with 'CID' are skipped.
    """
    z_list, mu_list, muerr_list = [], [], []
    with open(filepath, 'r') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith('#'):
                continue
            if s.startswith('CID'):
                # header line
                continue
            cols = s.split()
            try:
                # zero-based indexing: 2→zHD, 10→MU_SH0ES, 11→MU_SH0ES_ERR_DIAG
                z_val = float(cols[2])
                mu_val = float(cols[10])
                mu_err = float(cols[11])
            except (IndexError, ValueError):
                continue
            z_list.append(z_val)
            mu_list.append(mu_val)
            muerr_list.append(mu_err)
    return np.asarray(z_list, float), np.asarray(mu_list, float), np.asarray(muerr_list, float)


class PhotonJourney:
    """Simulate a photon’s redshift via the Energy Tariff principle along a path.

    1 + z = exp( k * ∫ (xi(r) - 1) dr ), with r in Mpc and k in 1/Mpc.
    """

    def __init__(self, k_coupling_mpc_inv: float):
        self.k = float(k_coupling_mpc_inv)
        self.a0 = float(A0)
        self.d_max = float(D_MAX)
        self._lookup: Tuple[np.ndarray, np.ndarray] | None = None  # (distances, z)

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
        for i in range(steps):
            r = i * dr
            xi = xi_rar_plateau(g_bar_fn(r), self.a0, self.d_max)
            accum += (xi - 1.0) * dr
        expo = self.k * accum
        return float(np.exp(expo) - 1.0)

    # ---------- Inverse: μ(z) via a precomputed lookup of z(r) ----------

    def _ensure_lookup(self, max_dist_mpc: float = 8000.0, n_points: int = 600) -> None:
        if self._lookup is not None:
            return
        d_grid = np.linspace(0.0, max(1.0, float(max_dist_mpc)), int(max(n_points, 100)))
        z_grid = np.array([self.redshift(d, steps=3000) for d in d_grid], dtype=float)
        # Ensure monotonic by slight smoothing if needed (model is monotonic; guard against tiny FP noise)
        z_grid = np.maximum.accumulate(z_grid)
        self._lookup = (d_grid, z_grid)

    def distance_modulus_at_z(self, z_values: np.ndarray) -> np.ndarray:
        """Predict μ(z) using inverse of z(r):
        - Build a lookup (r, z(r)) and linearly interpolate r for each z.
        - Convert r [Mpc] → parsecs and then μ = 5 log10(d_pc) − 5.
        """
        self._ensure_lookup()
        d_grid, z_grid = self._lookup  # type: ignore[misc]
        z_values = np.asarray(z_values, float)
        # Clamp z within table range
        z_min, z_max = float(z_grid[0]), float(z_grid[-1])
        z_safe = np.clip(z_values, z_min, z_max)
        r_mpc = np.interp(z_safe, z_grid, d_grid)
        d_pc = r_mpc * 1.0e6
        mu = np.full_like(d_pc, np.nan)
        mask = d_pc > 0
        mu[mask] = 5.0 * np.log10(d_pc[mask]) - 5.0
        return mu


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Energy Tariff redshift model using README RAR‑plateau xi")
    ap.add_argument("--distance-max", type=float, default=4000.0,
                    help="Max distance (Mpc) to simulate/plot (default 4000)")
    ap.add_argument("--steps", type=int, default=200,
                    help="Number of distance samples for the redshift curve (default 200)")
    ap.add_argument("--k-mpc-inv", type=float, default=None,
                    help="Override k (1/Mpc). If not provided, k is calibrated from --anchor-h0.")
    ap.add_argument("--anchor-h0", type=float, default=67.4,
                    help="H0 (km/s/Mpc) anchor to calibrate k when --k-mpc-inv is not given (default 67.4)")
    ap.add_argument("--data-file", type=str,
                    default="external_data/pantheon/Pantheon+SH0ES.dat",
                    help="Path to Pantheon+SH0ES .dat file to overlay (μ vs z)")
    ap.add_argument("--plot-hubble", action="store_true",
                    help="If set, build Hubble Diagram (μ vs z) with data overlay and model lines")
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

    # Distance grid for z(distance) curve
    dmax = max(float(args.distance_max), 0.0)
    n = int(max(args.steps, 10))
    distances = np.linspace(0.0, dmax, n)
    print("Computing redshift curve...")
    z_vals = np.array([sim.redshift(d, steps=4000) for d in distances], dtype=float)

    # Print sample values at 500 Mpc multiples
    for D in range(500, int(dmax)+1, 500):
        zD = sim.redshift(float(D), steps=4000)
        print(f"  z({D:4d} Mpc) = {zD:.6f}")

    # Plot distance vs redshift curve and linear Hubble for comparison (original plot)
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

    # Optional: Build Hubble Diagram (μ vs z) with data overlay
    if args.plot_hubble:
        try:
            z_data, mu_data, mu_err = load_pantheon_data(args.data_file)
            print(f"Loaded Pantheon+ data: N={len(z_data)} from '{args.data_file}'")
        except FileNotFoundError:
            print(f"Pantheon+ data file not found: '{args.data_file}'")
            return 1
        # Filter to finite and positive uncertainties
        mask = np.isfinite(z_data) & np.isfinite(mu_data) & np.isfinite(mu_err) & (mu_err > 0)
        z_use = z_data[mask]
        mu_use = mu_data[mask]
        mu_err_use = mu_err[mask]
        # Predict μ(z) from the Energy Tariff model
        mu_model = sim.distance_modulus_at_z(z_use)
        # Chi-squared
        valid = np.isfinite(mu_model)
        zv, muv, muev, mum = z_use[valid], mu_use[valid], mu_err_use[valid], mu_model[valid]
        dof = max(int(len(zv) - 1), 1)
        chi2 = float(np.sum(((muv - mum) / muev) ** 2))
        red_chi2 = chi2 / dof
        print("-" * 40)
        print("Hubble Diagram fit (Energy Tariff vs Pantheon+):")
        print(f"  N_used = {len(zv)}  (filtered from N_total = {len(z_data)})")
        print(f"  Chi^2 = {chi2:.2f}   Reduced Chi^2 = {red_chi2:.3f}")
        print("-" * 40)

        # Smooth model line across z-range of data
        z_smooth = np.logspace(np.log10(max(zv.min(), 1e-4)), np.log10(max(zv.max(), 1e-3)), 400)
        mu_smooth = sim.distance_modulus_at_z(z_smooth)
        # Linear Hubble μ(z) references (small-z d_L ≈ c/H0 * z)
        def mu_linear(z_arr: np.ndarray, H0: float) -> np.ndarray:
            d_mpc = (C_KM_S / H0) * z_arr
            d_pc = d_mpc * 1.0e6
            out = np.full_like(d_pc, np.nan)
            ok = d_pc > 0
            out[ok] = 5.0 * np.log10(d_pc[ok]) - 5.0
            return out
        mu_planck = mu_linear(z_smooth, h0_planck)
        mu_shoes = mu_linear(z_smooth, h0_shoes)

        plt.figure(figsize=(12, 8))
        plt.errorbar(zv, muv, yerr=muev, fmt='.', color='gray', alpha=0.45,
                     label=f'Pantheon+SH0ES (N={len(zv)})')
        plt.plot(z_smooth, mu_smooth, '-', color='crimson', lw=2.2, label='Energy Tariff (RAR‑plateau)')
        plt.plot(z_smooth, mu_planck, '--', color='steelblue', lw=1.5, label=f'Linear Hubble (Planck H0={h0_planck})')
        plt.plot(z_smooth, mu_shoes, ':', color='seagreen', lw=1.5, label=f'Linear Hubble (SH0ES H0={h0_shoes})')
        plt.xscale('log')
        plt.xlabel('Redshift z')
        plt.ylabel('Distance Modulus μ')
        plt.title('Hubble Diagram: Pantheon+ vs Energy Tariff (RAR‑plateau)')
        plt.grid(True, which='both', ls='--', alpha=0.5)
        plt.legend()
        out_hd = "hubble_diagram_with_data.png"
        plt.savefig(out_hd, dpi=150)
        print(f"Saved Hubble Diagram: {out_hd}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

