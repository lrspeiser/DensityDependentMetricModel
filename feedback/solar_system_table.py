#!/usr/bin/env python3
"""
feedback/solar_system_table.py

Generates Solar System constraints for Gravity Gates / RAR-like gating:
- Computes |xi - 1| as a function of heliocentric distance (0.2–50 AU) for
  circular orbits around the Sun using a weak-field RAR mapping:
    D(x) = 0.5 + sqrt(0.25 + a0 / g_bar),  xi ≡ D
  where g_bar = GM_sun / r^2.
- Outputs:
  - feedback/results/solar_system/solar_system_xi_table.csv
  - feedback/results/solar_system/solar_system_xi_table.json
  - feedback/results/solar_system/solar_system_xi.png (with Cassini/ephemeris/LLR bands)
  - feedback/results/solar_system/ppn_summary.json (PPN values; GR under screening)

Notes
- Pure NumPy/Matplotlib; no CuPy/GPU needed.
- Does not modify any existing files; all artifacts live under feedback/results/.
- If theory/relativistic.evaluate_ppn is available, we record its GR PPN table; otherwise we fall back to GR values by assumption under screening.

"""
from __future__ import annotations
import os
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import matplotlib.pyplot as plt

# Constants
G_SI = 6.67430e-11           # m^3 kg^-1 s^-2
M_SUN = 1.98847e30           # kg
AU_M = 1.495978707e11        # m
KPC_M = 3.085677581491367e19 # m per kpc

# Constraint bands
CASSINI_GAMMA_BAND = 2.3e-5    # |gamma - 1| upper bound
EPHEMERIDES_BAND = 1.0e-8      # illustrative |ΔG/G| or |xi-1| tolerance level
LLR_BAND = 1.0e-13             # illustrative |ΔG/G| or |xi-1| tolerance level

# Default a0 (m/s^2) for RAR-like mapping
DEFAULT_A0 = 1.2e-10


def xi_from_rar_mapping(a0_m_s2: float, r_m: np.ndarray) -> np.ndarray:
    """RAR-inspired weak-field boost on circular orbits around the Sun.
    D = 0.5 + sqrt(0.25 + a0 / g_bar), g_bar = GM/r^2; xi ≡ D.
    Returns xi array with xi >= 1.
    """
    r = np.asarray(r_m, float)
    g_bar = G_SI * M_SUN / np.maximum(r, 1.0)**2
    D = 0.5 + np.sqrt(0.25 + np.maximum(a0_m_s2, 0.0) / np.maximum(g_bar, 1e-30))
    D = np.where(np.isfinite(D), D, 1.0)
    return np.maximum(D, 1.0)


def try_ppn_summary() -> Dict[str, Any]:
    """Return PPN summary (GR values under screening) if available from theory.relativistic;
    otherwise return a safe GR fallback.
    """
    try:
        from theory.relativistic import evaluate_ppn
        ppn = evaluate_ppn(params={"c_T": 1.0}, radii_AU=[0.39, 1.0, 5.2, 9.5, 19.2, 30.0])
        # Serialize dataclasses to dicts if needed
        out = []
        for row in ppn:
            out.append({
                "gamma": getattr(row, "gamma", 1.0),
                "beta": getattr(row, "beta", 1.0),
                "alpha1": getattr(row, "alpha1", 0.0),
                "alpha2": getattr(row, "alpha2", 0.0),
                "theory_assumption": getattr(row, "theory_assumption", "screened → GR"),
            })
        return {"ppn": out, "source": "theory.relativistic.evaluate_ppn"}
    except Exception:
        return {
            "ppn": [{"gamma": 1.0, "beta": 1.0, "alpha1": 0.0, "alpha2": 0.0} for _ in range(6)],
            "source": "fallback_GR_screened_assumption"
        }


def main() -> int:
    out_dir = Path("feedback/results/solar_system")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Radii to evaluate
    r_planets_AU = np.array([0.39, 1.0, 5.2, 9.5, 19.2, 30.0], dtype=float)  # Merc, Earth, Jup, Sat, Ura, Nep
    r_sweep_AU = np.logspace(np.log10(0.2), np.log10(50.0), 400)

    # Parameters (could be made configurable later)
    a0 = float(os.environ.get("FEEDBACK_A0_M_S2", DEFAULT_A0))

    # Compute xi and bands
    r_planets_m = r_planets_AU * AU_M
    r_sweep_m = r_sweep_AU * AU_M

    xi_sweep = xi_from_rar_mapping(a0, r_sweep_m)
    xi_planets = xi_from_rar_mapping(a0, r_planets_m)

    # Tables
    table_csv = out_dir / "solar_system_xi_table.csv"
    table_json = out_dir / "solar_system_xi_table.json"

    with table_csv.open("w", encoding="utf-8") as f:
        f.write("r_AU,g_bar_m_s2,xi,xi_minus_1\n")
        for rAU in r_sweep_AU:
            r = rAU * AU_M
            g = G_SI * M_SUN / (r * r)
            xi = float(0.5 + np.sqrt(0.25 + a0 / g))
            f.write(f"{rAU:.8f},{g:.10e},{xi:.10e},{(xi-1.0):.10e}\n")

    data = {
        "a0_m_s2": a0,
        "r_AU": r_sweep_AU.tolist(),
        "xi": xi_sweep.tolist(),
        "xi_minus_1": (xi_sweep - 1.0).tolist(),
        "planets": {
            "r_AU": r_planets_AU.tolist(),
            "xi_minus_1": (xi_planets - 1.0).tolist(),
        },
        "bands": {
            "cassini_gamma": CASSINI_GAMMA_BAND,
            "ephemerides": EPHEMERIDES_BAND,
            "llr": LLR_BAND,
        },
    }
    table_json.write_text(json.dumps(data, indent=2), encoding="utf-8")

    # PPN summary
    ppn_json = out_dir / "ppn_summary.json"
    ppn_json.write_text(json.dumps(try_ppn_summary(), indent=2), encoding="utf-8")

    # Plot
    plt.figure(figsize=(7.0, 4.4))
    y = np.abs(xi_sweep - 1.0)
    plt.loglog(r_sweep_AU, y, label=r"|$\xi-1$| (RAR mapping)")
    # Bands
    plt.axhline(CASSINI_GAMMA_BAND, color="tab:green", ls=":", lw=1.6, label="Cassini |γ−1|<2.3e−5")
    plt.axhline(EPHEMERIDES_BAND, color="tab:orange", ls=":", lw=1.6, label="Ephemerides ~1e−8")
    plt.axhline(LLR_BAND, color="tab:red", ls=":", lw=1.6, label="LLR ~1e−13")
    # Planet markers
    yp = np.abs(xi_planets - 1.0)
    plt.scatter(r_planets_AU, yp, c="k", s=30, zorder=3, label="Planets")
    for rAU, yv, name in zip(r_planets_AU, yp, ["Mercury","Earth","Jupiter","Saturn","Uranus","Neptune"]):
        plt.annotate(name, xy=(rAU, yv), xytext=(5, 3), textcoords="offset points", fontsize=8)

    plt.xlabel("Heliocentric distance r (AU)")
    plt.ylabel(r"|$\xi-1$|")
    plt.title("Solar System screening check (RAR-like gating)")
    plt.grid(alpha=0.3, which="both")
    plt.legend(frameon=False)
    figp = out_dir / "solar_system_xi.png"
    plt.tight_layout(); plt.savefig(figp, dpi=140); plt.close()

    print(f"[feedback] Wrote: {table_csv}")
    print(f"[feedback] Wrote: {table_json}")
    print(f"[feedback] Wrote: {ppn_json}")
    print(f"[feedback] Wrote: {figp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

