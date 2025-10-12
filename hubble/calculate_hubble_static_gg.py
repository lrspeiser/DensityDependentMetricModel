#!/usr/bin/env python3
"""
calculate_hubble_static_gg.py — Static “gravitational resistance” toy model using the
RAR‑plateau gate from this repository’s main README.

What this script does
- Implements ξ(g_bar) from README.md (RAR‑plateau; D_max = 50; a0 = 1.2e−10 m/s^2)
  and applies it to a static “gravitational resistance” redshift law:
      dE/E = -k * (ξ_void - 1) * dr   with k in Mpc^-1
  which implies for small z:
      z ≈ k * (ξ_void - 1) * r,
      H0 = c * k * (ξ_void - 1).
- Reports:
  1) An illustrative estimate of g_bar in a void from a supercluster-like mass;
     the corresponding ξ at that estimate;
  2) The strict void limit (g_bar → 0) where ξ_void → D_max;
  3) For several observed H0 anchors (Planck, SH0ES, BAO+BBN/“DESI-era”), the
     required k to match each (and the implied per‑Mpc fractional energy loss);
  4) Optionally, if --k-mpc-inv is supplied, computes H0_pred and offsets
     relative to each anchor.

Academic caveat
- Without an independent theoretical prior for k, the pair (k, ξ_void) only
  constrains their product via H0 = c * k * (ξ_void - 1). Calibrating k to any
  one anchor will reproduce that anchor by construction; offsets versus other
  anchors reflect the known “H0 tension,” not a prediction of the gate itself.

Formula provenance (exact RAR‑plateau gate from README)
- README.md (repo root), “Box 1 — Exact weak‑field formula used in all figures”:
  ξ(R) = min[ 1/2 + sqrt(1/4 + a0_eff/g_bar(R)), D_max ], D_max = 50 (fiducial).
  We adopt the a0_eff → a0 constant form here (no environment terms) and apply
  the function to a representative g_bar value for intergalactic space and to
  the strict-void limit.

This script reads no web services, no secrets, and does not modify README.md.
"""
from __future__ import annotations

import argparse
import math
from typing import Dict, List, Tuple

# --- Physical constants and units ---
G_N = 6.67430e-11              # m^3 kg^-1 s^-2
C_KM_S = 299_792.458           # km/s
MPC_TO_M = 3.085677581491367e22  # m/Mpc
M_SUN = 1.98847e30             # kg

# --- DGG (from README) parameters ---
A0 = 1.2e-10   # m/s^2
D_MAX = 50.0   # plateau cap (paper preset)


def xi_rar_plateau(g_bar_m_s2: float, a0: float = A0, D_max: float = D_MAX) -> float:
    """RAR‑plateau gate from README (Box 1), applied to a scalar g_bar.
    ξ(g) = min[ 1/2 + sqrt(1/4 + a0/g), D_max ].
    - In the strict void limit g→0+, ξ→D_max.
    - If g <= 0, return D_max by definition of the plateau limit.
    """
    if not math.isfinite(g_bar_m_s2) or g_bar_m_s2 <= 0.0:
        return float(D_max)
    val = 0.5 + math.sqrt(0.25 + a0 / max(g_bar_m_s2, 1e-300))
    return float(min(val, D_max))


def estimate_intergalactic_g_bar(m_supercluster_msun: float = 1.0e17,
                                 distance_mpc: float = 50.0) -> float:
    """Illustrative Newtonian g_bar ≈ GM/r^2 in a void influenced by a nearby
    supercluster-like mass.
    NOTE: This is only a toy estimate; the strict void-limit behavior is used
    for k/H0 calibration (ξ_void = D_max) to reflect g→0.
    """
    mass_kg = m_supercluster_msun * M_SUN
    r_m = distance_mpc * MPC_TO_M
    return G_N * mass_kg / (r_m * r_m)


def parse_anchors(anchor_args: List[str]) -> List[Tuple[str, float]]:
    out: List[Tuple[str, float]] = []
    for s in anchor_args:
        try:
            label, val = s.split(":", 1)
            label = label.strip()
            h0 = float(val.strip())
            if not label:
                label = f"Anchor_{len(out)+1}"
            out.append((label, h0))
        except Exception:
            # ignore malformed entries
            continue
    return out


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Static gravitational‑resistance H0 toy model (RAR‑plateau ξ)")
    ap.add_argument(
        "--anchor",
        action="append",
        default=[],
        help="Anchor as 'Label:Value' with Value in km/s/Mpc. Repeatable.",
    )
    ap.add_argument(
        "--k-mpc-inv",
        type=float,
        default=None,
        help="If provided, compute H0_pred = c * k * (ξ_void - 1) and offsets vs anchors.",
    )
    ap.add_argument(
        "--void-mass",
        type=float,
        default=1.0e17,
        help="Supercluster-like mass [M_sun] used in the illustrative g_bar estimate (default 1e17).",
    )
    ap.add_argument(
        "--void-distance",
        type=float,
        default=50.0,
        help="Distance [Mpc] used in the illustrative g_bar estimate (default 50).",
    )
    args = ap.parse_args(argv)

    # Default anchors (academically relevant, 1-liners; editable via --anchor)
    anchors: List[Tuple[str, float]] = [
        ("Planck_2018", 67.4),  # CMB baseline
        ("SH0ES", 73.0),        # local distance ladder
        ("BAO+BBN_DESI", 68.5), # representative BAO+BBN combo
    ]
    user_anchors = parse_anchors(args.anchor)
    if user_anchors:
        anchors = user_anchors

    # 1) Illustrative void g_bar estimate and ξ at that estimate
    g_est = estimate_intergalactic_g_bar(args.void_mass, args.void_distance)
    xi_at_est = xi_rar_plateau(g_est)

    # 2) Strict void limit used for calibration (g→0 => ξ_void = D_MAX)
    xi_void = D_MAX

    print("--- Static gravitational-resistance model with RAR‑plateau ξ ---")
    print(f"README formula: xi(g) = min[ 1/2 + sqrt(1/4 + a0/g), D_max ] with a0={A0:.2e} m/s^2, D_max={D_MAX}")
    print("(We adopt the strict void limit ξ_void = D_max for calibration; the estimate below is illustrative.)")
    print("-")
    print("Illustrative intergalactic void estimate (toy):")
    print(f"  Mass ≈ {args.void_mass:.2e} M_sun, Distance ≈ {args.void_distance:.1f} Mpc")
    print(f"  g_bar_est ≈ {g_est:.3e} m/s^2;  a0/g_est ≈ {A0/g_est:.3e}")
    print(f"  xi(g_bar_est) ≈ {xi_at_est:.6g}")
    print("  Strict void limit: xi_void = D_max = {:.0f}".format(D_MAX))
    print("-")

    # 3) For each anchor H0, compute k and the implied per‑Mpc fractional loss
    # From H0 = c * k * (ξ_void - 1)  =>  k = H0 / [c * (ξ_void - 1)]
    # Fractional energy loss per Mpc: k * (ξ_void - 1) = H0 / c (unitless per Mpc)
    print("Calibrated k per anchor (using ξ_void = D_max):")
    print("  Columns: label | H0 [km/s/Mpc] | k [1/Mpc] | frac_loss_per_Mpc [%]")
    k_for_anchor: Dict[str, float] = {}
    for label, h0 in anchors:
        denom = C_KM_S * (xi_void - 1.0)
        k_mpc_inv = float(h0) / denom
        frac_loss = (float(h0) / C_KM_S) * 100.0  # percent per Mpc
        k_for_anchor[label] = k_mpc_inv
        print(f"  {label:14s} | {h0:8.3f} | {k_mpc_inv:9.6e} | {frac_loss:10.6f}")

    # 4) If user supplied k, compute H0_pred and offsets
    if args.k_mpc_inv is not None and math.isfinite(args.k_mpc_inv) and args.k_mpc_inv > 0.0:
        k_in = float(args.k_mpc_inv)
        h0_pred = C_KM_S * k_in * (xi_void - 1.0)
        print("-")
        print("Prediction from user-supplied k:")
        print(f"  k_in = {k_in:.9e} 1/Mpc  =>  H0_pred = {h0_pred:.6f} km/s/Mpc")
        print("  Offsets vs anchors:")
        for label, h0 in anchors:
            diff = h0_pred - float(h0)
            rel = diff / float(h0)
            print(f"    {label:14s}: Δ = {diff:+.3f} km/s/Mpc ({rel:+.3%})")

    print("-")
    print("Observational caveats (tired‑light‑like phenomenology): SN Ia time dilation, Tolman surface‑brightness test,\n"
          "CMB blackbody spectrum, and BBN all pose strong constraints that any static‑universe mechanism must meet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

