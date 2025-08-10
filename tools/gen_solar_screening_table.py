#!/usr/bin/env python3
"""
Generate a Solar-System screening comparison table for Supplement Table S3.
Computes |xi-1| for power-law screening and exponential screening at representative
Solar-System densities. Writes docs/table_solar_screening.md.

This is a theory-facing quick calculator: it does not integrate orbits; it simply
reports xi(ρ) for given parameter choices to illustrate screening behavior.
"""
from __future__ import annotations
from pathlib import Path
import math

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / 'docs' / 'table_solar_screening.md'

def xi_power(rho: float, rho_c: float, n_exp: float, A: float) -> float:
    # Power-law bounded: xi = 1 + A / (1 + (rho/rho_c)^n)
    ratio = max(rho, 1e-30) / max(rho_c, 1e-30)
    return 1.0 + A / (1.0 + ratio**n_exp)

def xi_exp(rho: float, rho_c: float, gamma: float, lambda_max: float) -> float:
    # Exponential screening: xi = 1 + lambda_max * exp(-(rho/rho_c)^gamma)
    ratio = max(rho, 1e-30) / max(rho_c, 1e-30)
    return 1.0 + lambda_max * math.exp(-(ratio ** gamma))

def main():
    # Representative densities (M_sun/kpc^3); rough orders of magnitude
    systems = [
        ("Laboratory (air)", 1e22),
        ("Earth orbit", 5e21),
        ("Jupiter orbit", 5e21),
        ("Saturn orbit", 2.3e21),
        ("Neptune orbit", 1e21),
    ]
    # Parameter choices (illustrative; to be updated alongside MW fits if needed)
    # Power-law variant
    rho_c_pow = 1e15
    n_pow = 3.0
    A_pow = 4.0
    # Exponential variant (gravitational color-like)
    rho_c_exp = 1e15
    gamma_exp = 3.0
    lambda_max = 4.0

    lines = []
    lines.append("# Solar-System Screening Comparison (Illustrative)")
    lines.append("")
    lines.append("Parameters:")
    lines.append(f"- Power-law: rho_c={rho_c_pow:.1e}, n={n_pow:.2f}, A={A_pow:.2f}")
    lines.append(f"- Exponential: rho_c={rho_c_exp:.1e}, gamma={gamma_exp:.2f}, lambda_max={lambda_max:.2f}")
    lines.append("")
    lines.append("| Environment | ρ [M_⊙/kpc³] | |ξ−1| (power) | |ξ−1| (exp) |")
    lines.append("|---|---:|---:|---:|")
    for name, rho in systems:
        xip = xi_power(rho, rho_c_pow, n_pow, A_pow)
        xie = xi_exp(rho, rho_c_exp, gamma_exp, lambda_max)
        lines.append(f"| {name} | {rho:.2e} | {abs(xip-1):.2e} | {abs(xie-1):.2e} |")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding='utf-8')
    print(f"Wrote {OUT}")

if __name__ == '__main__':
    main()
