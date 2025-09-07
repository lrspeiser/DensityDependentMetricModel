#!/usr/bin/env python3
"""
Minimal relativistic scaffolding for weak-field predictions and PPN export.

Assumption (explicit): we adopt a covariant subclass with
- c_T = 1 (GW170817-safe) in the relevant backgrounds
- no anisotropic stress in the weak-field, quasi-static regime so that Φ = Ψ

Consequences under this assumption:
- PPN γ = 1 and β = 1 in screened/high-acceleration regimes (Solar System)
- Preferred-frame parameters α1 = α2 = 0

This module provides programmatic PPN export and a guardrail to assert c_T = 1.
It does not derive these values from first principles here; instead, it encodes
the chosen subclass explicitly so manuscript figures are generated from a
single-theory path (dynamics and lensing) without any lensing-only scalars.

Notes
- Lensing in weak field uses Φ + Ψ = 2Φ for Φ = Ψ, matching our current
  stars+phantom mass mapping from xi(...) for ΔΣ and θ_E computations.
- Future work can replace these assumed-return values with explicit expressions
  of γ, β, α1, α2 for a concrete Lagrangian (e.g., restricted Horndeski/DHOST).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class PPNResult:
    gamma: float
    beta: float
    alpha1: float
    alpha2: float
    theory_assumption: str
    note: str


def check_c_T_guardrail(params: Dict[str, float]) -> bool:
    """Return True if tensor-speed constraint c_T == 1 is satisfied.

    Under the adopted subclass, c_T is identically 1 in the backgrounds of
    interest, so we assert True and allow an optional parameter to hard-fail
    if a user attempts to deviate (e.g., params.get('c_T') not in {None,1}).
    """
    cT = params.get('c_T', 1.0)
    return float(cT) == 1.0


def evaluate_ppn(params: Dict[str, float], radii_AU: List[float]) -> List[PPNResult]:
    """Evaluate PPN coefficients for the adopted relativistic subclass.

    For the Solar System radii provided, we return GR-consistent values
    (γ=1, β=1, α1=α2=0) with explicit theory assumption labels. This pairs with
    separate ΔG/G tables produced elsewhere from the xi-based gating.
    """
    assumption = 'weak-field Φ=Ψ, c_T=1; screened → GR locally'
    note = 'Adopted covariant subclass; see docs for derivation plan.'
    out = []
    for _ in radii_AU:
        out.append(PPNResult(
            gamma=1.0,
            beta=1.0,
            alpha1=0.0,
            alpha2=0.0,
            theory_assumption=assumption,
            note=note,
        ))
    return out
