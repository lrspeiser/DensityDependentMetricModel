# models/gas_profile.py
# -----------------------------------------------------------------------------
# Gas surface-density reconstruction for SPARC galaxies.
# Option A: Exponential disk constrained by MHI and RHI (Sigma(RHI) ~ 1 Msun/pc^2)
# Option B: Shape from V_gas(R) normalized to total gas mass (with helium factor)
# Also provides a truncated exponential variant to avoid unrealistically large
# central densities when enforcing Sigma(RHI)=1 with a finite total mass.
# -----------------------------------------------------------------------------
from __future__ import annotations
import math
from typing import Dict
import numpy as np

PC_PER_KPC = 1000.0


def _bisection_solve(f, lo: float, hi: float, tol: float = 1e-6, maxiter: int = 200) -> float:
    flo = f(lo)
    fhi = f(hi)
    if flo == 0.0:
        return lo
    if fhi == 0.0:
        return hi
    if flo * fhi > 0:
        # Try expanding hi
        for _ in range(40):
            hi *= 1.5
            fhi = f(hi)
            if flo * fhi <= 0:
                break
        if flo * fhi > 0:
            raise RuntimeError("Bisection failed to bracket root")
    for _ in range(maxiter):
        mid = 0.5 * (lo + hi)
        fmid = f(mid)
        if abs(fmid) < tol or (hi - lo) < tol:
            return mid
        if flo * fmid < 0:
            hi = mid
            fhi = fmid
        else:
            lo = mid
            flo = fmid
    return 0.5 * (lo + hi)


def reconstruct_gas_exponential(
    R_kpc: np.ndarray,
    MHI_1e9Msun: float,
    RHI_kpc: float,
    include_He: bool = True,
    min_Rd_kpc: float = 0.2,
    max_Rd_kpc: float = 20.0,
) -> Dict[str, np.ndarray]:
    """
    Infinite exponential (legacy): Σ(R) = Σ0 * exp(-R/Rd) with constraints
      Σ(RHI) = 1 Msun/pc^2
      M_gas = (1.33 if include_He else 1.0) * MHI = 2π Σ0 Rd^2
    Note: This can over-concentrate mass. Prefer the truncated version below.
    Returns dict with Sigma_gas (Msun/pc^2), Rd_kpc, Sigma0.
    """
    MHI = float(MHI_1e9Msun) * 1e9
    Mgas = 1.33 * MHI if include_He else MHI

    def f_rd(rd_kpc: float) -> float:
        if rd_kpc <= 0:
            return -1e99
        return 2.0 * math.pi * (rd_kpc ** 2) * math.exp(RHI_kpc / rd_kpc) - Mgas

    Rd_kpc = _bisection_solve(f_rd, min_Rd_kpc, max_Rd_kpc)
    Sigma0 = math.exp(RHI_kpc / Rd_kpc)

    R_pc = np.asarray(R_kpc, dtype=float) * PC_PER_KPC
    Rd_pc = Rd_kpc * PC_PER_KPC
    Sigma = Sigma0 * np.exp(-R_pc / Rd_pc)
    return {"Sigma_gas": Sigma, "Rd_kpc": np.array([Rd_kpc]), "Sigma0": np.array([Sigma0])}


def reconstruct_gas_exponential_truncated(
    R_kpc: np.ndarray,
    MHI_1e9Msun: float,
    RHI_kpc: float,
    include_He: bool = True,
    Rmax_mode: str = "RHI",   # "RHI" or "kRd"
    kRd: float = 3.0,         # only used if Rmax_mode=="kRd"
    rd_bracket_kpc=(0.1, 30.0),
    enforce_rd_bounds: bool = False,
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Truncated exponential: Σ(R)=Σ0 exp(-R/Rd) for 0<=R<=Rmax, with
      Σ(RHI)=1 [Msun/pc^2]  and  Mgas = 2π Σ0 Rd^2 [1 - e^{-x}(1+x)], x=Rmax/Rd.
    Units:
      - R, Rd, RHI, Rmax in kpc throughout
      - Σ in Msun/pc^2
    """
    import math as _m
    import numpy as _np
    PC_PER_KPC = 1_000.0
    MHI = float(MHI_1e9Msun) * 1e9
    Mgas = 1.33 * MHI if include_He else MHI

    def mass_residual(Rd_kpc: float) -> float:
        if Rd_kpc <= 0:
            return 1e99
        Sigma0 = _m.exp(RHI_kpc / Rd_kpc)  # Σ(RHI)=1 => Σ0=e^{RHI/Rd} [Msun/pc^2]
        Rmax_kpc = (RHI_kpc if Rmax_mode == "RHI" else kRd * Rd_kpc)
        x = Rmax_kpc / Rd_kpc
        # Truncated mass (unit conversion: kpc^2 -> pc^2)
        M_pred = 2.0 * _m.pi * Sigma0 * (Rd_kpc**2) * (1.0 - _m.exp(-x) * (1.0 + x)) * (PC_PER_KPC**2)
        return M_pred - Mgas

    lo, hi = rd_bracket_kpc
    # If enforcing bounds for RHI mode, clamp bracket to [0.5,10] kpc per conservative recipe
    if enforce_rd_bounds and Rmax_mode.upper() == "RHI":
        lo, hi = max(lo, 0.5), min(hi, 10.0)
        if lo >= hi:
            lo, hi = 0.5, 10.0

    # Attempt to find a sign change within bracket; if none, we will pick the minimizer inside [lo,hi]
    f_lo, f_hi = mass_residual(lo), mass_residual(hi)
    have_bracket = (f_lo * f_hi <= 0)

    # bracket growth if allowed and not enforcing strict bounds
    if (not have_bracket) and (not enforce_rd_bounds):
        grow = 0
        while f_lo * f_hi > 0 and hi < 200.0:
            hi *= 1.5
            f_hi = mass_residual(hi)
            grow += 1
            if grow > 40:
                break
        have_bracket = (f_lo * f_hi <= 0)

    Rd_kpc = None
    if have_bracket:
        # bisection
        lo_b, hi_b = lo, hi
        f_lo_b, f_hi_b = f_lo, f_hi
        for _ in range(200):
            mid = 0.5 * (lo_b + hi_b)
            f_mid = mass_residual(mid)
            if abs(f_mid) < 1e-6 or (hi_b - lo_b) < 1e-6:
                Rd_kpc = mid
                break
            if f_lo_b * f_mid < 0:
                hi_b = mid
                f_hi_b = f_mid
            else:
                lo_b = mid
                f_lo_b = f_mid
        if Rd_kpc is None:
            Rd_kpc = 0.5 * (lo_b + hi_b)
        mass_mismatch = 0.0
        penalty_mass = 0.0
    else:
        # No root in [lo,hi]; choose Rd that minimizes |mass_residual| within bounds and compute a soft penalty
        # Coarse scan then local refine
        grid = _np.linspace(lo, hi, 200)
        vals = _np.abs([mass_residual(x) for x in grid])
        idx = int(_np.argmin(vals))
        Rd_init = float(grid[idx])
        # Simple golden-section like refine
        a, b = max(lo, Rd_init - (hi-lo)*0.1), min(hi, Rd_init + (hi-lo)*0.1)
        for _ in range(80):
            m1 = a + (b - a) / 3.0
            m2 = b - (b - a) / 3.0
            f1 = abs(mass_residual(m1))
            f2 = abs(mass_residual(m2))
            if f1 < f2:
                b = m2
            else:
                a = m1
        Rd_kpc = 0.5 * (a + b)
        # Define a mass mismatch ratio and convert to a 0.2 dex Gaussian penalty equivalent
        M_pred = 2.0 * _m.pi * _m.exp(RHI_kpc / Rd_kpc) * (Rd_kpc**2) * (1.0 - _m.exp(-( (RHI_kpc if Rmax_mode == "RHI" else (kRd*Rd_kpc)) / Rd_kpc)) * (1.0 + ( (RHI_kpc if Rmax_mode == "RHI" else (kRd*Rd_kpc)) / Rd_kpc))) * (PC_PER_KPC**2)
        ratio = (M_pred / MHI) if MHI > 0 else 1.0
        # treat log10 ratio deviation with sigma=0.2 dex
        import math as __math
        dlog10 = __math.log10(max(ratio, 1e-30))
        penalty_mass = (dlog10 / 0.2) ** 2
        mass_mismatch = ratio

    Sigma0 = _m.exp(RHI_kpc / Rd_kpc)
    Rmax_kpc = (RHI_kpc if Rmax_mode == "RHI" else kRd * Rd_kpc)
    if verbose:
        print(f"[gas_profile] Truncated-A: Rd={Rd_kpc:.3f} kpc, Sigma0={Sigma0:.2f} Msun/pc^2, Rmax={Rmax_kpc:.3f} kpc")
        if not have_bracket:
            print(f"[gas_profile] Note: No exact mass solve within bounds; using best-effort Rd with mass_mismatch={mass_mismatch:.3f} and penalty≈{penalty_mass:.2f}")

    R_pc  = _np.asarray(R_kpc) * PC_PER_KPC
    Rd_pc = Rd_kpc * PC_PER_KPC
    Rmax_pc = Rmax_kpc * PC_PER_KPC
    Sigma = Sigma0 * _np.exp(-R_pc / Rd_pc)
    Sigma[R_pc > Rmax_pc] = 0.0
    return {"Sigma_gas": Sigma, "Rd_kpc": np.array([Rd_kpc]), "Sigma0": np.array([Sigma0]), "Rmax_kpc": np.array([Rmax_kpc]), "mass_mismatch": _np.array([mass_mismatch if have_bracket is False else 1.0]), "penalty_mass": _np.array([penalty_mass if have_bracket is False else 0.0])}


def reconstruct_gas_from_vgas(
    R_kpc: np.ndarray,
    Vgas_kms: np.ndarray,
    MHI_1e9Msun: float | None,
    include_He: bool = True,
    eps: float = 1e-6,
) -> Dict[str, np.ndarray]:
    """
    Shape-from-V_gas fallback: Σ~V_gas^2/R, normalized so total mass equals M_gas if MHI given.
    Returns dict with Sigma_gas (Msun/pc^2).
    """
    R = np.asarray(R_kpc, dtype=float)
    V = np.asarray(Vgas_kms, dtype=float)
    shape = (np.clip(V, 0, np.inf) ** 2) / np.maximum(R, eps)

    if shape.max() <= 0:
        return {"Sigma_gas": np.zeros_like(R)}

    if MHI_1e9Msun is not None:
        MHI = float(MHI_1e9Msun) * 1e9
        Mgas = 1.33 * MHI if include_He else MHI
        dR_kpc = np.gradient(R)
        R_pc = R * PC_PER_KPC
        dR_pc = dR_kpc * PC_PER_KPC
        denom = 2.0 * math.pi * np.sum(shape * R_pc * dR_pc)
        if denom <= 0:
            return {"Sigma_gas": np.zeros_like(R)}
        A = Mgas / denom
        Sigma = A * shape
    else:
        # Normalize such that Σ at last point ~ 1 Msun/pc^2 (rough)
        A = 1.0 / (shape[-1])
        Sigma = A * shape

    return {"Sigma_gas": Sigma}

