# Variant gate registry for sandboxed density vs acceleration analyses
# Does not modify the main codebase; lives under Paper DensityAccel Variants/.

from __future__ import annotations
import numpy as np
from typing import Callable

Gate = Callable[[np.ndarray, np.ndarray | None, np.ndarray | None], np.ndarray]


def xi_accel(
    gbar: np.ndarray,
    rho: np.ndarray | None = None,
    R_kpc: np.ndarray | None = None,
    *,
    a0: float = 1.93e-7,
    Dmax: float = 50.0,
    **_: float,
) -> np.ndarray:
    gb = np.maximum(np.asarray(gbar, dtype=float), 1e-99)
    xi = 0.5 + np.sqrt(0.25 + a0 / gb)
    if np.isfinite(Dmax):
        xi = np.minimum(xi, Dmax)
    return xi


def xi_density_only(
    gbar: np.ndarray,
    rho: np.ndarray,
    R_kpc: np.ndarray | None = None,
    *,
    rho_c: float = 1e-27,
    gamma: float = 1.5,
    Dmax: float = 50.0,
    **_: float,
) -> np.ndarray:
    r = np.maximum(np.asarray(rho, dtype=float), 1e-99)
    env = 1.0 / (1.0 + np.power(r / rho_c, gamma))
    return 1.0 + (Dmax - 1.0) * env


def xi_hybrid(
    gbar: np.ndarray,
    rho: np.ndarray,
    R_kpc: np.ndarray | None = None,
    *,
    a0: float = 1.93e-7,
    rho_c: float = 1e-27,
    gamma: float = 1.5,
    zeta: float = 1.0,
    Dmax: float = 50.0,
    **_: float,
) -> np.ndarray:
    gb = np.maximum(np.asarray(gbar, dtype=float), 1e-99)
    r = np.maximum(np.asarray(rho, dtype=float), 1e-99)
    env = 1.0 / (1.0 + np.power(r / rho_c, gamma))
    a0_eff = a0 * (1.0 + zeta * env)
    xi = 0.5 + np.sqrt(0.25 + a0_eff / gb)
    if np.isfinite(Dmax):
        xi = np.minimum(xi, Dmax)
    return xi


def xi_density_plateau(
    gbar: np.ndarray,
    rho: np.ndarray,
    R_kpc: np.ndarray | None = None,
    *,
    rho_c: float = 1e-27,
    gamma: float = 1.5,
    n: float = 2.0,
    Dmax: float = 50.0,
    **_: float,
) -> np.ndarray:
    """Bounded density gate with adjustable sharpness and plateau."""
    r = np.maximum(np.asarray(rho, dtype=float), 1e-99)
    exponent = float(gamma) * float(n)
    env = 1.0 / (1.0 + np.power(r / rho_c, exponent))
    return 1.0 + (Dmax - 1.0) * env


def build_gate(kind: str, **params) -> Gate:
    k = kind.lower()
    if k == "accel":
        return lambda gbar, rho=None, R=None: xi_accel(gbar, rho, R, **params)
    if k == "density":
        return lambda gbar, rho, R=None: xi_density_only(gbar, rho, R, **params)
    if k in ("density-plateau", "dg-plateau"):
        return lambda gbar, rho, R=None: xi_density_plateau(gbar, rho, R, **params)
    if k == "hybrid":
        return lambda gbar, rho, R=None: xi_hybrid(gbar, rho, R, **params)
    raise ValueError(f"Unknown gate kind: {kind}")
