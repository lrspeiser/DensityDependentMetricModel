"""Planetary ephemeris surrogates (code-based, reproducible)
Provide a simple two-body surrogate to estimate perihelion precession per orbit
under a small radial modulation of the Newtonian field: g(r) = GM (1+ε(r)) / r^2.

- ε(r) comes from the same xi function used elsewhere: ε ≡ ξ − 1 with ξ = 0.5+sqrt(0.25+a0_eff/g_N).
- This is a numerical surrogate (2D leapfrog) to give an order-of-magnitude
  precession consistent with the gating profile. It is not a production ephemeris.
- No web/API usage. All inputs are local and transparent.
"""
from __future__ import annotations
from typing import Dict, Any, Callable, Tuple
import math
import numpy as np

# Constants
G = 6.67430e-11
M_SUN = 1.98847e30
AU_M = 1.495978707e11

# Gate: same closed-form used in scripts/next_steps_from_run.py (rar-plateau)
ACC_M_S2_PER_KMS2_PER_KPC = 3.240779289e-14


def xi_rar_plateau_scalar(g_bar: float, *, a0_m_s2: float, zeta_env: float = 0.0,
                          W: float = 1.0, D_max: float | None = None) -> float:
    a0_eff = float(a0_m_s2) * (1.0 + float(zeta_env) * float(W))
    D = 0.5 + math.sqrt(0.25 + max(a0_eff, 0.0) / max(g_bar, 1e-30))
    if D_max is not None and float(D_max) > 1.0:
        D = min(D, float(D_max))
    return float(D)


def epsilon_from_xi(g_bar: float, *, a0_m_s2: float, zeta_env: float = 0.0,
                    W: float = 1.0, D_max: float | None = None) -> float:
    return xi_rar_plateau_scalar(g_bar, a0_m_s2=a0_m_s2, zeta_env=zeta_env, W=W, D_max=D_max) - 1.0


def make_eps_fn(rar_params: Dict[str, float]) -> Callable[[float], float]:
    a0 = float(rar_params.get('a0_m_s2', 1.2e-10))
    zeta = float(rar_params.get('zeta_env', 0.0))
    D_max = rar_params.get('D_max', None)
    # In Solar limit, use W≈1 (local constancy assumption)
    def eps(r_m: float) -> float:
        gN = G * M_SUN / max(r_m**2, 1.0)
        return epsilon_from_xi(gN, a0_m_s2=a0, zeta_env=zeta, W=1.0, D_max=D_max)
    return eps


def kepler_initial_conditions(a_AU: float, e: float) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return planar (x,v) at perihelion for a Kepler ellipse around the Sun.
    Ignores ε in the initial speed (sufficient at ≪10^-5 levels).
    """
    a = float(a_AU) * AU_M
    e = float(e)
    r_peri = a * (1.0 - e)
    mu = G * M_SUN
    v_peri = math.sqrt(mu * (1.0 + e) / max(r_peri, 1.0) / (1.0 - e))
    x0 = np.array([r_peri, 0.0], dtype=float)
    v0 = np.array([0.0, v_peri], dtype=float)
    T = 2.0 * math.pi * math.sqrt(a**3 / mu)
    return x0, v0, T


def leapfrog_precession(eps_fn: Callable[[float], float], a_AU: float, e: float,
                        n_orbits: int = 5, n_steps_per_orbit: int = 5000) -> Dict[str, Any]:
    x, v, T = kepler_initial_conditions(a_AU, e)
    dt = T / float(max(n_steps_per_orbit, 100))
    peri_angles = []
    prev_r = np.linalg.norm(x)
    for step in range(int(n_orbits * n_steps_per_orbit)):
        # State at the beginning of the step
        x_curr = x.copy()
        r = np.linalg.norm(x_curr)
        angle_curr = math.atan2(x_curr[1], x_curr[0])
        # Kick
        eps = float(eps_fn(r))
        acc = -(G * M_SUN * (1.0 + eps) / max(r**3, 1.0)) * x_curr
        v_half = v + 0.5 * dt * acc
        # Drift
        x = x_curr + dt * v_half
        # Kick
        r_new = np.linalg.norm(x)
        eps = float(eps_fn(r_new))
        acc_new = -(G * M_SUN * (1.0 + eps) / max(r_new**3, 1.0)) * x
        v = v_half + 0.5 * dt * acc_new
        # Perihelion detection: local minimum in r occurs at x_curr
        if step > 2 and r_new > r and r < prev_r:
            peri_angles.append(angle_curr)
        prev_r = r
        r = r_new
    # Compute precession per orbit from successive perihelia as change in periapsis angle
    dphis = []
    for i in range(1, len(peri_angles)):
        dphi = peri_angles[i] - peri_angles[i-1]
        # wrap to [-π, π]
        while dphi <= -math.pi:
            dphi += 2.0 * math.pi
        while dphi > math.pi:
            dphi -= 2.0 * math.pi
        dphis.append(dphi)
    if len(dphis) == 0:
        return {"status": "insufficient_perihelia"}
    mean_precession_rad = float(np.mean(dphis))
    arcsec_per_orbit = mean_precession_rad * (180.0/math.pi) * 3600.0
    return {
        "status": "ok",
        "a_AU": float(a_AU),
        "e": float(e),
        "arcsec_per_orbit": arcsec_per_orbit,
        "n_orbits": int(n_orbits),
        "n_steps_per_orbit": int(n_steps_per_orbit),
    }


def run_surrogate_suite(rar_params: Dict[str, float]) -> Dict[str, Any]:
    eps_fn = make_eps_fn(rar_params)
    cases = {
        "Mercury": {"a_AU": 0.387, "e": 0.206},
        "Earth":   {"a_AU": 1.000, "e": 0.017},
        "Saturn":  {"a_AU": 9.582, "e": 0.056},
    }
    out = {}
    for name, cfg in cases.items():
        res = leapfrog_precession(eps_fn, cfg["a_AU"], cfg["e"], n_orbits=5, n_steps_per_orbit=4000)
        out[name] = res
    return out


def main() -> None:
    # Example params matching paper preset; callers can pass actual run params
    rar = {"a0_m_s2": 3.5e-11, "zeta_env": 0.0, "D_max": 50.0}
    res = run_surrogate_suite(rar)
    import json, os
    os.makedirs("results/nature_readiness/solar_system", exist_ok=True)
    (open("results/nature_readiness/solar_system/perihelion_surrogate.json","w")).write(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

