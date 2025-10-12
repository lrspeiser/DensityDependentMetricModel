# Experimental variants for density-gated vs acceleration-gated analysis
# This module defines a density-gated xi(ρ) with a finite plateau, mirroring the
# role of Dmax in the acceleration (RAR) gate. Units for rho and rho_c are cgs
# (g cm^-3). The function works with numpy or jax.numpy if available.

try:
    import jax.numpy as jnp  # type: ignore
except Exception:  # pragma: no cover
    import numpy as jnp  # type: ignore


def xi_density_plateau(
    rho,
    rho_c=1e-25,
    gamma=1.0,
    xi_max=50.0,
    xi_min=1.0,
):
    """
    Density-gated boost function xi(ρ) in [xi_min, xi_max].

    Parameters
    ----------
    rho : array_like or scalar
        Local baryon density (g cm^-3). Safe for scalars or arrays.
    rho_c : float
        Characteristic density scale (g cm^-3) where gating transitions.
    gamma : float
        Slope/steepness parameter for the transition.
    xi_max : float
        Low-density plateau value (deep-gate limit).
    xi_min : float
        High-density limit (typically 1.0 to recover Newton/GR).

    Returns
    -------
    xi : same shape as rho
        The multiplicative boost factor applied to the baryonic field.

    Notes
    -----
    xi(rho) = xi_min + (xi_max - xi_min) * exp(-(rho/rho_c)^gamma)
    Monotone decreasing with rho; xi(rho->∞)→xi_min; xi(rho->0)→xi_max.
    """
    rho = jnp.asarray(rho)
    # Avoid log/exp issues – clamp to tiny positive value respecting dtype
    tiny = getattr(rho, "dtype", None)
    tiny = jnp.finfo(rho.dtype).tiny if tiny is not None else 1e-300
    rho = jnp.maximum(rho, tiny)
    boost = jnp.exp(-jnp.power(rho / rho_c, gamma))
    return xi_min + (xi_max - xi_min) * boost
