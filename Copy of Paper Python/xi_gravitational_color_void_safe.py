#!/usr/bin/env python3
"""
xi_gravitational_color_void_safe.py - JAX implementation of a void-safe
exponential screening function for gravitational color.

Form:
  xi(ρ) = 1 + λ_g * exp(- (ρ/ρ_c)^γ)

Properties:
- xi -> 1 + λ_g as ρ -> 0 (max enhancement in low-density regions)
- xi -> 1 as ρ >> ρ_c (Solar System/lab screening)
- Numerically stable for extreme ρ, with caps in caller as needed
"""
import jax
import jax.numpy as jnp

@jax.jit
def xi_gravitational_color_void_safe(rho, rho_c, gamma, lambda_g):
    rho_arr = jnp.atleast_1d(rho)
    rho_c_safe = jnp.maximum(rho_c, 1e-30)
    ratio = jnp.maximum(rho_arr, 0.0) / rho_c_safe
    exp_arg = -jnp.power(ratio, gamma)
    xi = 1.0 + lambda_g * jnp.exp(exp_arg)
    # Ensure numerical safety
    xi = jnp.where(jnp.isfinite(xi), xi, 1.0)
    return xi

