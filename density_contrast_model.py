#!/usr/bin/env python3
"""
density_contrast_model.py - Testing density CONTRAST as the driver of modified gravity.

Core hypothesis: Gravity modification depends on density gradients/contrasts,
not absolute density. This explains why:
- Solar system has normal gravity (uniform density)
- Galaxy edges show modification (huge density contrast)
- Bulge/disk show less effect (gradual density change)
"""

import cupy as cp
import numpy as np

# Configuration
DEFAULT_DTYPE = cp.float32

def to_cupy_array(arr):
    """Convert to CuPy array."""
    if isinstance(arr, cp.ndarray):
        return arr
    return cp.asarray(arr, dtype=DEFAULT_DTYPE)

def to_numpy_array(arr):
    """Convert to numpy array."""
    if isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return np.asarray(arr)

# ============================================================================
# BARYONIC PHYSICS (Minimal - just what we need)
# ============================================================================

def v_baryon_simple_kms(R_kpc, M_disk, R_disk, M_bulge, R_bulge):
    """Simple baryonic velocity for Milky Way."""
    R = cp.asarray(R_kpc, dtype=DEFAULT_DTYPE)
    
    # Disk contribution (exponential)
    v_disk_sq = (4.302e-3 * M_disk / R) * (1.0 - cp.exp(-R/R_disk) * (1.0 + R/R_disk))
    
    # Bulge contribution (Hernquist)
    v_bulge_sq = (4.302e-3 * M_bulge / R) * (R / (R + R_bulge))**2
    
    v_total_sq = v_disk_sq + v_bulge_sq
    return cp.sqrt(cp.maximum(v_total_sq, 0.0))

def density_profile(R_kpc, M_disk, R_disk, hz_disk, M_bulge, R_bulge):
    """Simple density profile for contrast calculation."""
    R = cp.asarray(R_kpc, dtype=DEFAULT_DTYPE)
    
    # Disk density (exponential)
    Sigma0_disk = M_disk / (2.0 * cp.pi * R_disk**2)
    rho_disk = (Sigma0_disk / (2.0 * hz_disk)) * cp.exp(-R / R_disk)
    
    # Bulge density (Hernquist)
    R_safe = cp.maximum(R, 1e-6)
    rho_bulge = (M_bulge / (2.0 * cp.pi)) * (R_bulge / (R_safe * (R_safe + R_bulge)**3))
    
    return rho_disk + rho_bulge

# ============================================================================
# DENSITY CONTRAST XI FUNCTIONS - The Core Innovation
# ============================================================================

@cp.fuse()
def xi_density_contrast(R_kpc, rho_local, params):
    """
    Density CONTRAST based enhancement - the key insight.
    
    Enhancement depends on:
    1. Local density gradient (change rate)
    2. Density contrast ratio (high/low)
    3. Distance scales where contrasts occur
    
    NO enhancement in uniform regions (Cassini constraint satisfied).
    """
    R = cp.asarray(R_kpc, dtype=DEFAULT_DTYPE)
    rho = cp.asarray(rho_local, dtype=DEFAULT_DTYPE)
    
    # Parameters
    gradient_scale = params.get('gradient_scale_kpc', 1.0)  # Scale for gradient calculation
    contrast_threshold = params.get('contrast_threshold', 10.0)  # Min contrast for activation
    A_max = params.get('A_contrast', 5.0)  # Maximum enhancement
    transition_width = params.get('transition_width', 2.0)  # Width of transition zones
    
    # Calculate density gradient (approximate using finite difference)
    dR = gradient_scale
    R_plus = R + dR
    R_minus = cp.maximum(R - dR, 0.1)
    
    # Get densities at nearby points (would be calculated from model)
    # For now, use power law approximation
    rho_plus = rho * (R / R_plus)**2  # Approximate
    rho_minus = rho * (R / R_minus)**2
    
    # Logarithmic gradient (more stable than linear)
    grad_log_rho = cp.abs(cp.log10(rho_plus + 1e-10) - cp.log10(rho_minus + 1e-10)) / (2 * dR)
    
    # Density contrast ratio (local vs background)
    # Define background as density at 2x the radius
    rho_background = rho * (R / (2*R))**2  # Approximate background
    contrast_ratio = rho / (rho_background + 1e-10)
    
    # Enhancement based on BOTH gradient AND contrast
    # Key insight: Need both high gradient AND high contrast
    
    # Gradient factor: peaks where density changes rapidly
    gradient_factor = 1.0 - cp.exp(-grad_log_rho**2)
    
    # Contrast factor: activates above threshold
    contrast_factor = cp.tanh((cp.log10(contrast_ratio) - cp.log10(contrast_threshold)) / 0.5)
    contrast_factor = cp.maximum(contrast_factor, 0.0)
    
    # Combined enhancement
    xi = 1.0 + A_max * gradient_factor * contrast_factor
    
    # CRITICAL: Suppress enhancement in high-density uniform regions
    # This is key for Cassini constraint
    high_density_suppression = cp.exp(-rho / 1e8)  # Suppress above 10^8 M_sun/kpc^3
    xi = 1.0 + (xi - 1.0) * high_density_suppression
    
    # Also suppress at very small radii (nuclear region)
    nuclear_suppression = cp.tanh(R / 1.0)  # Smooth turn-on beyond 1 kpc
    xi = 1.0 + (xi - 1.0) * nuclear_suppression
    
    return xi

@cp.fuse()
def xi_gradient_bands(R_kpc, rho_local, params):
    """
    Banded enhancement based on logarithmic density bands.
    Simpler version using the Richter-scale concept.
    """
    R = cp.asarray(R_kpc, dtype=DEFAULT_DTYPE)
    rho = cp.asarray(rho_local, dtype=DEFAULT_DTYPE)
    
    # Parameters for band transitions
    band_width = params.get('band_width_dex', 1.0)  # Width in decades
    A_per_band = params.get('A_per_band', 2.0)  # Enhancement per band
    rho_ref = params.get('rho_ref', 1e6)  # Reference density
    
    # Calculate which logarithmic band we're in
    log_rho = cp.log10(rho + 1e-10)
    log_ref = cp.log10(rho_ref)
    
    # Number of bands below reference
    n_bands = cp.floor((log_ref - log_rho) / band_width)
    n_bands = cp.maximum(n_bands, 0)  # Only enhance for lower densities
    
    # Smooth transition between bands
    band_fraction = ((log_ref - log_rho) % band_width) / band_width
    smooth_bands = n_bands + cp.tanh(band_fraction * 3 - 1.5) * 0.5
    
    # Enhancement increases with each band
    xi = 1.0 + A_per_band * smooth_bands
    
    # Critical: No enhancement above reference density (Cassini)
    xi = cp.where(rho > rho_ref, 1.0, xi)
    
    # Suppress near center
    xi = cp.where(R < 1.0, 1.0, xi)
    
    return cp.minimum(xi, 20.0)  # Cap maximum enhancement

@cp.fuse()
def xi_boundary_detection(R_kpc, rho_local, params):
    """
    Enhancement at detected density boundaries.
    Most direct implementation of the boundary concept.
    """
    R = cp.asarray(R_kpc, dtype=DEFAULT_DTYPE)
    rho = cp.asarray(rho_local, dtype=DEFAULT_DTYPE)
    
    # Boundary detection parameters
    boundary_ratio = params.get('boundary_ratio', 100.0)  # Density drop for boundary
    boundary_width = params.get('boundary_width_kpc', 3.0)
    A_boundary = params.get('A_boundary', 8.0)
    
    # Known boundary locations for Milky Way (approximate)
    # These could be determined dynamically from density profile
    boundaries_kpc = cp.array([3.0, 10.0, 25.0], dtype=DEFAULT_DTYPE)  # Bulge edge, disk edge, halo edge
    
    xi = cp.ones_like(R)
    
    # Add enhancement at each boundary
    for boundary_r in boundaries_kpc:
        distance_from_boundary = cp.abs(R - boundary_r)
        boundary_enhancement = A_boundary * cp.exp(-(distance_from_boundary**2) / (2 * boundary_width**2))
        
        # Only enhance if we're in the low-density side
        is_outside = R > boundary_r
        xi = xi + boundary_enhancement * is_outside
    
    # Suppress in uniform high-density regions
    xi = cp.where(rho > 1e7, 1.0, xi)
    
    return xi

# ============================================================================
# MAIN VELOCITY FUNCTION
# ============================================================================

def v_total_contrast(R_kpc, params, contrast_type='gradient'):
    """
    Calculate total velocity with density contrast-based enhancement.
    
    Parameters:
    -----------
    R_kpc : array
        Galactic radius in kpc
    params : dict
        Model parameters including masses and contrast parameters
    contrast_type : str
        'gradient' - continuous gradient-based
        'bands' - logarithmic bands
        'boundaries' - discrete boundaries
    """
    R = to_cupy_array(R_kpc)
    
    # Extract baryonic parameters
    M_disk = params.get('M_disk_solar', 5e10)
    R_disk = params.get('R_disk_kpc', 3.0)
    hz_disk = params.get('hz_disk_kpc', 0.3)
    M_bulge = params.get('M_bulge_solar', 1e10)
    R_bulge = params.get('R_bulge_kpc', 0.5)
    
    # Calculate Newtonian velocity
    v_newton = v_baryon_simple_kms(R, M_disk, R_disk, M_bulge, R_bulge)
    
    # Calculate density for contrast calculation
    rho = density_profile(R, M_disk, R_disk, hz_disk, M_bulge, R_bulge)
    
    # Calculate enhancement based on density contrast
    if contrast_type == 'gradient':
        xi = xi_density_contrast(R, rho, params)
    elif contrast_type == 'bands':
        xi = xi_gradient_bands(R, rho, params)
    elif contrast_type == 'boundaries':
        xi = xi_boundary_detection(R, rho, params)
    else:
        xi = cp.ones_like(R)  # No enhancement (GR)
    
    # Apply enhancement
    v_total = v_newton * cp.sqrt(xi)
    
    # Safety checks
    v_total = cp.nan_to_num(v_total, nan=0.0, posinf=0.0, neginf=0.0)
    
    return v_total

# ============================================================================
# CASSINI CONSTRAINT CHECK
# ============================================================================

def check_cassini_constraint(params, contrast_type='gradient'):
    """Check if model satisfies Cassini constraint at Solar position."""
    R_sun = cp.array([8.5], dtype=DEFAULT_DTYPE)  # Sun at 8.5 kpc
    
    # Solar neighborhood parameters
    M_disk = params.get('M_disk_solar', 5e10)
    R_disk = params.get('R_disk_kpc', 3.0) 
    hz_disk = params.get('hz_disk_kpc', 0.3)
    M_bulge = params.get('M_bulge_solar', 1e10)
    R_bulge = params.get('R_bulge_kpc', 0.5)
    
    # Get density at Sun
    rho_sun = density_profile(R_sun, M_disk, R_disk, hz_disk, M_bulge, R_bulge)
    
    # Calculate xi at Sun
    if contrast_type == 'gradient':
        xi_sun = xi_density_contrast(R_sun, rho_sun, params)
    elif contrast_type == 'bands':
        xi_sun = xi_gradient_bands(R_sun, rho_sun, params)
    elif contrast_type == 'boundaries':
        xi_sun = xi_boundary_detection(R_sun, rho_sun, params)
    else:
        xi_sun = cp.array([1.0])
    
    gamma_minus_one = float(xi_sun[0] - 1.0)
    cassini_limit = 2.3e-5
    
    return {
        'passes': abs(gamma_minus_one) < cassini_limit,
        'xi_sun': float(xi_sun[0]),
        'gamma_minus_one': gamma_minus_one,
        'violation_factor': abs(gamma_minus_one) / cassini_limit if cassini_limit > 0 else np.inf
    }
    