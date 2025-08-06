# Core Module Documentation

This directory contains the fundamental physics engines and data handling infrastructure for the DDMM.

## Files Overview

### 1. density_metric2.py - Main JAX Physics Engine
- Core DDMM physics using JAX for GPU/CPU acceleration
- Functions: v_total_kms(), v_baryon_total_newtonian_kms()
- XI_FUNCTION_MAP: Different ξ(ρ) functional forms
- Used by most runners and analysis scripts

### 2. density_metric_cupy.py - CuPy GPU Version
- Alternative GPU implementation using CuPy
- Direct CUDA kernel support
- Functions: v_total_kms_cupy(), v_baryon_comprehensive_kms_cupy()
- Used by GPU-specific runners

### 3. density_contrast_model.py - Density Contrast
- Density field reconstruction
- Contrast metric calculations
- Gaia stellar density integration

### 4. enhanced_light_propagation.py - Light Ray Tracing
- Geodesic integration in modified spacetime
- Redshift calculations with DDMM corrections
- Luminosity distance modifications

### 5. data_io.py - Critical Data Infrastructure
- Central data loading hub
- Functions: load_gaia(), load_all_sky_gaia_slices()
- Used by 19+ other files across project
