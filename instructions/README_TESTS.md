# Tests Module Documentation

This directory contains comprehensive test suites for validating the DDMM implementation and ensuring correctness.

## Core Model Tests

### 1. **test_contrast_model.py** - Density Contrast Model Tests
- **Tests**: density_contrast_model.py functionality
- **Coverage**: Density reconstruction, contrast metrics, Gaia integration
- **Special Tests**: Compares against analytical solutions, tests convergence
- **Differences**: Focuses on spatial density variations vs. velocity tests

### 2. **test_comprehensive_model.py** - CuPy Implementation Tests
- **Tests**: density_metric_cupy.py GPU implementation
- **Validation Against**: CPU reference, analytical cases, precision limits
- **Performance Tests**: GPU memory usage, kernel execution times
- **Key Difference**: GPU-specific optimizations and CUDA compatibility

### 3. **test_improvements.py** - JAX Implementation Tests
- **Tests**: density_metric2.py improvements and optimizations
- **Focus Areas**: JIT compilation, automatic differentiation, vectorization
- **Unique Tests**: JAX-specific features like vmap and grad

### 4. **test_xi_models.py** - ξ(ρ) Function Tests
- **Tests**: All XI_FUNCTION_MAP implementations
- **Models Tested**: Power law (ξ ∝ ρ^α), Exponential (ξ ∝ exp(βρ)), Logarithmic
- **Validation**: Continuity, smoothness, asymptotic behavior, parameter sensitivity
- **Differences**: Mathematical function testing vs. physics testing

## Cosmological Tests

### 5. **test_ddmm_cosmological_redshift.py** - Redshift Evolution Tests
- **Purpose**: Tests DDMM effects on cosmological redshift
- **Test Cases**: Hubble diagram modifications, distance-redshift relations
- **Datasets**: Uses Pantheon SNe for validation
- **Key Tests**: z < 2 regime where DDMM effects are strongest

### 6. **test_ddmm_cosmological_redshift_with_voids.py** - Void Effects
- **Purpose**: Tests DDMM in low-density regions
- **Unique Tests**: Void lensing signals, ISW modifications, void-galaxy correlations
- **Validation**: Against N-body simulations
- **Differences**: Tests underdense vs. overdense regions

### 7. **ddmm_test.py** - Integration Test Suite
- **Purpose**: End-to-end testing of full pipeline
- **Tests**: Data loading → Processing → Analysis → Results
- **Regression Tests**: Ensures updates don't break existing functionality
- **Differences**: Full pipeline vs. unit tests

## System and Infrastructure Tests

### 8. **test_gpu_detection.py** - GPU Availability Tests
- **Purpose**: Validates GPU setup and CUDA installation
- **Checks**: CUDA version, GPU memory, multi-GPU config, CuPy installation
- **Output**: GPU capability report
- **Use**: Run before GPU-accelerated tests

### 9. **test_resource_monitor.py** - Resource Monitoring Tests
- **Tests**: resource_monitor.py functionality
- **Validation**: CPU usage, memory leaks, GPU utilization, I/O tracking
- **Differences**: Infrastructure vs. physics testing

### 10. **test_new_run_system.py** - Runner Infrastructure Tests
- **Tests**: New features in run system
- **Coverage**: Checkpoint saving/loading, parallel execution, error recovery
- **Focus**: Runner robustness and reliability

### 11. **test_split_regions.py** - Regional Analysis Tests
- **Tests**: run_dynesty_split_regions.py functionality
- **Validation**: Region boundaries, parameter continuity, statistical consistency
- **Unique**: Tests spatial subdivision strategies

### 12. **test_fixes.py** - Bug Fix Validation
- **Purpose**: Regression tests for fixed bugs
- **Contains**: Specific cases that triggered bugs, edge cases
- **Prevents**: Reintroduction of fixed issues

## Test Data Directories

### **test_debug/** - Debugging Test Cases
- Contains minimal test cases for debugging
- Simplified data for quick iteration
- Used for: Isolating specific issues

### **test_grav_color_debug/** - Gravitational Field Tests
- Test data for gravitational field calculations
- Color-coded visualization data
- Focus: Visual debugging of field calculations

### **test_grav_color_stronger/** - Enhanced Gravity Tests
- Tests for strong-field regime
- Non-linear effect validation
- Differences: Pushes numerical limits

### **gaussian_test_params/** - Gaussian Mock Data
- Synthetic Gaussian datasets
- Known statistical properties
- Advantages: Exact analytical solutions available

### **quick_test/** - Rapid Testing Suite
- Subset of full tests for quick validation
- Runs in < 1 minute
- Use: Pre-commit checks, rapid development

## Running Tests

### Full Test Suite
```bash
python -m pytest tests/ -v
```

### Specific Test Categories
```bash
# Core model tests only
python -m pytest tests/test_contrast_model.py tests/test_comprehensive_model.py -v

# Cosmological tests
python -m pytest tests/test_ddmm_cosmological*.py -v

# Quick smoke tests
python -m pytest tests/quick_test/ -v
```

### GPU Tests (requires CUDA)
```bash
python tests/test_gpu_detection.py
python tests/test_comprehensive_model.py --gpu
```

## Test Coverage Goals

- **Unit Tests**: >90% code coverage
- **Integration Tests**: All major pipelines
- **Regression Tests**: All fixed bugs
- **Performance Tests**: No degradation > 10%
- **Numerical Tests**: Machine precision where applicable
