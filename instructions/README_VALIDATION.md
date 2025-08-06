# Validation Module Documentation

This directory contains validation tools and scripts for verifying the DDMM implementation against known standards and observational data.

## Primary Validation Scripts

### 1. **validate_ddmm.py** - Comprehensive DDMM Validation
- **Purpose**: Full validation suite for DDMM implementation
- **Validation Tests**: Newtonian limit (ξ → 0), Solar system constraints, Galaxy rotation curves
- **Reference Data**: Analytical solutions, N-body simulations, observational benchmarks
- **Output**: Validation report with pass/fail criteria

### 2. **validate_gr_simple.py** - GR Limit Validation
- **Purpose**: Verifies code reduces to GR when ξ = 0
- **Tests**: Schwarzschild metric recovery, FLRW cosmology, weak field approximation
- **Tolerance**: Numerical precision (typically 1e-10)
- **Differences**: Tests limiting behavior vs. full validation

### 3. **cassini.py** - Cassini Spacecraft Constraint
- **Purpose**: Tests against Cassini radio tracking data
- **Constraint**: |γ - 1| < 2.3 × 10^-5
- **Implementation**: Shapiro delay calculation, light deflection, PPN parameter extraction
- **Significance**: Strongest solar system test of GR
- **Differences**: Solar system vs. cosmological scales

## Data Coverage and Quality Checks

### 4. **check.py** - General Consistency Checks
- **Purpose**: Basic sanity checks on results
- **Checks**: Parameter bounds, physical constraints, energy conditions, causality
- **Use**: Quick validation during development
- **Differences**: Basic checks vs. comprehensive validation

### 5. **check_data_coverage.py** - Data Completeness Analysis
- **Purpose**: Verifies data coverage for analysis
- **Checks**: Sky coverage maps, redshift distributions, sample completeness
- **Output**: Coverage statistics and warning flags
- **Differences**: Data quality vs. physics validation

### 6. **check_resources.py** - Computational Resource Validation
- **Purpose**: Ensures adequate computational resources
- **Validates**: Memory requirements, CPU/GPU availability, disk space
- **Prevents**: Runtime failures due to resource limits
- **Differences**: System vs. science validation

### 7. **check_xi_contribution.py** - ξ(ρ) Effect Analysis
- **Purpose**: Quantifies DDMM contributions to observables
- **Analyzes**: Fractional velocity corrections, metric perturbation amplitudes
- **Output**: Contribution maps and significance regions
- **Differences**: DDMM-specific vs. general tests

## Supporting Files and Data

### **validation_results/** - Validation Output Directory
- Contains validation reports
- Benchmark comparison plots
- Test case results
- Historical validation records

### **cassini_passing_formulas.json** - Cassini Test Data
- Radio tracking measurements
- Ephemeris data
- Expected signals
- Systematic corrections

## Validation Workflow

### Standard Validation Pipeline
```bash
# 1. Run comprehensive validation
python validate_ddmm.py --full --output validation_report.pdf

# 2. Check GR limit
python validate_gr_simple.py --tolerance 1e-10

# 3. Solar system constraints
python cassini.py --data cassini_passing_formulas.json

# 4. Data quality checks
python check_data_coverage.py --dataset all
```

### Quick Validation
```bash
# Basic consistency check
python check.py results.json

# Resource availability
python check_resources.py --estimate-runtime

# ξ contribution analysis
python check_xi_contribution.py --parameter-file params.json
```

## Validation Criteria

### Physics Tests
| Test | Criterion | Status |
|------|-----------|--------|
| Newtonian limit | ξ → 0 recovers Newton | ✓ |
| GR limit | Matches GR to 1e-10 | ✓ |
| Cassini bound | γ within constraints | ✓ |
| Energy conditions | No violations | ✓ |
| Causality | No superluminal propagation | ✓ |

### Numerical Tests
| Test | Criterion | Status |
|------|-----------|--------|
| Convergence | Results stable with resolution | ✓ |
| Precision | Machine precision where applicable | ✓ |
| Symmetries | Conserved quantities preserved | ✓ |
| Boundaries | Correct boundary conditions | ✓ |

## Performance Benchmarks

Typical validation times:
- Quick check: < 1 minute
- Standard validation: 5-10 minutes
- Full validation suite: 30-60 minutes
- Monte Carlo validation: 2-4 hours
