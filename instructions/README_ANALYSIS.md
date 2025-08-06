# Analysis Module Documentation

This directory contains scripts for analyzing results, extracting insights, and creating visualizations from DDMM runs.

## Primary Analysis Scripts

### 1. **analyze_results.py** - Main Results Analysis
- **Purpose**: Comprehensive analysis of Dynesty/MCMC outputs
- **Key Features**:
  - Parameter estimation and uncertainties
  - Correlation analysis
  - Model comparison statistics
  - Goodness-of-fit metrics
- **Outputs**:
  - Corner plots
  - Trace plots
  - Summary statistics tables
  - LaTeX-formatted results
- **Differences**: Most comprehensive analysis tool vs. specialized scripts

### 2. **analyze_gaia_distribution.py** - Gaia Data Analysis
- **Purpose**: Analyzes stellar density distributions from Gaia
- **Unique Analyses**:
  - 3D density reconstruction
  - Spiral arm tracing
  - Local bubble structure
  - Velocity dispersion maps
- **Products**:
  - Density slice plots
  - Statistical summaries
  - Comparison with models
- **Differences**: Focuses on stellar data vs. cosmological data

### 3. **analyze_gr_results.py** - GR Comparison Analysis
- **Purpose**: Compares DDMM results with General Relativity baseline
- **Key Metrics**:
  - Bayes factors
  - Chi-squared improvements
  - Residual patterns
  - Parameter shifts
- **Outputs**:
  - Comparison plots
  - Statistical significance tests
  - Systematic deviation maps
- **When to Use**: Model comparison and validation

### 4. **analyse_checkpoints.py** - Checkpoint Analysis
- **Purpose**: Analyzes saved checkpoints from long runs
- **Features**:
  - Convergence diagnostics over time
  - Live point evolution
  - Evidence accumulation
  - Parameter space exploration
- **Use Case**: Debugging convergence issues
- **Differences**: Time-series analysis vs. final results

## Visualization Scripts

### 5. **visualization.py** - Main Visualization Module
- **Purpose**: Creates publication-quality plots
- **Plot Types**:
  - Hubble diagrams
  - Rotation curves
  - Density maps
  - Likelihood contours
- **Customization**:
  - Multiple color schemes
  - Journal-specific formatting
  - Interactive HTML outputs
- **Differences**: Full-featured vs. simple plotting

### 6. **plot_gr_simple.py** - Simple GR Comparison Plots
- **Purpose**: Quick visualization of GR vs DDMM
- **Plots**:
  - Distance modulus residuals
  - H(z) evolution
  - Growth factor
  - Simple 2-panel comparisons
- **Use**: Rapid visual inspection
- **Differences**: Quick plots vs. publication quality

## Data Extraction Scripts

### 7. **extract_results.py** - General Result Extraction
- **Purpose**: Extracts key results from chain files
- **Extracts**:
  - Best-fit parameters
  - Credible intervals
  - Evidence values
  - Derived parameters
- **Output Formats**: JSON, CSV, HDF5
- **Use**: General purpose extraction

### 8. **extract_tier_1.py** - Tier-1 Results Extraction
- **Purpose**: Extracts results from two-tier analysis
- **Specific to**:
  - First tier exploration results
  - Parameter space boundaries
  - Region of interest identification
- **Feeds Into**: Tier-2 analysis setup
- **Differences**: Tier-specific vs. general extraction

### 9. **extract_gr_baseline_results.py** - GR Baseline Extraction
- **Purpose**: Standardized extraction of GR-only fits
- **Extracts**:
  - Cosmological parameters (H0, Ωm, etc.)
  - Nuisance parameters
  - Systematic corrections
- **Format**: Compatible with comparison tools
- **Differences**: GR-specific formatting

## Report Generation

### 10. **gr_final_report.py** - Comprehensive GR Comparison Report
- **Purpose**: Generates full comparison report
- **Sections**:
  - Executive summary
  - Detailed statistics
  - All plots and tables
  - Appendices with diagnostics
- **Output**: PDF report via LaTeX
- **Use**: Final publication-ready reports

### 11. **create_json_file_ddmm_results.py** - JSON Results Compiler
- **Purpose**: Creates machine-readable result files
- **Contents**:
  - All parameter estimates
  - Covariance matrices
  - Metadata and configuration
  - Quality flags
- **Use**: Input for other codes, web APIs
- **Differences**: Machine-readable vs. human-readable

### 12. **inspect_results.py** - Interactive Result Inspector
- **Purpose**: Interactive exploration of results
- **Features**:
  - Command-line interface
  - Quick queries
  - Data filtering
  - Simple calculations
- **Use**: Debugging and exploration
- **Differences**: Interactive vs. batch processing
