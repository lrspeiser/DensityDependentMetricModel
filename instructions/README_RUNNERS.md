# Runners Module Documentation

This directory contains execution scripts that orchestrate model runs, parameter estimation, and Bayesian inference.

## Main Runner Files

### 1. **run_dynesty.py** - Primary CPU-Based Bayesian Inference
- **Purpose**: Main production runner using Dynesty nested sampling
- **Key Features**:
  - Multi-threaded CPU execution
  - Real-time monitoring dashboard integration
  - Checkpoint saving for long runs
  - Automatic convergence checking
- **Usage**: `python run_dynesty.py --nlive 500 --threads 8`
- **Output**: Posterior samples, evidence estimates, chain diagnostics

### 2. **run_dynesty_cupy.py** - GPU-Accelerated Bayesian Inference
- **Purpose**: GPU version of Dynesty runner using CuPy
- **Key Differences from run_dynesty.py**:
  - 10-50x faster for large datasets
  - Requires NVIDIA GPU with CUDA
  - Limited to single-GPU execution
  - Different memory management strategy
- **Usage**: `python run_dynesty_cupy.py --gpu 0 --nlive 1000`
- **Best For**: Quick exploration runs, large parameter spaces

### 3. **run_dynesty_split_regions.py** - Regional Analysis Runner
- **Purpose**: Splits sky/parameter space into regions for parallel analysis
- **Unique Features**:
  - Divides data by sky regions or redshift bins
  - Runs independent chains per region
  - Combines results for global inference
- **Applications**: Testing spatial variations in DDMM parameters

### 4. **main2.py** - Alternative MCMC Pipeline
- **Purpose**: Emcee-based MCMC sampling (alternative to Dynesty)
- **Key Differences**:
  - Uses ensemble sampling instead of nested sampling
  - Better for posterior exploration than evidence calculation
  - Different convergence diagnostics (Gelman-Rubin, autocorrelation)

### 5. **run_gr_baseline.py** - General Relativity Baseline
- **Purpose**: Runs standard GR model for comparison
- **Features**:
  - Same data, likelihood, but ξ(ρ) = 0 (no DDMM effects)
  - Provides baseline chi-squared and residuals
  - Used for model comparison statistics

### 6. **run_two_tier.py** - Hierarchical Analysis
- **Purpose**: Two-stage analysis with fast then detailed sampling
- **Strategy**:
  - Tier 1: Quick exploration with few live points
  - Tier 2: Refined sampling around best regions

## Monitoring Scripts

### 7. **monitor_dashboard.py** - Real-Time Run Monitoring
- **Purpose**: Web dashboard for monitoring long runs
- **Features**:
  - Live parameter trace plots
  - Convergence metrics
  - Resource usage (CPU, memory, GPU)
- **Access**: Opens browser at localhost:8050 during runs

### 8. **resource_monitor.py** - System Resource Tracking
- **Purpose**: Logs resource usage for performance optimization
- **Monitors**: CPU, memory, GPU utilization, disk I/O
- **Output**: JSON logs for analysis, performance reports

### 9. **monitor_run_dynesty.py** - Command-Line Monitor
- **Purpose**: Terminal-based monitoring for remote runs
- **Features**: ASCII progress bars, key statistics updates
- **Best For**: SSH sessions, cluster jobs

## Batch Scripts

### 10. **run_dynesty_safe.bat** - Windows Batch Launcher
- **Purpose**: Safe execution wrapper for Windows
- **Features**: Environment setup, error handling, automatic restart

### 11. **run_dynesty_safe.ps1** - PowerShell Launcher
- **Purpose**: Advanced Windows automation
- **Features**: Parameter validation, email notifications, cloud backup

## Runner Selection Guide

| Use Case | Recommended Runner | Reason |
|----------|-------------------|---------|
| Production runs | run_dynesty.py | Stable, well-tested |
| Quick exploration | run_dynesty_cupy.py | GPU acceleration |
| Evidence calculation | run_dynesty.py | Nested sampling |
| Posterior only | main2.py | Efficient MCMC |
| Spatial analysis | run_dynesty_split_regions.py | Regional parameters |
| Model comparison | run_gr_baseline.py | Baseline metrics |

## Common Parameters

All runners share common command-line arguments:
- `--nlive`: Number of live points (Dynesty)
- `--nwalkers`: Number of walkers (Emcee)
- `--threads`: CPU threads to use
- `--gpu`: GPU device ID
- `--checkpoint`: Save interval (iterations)
- `--max-iter`: Maximum iterations
- `--evidence-tol`: Evidence tolerance for convergence
