# Data Directory Documentation

This README documents all data sources, formats, and conversion scripts for the DensityDependentMetricModel project.

## Primary Testing Focus

### What We're Testing:
1. **Galaxy Rotation Curves** (SPARC: 175 galaxies) - ✅ Data Complete
   - Test if geometric enhancement explains flat rotation curves
   - Validate BTFR (Baryonic Tully-Fisher Relation)
   
2. **Cluster Lensing** (NOT dynamics) - 🔬 Using synthetic data
   - Test if different photon coupling can explain M_lens >> M_gas
   - This is where MOND fails but our model might succeed
   
3. **Milky Way Constraints** (Gaia DR3) - ✅ Data Available
   - Local validation using high-precision stellar kinematics

## Data Sources Overview

### 1. SPARC Galaxy Data (✅ COMPLETE - 175 galaxies)
- **Location**: `external_data/Rotmod_LTG/` 
- **Source**: [Stony Brook SPARC Database](https://astroweb.cwru.edu/SPARC)
- **Content**: Full dataset of 175 late-type galaxies with rotation curves
- **Status**: ✅ FULLY DOWNLOADED AND AVAILABLE
- **Key Files**:
  - `data/SPARC_Lelli2016c.mrt` - Master catalog with galaxy properties
  - `external_data/Rotmod_LTG/MasterSheet_SPARC.csv` - Converted master sheet
  - Individual galaxy files (175 total):
    - `{GALAXY}_rotmod.dat` - Rotation curve data
    - `{GALAXY}.dens` - Density profiles
    - `{GALAXY}.sfb` - Surface brightness
  - Examples: NGC0024, NGC0055, NGC0100, NGC3198, NGC2403, etc.
- **Usage**: For testing rotation curve fits and BTFR relations
- **Fetch Script**: `scripts/fetch_sparc.py` - Already executed successfully

### 2. Gaia Milky Way Data
- **Location**: `data/gaia_query_cache_*`
- **Source**: Gaia DR3 archive
- **Content**: Stellar kinematics for Milky Way rotation curve
- **Files**:
  - `gaia_query_cache_DR3_raw.csv` - Raw Gaia query results
  - `gaia_query_cache_DR3_processed_for_fit.parquet` - Processed for fitting
  - `mw_binned_velocities.csv` - Binned velocity data
- **Generation Script**: `scripts/generate_mw_triplet.py` - Creates MW data triplet

### 3. Lensing Data
- **CASTLES Lenses**: Strong gravitational lenses
  - **Conversion Script**: `scripts/convert_castles_small_to_rar_lenses.py`
  - Converts CASTLES CSV to format for lensing analysis
  - Derives stellar mass from Faber-Jackson scaling
  - Estimates Einstein radius from image separation
  - Example lenses: PG1115+080, B1608+656, Q0957+561
  
- **SLACS-style Lenses**: From `docs/lensing_targets.csv`
  - Format: `z_l,z_s,Re_kpc,log10M,theta_E_obs_arcsec`

- **Results Files**: `results/next_steps/*/lensing_*.csv`
  - Contains lensing predictions from various model runs
  - Scripts to combine: `scripts/combine_lensing_tables.py`
  - Scripts to summarize: `scripts/summarize_lensing_runs.py`

### 4. Cluster Data (Focus: LENSING ONLY)
**Note on Cluster Dynamics**: Cluster dynamics may not follow pure GR - the same "dark matter" 
needed for lensing is often invoked for dynamics too. However, cluster orbital timescales are 
very long (>Gyr) making dynamical measurements uncertain. Therefore we focus on cluster LENSING 
which provides cleaner, instantaneous mass measurements.

**Synthetic Data Generated**: `scripts/generate_cluster_profiles.py`
- Generated synthetic profiles for: ABELL_1689, ABELL_2029, A478, A1795, A2029, ABELL_0426
- Based on typical beta-model parameters from literature
- Files created in `data/` directory:
  - `{CLUSTER}_gas_profile.csv` - Gas density profiles
  - `{CLUSTER}_temperature_profile.csv` - Temperature profiles  
  - `{CLUSTER}_clumping_profile.csv` - Clumping factors
  
**Data Loader**: `data_loaders/frontier_lensing_loader.py`
- Loads Hubble Frontier Fields data for MACS0416
- Expected location: `hlsp_frontier/` directory
- Provides convergence maps, shear maps, deflection fields

### 5. External Data
- **Location**: `external_data/`
- **Contents**:
  - `Rotmod_LTG/` - SPARC rotation curves (fetched by script)
  - `gaia_sky_slices/` - Gaia sky region data  
  - `pantheon/` - Type Ia supernovae for cosmology
  - `cassini_passing_formulas.json` - Solar system constraints
  
## Data Processing Scripts

### Fetching & Conversion
1. **`scripts/fetch_sparc.py`** - Download SPARC galaxy data
   ```bash
   python scripts/fetch_sparc.py external_data/Rotmod_LTG
   ```

2. **`scripts/fetch_sparc_direct.py`** - Alternative SPARC fetcher
3. **`scripts/fetch_sparc_hirad_sb_v2.py`** - Fetch HIrad/SB profiles

4. **`scripts/convert_castles_small_to_rar_lenses.py`** - Convert CASTLES lenses
   ```bash
   python scripts/convert_castles_small_to_rar_lenses.py --in castles.csv --out lenses.csv
   ```

### Analysis & Aggregation
1. **`scripts/combine_lensing_tables.py`** - Combine multiple lensing runs
   - Takes multiple `lensing_rar_table.csv` files
   - Creates combined long-format table with run labels
   - Generates pivot tables for key predictions

2. **`scripts/summarize_lensing_runs.py`** - Summarize lensing predictions
   - Compares predictions across different model runs
   - Calculates residuals vs observations
   - Outputs summary CSV and console table

3. **`scripts/aggregate_metrics.py`** - Aggregate accuracy metrics across runs

4. **`scripts/metrics_from_combined.py`** - Extract metrics from combined tables

### Plotting & Visualization
- `scripts/plot_lensing_grid_metrics.py` - Plot lensing parameter grids
- `scripts/plot_sparc_rotation_overlay.py` - Overlay SPARC rotation curves
- `scripts/plot_mw_overlay_triplet.py` - Plot Milky Way data triplet
- `scripts/plot_comprehensive_comparison.py` - Comprehensive model comparison

## Data Formats

### CSV Column Specifications

#### Gas Profile (Clusters)
```csv
radius_kpc,density_kg_m3,mass_integrated
```

#### Temperature Profile (Clusters)  
```csv
radius_kpc,temperature_K
```

#### Lensing Table
```csv
lens_id,z_l,z_s,log10M_star,Re_kpc,n_sersic,theta_E_obs_arcsec,theta_E_GR_arcsec,theta_E_RAR_arcsec
```

#### SPARC Rotation Curves
- HIrad.dat: HI gas rotation data
- SB.dat: Surface brightness profile

## Data Quality Notes

1. **SPARC Data**: Well-validated, widely used benchmark dataset
2. **Gaia Data**: High-precision astrometry, processed for circular velocity
3. **Lensing Data**: Mix of observational (CASTLES) and model predictions
4. **Cluster Data**: Currently MISSING - need to obtain from X-ray observations

## Required Data for Lensing Analysis

To properly test cluster lensing (where MOND fails), we need:
1. **Gas density profiles** from X-ray observations
2. **Temperature profiles** from spectroscopy
3. **Clumping factor profiles** if available
4. **Known Einstein radii** from strong lensing observations

Potential sources:
- Chandra X-ray Observatory archive
- XMM-Newton archive
- Published cluster catalogs (e.g., ACCEPT, HIFLUGCS)
- Hubble Frontier Fields (partially available via frontier_lensing_loader.py)

## Running the GPU Optimizer

To run the GPU-accelerated multi-system optimizer with correct data paths:

```python
# From root-m/pde directory:
python gpu_multi_optimizer.py

# Note: The optimizer expects:
# - SPARC data in: external_data/Rotmod_LTG/
# - Gaia data in: data/gaia/
# - Cluster data in: data/
```

### Correct SPARC Data Path
The SPARC galaxies are in `external_data/Rotmod_LTG/` with files like:
- `NGC0024_rotmod.dat`
- `NGC0024.dens`
- `NGC0024.sfb`

The optimizer needs to be called with `--data-dir C:/Users/henry/Documents/GitHub/DensityDependentMetricModel/external_data`

## How to Add New Data

1. Place raw data files in appropriate directory
2. Create or modify a data loader in `data_loaders/`
3. Add conversion script to `scripts/` if needed
4. Update this README with new data source
5. Test with validation scripts in `validation/`

## Current Data Gaps

⚠️ **Critical Missing Data**:
- Cluster gas profiles (needed for lensing analysis)
- Cluster temperature profiles
- Complete Hubble Frontier Fields data
- Full SLACS lens sample

These are essential for testing the model's ability to explain the cluster lensing mass discrepancy that MOND cannot address.