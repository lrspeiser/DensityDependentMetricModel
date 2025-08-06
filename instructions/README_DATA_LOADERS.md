# Data Loaders Module Documentation

This directory contains specialized data loading modules for different astronomical surveys and datasets.

## Survey-Specific Loaders

### 1. **all_data_loader.py** - Multi-Dataset Aggregator
- **Purpose**: Loads and combines multiple datasets for joint analysis
- **Datasets Combined**: Planck CMB, Pantheon SNe, BAO measurements
- **Key Functions**: `load_combined_cosmology()`, `apply_systematic_corrections()`
- **Output Format**: Unified DataFrame with error covariances
- **When to Use**: Joint cosmological constraints requiring multiple probes

### 2. **sparc_data_loader.py** - SPARC Galaxy Database
- **Purpose**: Loads rotation curves from SPARC (Spitzer Photometry & Accurate Rotation Curves)
- **Data Content**: 175 galaxy rotation curves, photometric profiles, mass distributions
- **Unique Features**: Quality flags, multiple distance estimates, bulge/disk decomposition
- **Use Case**: Testing DDMM on galactic scales
- **Differences**: Focuses on individual galaxy dynamics vs. cosmological scales

### 3. **bao_data_loader.py** - Baryon Acoustic Oscillations
- **Purpose**: Loads BAO measurements from various surveys
- **Surveys Included**: SDSS DR12, 6dFGS, WiggleZ, DES Y3
- **Data Types**: DV/rd measurements, DA/rd and H*rd (anisotropic), full covariance matrices
- **Redshift Range**: 0.1 < z < 2.4
- **Key Difference**: Provides standard ruler measurements for expansion history

### 4. **des_y3_loader.py** - Dark Energy Survey Year 3
- **Purpose**: Loads DES Y3 cosmological data products
- **Data Products**: Weak lensing shear catalogs, galaxy clustering, cross-correlations
- **Special Handling**: Complex selection functions, photo-z uncertainty propagation
- **Coverage**: 5000 square degrees
- **Differences**: Most comprehensive weak lensing dataset

### 5. **frontier_lensing_loader.py** - Frontier Fields Lensing
- **Purpose**: Strong and weak lensing data from Hubble Frontier Fields
- **Data Content**: Lensing mass maps, critical curves, magnification maps
- **Unique Aspects**: High-resolution cluster lensing, multiple lens plane modeling
- **Best For**: Testing DDMM in strong gravitational fields

### 6. **kids_loader.py** - Kilo-Degree Survey
- **Purpose**: Loads KiDS weak lensing and photometry data
- **Coverage**: 1350 square degrees
- **Data Products**: Shear catalogs, photometric redshifts (9-band), survey masks
- **Quality Features**: Blind analysis flags, multiple shape measurement methods
- **Differences from DES**: Different systematic control, complementary sky coverage

### 7. **load_existing_gaia.py** - Gaia Stellar Density
- **Purpose**: Loads pre-processed Gaia stellar density maps
- **Differences from data_io.load_gaia()**:
  - Loads cached/pre-computed density fields
  - Includes proper motion corrections
  - 100x faster than processing raw Gaia data
- **Data Products**: 3D stellar density cubes, velocity dispersion maps
- **Performance**: Optimized for repeated access

## Data Format Standardization

All loaders output standardized formats:
```python
{
    'data': numpy.ndarray,      # Measurements
    'errors': numpy.ndarray,     # Uncertainties
    'covariance': numpy.ndarray, # Full covariance (if available)
    'redshift': numpy.ndarray,   # Redshift array
    'metadata': dict,            # Survey-specific info
    'masks': numpy.ndarray,      # Quality/selection masks
}
```

## Loader Selection Guide

| Data Type | Loader | Best For |
|-----------|--------|----------|
| Combined cosmology | all_data_loader.py | Joint constraints |
| Galaxy dynamics | sparc_data_loader.py | Dark matter tests |
| Large-scale structure | bao_data_loader.py | Expansion history |
| Weak lensing | des_y3_loader.py, kids_loader.py | Growth of structure |
| Strong lensing | frontier_lensing_loader.py | Cluster mass profiles |
| Stellar density | load_existing_gaia.py | Local density field |

## Error Handling

All loaders implement consistent error handling:
- Missing data files raise `FileNotFoundError`
- Corrupted data raises `DataIntegrityError`
- Version mismatches logged as warnings
- Automatic fallback to cached data when available

## Caching Strategy

- First load: Downloads and caches processed data
- Subsequent loads: Uses cached version (100x faster)
- Cache location: `~/.ddmm_cache/[survey_name]/`
- Cache invalidation: Based on file hash and version

## Configuration

Key configuration options:
- `DATA_ROOT`: Base path for all data
- `USE_CACHE`: Enable/disable caching
- `DOWNLOAD_MISSING`: Auto-download missing data
- `QUALITY_CUTS`: Apply standard quality cuts
