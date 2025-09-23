# Observational Baseline Data

This file contains the raw observational data that our model must explain.

## 1. Cluster Lensing Data

### Mass Discrepancy Problem
Clusters lens approximately 5× more than their visible (gas) mass.

| Cluster | z | M_gas [10¹⁴ M☉] | M_lens [10¹⁴ M☉] | M_lens/M_gas | R_Einstein [arcsec] |
|---------|---|------------------|-------------------|--------------|--------------------|
| Abell 1689 | 0.184 | 1.2 | 5.8 | 4.8 | 47.0 |
| Abell 2029 | 0.077 | 0.8 | 3.2 | 4.0 | 28.0 |
| A478 | 0.088 | 0.9 | 3.5 | 3.9 | 31.0 |
| MACS J0416 | 0.396 | 1.5 | 8.2 | 5.5 | 35.0 |
| Bullet | 0.296 | 2.0 | 15.0 | 7.5 | 55.0 |

### Key Insight
- **MOND fails** here because it doesn't modify light deflection
- **Dark matter** invokes ~5× more invisible mass
- **Our model** needs different photon vs matter coupling

## 2. Galaxy Rotation Curves (SPARC)

### The Flat Rotation Curve Problem
Galaxy rotation curves stay flat instead of declining as expected from visible matter.

### NGC3198
```python
R_kpc = [0.32, 0.64, 0.96, 1.28, 1.61, 1.93, 2.24, 2.57, 2.89, 3.21]...  # 43 points
V_obs = [24.4, 43.3, 45.5, 58.5, 68.8, 76.9, 82.0, 86.9, 97.6, 100.0]...  # km/s
V_bar = [63.28, 73.66, 78.98, 82.70074062546236, 84.22013357861645, 83.17001502945638, 87.04126894755154, 88.91507521224958, 88.99149004258778, 93.81692651115789]...  # Baryonic expectation
```

### NGC2403
```python
R_kpc = [0.16, 0.26, 0.36, 0.46, 0.56, 0.66, 0.76, 0.86, 0.96, 1.06]...  # 73 points
V_obs = [24.5, 35.3, 43.2, 52.0, 60.9, 65.8, 71.7, 74.6, 74.6, 76.6]...  # km/s
V_bar = [23.21, 35.33, 47.00922569028339, 56.726241722856976, 63.8246229914443, 67.62657465819188, 70.9087053893949, 72.89295164829038, 74.9778453944897, 77.24371366007722]...  # Baryonic expectation
```

### NGC7331
```python
R_kpc = [2.67, 3.21, 3.74, 4.27, 4.81, 5.35, 5.88, 6.41, 7.48, 8.55]...  # 36 points
V_obs = [221.0, 237.0, 249.0, 250.0, 253.0, 257.0, 257.0, 257.0, 257.0, 255.0]...  # km/s
V_bar = [346.30922020645073, 366.1894397712747, 383.58118449684156, 384.6465190275352, 381.81582588991773, 383.73184504286326, 376.5703367234334, 370.806201269612, 355.2913137131275, 333.54086915998766]...  # Baryonic expectation
```

### NGC2976
```python
R_kpc = [0.11, 0.17, 0.24, 0.31, 0.38, 0.45, 0.52, 0.59, 0.66, 0.73]...  # 27 points
V_obs = [6.8, 9.5, 14.0, 19.8, 26.1, 28.7, 28.7, 31.7, 35.5, 39.4]...  # km/s
V_bar = [12.02680339907492, 15.5399871299818, 19.675652466945028, 23.294999463404157, 26.475214824435326, 29.4515279739439, 32.4975029809984, 35.56337020024959, 38.46211772640711, 41.246339231500286]...  # Baryonic expectation
```

### F568-3
```python
R_kpc = [0.64, 1.18, 1.8, 2.03, 2.78, 3.0, 3.64, 4.19, 4.6, 5.14]...  # 18 points
V_obs = [12.4, 21.9, 32.3, 37.9, 53.1, 62.4, 68.0, 78.0, 82.3, 85.7]...  # km/s
V_bar = [13.611245350811954, 21.226667190117247, 27.76613765002255, 29.79981879139536, 36.42926845271532, 38.11444345651658, 42.50319517401015, 45.735798888835426, 46.93404414707942, 48.025644191410905]...  # Baryonic expectation
```

### DDO154
```python
R_kpc = [0.49, 0.99, 1.48, 1.97, 2.47, 2.96, 3.46, 3.95, 4.44, 4.94]...  # 12 points
V_obs = [13.8, 21.6, 28.9, 34.3, 38.2, 42.0, 44.6, 46.3, 47.4, 48.2]...  # km/s
V_bar = [12.865601423952167, 16.350966332299752, 16.907377088123397, 17.623677255328978, 17.938375065763342, 18.61036270468687, 19.48889170784219, 19.522832786253126, 18.93923176900267, 18.278320491773854]...  # Baryonic expectation
```

## 3. Milky Way Rotation Curve

### Our Local Laboratory
The Milky Way rotation curve from Gaia DR3 shows no Keplerian decline.

```python
# Galactocentric radius [kpc]
R_kpc = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20]

# Observed circular velocity [km/s]
V_obs = [195, 210, 220, 225, 228, 230, 232, 235, 233, 230, 225, 220, 215, 210, 205]

# Baryonic expectation [km/s]
V_bar = [107.8, 123.0, 124.8, 123.1, 120.7, 118.6, 117.2, 116.3, 115.8, 115.6, 115.7, 116.0, 116.4, 116.8, 117.1]

# Discrepancy
V_missing = V_obs - V_bar  # ~100 km/s at R > 10 kpc
```

## Summary of Observational Challenges

1. **Clusters**: Lens 5× more than gas mass (100-1000 kpc scale)
2. **Galaxies**: Flat rotation curves (1-50 kpc scale)
3. **Milky Way**: No Keplerian decline (1-20 kpc scale)

All three require either:
- Dark matter (adds invisible mass)
- Modified gravity (changes force law)
- **Geometric enhancement** (our approach - modifies spacetime)

## Data Files

- Plots: `images/baseline/`
- SPARC data: `external_data/Rotmod_LTG/`
- Cluster profiles: `data/*_gas_profile.csv`
- This file: `data/baseline.md`
