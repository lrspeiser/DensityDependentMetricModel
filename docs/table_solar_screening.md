# Solar-System Screening Comparison (Illustrative)

Parameters:
- Power-law: rho_c=1.0e+15, n=3.00, A=4.00
- Exponential: rho_c=1.0e+15, gamma=3.00, lambda_max=4.00

| Environment | ρ [M_⊙/kpc³] | |ξ−1| (power) | |ξ−1| (exp) |
|---|---:|---:|---:|
| Laboratory (air) | 1.00e+22 | 0.00e+00 | 0.00e+00 |
| Earth orbit | 5.00e+21 | 0.00e+00 | 0.00e+00 |
| Jupiter orbit | 5.00e+21 | 0.00e+00 | 0.00e+00 |
| Saturn orbit | 2.30e+21 | 0.00e+00 | 0.00e+00 |
| Neptune orbit | 1.00e+21 | 0.00e+00 | 0.00e+00 |
---

## Mapping to ΔGM/GM (weak-field) at AU radii
We adopt the simple weak-field rule ΔGM/GM ≈ |ξ−1|. Values below are medians across the five galaxies’ posteriors with [IQR] across galaxies. Densities are representative placeholders for the vacuum/plasma environment near the listed AU radii.

| Radius | ρ [M_⊙/kpc³] | ΔGM/GM (exp) median [IQR] | ΔGM/GM (power) median [IQR] |
|---|---:|---:|---:|
| 1 AU | 1.00e+19 | 0.00e+00 [0.00e+00,0.00e+00] | 8.75e-12 [6.34e-12,1.01e-11] |
| 9.5 AU | 5.00e+17 | 0.00e+00 [0.00e+00,0.00e+00] | 1.04e-07 [7.16e-08,1.06e-07] |
| 19 AU | 2.00e+17 | 0.00e+00 [0.00e+00,0.00e+00] | 1.53e-06 [1.13e-06,1.58e-06] |
| 30 AU | 1.00e+17 | 0.00e+00 [0.00e+00,0.00e+00] | 1.09e-05 [7.99e-06,1.10e-05] |

Reference: canonical two-way ranging/ephemeris limits suggest |ΔGM/GM| ≲ O(10^{-12}) at outer-planet scales; see, e.g., Adelberger et al. (2003) and follow-ups.
