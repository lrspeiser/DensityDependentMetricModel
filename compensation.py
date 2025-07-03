# Create a "compensation plot" showing how M_total × average(ξ) might be conserved
import numpy as np
import matplotlib.pyplot as plt

# Calculate average ξ over relevant radial range (5-15 kpc)
R_range = np.linspace(5, 15, 100)
rho_range = np.logspace(8, 7.3, 100)  # Approximate densities

xi_single_avg = np.mean([1/(1+(rho/1.64e9)**1.56) for rho in rho_range])
xi_thick_avg = np.mean([1/(1+(rho/1.86e8)**0.724) for rho in rho_range])

M_single = 1.27e11
M_thick = 1.67e11

print(f"Single disk: M × <ξ> = {M_single:.2e} × {xi_single_avg:.3f} = {M_single * xi_single_avg:.2e}")
print(f"Thin+Thick:  M × <ξ> = {M_thick:.2e} × {xi_thick_avg:.3f} = {M_thick * xi_thick_avg:.2e}")
print(f"Ratio: {(M_single * xi_single_avg)/(M_thick * xi_thick_avg):.2f}")