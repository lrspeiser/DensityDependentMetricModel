import numpy as np
import corner
import matplotlib.pyplot as plt

# Load the final stage results
data = np.load('chains_dynesty/dynesty_curriculum_stage_3_power_samples.npz')
samples = data['samples']
weights = data['weights']

# Resample to equal weights for corner plot
from dynesty import utils as dyfunc
samples_equal = dyfunc.resample_equal(samples, weights)

# Make corner plot
labels = ['ρ_c (M☉/kpc³)', 'n', 'M_thin (M☉)', 'R_thin (kpc)', 'h_thin (kpc)']
fig = corner.corner(samples_equal, labels=labels, show_titles=True, 
                    quantiles=[0.16, 0.5, 0.84])
plt.savefig('final_corner_plot.png', dpi=300)
plt.show()

# Check what xi looks like
rho_c = np.median(samples_equal[:, 0])
n = np.median(samples_equal[:, 1])
print(f"\nXi function parameters:")
print(f"ρ_c = {rho_c:.2e} M☉/kpc³")
print(f"n = {n:.3f}")

# Calculate xi at different densities
rho_test = np.logspace(6, 10, 100)  # M☉/kpc³
xi = 1 / (1 + (rho_test/rho_c)**n)
plt.figure()
plt.loglog(rho_test, xi)
plt.xlabel('ρ (M☉/kpc³)')
plt.ylabel('ξ')
plt.title('Density correction factor')
plt.grid(True)
plt.savefig('xi_function.png')
plt.show()