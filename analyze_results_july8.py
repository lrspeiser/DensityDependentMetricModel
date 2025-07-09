import numpy as np
import matplotlib.pyplot as plt
import corner

# Load results
data = np.load('chains_truly_data_driven/dynesty_mw_power_Bf_DTf_DKf_Gf_samples.npz')
samples = data['samples']
weights = data['weights']

# Make corner plot
labels = ['ρ_c', 'n', 'M_thin', 'R_d_thin', 'h_z_thin', 
          'M_thick', 'R_d_thick', 'h_z_thick',
          'M_bulge', 'a_bulge', 'M_gas', 'R_d_gas', 'h_z_gas']
fig = corner.corner(samples, weights=weights, labels=labels, 
                   quantiles=[0.16, 0.5, 0.84], show_titles=True)
plt.savefig('parameter_posteriors.png', dpi=300)