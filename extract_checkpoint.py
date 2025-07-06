import pickle
import numpy as np

# This avoids the need to import dynesty by just loading the raw data
with open('chains_full_precision/stage_3/dynesty_checkpoint.pkl', 'rb') as f:
    res = pickle.load(f)

print('From checkpoint file:')
print(f'  Number of samples: {len(res.samples)}')
print(f'  Final log(Z): {res.logz[-1]:.3f}')
print(f'  Log(Z) error: {res.logzerr[-1]:.3f}')

# Calculate dlogz
if len(res.logz) > 1:
    dlogz_values = np.diff(res.logz)
    print(f'  Final dlogz: {dlogz_values[-1]:.4f}')
    print(f'  Last 5 dlogz values: {dlogz_values[-5:]}')

# Calculate weights
weights = np.exp(res.logwt - res.logz[-1])
print(f'  Sum of weights: {np.sum(weights):.3f}')
print(f'  Effective samples: {1.0 / np.sum(weights**2):.1f}')

# Get parameter estimates
weighted_mean = np.average(res.samples, weights=weights, axis=0)
weighted_std = np.sqrt(np.average((res.samples - weighted_mean)**2, weights=weights, axis=0))

print('\nFinal parameters from checkpoint:')
param_names = ['rho_c', 'n', 'M_thin', 'R_thin', 'h_thin', 'M_thick', 'R_thick', 'h_thick', 
               'M_bulge', 'a_bulge', 'M_gas', 'R_gas', 'h_gas']

for i in range(len(weighted_mean)):
    if i < len(param_names):
        print(f'  {param_names[i]}: {weighted_mean[i]:.3e} ± {weighted_std[i]:.3e}')
    else:
        print(f'  Param {i}: {weighted_mean[i]:.3e} ± {weighted_std[i]:.3e}')

# Save the proper results
np.savez('chains_full_precision/final_results_stage3_corrected.npz',
         samples=res.samples,
         weights=weights,
         logl=res.logl,
         logz=res.logz,
         logzerr=res.logzerr,
         logwt=res.logwt,
         ncall=res.ncall if hasattr(res, 'ncall') else None)

print(f'\nSaved corrected results to final_results_stage3_corrected.npz')
print(f'This file contains {len(res.samples)} samples with proper weights')

