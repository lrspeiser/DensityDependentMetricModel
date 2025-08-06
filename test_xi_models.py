import cupy as cp
import numpy as np
from density_metric_cupy import v_total_kms_cupy

# More realistic galaxy parameters
params_realistic = {
    'M_thin_disk_solar': 6e10,    # Milky Way thin disk mass
    'R_thin_disk_kpc': 2.6,        # Scale length
    'hz_thin_disk_kpc': 0.3,       
    'M_thick_disk_solar': 1e10,    # ~15% of thin disk
    'R_thick_disk_kpc': 3.5,       
    'hz_thick_disk_kpc': 0.9,      
    'M_bulge_solar': 2e10,         # Central bulge
    'R_bulge_kpc': 0.5,            
    'M_gas_solar': 1e10,           # Gas disk
    'R_gas_kpc': 4.0,              
    'hz_gas_kpc': 0.15,
}

# Test at different radii
test_radii = [1.0, 5.0, 8.5, 15.0, 25.0]  # kpc

print("Testing velocities at different radii:\n")
print("Radius | GR    | Power | Enhanced | Sigmoid | Peak  | Broken")
print("-" * 70)

for R in test_radii:
    R_test = cp.array([R])
    velocities = []
    
    # Test each model with appropriate parameters
    for xi_type in ['gr', 'power', 'enhanced', 'sigmoid', 'peak', 'broken']:
        params = params_realistic.copy()
        
        # Add xi-specific parameters
        if xi_type == 'power':
            params.update({'rho_c_solar_kpc3': 1e8, 'n_exp': 1.2, 'A': 3.0})
        elif xi_type == 'enhanced':
            params.update({'rho_c_solar_kpc3': 1e15, 'n_exp': 1.2, 'A': 5.0})
        elif xi_type == 'sigmoid':
            params.update({'rho_c_solar_kpc3': 1e6, 'n_exp': 1.5, 'A': 2.0})
        elif xi_type == 'peak':
            params.update({'rho_peak_solar_kpc3': 1e4, 'width_log': 1.5, 'A': 3.0})
        elif xi_type == 'broken':
            params.update({'rho_break_solar_kpc3': 1e5, 'n_low': 1.5, 'n_high': 0.5, 'A': 3.0})
        
        try:
            v = v_total_kms_cupy(R_test, params, xi_type=xi_type)
            v_val = float(v[0])
            velocities.append(f"{v_val:6.1f}")
        except Exception as e:
            velocities.append(" ERROR")
    
    print(f"{R:5.1f} | " + " | ".join(velocities))

# Expected values at 8.5 kpc (Sun's position) should be ~220 km/s