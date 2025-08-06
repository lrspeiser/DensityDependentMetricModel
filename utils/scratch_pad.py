# Save as test_enhanced.py
import sys
sys.path.append('.')  # Ensure we can import local modules

from density_metric2 import XI_FUNCTION_MAP, v_total_kms
import numpy as np

# Test parameters
params = {
    'rho_c_solar_kpc3': 1e13,
    'n_exp': 1.5,
    'A': 1.0,
    'M_disk_thin_solar': 4e10,
    'R_d_thin_kpc': 2.5,
    'h_z_thin_kpc': 0.3,
    'include_disk_thin': True,
    'include_disk_thick': False,
    'include_bulge': False,
    'include_gas': False
}

# Test at solar radius
R_test = np.array([8.0])
print(f"Testing enhanced model at R={R_test[0]} kpc...")

try:
    v = v_total_kms(R_test, params, xi_type='enhanced')
    print(f"Success! v = {v[0]:.1f} km/s")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()