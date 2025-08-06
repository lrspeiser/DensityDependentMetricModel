# test_xi_models.py - Check if functions exist
import cupy as cp
from density_metric_cupy import v_total_kms_cupy

# Test basic parameters
R_test = cp.array([8.0])
params_test = {
    'M_thin_disk_solar': 5e10,
    'R_thin_disk_kpc': 3.0,
    'hz_thin_disk_kpc': 0.3,
    'M_thick_disk_solar': 5e9,
    'R_thick_disk_kpc': 4.0,
    'hz_thick_disk_kpc': 0.8,
    'M_bulge_solar': 5e9,
    'R_bulge_kpc': 1.0,
    'M_gas_solar': 1e10,
    'R_gas_kpc': 7.0,
    'hz_gas_kpc': 0.15,
    'rho_c_solar_kpc3': 1e15,
    'n_exp': 2.0,
    'A': 5.0
}

# Test each xi type
for xi_type in ['gr', 'power', 'enhanced', 'sigmoid', 'peak', 'broken']:
    try:
        v = v_total_kms_cupy(R_test, params_test, xi_type=xi_type)
        print(f"{xi_type}: v = {float(v[0]):.2f} km/s")
    except Exception as e:
        print(f"{xi_type}: FAILED - {e}")