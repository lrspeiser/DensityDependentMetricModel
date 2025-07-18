# save_params.py
import json

params = {
    "rho_c_solar_kpc3": 1.02e8,
    "n_exp": 3.174,  # This is your gamma_exp
    "M_disk_thin_solar": 8.51e10,
    "R_d_thin_kpc": 6.271,
    "h_z_thin_kpc": 0.445,
    "M_disk_thick_solar": 1.90e10,
    "R_d_thick_kpc": 1.800,
    "h_z_thick_kpc": 0.065,
    "M_bulge_solar": 3.71e10,
    "a_bulge_kpc": 14.377,
    "M_gas_solar": 5.69e9,
    "include_disk_thin": True,
    "include_disk_thick": True,
    "include_bulge": True,
    "include_gas": True,
    "xi_type": "power"
}

with open("ddmm_params.json", "w") as f:
    json.dump(params, f, indent=2)