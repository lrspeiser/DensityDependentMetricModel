# fix_json.py
import json

# Load the summary
with open('chains_DDMM_power_best_fit/summary_power.json', 'r') as f:
    data = json.load(f)

# Create flattened version for validator
validator_params = {
    'xi_type': 'power',
    'A': 1.0  # Default enhancement factor
}

# Extract median values from parameters
for param_name, param_data in data['parameters'].items():
    validator_params[param_name] = param_data['median']

# Add component flags based on what parameters are present
validator_params['include_disk_thin'] = True  # Default thin disk
validator_params['include_disk_thick'] = 'M_disk_thick_solar' in data['parameters']
validator_params['include_bulge'] = 'M_bulge_solar' in data['parameters'] 
validator_params['include_gas'] = 'M_gas_solar' in data['parameters']

# Save for validator
with open('chains_DDMM_power_best_fit/params_for_validation.json', 'w') as f:
    json.dump(validator_params, f, indent=2)

print("Created params_for_validation.json with structure:")
for k, v in validator_params.items():
    if isinstance(v, float) and v > 1e6:
        print(f"  {k}: {v:.2e}")
    else:
        print(f"  {k}: {v}")