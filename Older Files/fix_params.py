#!/usr/bin/env python3
"""
Fix the xi function selection issue
"""
import json
import re

print("ISSUE FOUND: Validation is using 'power' instead of 'gaussian'!")
print("="*60)

# Check how validation selects the xi function
with open('validate_ddmm.py', 'r') as f:
    content = f.read()

# Find xi_func initialization
xi_init_match = re.search(r"self\.xi_func = XI_FUNCTION_MAP\.get\(\s*([^,)]+)[^)]*\)", content)

if xi_init_match:
    print("Found xi_func initialization:")
    print(f"  {xi_init_match.group(0)}")
    
    param_used = xi_init_match.group(1).strip()
    print(f"\nValidation is looking for: {param_used}")
    
    if 'xi_type' in param_used:
        print("✓ It's looking for 'xi_type' parameter")
        param_key = 'xi_type'
    elif 'xi_function' in param_used:
        print("✓ It's looking for 'xi_function' parameter")  
        param_key = 'xi_function'
    else:
        print("⚠️  Unknown parameter lookup")
        param_key = None

# Find the default
default_match = re.search(r"XI_FUNCTION_MAP\[['\"](.*?)['\"]\]", xi_init_match.group(0))
if default_match:
    default_func = default_match.group(1)
    print(f"\nDefault xi function: '{default_func}'")
    
    if default_func == 'power':
        print("⚠️  This explains it! Default is 'power', not 'gaussian'")

print("\n" + "="*60)
print("Creating fixed parameter file...")

# Load current parameters
with open('gaussian_params_best.json', 'r') as f:
    params = json.load(f)

# Create fixed version
params_fixed = params.copy()

# Add the parameter that validation is looking for
if param_key:
    params_fixed[param_key] = 'gaussian'
    print(f"✓ Added '{param_key}': 'gaussian' to parameters")

# Also add xi_type just in case
if 'xi_type' not in params_fixed:
    params_fixed['xi_type'] = 'gaussian'
    print("✓ Added 'xi_type': 'gaussian' as backup")

# Save fixed parameters
with open('gaussian_params_fixed_xi_type.json', 'w') as f:
    json.dump(params_fixed, f, indent=2)

print("\n✓ Created gaussian_params_fixed_xi_type.json")
print("\nThis file has:")
for key in ['xi_function', 'xi_type']:
    print(f"  '{key}': '{params_fixed.get(key, 'NOT SET')}'")

print("\n" + "="*60)
print("Run validation with fixed parameters:")
print("  python3 validate_ddmm.py gaussian_params_fixed_xi_type.json --output_dir validation_gaussian_fixed")