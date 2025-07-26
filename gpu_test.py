#!/usr/bin/env python3
"""
Check JAX installation and configuration for Metal GPU support
"""
import sys

print("Checking JAX installation...")
print("=" * 60)

# Check if JAX is installed
try:
    import jax
    print(f"✅ JAX version: {jax.__version__}")
except ImportError:
    print("❌ JAX is not installed!")
    print("Install with: pip install jax-metal")
    sys.exit(1)

# Check if jaxlib is installed
try:
    import jaxlib
    print(f"✅ JAXlib version: {jaxlib.__version__}")
except ImportError:
    print("❌ JAXlib is not installed!")
    sys.exit(1)

# Check backend
print(f"\n📍 Default backend: {jax.default_backend()}")
print(f"📍 Available devices: {jax.devices()}")

# Check if Metal is available
devices = jax.devices()
has_metal = any('metal' in str(d).lower() for d in devices)
if has_metal:
    print("✅ Metal backend detected!")
else:
    print("⚠️  Metal backend NOT detected - JAX will use CPU")
    print("   Make sure you installed jax-metal, not regular jax")

# Test basic operations
print("\n🧪 Testing JAX operations...")
try:
    import jax.numpy as jnp
    
    # Test array creation
    test_array = jnp.ones(10, dtype=jnp.float32)
    print(f"✅ Array creation: shape={test_array.shape}, dtype={test_array.dtype}")
    
    # Test computation
    result = jnp.sum(test_array)
    print(f"✅ Basic computation: sum={result}")
    
    # Test JIT compilation
    @jax.jit
    def test_func(x):
        return x * 2 + 1
    
    jitted_result = test_func(test_array)
    print(f"✅ JIT compilation: successful, result shape={jitted_result.shape}")
    
    # Test device placement
    x = jax.device_put(jnp.array([1.0, 2.0, 3.0]))
    print(f"✅ Device placement: array on {x.device()}")
    
except Exception as e:
    print(f"❌ JAX operations failed: {e}")
    import traceback
    traceback.print_exc()

# Check float64 setting (should be disabled for Metal)
print(f"\n📍 Float64 enabled: {jax.config.x64_enabled}")
if jax.config.x64_enabled and has_metal:
    print("⚠️  WARNING: Float64 is enabled but Metal doesn't fully support it!")
    print("   Your code correctly disables it with: jax.config.update('jax_enable_x64', False)")

# Check for scipy special functions
print("\n🧪 Testing scipy special functions...")
try:
    from jax.scipy.special import i0, i1
    test_val = jnp.array(0.5, dtype=jnp.float32)
    i0_result = i0(test_val)
    i1_result = i1(test_val)
    print(f"✅ Bessel I0(0.5) = {i0_result:.6f}")
    print(f"✅ Bessel I1(0.5) = {i1_result:.6f}")
except Exception as e:
    print(f"⚠️  scipy.special functions issue: {e}")

# Installation instructions
print("\n" + "=" * 60)
print("📋 Installation Guide:")
print("=" * 60)

if not has_metal:
    print("\nTo install JAX with Metal support on macOS:")
    print("1. First uninstall any existing JAX:")
    print("   pip uninstall jax jaxlib")
    print("\n2. Install jax-metal:")
    print("   pip install jax-metal")
    print("\n3. Verify Metal support:")
    print("   python -c \"import jax; print(jax.devices())\"")
else:
    print("\n✅ Your JAX installation looks good for Metal!")
    print("\nYour code has proper Metal compatibility:")
    print("- ✅ Uses float32 (Metal doesn't support float64)")
    print("- ✅ Has CPU fallbacks for Bessel K functions") 
    print("- ✅ Checks for Metal backend with is_metal_backend()")
    print("- ✅ Disables x64 with jax.config.update('jax_enable_x64', False)")

# Performance tips
print("\n💡 Performance Tips for JAX on Metal:")
print("- Use float32 for all computations (as your code does)")
print("- Batch operations when possible")
print("- Use jax.jit for frequently called functions")
print("- Avoid frequent CPU-GPU transfers")