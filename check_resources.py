#!/usr/bin/env python3
# save as: check_resources.py
import psutil
import multiprocessing
import jax
import resource

print("=== SYSTEM RESOURCES ===")
print(f"CPU cores: {multiprocessing.cpu_count()}")
print(f"Total RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
print(f"Available RAM: {psutil.virtual_memory().available / (1024**3):.1f} GB")

print(f"\n=== PROCESS LIMITS ===")
soft, hard = resource.getrlimit(resource.RLIMIT_AS)
print(f"Memory limit: {soft if soft != resource.RLIM_INFINITY else 'unlimited'}")

soft, hard = resource.getrlimit(resource.RLIMIT_CPU)
print(f"CPU time limit: {soft if soft != resource.RLIM_INFINITY else 'unlimited'} seconds")

print(f"\n=== JAX CONFIGURATION ===")
print(f"Default backend: {jax.default_backend()}")
print(f"Devices: {jax.devices()}")

# Test memory allocation
print(f"\n=== MEMORY TEST ===")
try:
    for size_gb in [1, 2, 4, 8, 16]:
        size = int(size_gb * 1024**3 / 8)  # 8 bytes per float64
        arr = jax.numpy.zeros(size)
        print(f"✓ Successfully allocated {size_gb} GB JAX array")
        del arr
except Exception as e:
    print(f"✗ Failed at {size_gb} GB: {e}")