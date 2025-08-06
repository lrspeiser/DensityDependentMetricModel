#!/usr/bin/env python3
"""
test_gpu_detection.py - Diagnostic script to test GPU detection methods

This script tests various GPU detection methods to identify why the NVIDIA 5090
isn't being detected by the resource monitor.
"""

import sys
import os
import subprocess
import json
from pathlib import Path

def test_nvidia_smi():
    """Test if nvidia-smi command is available and working."""
    print("=== Testing nvidia-smi ===")
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✓ nvidia-smi is available and working:")
            print(result.stdout.strip())
            return True
        else:
            print(f"✗ nvidia-smi failed with return code {result.returncode}")
            print(f"Error: {result.stderr}")
            return False
    except FileNotFoundError:
        print("✗ nvidia-smi not found in PATH")
        return False
    except subprocess.TimeoutExpired:
        print("✗ nvidia-smi timed out")
        return False
    except Exception as e:
        print(f"✗ nvidia-smi error: {e}")
        return False

def test_pynvml():
    """Test pynvml library."""
    print("\n=== Testing pynvml ===")
    try:
        import pynvml
        print("✓ pynvml imported successfully")
        
        # Initialize
        pynvml.nvmlInit()
        print("✓ pynvml initialized successfully")
        
        # Get device count
        gpu_count = pynvml.nvmlDeviceGetCount()
        print(f"✓ Found {gpu_count} NVIDIA GPU(s)")
        
        # Get device details
        for i in range(gpu_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            
            print(f"  GPU {i}: {name}")
            print(f"    Memory: {memory.total / (1024**3):.1f} GB total, {memory.free / (1024**3):.1f} GB free")
            print(f"    Utilization: {util.gpu}%")
        
        pynvml.nvmlShutdown()
        print("✓ pynvml shutdown successfully")
        return True
        
    except ImportError:
        print("✗ pynvml not installed")
        return False
    except Exception as e:
        print(f"✗ pynvml error: {e}")
        return False

def test_gputil():
    """Test GPUtil library."""
    print("\n=== Testing GPUtil ===")
    try:
        import GPUtil
        print("✓ GPUtil imported successfully")
        
        gpus = GPUtil.getGPUs()
        print(f"✓ Found {len(gpus)} GPU(s) via GPUtil")
        
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu.name}")
            print(f"    Memory: {gpu.memoryTotal} MB total, {gpu.memoryFree} MB free")
            print(f"    Load: {gpu.load*100:.1f}%")
            print(f"    Temperature: {gpu.temperature}°C")
        
        return True
        
    except ImportError:
        print("✗ GPUtil not installed")
        return False
    except Exception as e:
        print(f"✗ GPUtil error: {e}")
        return False

def test_jax():
    """Test JAX GPU detection."""
    print("\n=== Testing JAX ===")
    try:
        import jax
        print("✓ JAX imported successfully")
        
        devices = jax.devices()
        backend = jax.default_backend()
        
        print(f"✓ JAX backend: {backend}")
        print(f"✓ JAX devices: {len(devices)}")
        
        for i, device in enumerate(devices):
            print(f"  Device {i}: {device}")
        
        return True
        
    except ImportError:
        print("✗ JAX not installed")
        return False
    except Exception as e:
        print(f"✗ JAX error: {e}")
        return False

def test_torch():
    """Test PyTorch GPU detection."""
    print("\n=== Testing PyTorch ===")
    try:
        import torch
        print("✓ PyTorch imported successfully")
        
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.is_available()}")
            print(f"✓ CUDA device count: {torch.cuda.device_count()}")
            print(f"✓ Current device: {torch.cuda.current_device()}")
            
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                print(f"  GPU {i}: {name} ({memory:.1f} GB)")
        else:
            print("✗ CUDA not available in PyTorch")
        
        return True
        
    except ImportError:
        print("✗ PyTorch not installed")
        return False
    except Exception as e:
        print(f"✗ PyTorch error: {e}")
        return False

def test_tensorflow():
    """Test TensorFlow GPU detection."""
    print("\n=== Testing TensorFlow ===")
    try:
        import tensorflow as tf
        print("✓ TensorFlow imported successfully")
        
        gpus = tf.config.list_physical_devices('GPU')
        print(f"✓ TensorFlow found {len(gpus)} GPU(s)")
        
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu.name}")
        
        return True
        
    except ImportError:
        print("✗ TensorFlow not installed")
        return False
    except Exception as e:
        print(f"✗ TensorFlow error: {e}")
        return False

def test_system_info():
    """Test system information."""
    print("\n=== System Information ===")
    try:
        import psutil
        print(f"✓ Platform: {sys.platform}")
        print(f"✓ CPU cores: {psutil.cpu_count()}")
        print(f"✓ Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
        
        # Check for NVIDIA drivers in Windows
        if sys.platform == "win32":
            nvidia_paths = [
                r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
                r"C:\Windows\System32\nvidia-smi.exe"
            ]
            for path in nvidia_paths:
                if os.path.exists(path):
                    print(f"✓ Found nvidia-smi at: {path}")
                    break
            else:
                print("✗ nvidia-smi not found in common Windows locations")
        
        return True
        
    except Exception as e:
        print(f"✗ System info error: {e}")
        return False

def main():
    """Run all GPU detection tests."""
    print("GPU Detection Diagnostic Tool")
    print("=" * 50)
    
    results = {
        'nvidia_smi': test_nvidia_smi(),
        'pynvml': test_pynvml(),
        'gputil': test_gputil(),
        'jax': test_jax(),
        'torch': test_torch(),
        'tensorflow': test_tensorflow(),
        'system_info': test_system_info()
    }
    
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    for test, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test:15}: {status}")
    
    # Save results
    with open("gpu_detection_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: gpu_detection_results.json")

if __name__ == "__main__":
    main() 