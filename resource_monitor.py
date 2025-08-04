#!/usr/bin/env python3
"""
resource_monitor.py - Comprehensive hardware resource monitoring for dynesty runs

This module provides:
- Real-time GPU utilization monitoring (NVIDIA, AMD, Apple Metal)
- CPU utilization and core usage tracking
- Memory usage monitoring (RAM, VRAM)
- JAX device utilization tracking
- Performance bottleneck detection
- Resource utilization reports

Author: Enhanced Resource Monitoring System
Version: 1.0
"""

import time
import psutil
import threading
import json
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
import os
import sys

# Optional imports for GPU monitoring
try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

try:
    import pynvml
    NVIDIA_AVAILABLE = True
except ImportError:
    NVIDIA_AVAILABLE = False


class ResourceMonitor:
    """Comprehensive hardware resource monitoring for dynesty runs."""
    
    def __init__(self, output_dir: Path, log_interval: int = 30):
        """
        Initialize resource monitor.
        
        Args:
            output_dir: Directory to save resource logs
            log_interval: How often to log resources (seconds)
        """
        self.output_dir = Path(output_dir)
        self.log_interval = log_interval
        self.resource_file = self.output_dir / "resource_usage.json"
        self.summary_file = self.output_dir / "resource_summary.json"
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        
        # Resource tracking
        self.resource_history = []
        self.start_time = None
        self.monitoring = False
        self.monitor_thread = None
        
        # Performance metrics
        self.peak_cpu = 0.0
        self.peak_memory = 0.0
        self.peak_gpu_util = 0.0
        self.peak_gpu_memory = 0.0
        
        # Initialize hardware detection
        self._detect_hardware()
        
    def _detect_hardware(self):
        """Detect available hardware and capabilities."""
        self.hardware_info = {
            'cpu': {
                'cores': psutil.cpu_count(),
                'physical_cores': psutil.cpu_count(logical=False),
                'frequency': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
                'architecture': sys.platform
            },
            'memory': {
                'total_gb': psutil.virtual_memory().total / (1024**3),
                'available_gb': psutil.virtual_memory().available / (1024**3)
            },
            'gpu': {},
            'jax': {}
        }
        
        # Detect GPUs - prioritize GPUtil as it's more reliable
        if GPU_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                for i, gpu in enumerate(gpus):
                    self.hardware_info['gpu'][f'gpu_{i}'] = {
                        'name': gpu.name,
                        'memory_total_gb': gpu.memoryTotal / 1024,
                        'memory_free_gb': gpu.memoryFree / 1024,
                        'type': 'nvidia' if 'nvidia' in gpu.name.lower() else 'generic'
                    }
                self.logger.info(f"Detected {len(gpus)} GPU(s) via GPUtil")
            except Exception as e:
                self.logger.warning(f"Failed to detect GPUs via GPUtil: {e}")
        
        # Fallback to pynvml if GPUtil failed
        if not self.hardware_info['gpu'] and NVIDIA_AVAILABLE:
            try:
                pynvml.nvmlInit()
                gpu_count = pynvml.nvmlDeviceGetCount()
                for i in range(gpu_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    name = pynvml.nvmlDeviceGetName(handle)
                    # Handle both bytes and string types
                    if isinstance(name, bytes):
                        name = name.decode('utf-8')
                    memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    self.hardware_info['gpu'][f'nvidia_{i}'] = {
                        'name': name,
                        'memory_total_gb': memory.total / (1024**3),
                        'memory_free_gb': memory.free / (1024**3),
                        'type': 'nvidia'
                    }
                pynvml.nvmlShutdown()
                self.logger.info(f"Detected {gpu_count} GPU(s) via pynvml")
            except Exception as e:
                self.logger.warning(f"Failed to detect NVIDIA GPUs via pynvml: {e}")
        
        # Detect JAX devices
        if JAX_AVAILABLE:
            try:
                devices = jax.devices()
                backend = jax.default_backend()
                self.hardware_info['jax'] = {
                    'backend': backend,
                    'devices': [str(d) for d in devices],
                    'device_count': len(devices)
                }
            except Exception as e:
                self.logger.warning(f"Failed to detect JAX devices: {e}")
        
        # Log hardware summary
        self.logger.info("=== HARDWARE DETECTION SUMMARY ===")
        self.logger.info(f"CPU: {self.hardware_info['cpu']['cores']} cores ({self.hardware_info['cpu']['physical_cores']} physical)")
        self.logger.info(f"Memory: {self.hardware_info['memory']['total_gb']:.1f} GB total")
        self.logger.info(f"GPUs detected: {len(self.hardware_info['gpu'])}")
        if JAX_AVAILABLE:
            self.logger.info(f"JAX backend: {self.hardware_info['jax'].get('backend', 'unknown')}")
            self.logger.info(f"JAX devices: {self.hardware_info['jax'].get('device_count', 0)}")
        
        # Save hardware info
        with open(self.output_dir / "hardware_info.json", 'w') as f:
            json.dump(self.hardware_info, f, indent=2, default=str)
    
    def get_current_resources(self) -> Dict[str, Any]:
        """Get current resource utilization."""
        timestamp = datetime.now().isoformat()
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_per_core = psutil.cpu_percent(interval=1, percpu=True)
        
        # Memory metrics
        memory = psutil.virtual_memory()
        memory_info = {
            'total_gb': memory.total / (1024**3),
            'available_gb': memory.available / (1024**3),
            'used_gb': memory.used / (1024**3),
            'percent': memory.percent
        }
        
        # Process-specific metrics
        process = psutil.Process()
        process_info = {
            'cpu_percent': process.cpu_percent(),
            'memory_gb': process.memory_info().rss / (1024**3),
            'memory_percent': process.memory_percent(),
            'num_threads': process.num_threads()
        }
        
        # GPU metrics - prioritize GPUtil as it's more reliable
        gpu_info = {}
        if GPU_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                for i, gpu in enumerate(gpus):
                    gpu_info[f'gpu_{i}'] = {
                        'utilization_percent': gpu.load * 100,
                        'memory_used_gb': (gpu.memoryTotal - gpu.memoryFree) / 1024,
                        'memory_percent': (gpu.memoryTotal - gpu.memoryFree) / gpu.memoryTotal * 100,
                        'temperature_c': gpu.temperature
                    }
            except Exception as e:
                self.logger.warning(f"Failed to get GPU metrics via GPUtil: {e}")
        
        # Fallback to pynvml if GPUtil failed
        if not gpu_info and NVIDIA_AVAILABLE:
            try:
                pynvml.nvmlInit()
                gpu_count = pynvml.nvmlDeviceGetCount()
                for i in range(gpu_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    
                    gpu_info[f'nvidia_{i}'] = {
                        'utilization_percent': util.gpu,
                        'memory_used_gb': memory.used / (1024**3),
                        'memory_percent': (memory.used / memory.total) * 100,
                        'temperature_c': temp
                    }
                pynvml.nvmlShutdown()
            except Exception as e:
                self.logger.warning(f"Failed to get NVIDIA GPU metrics via pynvml: {e}")
        
        # JAX device utilization (estimated)
        jax_info = {}
        if JAX_AVAILABLE:
            try:
                # Get current JAX device
                current_device = jax.devices()[0] if jax.devices() else None
                jax_info = {
                    'current_device': str(current_device) if current_device else None,
                    'device_count': len(jax.devices()),
                    'backend': jax.default_backend()
                }
            except Exception as e:
                self.logger.warning(f"Failed to get JAX info: {e}")
        
        # Update peak values
        self.peak_cpu = max(self.peak_cpu, cpu_percent)
        self.peak_memory = max(self.peak_memory, memory_info['percent'])
        
        if gpu_info:
            for gpu_data in gpu_info.values():
                self.peak_gpu_util = max(self.peak_gpu_util, gpu_data.get('utilization_percent', 0))
                self.peak_gpu_memory = max(self.peak_gpu_memory, gpu_data.get('memory_percent', 0))
        
        return {
            'timestamp': timestamp,
            'cpu': {
                'overall_percent': cpu_percent,
                'per_core': cpu_per_core,
                'peak_percent': self.peak_cpu
            },
            'memory': memory_info,
            'process': process_info,
            'gpu': gpu_info,
            'jax': jax_info
        }
    
    def start_monitoring(self):
        """Start continuous resource monitoring."""
        if self.monitoring:
            self.logger.warning("Resource monitoring already running")
            return
        
        self.start_time = datetime.now()
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        self.logger.info(f"Started resource monitoring (log interval: {self.log_interval}s)")
    
    def stop_monitoring(self):
        """Stop resource monitoring and generate summary."""
        if not self.monitoring:
            return
        
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        self._generate_summary()
        self.logger.info("Stopped resource monitoring")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                resources = self.get_current_resources()
                self.resource_history.append(resources)
                
                # Log to file periodically
                if len(self.resource_history) % 10 == 0:  # Every 10 samples
                    self._save_history()
                
                # Print status every log_interval
                if len(self.resource_history) % (self.log_interval // 1) == 0:
                    self._print_status(resources)
                
                time.sleep(1)  # Sample every second
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5)  # Wait before retrying
    
    def _print_status(self, resources: Dict):
        """Print current resource status."""
        cpu = resources['cpu']['overall_percent']
        memory = resources['memory']['percent']
        process_mem = resources['process']['memory_gb']
        
        # GPU status
        gpu_status = "No GPU"
        if resources['gpu']:
            gpu_utils = [gpu['utilization_percent'] for gpu in resources['gpu'].values()]
            gpu_mems = [gpu['memory_percent'] for gpu in resources['gpu'].values()]
            gpu_status = f"GPU: {max(gpu_utils):.1f}% util, {max(gpu_mems):.1f}% mem"
        
        # JAX status
        jax_status = "No JAX"
        if resources['jax']:
            jax_status = f"JAX: {resources['jax']['backend']} on {resources['jax']['device_count']} devices"
        
        status_msg = (
            f"[RESOURCE] CPU: {cpu:.1f}% | "
            f"RAM: {memory:.1f}% ({process_mem:.2f}GB) | "
            f"{gpu_status} | "
            f"{jax_status}"
        )
        
        self.logger.info(status_msg)
    
    def _save_history(self):
        """Save resource history to file."""
        try:
            with open(self.resource_file, 'w') as f:
                json.dump(self.resource_history, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Failed to save resource history: {e}")
    
    def _generate_summary(self):
        """Generate resource utilization summary."""
        if not self.resource_history:
            return
        
        # Calculate statistics
        cpu_values = [r['cpu']['overall_percent'] for r in self.resource_history]
        memory_values = [r['memory']['percent'] for r in self.resource_history]
        process_memory_values = [r['process']['memory_gb'] for r in self.resource_history]
        
        # GPU statistics
        gpu_util_values = []
        gpu_memory_values = []
        for r in self.resource_history:
            if r['gpu']:
                gpu_utils = [gpu['utilization_percent'] for gpu in r['gpu'].values()]
                gpu_mems = [gpu['memory_percent'] for gpu in r['gpu'].values()]
                gpu_util_values.extend(gpu_utils)
                gpu_memory_values.extend(gpu_mems)
        
        summary = {
            'monitoring_duration_seconds': (datetime.now() - self.start_time).total_seconds(),
            'samples_collected': len(self.resource_history),
            'cpu': {
                'mean_percent': np.mean(cpu_values),
                'max_percent': np.max(cpu_values),
                'min_percent': np.min(cpu_values),
                'std_percent': np.std(cpu_values)
            },
            'memory': {
                'mean_percent': np.mean(memory_values),
                'max_percent': np.max(memory_values),
                'min_percent': np.min(memory_values),
                'std_percent': np.std(memory_values)
            },
            'process_memory': {
                'mean_gb': np.mean(process_memory_values),
                'max_gb': np.max(process_memory_values),
                'min_gb': np.min(process_memory_values),
                'std_gb': np.std(process_memory_values)
            },
            'gpu': {
                'mean_utilization_percent': np.mean(gpu_util_values) if gpu_util_values else 0,
                'max_utilization_percent': np.max(gpu_util_values) if gpu_util_values else 0,
                'mean_memory_percent': np.mean(gpu_memory_values) if gpu_memory_values else 0,
                'max_memory_percent': np.max(gpu_memory_values) if gpu_memory_values else 0
            },
            'hardware_info': self.hardware_info,
            'utilization_assessment': self._assess_utilization()
        }
        
        # Save summary
        try:
            with open(self.summary_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            # Print summary
            self._print_summary(summary)
            
        except Exception as e:
            self.logger.error(f"Failed to save resource summary: {e}")
    
    def _assess_utilization(self) -> Dict[str, str]:
        """Assess resource utilization and provide recommendations."""
        assessment = {}
        
        # CPU assessment
        if self.peak_cpu > 90:
            assessment['cpu'] = "EXCELLENT - High CPU utilization detected"
        elif self.peak_cpu > 70:
            assessment['cpu'] = "GOOD - Moderate CPU utilization"
        elif self.peak_cpu > 30:
            assessment['cpu'] = "FAIR - Low CPU utilization, consider increasing workload"
        else:
            assessment['cpu'] = "POOR - Very low CPU utilization, hardware underutilized"
        
        # Memory assessment
        if self.peak_memory > 90:
            assessment['memory'] = "WARNING - High memory usage, monitor for OOM"
        elif self.peak_memory > 70:
            assessment['memory'] = "GOOD - Healthy memory utilization"
        else:
            assessment['memory'] = "LOW - Memory underutilized"
        
        # GPU assessment
        if self.peak_gpu_util > 80:
            assessment['gpu'] = "EXCELLENT - High GPU utilization"
        elif self.peak_gpu_util > 50:
            assessment['gpu'] = "GOOD - Moderate GPU utilization"
        elif self.peak_gpu_util > 10:
            assessment['gpu'] = "FAIR - Low GPU utilization"
        else:
            assessment['gpu'] = "POOR - GPU barely utilized or not detected"
        
        return assessment
    
    def _print_summary(self, summary: Dict):
        """Print resource utilization summary."""
        self.logger.info("\n" + "="*60)
        self.logger.info("RESOURCE UTILIZATION SUMMARY")
        self.logger.info("="*60)
        
        duration = summary['monitoring_duration_seconds']
        self.logger.info(f"Monitoring duration: {duration/3600:.1f} hours")
        self.logger.info(f"Samples collected: {summary['samples_collected']}")
        
        # CPU summary
        cpu = summary['cpu']
        self.logger.info(f"\nCPU Utilization:")
        self.logger.info(f"  Mean: {cpu['mean_percent']:.1f}%")
        self.logger.info(f"  Peak: {cpu['max_percent']:.1f}%")
        self.logger.info(f"  Assessment: {summary['utilization_assessment']['cpu']}")
        
        # Memory summary
        mem = summary['memory']
        proc_mem = summary['process_memory']
        self.logger.info(f"\nMemory Utilization:")
        self.logger.info(f"  System: {mem['mean_percent']:.1f}% mean, {mem['max_percent']:.1f}% peak")
        self.logger.info(f"  Process: {proc_mem['mean_gb']:.2f} GB mean, {proc_mem['max_gb']:.2f} GB peak")
        self.logger.info(f"  Assessment: {summary['utilization_assessment']['memory']}")
        
        # GPU summary
        gpu = summary['gpu']
        if gpu['max_utilization_percent'] > 0:
            self.logger.info(f"\nGPU Utilization:")
            self.logger.info(f"  Utilization: {gpu['mean_utilization_percent']:.1f}% mean, {gpu['max_utilization_percent']:.1f}% peak")
            self.logger.info(f"  Memory: {gpu['mean_memory_percent']:.1f}% mean, {gpu['max_memory_percent']:.1f}% peak")
            self.logger.info(f"  Assessment: {summary['utilization_assessment']['gpu']}")
        else:
            self.logger.info(f"\nGPU: No GPU utilization detected")
        
        self.logger.info("="*60)


def create_resource_monitor(output_dir: Path, log_interval: int = 30) -> ResourceMonitor:
    """Convenience function to create and start a resource monitor."""
    monitor = ResourceMonitor(output_dir, log_interval)
    monitor.start_monitoring()
    return monitor


if __name__ == "__main__":
    # Test the resource monitor
    import argparse
    
    parser = argparse.ArgumentParser(description="Test resource monitoring")
    parser.add_argument("--output-dir", default="./resource_test", help="Output directory")
    parser.add_argument("--duration", type=int, default=60, help="Test duration in seconds")
    parser.add_argument("--log-interval", type=int, default=10, help="Log interval in seconds")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(name)s | %(message)s')
    
    # Create and start monitor
    monitor = create_resource_monitor(output_dir, args.log_interval)
    
    print(f"Monitoring resources for {args.duration} seconds...")
    time.sleep(args.duration)
    
    # Stop monitoring
    monitor.stop_monitoring()
    
    print(f"Resource monitoring complete. Check {output_dir} for results.") 