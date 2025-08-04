#!/usr/bin/env python3
"""
Test script for resource monitoring functionality
"""

import time
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(name)s | %(message)s')
logger = logging.getLogger(__name__)

def test_resource_monitor():
    """Test the resource monitoring functionality."""
    
    # Create test output directory
    output_dir = Path("./test_resource_monitor")
    output_dir.mkdir(exist_ok=True)
    
    logger.info("Testing resource monitoring...")
    
    try:
        from resource_monitor import ResourceMonitor
        
        # Create and start monitor
        monitor = ResourceMonitor(output_dir, log_interval=5)
        monitor.start_monitoring()
        
        logger.info("Resource monitoring started. Running for 30 seconds...")
        
        # Simulate some CPU work
        start_time = time.time()
        while time.time() - start_time < 30:
            # Do some CPU-intensive work
            _ = sum(i**2 for i in range(10000))
            time.sleep(0.1)
        
        # Stop monitoring
        monitor.stop_monitoring()
        
        logger.info("Resource monitoring test completed!")
        logger.info(f"Check {output_dir} for resource usage files:")
        logger.info(f"  - resource_usage.json: Detailed resource history")
        logger.info(f"  - resource_summary.json: Utilization summary")
        logger.info(f"  - hardware_info.json: Hardware detection results")
        
        # Check if files were created
        files_created = []
        for file_name in ["resource_usage.json", "resource_summary.json", "hardware_info.json"]:
            file_path = output_dir / file_name
            if file_path.exists():
                files_created.append(file_name)
                logger.info(f"✓ {file_name} created successfully")
            else:
                logger.warning(f"✗ {file_name} not found")
        
        if len(files_created) == 3:
            logger.info("SUCCESS: All resource monitoring files created!")
            return True
        else:
            logger.warning(f"PARTIAL: Only {len(files_created)}/3 files created")
            return False
            
    except ImportError as e:
        logger.error(f"Failed to import resource_monitor: {e}")
        logger.info("Make sure you have the required dependencies:")
        logger.info("  pip install psutil")
        logger.info("  pip install GPUtil  # for GPU monitoring")
        logger.info("  pip install pynvml  # for NVIDIA GPU monitoring")
        return False
    except Exception as e:
        logger.error(f"Resource monitoring test failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = test_resource_monitor()
    if success:
        print("\n✅ Resource monitoring test PASSED!")
    else:
        print("\n❌ Resource monitoring test FAILED!")
        print("Check the logs above for details.") 