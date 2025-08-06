#!/usr/bin/env python3
# save as: monitor_run.py
import subprocess
import sys
import time
import psutil
import os
from datetime import datetime

def monitor_dynesty_run(cmd_args):
    """Monitor a dynesty run and log why it stopped."""
    
    log_file = f"run_monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    with open(log_file, 'w') as log:
        log.write(f"Starting monitored run at {datetime.now()}\n")
        log.write(f"Command: {' '.join(cmd_args)}\n\n")
        
        # Start the process
        process = subprocess.Popen(
            cmd_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1
        )
        
        # Monitor memory usage
        proc_info = psutil.Process(process.pid)
        max_memory = 0
        
        while process.poll() is None:
            try:
                # Check memory
                mem_info = proc_info.memory_info()
                current_mem_gb = mem_info.rss / (1024**3)
                max_memory = max(max_memory, current_mem_gb)
                
                # Log every 60 seconds
                if int(time.time()) % 60 == 0:
                    cpu_percent = proc_info.cpu_percent(interval=1)
                    log.write(f"[{datetime.now()}] CPU: {cpu_percent:.1f}%, Memory: {current_mem_gb:.2f} GB\n")
                    log.flush()
                
                time.sleep(1)
                
            except psutil.NoSuchProcess:
                break
        
        # Get exit code and final output
        stdout, stderr = process.communicate()
        exit_code = process.returncode
        
        log.write(f"\n=== PROCESS ENDED ===\n")
        log.write(f"Exit code: {exit_code}\n")
        log.write(f"Max memory used: {max_memory:.2f} GB\n")
        log.write(f"\nFinal stdout:\n{stdout[-5000:]}\n")  # Last 5000 chars
        log.write(f"\nFinal stderr:\n{stderr}\n")
        
        # Interpret exit code
        if exit_code == 0:
            log.write("✅ Normal completion\n")
        elif exit_code == -9 or exit_code == 137:
            log.write("❌ KILLED by system (likely OOM - Out of Memory)\n")
        elif exit_code == -15 or exit_code == 143:
            log.write("❌ TERMINATED by SIGTERM (walltime limit or manual kill)\n")
        elif exit_code == 1:
            log.write("❌ Python exception (check stderr above)\n")
        else:
            log.write(f"❌ Abnormal exit with code {exit_code}\n")
    
    print(f"Monitor log saved to: {log_file}")
    return exit_code

if __name__ == "__main__":
    exit_code = monitor_dynesty_run(sys.argv[1:])
    sys.exit(exit_code)