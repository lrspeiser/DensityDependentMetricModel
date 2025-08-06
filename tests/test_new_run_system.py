#!/usr/bin/env python3
"""
Test script to demonstrate the new run system with unique folders
"""

import subprocess
import sys
from pathlib import Path

def test_new_run_system():
    """Test the new run system with different parameters"""
    
    print("=== TESTING NEW RUN SYSTEM ===")
    print("This will create unique folders for each run with:")
    print("- CLI command stored in cli_command.txt")
    print("- Run stats saved every 60 seconds")
    print("- Progress monitoring")
    print()
    
    # Test commands
    test_commands = [
        # GR Baseline with expanded bounds
        "py run_dynesty_cupy.py --xi gr --nlive 1000 --maxcall 500000 --dlogz_target 0.01",
        
        # Enhanced model
        "py run_dynesty_cupy.py --xi enhanced --nlive 1000 --maxcall 500000 --dlogz_target 0.01",
        
        # Power law model
        "py run_dynesty_cupy.py --xi power --nlive 1000 --maxcall 500000 --dlogz_target 0.01",
        
        # Gravitational color model
        "py run_dynesty_cupy.py --xi grav_color --nlive 1000 --maxcall 500000 --dlogz_target 0.01",
        
        # Quick test run
        "py run_dynesty_cupy.py --xi gr --nlive 100 --maxcall 10000 --dlogz_target 0.1"
    ]
    
    print("Available test commands:")
    for i, cmd in enumerate(test_commands, 1):
        print(f"{i}. {cmd}")
    
    print("\nEach run will create a folder like:")
    print("runs/gr_20250804_143022/")
    print("runs/enhanced_20250804_143045/")
    print("etc.")
    
    print("\nFiles created in each run folder:")
    print("- cli_command.txt (original command)")
    print("- run_stats.json (current stats)")
    print("- run_stats_history.json (all stats)")
    print("- dynesty_progress.json (detailed progress)")
    print("- posterior_samples.npz (final results)")
    print("- dynesty_checkpoint.pkl (checkpoint)")
    print("- resource_usage.json (hardware monitoring)")
    
    print("\nTo run a test, choose a command number or enter 'q' to quit:")
    
    try:
        choice = input("Enter choice (1-5, or 'q'): ").strip()
        
        if choice.lower() == 'q':
            print("Exiting...")
            return
        
        choice_num = int(choice)
        if 1 <= choice_num <= len(test_commands):
            selected_cmd = test_commands[choice_num - 1]
            print(f"\nRunning: {selected_cmd}")
            print("Press Ctrl+C to stop the run")
            print()
            
            # Run the command
            subprocess.run(selected_cmd.split(), check=True)
            
        else:
            print("Invalid choice!")
            
    except KeyboardInterrupt:
        print("\nRun interrupted by user")
    except ValueError:
        print("Invalid input!")
    except subprocess.CalledProcessError as e:
        print(f"Command failed with error: {e}")

def show_folder_structure():
    """Show the expected folder structure"""
    
    print("\n=== EXPECTED FOLDER STRUCTURE ===")
    print("After running, you'll have:")
    print()
    print("runs/")
    print("├── gr_20250804_143022/")
    print("│   ├── cli_command.txt")
    print("│   ├── run_stats.json")
    print("│   ├── run_stats_history.json")
    print("│   ├── dynesty_progress.json")
    print("│   ├── posterior_samples.npz")
    print("│   ├── dynesty_checkpoint.pkl")
    print("│   ├── resource_usage.json")
    print("│   └── hardware_info.json")
    print("├── enhanced_20250804_143045/")
    print("│   └── [same files]")
    print("├── power_20250804_143108/")
    print("│   └── [same files]")
    print("└── grav_color_20250804_143131/")
    print("    └── [same files]")
    print()

if __name__ == "__main__":
    show_folder_structure()
    test_new_run_system() 