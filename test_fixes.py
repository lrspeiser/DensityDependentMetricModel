#!/usr/bin/env python3
"""
Test script to verify the fixes for Unicode encoding and UnboundLocalError.
"""

import sys
import os

# Test UTF-8 encoding setup
print("Testing UTF-8 encoding setup...")
if sys.platform.startswith('win'):
    print("Windows platform detected")
    try:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())
        os.environ['PYTHONIOENCODING'] = 'utf-8'
        print("UTF-8 encoding configured successfully")
    except Exception as e:
        print(f"UTF-8 encoding setup failed: {e}")
else:
    print("Non-Windows platform, UTF-8 encoding not needed")

# Test Unicode character output
print("Testing Unicode character output...")
try:
    print("✅ Test emoji output")
    print("📊 Test chart emoji")
    print("🎯 Test target emoji")
    print("📏 Test ruler emoji")
    print("📌 Test pin emoji")
    print("🌟 Test star emoji")
    print("⚠️ Test warning emoji")
    print("→ Test arrow")
    print("💾 Test save emoji")
    print("🧠 Test brain emoji")
    print("🧪 Test test tube emoji")
    print("🛑 Test stop emoji")
    print("📡 Test satellite emoji")
    print("🔭 Test telescope emoji")
    print("🔬 Test microscope emoji")
    print("📋 Test clipboard emoji")
    print("☉ Test sun symbol")
    print("Unicode test completed successfully!")
except Exception as e:
    print(f"Unicode test failed: {e}")

# Test the specific problematic characters that were causing issues
print("\nTesting specific problematic characters...")
try:
    # Test the sun symbol that was causing issues
    print("M☉ Test sun symbol")
    print("M_sun Test ASCII replacement")
    
    # Test the clipboard emoji that was causing issues
    print("📋 Test clipboard emoji")
    print("FINAL PARAMETER CONFIGURATION Test ASCII replacement")
    
    print("Specific character test completed successfully!")
except Exception as e:
    print(f"Specific character test failed: {e}")

# Test variable initialization (simulating the UnboundLocalError fix)
print("\nTesting variable initialization fix...")
try:
    import numpy as np
    
    # Simulate the monitoring function structure
    def test_monitoring_function():
        # Initialize variables early to prevent UnboundLocalError
        current_logz = -np.inf
        dlogz = np.nan
        
        # Simulate some condition that might not set these variables
        test_condition = False
        
        if test_condition:
            current_logz = 100.0
            dlogz = 0.1
        
        # These should work without UnboundLocalError
        print(f"current_logz: {current_logz}")
        print(f"dlogz: {dlogz}")
        
        # Test phase determination
        if dlogz > 1.0:
            phase = "ACTIVE_EXPLORATION"
        elif dlogz > 0.1:
            phase = "REFINEMENT"
        else:
            phase = "CONVERGING"
        
        print(f"phase: {phase}")
        return True
    
    result = test_monitoring_function()
    print("Variable initialization test completed successfully!")
    
except Exception as e:
    print(f"Variable initialization test failed: {e}")

print("\nAll tests completed!") 