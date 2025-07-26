import sys
import pickle
import numpy as np
from pathlib import Path

# Import the necessary functions from run_dynesty
sys.path.insert(0, '.')  # Ensure current directory is in path
from run_dynesty import log_likelihood_dynesty, prior_transform_dynesty

checkpoint_path = Path("chains_DDMM_power_best_fit/dynesty_checkpoint.pkl")

print(f"Loading checkpoint from: {checkpoint_path}")
print("=" * 80)

try:
    with open(checkpoint_path, 'rb') as f:
        checkpoint_data = pickle.load(f)
    
    print("✅ Checkpoint loaded successfully!")
    
    if 'sampler' in checkpoint_data:
        sampler = checkpoint_data['sampler']
        print(f"\n📊 Sampler type: {type(sampler).__name__}")
        
        # Check for _run_config
        if hasattr(sampler, '_run_config'):
            print("\n✅ Found _run_config:")
            for key, value in sampler._run_config.items():
                print(f"  {key}: {value}")
        else:
            print("\n⚠️  No _run_config found")
        
        # Check for results
        if hasattr(sampler, 'results'):
            res = sampler.results
            print("\n✅ Found results object")
            
            if hasattr(res, 'samples') and res.samples is not None:
                print(f"  Samples shape: {res.samples.shape}")
                print(f"  Number of samples: {len(res.samples)}")
                print(f"  Number of parameters: {res.samples.shape[1]}")
            else:
                print("  ⚠️  No samples in results")
                
            if hasattr(res, 'logz') and res.logz is not None:
                if hasattr(res.logz, '__len__'):
                    print(f"  LogZ iterations: {len(res.logz)}")
                    if len(res.logz) > 0:
                        print(f"  Current logZ: {res.logz[-1]:.3f}")
                else:
                    print(f"  LogZ: {res.logz:.3f}")
        else:
            print("\n⚠️  No results attribute")
        
        # Check sampler state
        print("\n🔍 Sampler state:")
        state_attrs = ['it', 'ncall', 'nbound', 'added_live', 'nlive', 'batch']
        for attr in state_attrs:
            if hasattr(sampler, attr):
                val = getattr(sampler, attr)
                if isinstance(val, (int, float)):
                    print(f"  {attr}: {val}")
                elif hasattr(val, '__len__'):
                    print(f"  {attr}: length {len(val)}")
        
        # Check for parameter info
        if hasattr(sampler, 'ptform_args') and sampler.ptform_args:
            print("\n📋 Prior transform args:")
            if len(sampler.ptform_args) > 0:
                print(f"  fitted_param_names: {sampler.ptform_args[0]}")
                
        if hasattr(sampler, 'logl_args') and sampler.logl_args:
            print("\n📋 Likelihood args:")
            if len(sampler.logl_args) > 0:
                print(f"  fitted_param_names: {sampler.logl_args[0]}")
        
        # Check saved arrays
        print("\n💾 Saved data:")
        saved_attrs = ['saved_u', 'saved_v', 'saved_logl', 'saved_logz']
        for attr in saved_attrs:
            if hasattr(sampler, attr):
                val = getattr(sampler, attr)
                if hasattr(val, 'shape'):
                    print(f"  {attr}: shape {val.shape}")
                elif hasattr(val, '__len__'):
                    print(f"  {attr}: length {len(val)}")
    
    print(f"\n📁 Checkpoint file size: {checkpoint_path.stat().st_size/1024:.1f} KB")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)