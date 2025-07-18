import pandas as pd
import numpy as np

# Load the data
df = pd.read_parquet('gaia_query_cache_DR3_processed_for_fit.parquet')

# Check what columns we have
print("Current columns:", df.columns.tolist())

# If phi_rad is missing, we can calculate it or skip that check
if 'phi_rad' not in df.columns:
    print("phi_rad missing - this validation will be skipped")
    # Could calculate it from X_gc, Y_gc if needed:
    # df['phi_rad'] = np.arctan2(df['Y_gc_kpc'], df['X_gc_kpc'])
