
# Monitor what the data is telling us
import pandas as pd
import numpy as np

# Load the actual data being fitted
data = pd.read_parquet('gaia_query_cache_DR3_processed_for_fit.parquet')

print(f"Data characteristics:")
print(f"  N_stars: {len(data):,}")
print(f"  R range: {data['R_kpc'].min():.1f} - {data['R_kpc'].max():.1f} kpc")
print(f"  <v(R=8kpc)>: {data[np.abs(data['R_kpc']-8)<0.5]['v_obs'].mean():.1f} km/s")
print(f"\nLet the data speak!")
