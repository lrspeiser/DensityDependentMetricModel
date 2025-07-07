#!/usr/bin/env python3
"""
fix_gaia_timeout.py - Debug and fix Gaia query timeout issues
"""
import time
from astroquery.gaia import Gaia
import warnings
warnings.filterwarnings('ignore')

def test_gaia_with_timeout():
    """Test Gaia with different timeout settings"""
    
    print("Testing Gaia TAP service...")
    
    # Check astroquery version and timeout handling
    print(f"Astroquery Gaia module: {Gaia}")
    
    # Different versions of astroquery handle timeouts differently
    # Try to set timeout in different ways
    try:
        # Newer versions
        if hasattr(Gaia, 'TIMEOUT'):
            original_timeout = Gaia.TIMEOUT
            Gaia.TIMEOUT = 600
            print(f"Set Gaia.TIMEOUT to {Gaia.TIMEOUT} seconds")
        else:
            print("Gaia.TIMEOUT not found, checking for alternative methods...")
            
        # Check if we can set on the TAP service
        if hasattr(Gaia, 'tap') and hasattr(Gaia.tap, 'timeout'):
            original_timeout = Gaia.tap.timeout
            Gaia.tap.timeout = 600
            print(f"Set Gaia.tap.timeout to {Gaia.tap.timeout} seconds")
        else:
            print("Using default timeout settings")
            original_timeout = None
            
    except Exception as e:
        print(f"Could not modify timeout: {e}")
        original_timeout = None
    
    # Test with a tiny query first
    test_query = """
    SELECT TOP 10 
        source_id, ra, dec, parallax
    FROM gaiadr3.gaia_source
    WHERE parallax > 10
    """
    
    try:
        print("\n1. Testing basic connectivity with 10 stars...")
        start = time.time()
        job = Gaia.launch_job(test_query)
        results = job.get_results()
        elapsed = time.time() - start
        print(f"✅ Success! Got {len(results)} rows in {elapsed:.1f} seconds")
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False
    
    # Try a simpler query for 1000 stars
    simple_query_1000 = """
    SELECT TOP 1000
        source_id,
        l, b,
        parallax, parallax_error,
        pmra, pmra_error,
        pmdec, pmdec_error,
        radial_velocity, radial_velocity_error,
        phot_g_mean_mag
    FROM gaiadr3.gaia_source
    WHERE l BETWEEN 60 AND 120
        AND b BETWEEN -3 AND 3
        AND parallax > 0.1
        AND parallax_error < 0.1 * parallax
        AND radial_velocity IS NOT NULL
        AND radial_velocity_error < 10
        AND radial_velocity_error > 0
        AND phot_g_mean_mag < 18
    ORDER BY source_id
    """
    
    try:
        print("\n2. Testing with 1000 stars (simplified query)...")
        start = time.time()
        
        # Use launch_job_async for better control
        job = Gaia.launch_job_async(simple_query_1000)
        print("   Query launched, waiting for completion...")
        
        # Monitor progress with timeout
        max_wait = 300  # 5 minutes
        check_interval = 5
        elapsed = 0
        
        while elapsed < max_wait:
            phase = job.get_phase()
            if phase == 'COMPLETED':
                break
            elif phase == 'ERROR':
                print(f"\n❌ Query failed with phase: {phase}")
                return False
            
            elapsed = time.time() - start
            print(f"   Status: {phase} ({elapsed:.0f}s elapsed)", end='\r')
            time.sleep(check_interval)
        
        if job.get_phase() != 'COMPLETED':
            print(f"\n❌ Query timed out after {elapsed:.0f} seconds")
            return False
            
        results = job.get_results()
        elapsed = time.time() - start
        print(f"\n✅ Success! Got {len(results)} rows in {elapsed:.1f} seconds")
        
        # Convert and process
        df = results.to_pandas()
        
        # Calculate simple derived quantities for testing
        # Convert to galactocentric cylindrical coordinates (simplified)
        R_sun = 8.0  # kpc
        v_sun = 220.0  # km/s
        
        # Simple distance from parallax
        df['distance_kpc'] = 1.0 / df['parallax']  # parallax in mas
        
        # Very simplified R calculation (just for testing)
        df['R_kpc'] = R_sun  # Assume everything at solar radius for now
        df['v_obs'] = df['radial_velocity'] + v_sun  # Simplified!
        df['sigma_v'] = df['radial_velocity_error']
        
        # Filter to reasonable values
        mask = (df['distance_kpc'] > 0.1) & (df['distance_kpc'] < 20) & (df['sigma_v'] < 50)
        df_clean = df[mask].copy()
        
        print(f"   After filtering: {len(df_clean)} stars")
        
        # Save processed data
        output_file = 'gaia_query_cache_DR3_processed_for_fit.parquet'
        df_clean[['R_kpc', 'v_obs', 'sigma_v']].to_parquet(output_file)
        print(f"✅ Saved to {output_file}")
        
    except Exception as e:
        print(f"\n❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        
    print("\n" + "="*50)
    print("If queries are still timing out, try:")
    print("1. Using a VPN to connect from a different location")
    print("2. Running during off-peak hours (not European daytime)")
    print("3. Using the synthetic data generator instead")
    print("="*50)

if __name__ == "__main__":
    test_gaia_with_timeout()