# kids_loader.py
import numpy as np
from astropy.io import fits
from pathlib import Path

class KiDSLoader:
    """Load KiDS DR4.1 weak lensing catalog"""
    
    def __init__(self, kids_dir: str = 'Kids'):
        self.kids_dir = Path(kids_dir)
        self.catalog = None
        
    def load_wl_catalog(self):
        """Load the weak lensing shape catalog"""
        cat_file = self.kids_dir / 'KiDS_DR4.1_ugriZYJHKs_SOM_gold_WL_cat.fits'
        
        if not cat_file.exists():
            raise FileNotFoundError(f"KiDS catalog not found: {cat_file}")
        
        print(f"Loading KiDS catalog (this may take a moment)...")
        with fits.open(cat_file) as hdul:
            # Print structure
            # print("KiDS catalog structure:")
            # hdul.info()
            
            # Load the main catalog
            self.catalog = hdul[1].data
            
            # Print available columns
            print(f"\nAvailable columns ({len(hdul[1].columns)} total):")
            important_cols = ['e1', 'e2', 'weight', 'Z_B', 'MAG_GAAP_u', 
                            'MAG_GAAP_r', 'ALPHA_J2000', 'DELTA_J2000']
            for col in important_cols:
                if col in hdul[1].columns.names:
                    print(f"  ✓ {col}")
            
            print(f"\nLoaded {len(self.catalog)} galaxies")
        
        return self.catalog
    
    def get_shear_statistics(self, n_bins: int = 10):
        """Calculate shear 2-point statistics"""
        if self.catalog is None:
            self.load_wl_catalog()
        
        # Get ellipticities
        e1 = self.catalog['e1']
        e2 = self.catalog['e2']
        weights = self.catalog['weight']
        
        # Photo-z
        z_phot = self.catalog['Z_B']
        
        # Binning in redshift
        z_bins = np.linspace(0.1, 1.2, n_bins + 1)
        
        stats = []
        for i in range(n_bins):
            mask = (z_phot > z_bins[i]) & (z_phot <= z_bins[i+1])
            if np.sum(mask) > 100:
                stats.append({
                    'z_mean': np.mean(z_phot[mask]),
                    'n_gal': np.sum(mask),
                    'e1_mean': np.average(e1[mask], weights=weights[mask]),
                    'e2_mean': np.average(e2[mask], weights=weights[mask]),
                    'sigma_e': np.sqrt(np.average((e1[mask]**2 + e2[mask]**2), 
                                                 weights=weights[mask]))
                })
        
        return stats