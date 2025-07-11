# des_y3_loader.py
import numpy as np
from astropy.io import fits
from pathlib import Path

class DESY3Loader:
    """Load DES Y3 2-point correlation functions"""
    
    def __init__(self, des_dir: str = 'DES_Y3'):
        self.des_dir = Path(des_dir)
        self.data = {}
        
    def load_2pt_data(self):
        """Load the 2-point correlation function data"""
        twopoint_file = self.des_dir / '2pt_NG_final_2ptunblind_02_26_21_wnz_maglim_covupdate.fits'
        
        if not twopoint_file.exists():
            raise FileNotFoundError(f"DES Y3 2pt file not found: {twopoint_file}")
        
        with fits.open(twopoint_file) as hdul:
            # Print structure to understand the file
            print("DES Y3 file structure:")
            hdul.info()
            
            # Usually contains:
            # - xi_plus: ξ₊(θ) correlation function
            # - xi_minus: ξ₋(θ) correlation function  
            # - gammat: galaxy-galaxy lensing
            # - wtheta: galaxy clustering
            # - covariance matrix
            
            self.data = {}
            for hdu in hdul[1:]:  # Skip primary HDU
                name = hdu.name
                self.data[name] = {
                    'data': hdu.data,
                    'columns': hdu.columns.names if hasattr(hdu, 'columns') else []
                }
                print(f"  Extension '{name}': {len(hdu.data)} rows")
        
        return self.data
    
    def get_cosmic_shear_data(self):
        """Extract cosmic shear measurements"""
        # Look for shear correlation functions
        if 'xi_plus' in self.data or 'xip' in self.data:
            xi_plus_name = 'xi_plus' if 'xi_plus' in self.data else 'xip'
            xi_minus_name = 'xi_minus' if 'xi_minus' in self.data else 'xim'
            
            return {
                'xi_plus': self.data.get(xi_plus_name),
                'xi_minus': self.data.get(xi_minus_name),
                'has_covariance': 'COVMAT' in self.data or 'covariance' in self.data
            }
        
        return None