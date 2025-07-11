# frontier_lensing_loader.py
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from pathlib import Path
from typing import Dict, Tuple, Optional

class FrontierFieldsLoader:
    """Load Hubble Frontier Fields lensing data for MACS0416"""
    
    def __init__(self, frontier_dir: str = 'hlsp_frontier'):
        self.frontier_dir = Path(frontier_dir)
        self.data = {}
        
    def load_convergence_map(self) -> Dict:
        """Load the convergence (kappa) map - this is the key for DDMM testing"""
        kappa_file = self.frontier_dir / 'hlsp_frontier_model_macs0416_cats_v4.1_kappa.fits'
        
        if not kappa_file.exists():
            raise FileNotFoundError(f"Convergence map not found: {kappa_file}")
        
        with fits.open(kappa_file) as hdul:
            kappa_data = hdul[0].data
            header = hdul[0].header
            wcs = WCS(header)
            
        # Get physical scales
        pixel_scale = abs(header.get('CDELT1', 0.065)) * 3600  # arcsec/pixel
        
        self.data['kappa'] = {
            'data': kappa_data,
            'wcs': wcs,
            'pixel_scale': pixel_scale,
            'header': header,
            'shape': kappa_data.shape
        }
        
        print(f"Loaded convergence map: {kappa_data.shape}")
        print(f"  Pixel scale: {pixel_scale:.3f} arcsec/pixel")
        print(f"  κ range: [{np.min(kappa_data):.3f}, {np.max(kappa_data):.3f}]")
        
        return self.data['kappa']
    
    def load_shear_maps(self) -> Dict:
        """Load shear components (gamma1, gamma2)"""
        gamma_file = self.frontier_dir / 'hlsp_frontier_model_macs0416_cats_v4.1_gamma.fits'
        
        if gamma_file.exists():
            with fits.open(gamma_file) as hdul:
                # Usually gamma1 and gamma2 are in different extensions
                if len(hdul) > 1:
                    gamma1 = hdul[0].data
                    gamma2 = hdul[1].data
                else:
                    # Or might be a single 3D array
                    data = hdul[0].data
                    if data.ndim == 3:
                        gamma1 = data[0]
                        gamma2 = data[1]
                    else:
                        gamma1 = data
                        gamma2 = np.zeros_like(data)
                
            self.data['shear'] = {
                'gamma1': gamma1,
                'gamma2': gamma2,
                'gamma_abs': np.sqrt(gamma1**2 + gamma2**2)
            }
            
            print(f"Loaded shear maps")
            print(f"  |γ| range: [{np.min(self.data['shear']['gamma_abs']):.3f}, "
                  f"{np.max(self.data['shear']['gamma_abs']):.3f}]")
        
        return self.data.get('shear')
    
    def load_deflection_fields(self) -> Dict:
        """Load deflection angle maps"""
        x_file = self.frontier_dir / 'hlsp_frontier_model_macs0416_cats_v4.1_x-arcsec-deflect.fits'
        y_file = self.frontier_dir / 'hlsp_frontier_model_macs0416_cats_v4.1_y-arcsec-deflect.fits'
        
        if x_file.exists() and y_file.exists():
            with fits.open(x_file) as hdul:
                alpha_x = hdul[0].data
            with fits.open(y_file) as hdul:
                alpha_y = hdul[0].data
                
            self.data['deflection'] = {
                'alpha_x': alpha_x,
                'alpha_y': alpha_y,
                'alpha_abs': np.sqrt(alpha_x**2 + alpha_y**2)
            }
            
            print(f"Loaded deflection fields")
            print(f"  |α| range: [{np.min(self.data['deflection']['alpha_abs']):.1f}, "
                  f"{np.max(self.data['deflection']['alpha_abs']):.1f}] arcsec")
        
        return self.data.get('deflection')
    
    def convert_to_physical_units(self, z_lens: float = 0.396) -> Dict:
        """Convert from observables to physical units for DDMM comparison"""
        # MACS0416 redshift
        from astropy.cosmology import Planck18
        
        # Angular diameter distance
        D_lens = Planck18.angular_diameter_distance(z_lens).value  # Mpc
        
        # Convert pixel scale to physical scale
        kpc_per_arcsec = D_lens * 1000 * np.pi / (180 * 3600)  # kpc/arcsec
        
        if 'kappa' in self.data:
            pixel_scale_kpc = self.data['kappa']['pixel_scale'] * kpc_per_arcsec
            
            # Create coordinate arrays
            ny, nx = self.data['kappa']['shape']
            x_kpc = (np.arange(nx) - nx/2) * pixel_scale_kpc
            y_kpc = (np.arange(ny) - ny/2) * pixel_scale_kpc
            
            self.data['physical'] = {
                'x_kpc': x_kpc,
                'y_kpc': y_kpc,
                'pixel_scale_kpc': pixel_scale_kpc,
                'z_lens': z_lens,
                'D_lens_Mpc': D_lens
            }
            
            print(f"\nPhysical units:")
            print(f"  Lens redshift: {z_lens}")
            print(f"  Angular diameter distance: {D_lens:.1f} Mpc")
            print(f"  Scale: {kpc_per_arcsec:.2f} kpc/arcsec")
            print(f"  Map size: {x_kpc.max()-x_kpc.min():.1f} × {y_kpc.max()-y_kpc.min():.1f} kpc")
        
        return self.data.get('physical')