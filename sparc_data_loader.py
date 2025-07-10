# sparc_data_loader.py
import numpy as np
import pandas as pd
from pathlib import Path
import re
from typing import Dict, List, Optional

class SPARCDataLoader:
    """Load and process SPARC rotation curve data"""
    
    def __init__(self, sparc_dir: str):
        """
        Initialize with the directory containing SPARC data files.
        
        Parameters
        ----------
        sparc_dir : str
            Path to directory containing *_rotmod.dat files (e.g., "Rotmod_LTG")
        """
        self.sparc_dir = Path(sparc_dir)
        if not self.sparc_dir.exists():
            raise FileNotFoundError(f"SPARC directory not found: {sparc_dir}")
        
        self.galaxies = {}
        print(f"Initialized SPARCDataLoader with directory: {self.sparc_dir}")
    
    def load_galaxy(self, filename: str) -> Dict:
        """
        Load single galaxy data file.
        
        Parameters
        ----------
        filename : str
            Name of the file (e.g., 'NGC6789_rotmod.dat')
            
        Returns
        -------
        dict
            Galaxy data including velocities, radii, etc.
        """
        # This is where we use self.sparc_dir!
        filepath = self.sparc_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Galaxy file not found: {filepath}")
        
        # Parse header for distance
        distance_mpc = None
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith('# Distance'):
                    match = re.search(r'([\d.]+)\s*Mpc', line)
                    if match:
                        distance_mpc = float(match.group(1))
                    break
        
        # Load data - skip comment lines
        data = pd.read_csv(filepath, comment='#', delim_whitespace=True,
                          names=['Rad_kpc', 'Vobs', 'errV', 'Vgas', 'Vdisk', 
                                'Vbul', 'SBdisk', 'SBbul'])
        
        # Calculate total baryonic velocity
        v_baryon_sq = data['Vgas']**2 + data['Vdisk']**2 + data['Vbul']**2
        v_baryon = np.sqrt(np.maximum(v_baryon_sq, 0))
        
        # Extract galaxy name from filename
        galaxy_name = filename.replace('_rotmod.dat', '')
        
        return {
            'name': galaxy_name,
            'distance_mpc': distance_mpc,
            'r_kpc': data['Rad_kpc'].values,
            'v_obs': data['Vobs'].values,
            'v_err': data['errV'].values,
            'v_gas': data['Vgas'].values,
            'v_disk': data['Vdisk'].values,
            'v_bulge': data['Vbul'].values,
            'v_baryon': v_baryon,
            'sb_disk': data['SBdisk'].values,
            'sb_bulge': data['SBbul'].values,
            'filename': filename,
            'filepath': str(filepath)
        }
    
    def load_all_galaxies(self) -> Dict[str, Dict]:
        """
        Load all galaxy files in the directory.
        
        Returns
        -------
        dict
            Dictionary mapping galaxy names to their data
        """
        # This is where self.sparc_dir is crucial!
        rotmod_files = list(self.sparc_dir.glob('*_rotmod.dat'))
        
        if not rotmod_files:
            print(f"WARNING: No *_rotmod.dat files found in {self.sparc_dir}")
            return {}
        
        print(f"Found {len(rotmod_files)} galaxy files in {self.sparc_dir}")
        
        for filepath in rotmod_files:
            try:
                galaxy_data = self.load_galaxy(filepath.name)
                self.galaxies[galaxy_data['name']] = galaxy_data
            except Exception as e:
                print(f"Error loading {filepath.name}: {e}")
        
        print(f"Successfully loaded {len(self.galaxies)} galaxies")
        return self.galaxies
    
    def get_galaxy_by_name(self, name: str) -> Optional[Dict]:
        """Get specific galaxy data by name"""
        return self.galaxies.get(name)
    
    def list_galaxies(self) -> List[str]:
        """List all loaded galaxy names"""
        return list(self.galaxies.keys())