# all_data_loader.py
import json
import numpy as np
from pathlib import Path
import pandas as pd
from astropy.io import fits

class UniversalDataLoader:
    """Load all validation datasets"""
    
    def __init__(self, base_dir='.'):
        self.base_dir = Path(base_dir)
        self.data = {}
        
    def load_planck_data(self, planck_dir='planck_data'):
        """Load Planck CMB power spectrum"""
        planck_path = self.base_dir / planck_dir
        
        # Look for TT spectrum file
        tt_files = list(planck_path.glob('*TT*.txt')) + \
                   list(planck_path.glob('*tt*.txt'))
        
        if tt_files:
            # Load TT power spectrum
            data = np.loadtxt(tt_files[0])
            self.data['planck'] = {
                'ell': data[:, 0],
                'D_ell': data[:, 1],  # D_ell = ell(ell+1)C_ell/2π
                'error': data[:, 2] if data.shape[1] > 2 else None,
                'file': str(tt_files[0])
            }
            print(f"Loaded Planck data from {tt_files[0]}")
        else:
            print(f"Warning: No Planck TT spectrum found in {planck_path}")
    
    def load_pantheon_data(self, pantheon_dir='pantheon'):
        """Load Pantheon+ supernova data"""
        pantheon_path = self.base_dir / pantheon_dir
        
        # Look for main data file
        data_files = list(pantheon_path.glob('*.txt')) + \
                     list(pantheon_path.glob('*.dat'))
        
        if data_files:
            # Try to read the first file
            try:
                data = pd.read_csv(data_files[0], delim_whitespace=True, 
                                  comment='#')
                self.data['pantheon'] = {
                    'data': data,
                    'file': str(data_files[0])
                }
                print(f"Loaded Pantheon data from {data_files[0]}")
                print(f"  Columns: {list(data.columns)}")
            except Exception as e:
                print(f"Error loading Pantheon data: {e}")
        else:
            print(f"Warning: No Pantheon data found in {pantheon_path}")
    
    def load_bao_data(self, bao_file='bao_data.json'):
        """Load BAO measurements"""
        bao_path = self.base_dir / bao_file
        if bao_path.exists():
            with open(bao_path, 'r') as f:
                self.data['bao'] = json.load(f)
            print(f"Loaded BAO data from {bao_file}")
        else:
            print(f"Warning: {bao_file} not found")
    
    def load_bullet_cluster(self, bullet_file='bullet_cluster_profile.json'):
        """Load Bullet Cluster profile"""
        bullet_path = self.base_dir / bullet_file
        if bullet_path.exists():
            with open(bullet_path, 'r') as f:
                self.data['bullet'] = json.load(f)
            print(f"Loaded Bullet Cluster data from {bullet_file}")
        else:
            print(f"Warning: {bullet_file} not found")
    
    def load_laboratory_constraints(self, lab_file='laboratory_constraints.json'):
        """Load laboratory test constraints"""
        lab_path = self.base_dir / lab_file
        if lab_path.exists():
            with open(lab_path, 'r') as f:
                self.data['laboratory'] = json.load(f)
            print(f"Loaded laboratory constraints from {lab_file}")
        else:
            print(f"Warning: {lab_file} not found")
    
    def load_all(self):
        """Load all available datasets"""
        self.load_planck_data()
        self.load_pantheon_data()
        self.load_bao_data()
        self.load_bullet_cluster()
        self.load_laboratory_constraints()
        return self.data

# Test the loader
if __name__ == "__main__":
    loader = UniversalDataLoader()
    
    # First create the JSON data files
    print("Creating data files...")
    import subprocess
    subprocess.run([sys.executable, "create_bao_data.py"])
    subprocess.run([sys.executable, "create_bullet_cluster_data.py"])
    subprocess.run([sys.executable, "create_laboratory_constraints.py"])
    
    # Then load everything
    print("\nLoading all data...")
    all_data = loader.load_all()
    
    print(f"\nLoaded datasets: {list(all_data.keys())}")