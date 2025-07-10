# bao_data_loader.py
import numpy as np
from pathlib import Path
import re
from typing import Dict, List, Tuple

class BAODataLoader:
    """Load SDSS BAO measurements from DR16 cosmo release"""
    
    def __init__(self, bao_dir: str = 'bao'):
        self.bao_dir = Path(bao_dir)
        self.measurements = {}
        
    def parse_bao_file(self, filename: str) -> Dict:
        """Parse a single BAO measurement file"""
        filepath = self.bao_dir / filename
        
        if not filepath.exists():
            print(f"Warning: {filepath} not found")
            return None
            
        # Extract info from filename
        parts = filename.split('_')
        survey = parts[1]  # DR12, DR16, etc.
        tracer = parts[2]  # LRG, QSO, MGS, LYAUTO, LYxQSO
        
        data = {
            'filename': filename,
            'survey': survey,
            'tracer': tracer,
            'measurements': []
        }
        
        with open(filepath, 'r') as f:
            lines = f.readlines()
            
        # Parse header to understand format
        in_data = False
        headers = []
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
                
            # Comments often contain redshift info
            if line.startswith('#'):
                if 'z =' in line or 'redshift' in line.lower():
                    z_match = re.search(r'z\s*=\s*([\d.]+)', line)
                    if z_match:
                        data['redshift'] = float(z_match.group(1))
                continue
            
            # First non-comment line is usually headers
            if not in_data and not line.startswith('#'):
                headers = line.split()
                in_data = True
                continue
                
            # Data lines
            if in_data:
                values = [float(x) for x in line.split()]
                if len(values) >= 2:
                    measurement = dict(zip(headers, values))
                    data['measurements'].append(measurement)
        
        # For grid files (Lyman-alpha), extract the best-fit values
        if 'grid' in filename:
            data['type'] = 'grid'
            # These files contain chi2 grids, need to find minimum
            if data['measurements']:
                chi2_values = [m.get('chi2', float('inf')) for m in data['measurements']]
                min_idx = np.argmin(chi2_values)
                data['best_fit'] = data['measurements'][min_idx]
        else:
            data['type'] = 'measurement'
            # Regular files have direct measurements
            if data['measurements']:
                data['best_fit'] = data['measurements'][0]
        
        return data
    
    def load_all_measurements(self) -> Dict[str, Dict]:
        """Load all BAO measurements from the directory"""
        
        bao_files = list(self.bao_dir.glob('*.txt'))
        print(f"Found {len(bao_files)} BAO measurement files")
        
        for filepath in bao_files:
            filename = filepath.name
            data = self.parse_bao_file(filename)
            
            if data:
                # Key by tracer and survey
                key = f"{data['survey']}_{data['tracer']}"
                self.measurements[key] = data
                
                # Print summary
                if 'best_fit' in data:
                    bf = data['best_fit']
                    z = data.get('redshift', 'unknown')
                    print(f"  {key}: z={z}, measurements: {list(bf.keys())}")
        
        return self.measurements
    
    def get_distance_measurements(self) -> List[Dict]:
        """Extract distance measurements for cosmological tests"""
        results = []
        
        for key, data in self.measurements.items():
            if 'best_fit' not in data:
                continue
                
            bf = data['best_fit']
            entry = {
                'name': key,
                'tracer': data['tracer'],
                'survey': data['survey']
            }
            
            # Extract redshift
            if 'redshift' in data:
                entry['z'] = data['redshift']
            elif 'z' in bf:
                entry['z'] = bf['z']
            else:
                # Try to infer from tracer type
                z_estimates = {
                    'MGS': 0.15,
                    'LRG': 0.7,
                    'QSO': 1.5,
                    'LYAUTO': 2.3,
                    'LYxQSO': 2.3
                }
                entry['z'] = z_estimates.get(data['tracer'], 1.0)
            
            # Extract distance measurements
            # DV: spherically averaged distance
            # DM: comoving angular diameter distance  
            # DH: Hubble distance (c/H(z))
            
            if 'DV/rd' in bf:
                entry['DV_over_rd'] = bf['DV/rd']
                entry['DV_over_rd_err'] = bf.get('DV/rd_err', bf.get('sigma_DV/rd', 0))
            elif 'DV_over_rd' in bf:
                entry['DV_over_rd'] = bf['DV_over_rd']
                entry['DV_over_rd_err'] = bf.get('sigma_DV_over_rd', 0)
                
            if 'DM/rd' in bf:
                entry['DM_over_rd'] = bf['DM/rd']
                entry['DM_over_rd_err'] = bf.get('DM/rd_err', bf.get('sigma_DM/rd', 0))
            elif 'DM_over_rd' in bf:
                entry['DM_over_rd'] = bf['DM_over_rd']
                entry['DM_over_rd_err'] = bf.get('sigma_DM_over_rd', 0)
                
            if 'DH/rd' in bf:
                entry['DH_over_rd'] = bf['DH/rd']
                entry['DH_over_rd_err'] = bf.get('DH/rd_err', bf.get('sigma_DH/rd', 0))
            elif 'DH_rd' in bf:
                entry['DH_rd'] = bf['DH_rd']
                entry['DH_rd_err'] = bf.get('sigma_DH_rd', 0)
            
            # Growth rate measurements
            if 'fs8' in bf:
                entry['fs8'] = bf['fs8']
                entry['fs8_err'] = bf.get('fs8_err', bf.get('sigma_fs8', 0))
            elif 'f_sigma8' in bf:
                entry['fs8'] = bf['f_sigma8']
                entry['fs8_err'] = bf.get('sigma_f_sigma8', 0)
            
            results.append(entry)
        
        # Sort by redshift
        results.sort(key=lambda x: x['z'])
        
        return results

# Test the loader
if __name__ == "__main__":
    loader = BAODataLoader('bao')
    measurements = loader.load_all_measurements()
    
    print(f"\nLoaded {len(measurements)} BAO measurements")
    
    # Get distance measurements
    distances = loader.get_distance_measurements()
    
    print("\nDistance measurements summary:")
    print(f"{'Tracer':<15} {'z':<6} {'DV/rd':<10} {'DM/rd':<10} {'DH/rd':<10} {'fs8':<10}")
    print("-" * 70)
    
    for d in distances:
        dv = d.get('DV_over_rd', '-')
        dm = d.get('DM_over_rd', '-') 
        dh = d.get('DH_over_rd', '-')
        fs8 = d.get('fs8', '-')
        
        # Format numbers
        if isinstance(dv, float): dv = f"{dv:.2f}"
        if isinstance(dm, float): dm = f"{dm:.2f}"
        if isinstance(dh, float): dh = f"{dh:.2f}"
        if isinstance(fs8, float): fs8 = f"{fs8:.3f}"
        
        print(f"{d['tracer']:<15} {d['z']:<6.2f} {dv:<10} {dm:<10} {dh:<10} {fs8:<10}")