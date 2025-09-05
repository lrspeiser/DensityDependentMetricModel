#!/usr/bin/env python3
"""
load_existing_gaia_lmc_smc.py

Offline converter for Gaia DR3 regional slices (LMC/SMC). This tool does not
query the Gaia Archive; instead it provides ADQL templates and converts your
manually downloaded CSV/FITS files into Parquet for faster local processing.

Per your rule, any use of web services or API keys must be documented in a
README. This module includes comments pointing to docs/gaia_slices_readme.md
with ADQL and step-by-step download instructions.

ADQL templates (run in Gaia Archive web UI and download the results locally)
- LMC (12° cone):

SELECT source_id, ra, dec, parallax, pmra, pmdec,
       phot_g_mean_mag, bp_rp, radial_velocity, radial_velocity_error
FROM gaiadr3.gaia_source
WHERE 1=CONTAINS(
  POINT('ICRS', ra, dec),
  CIRCLE('ICRS', 80.894, -69.756, 12)
)

- SMC (7° cone): change center to (13.186, -72.828), radius to 7.

Usage
  python -m data_loaders.load_existing_gaia_lmc_smc \
    --input "/path/to/downloads/*.csv" \
    --object LMC \
    --out-dir data/gaia_slices

Dependencies
- pandas (read_csv)
- pyarrow (Parquet writer)
- astropy (optional; FITS reader)
"""
from __future__ import annotations

import argparse
import glob
import logging
from pathlib import Path
from typing import List

import pandas as pd

try:
    import pyarrow  # noqa: F401
    _HAS_PARQUET = True
except Exception:
    _HAS_PARQUET = False

try:
    from astropy.table import Table as AstroTable
    _HAS_ASTROPY = True
except Exception:
    _HAS_ASTROPY = False


def setup_logger(debug: bool = False) -> None:
    logging.basicConfig(level=logging.DEBUG if debug else logging.INFO, format='[%(levelname)s] %(message)s')


def read_any(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in ('.csv', '.tsv'):
        sep = ',' if suffix == '.csv' else '\t'
        return pd.read_csv(path, sep=sep)
    if suffix in ('.fits', '.fit'):
        if not _HAS_ASTROPY:
            raise RuntimeError('FITS input requires astropy; install astropy to proceed')
        t = AstroTable.read(str(path))
        return t.to_pandas()
    raise ValueError(f'Unsupported input type: {path.suffix}')


def write_parquet(df: pd.DataFrame, out_path: Path) -> None:
    if not _HAS_PARQUET:
        raise RuntimeError('Parquet output requires pyarrow; install pyarrow to proceed')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)


def main():
    ap = argparse.ArgumentParser(description='Convert downloaded Gaia DR3 LMC/SMC CSV/FITS to Parquet')
    ap.add_argument('--input', required=True, help='Glob or file path(s) of Gaia outputs (CSV/FITS)')
    ap.add_argument('--object', choices=['LMC', 'SMC'], required=True, help='Target cloud for metadata tagging')
    ap.add_argument('--out-dir', default='data/gaia_slices', help='Parquet output directory')
    ap.add_argument('--debug', action='store_true')
    args = ap.parse_args()

    setup_logger(args.debug)

    files: List[str] = []
    if any(ch in args.input for ch in ['*', '?', '[']):
        files = glob.glob(args.input)
    else:
        files = [args.input]

    if not files:
        logging.error('No input files matched')
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for fp in files:
        p = Path(fp)
        try:
            df = read_any(p)
            df['cloud'] = args.object
            # Minimal schema sanity
            logging.info(f"Read {p.name}: rows={len(df):,} cols={len(df.columns)}")
            out_path = out_dir / (p.stem + '.parquet')
            write_parquet(df, out_path)
            logging.info(f"Wrote {out_path}")
        except Exception as e:
            logging.warning(f"Skip {p}: {e}")


if __name__ == '__main__':
    main()
