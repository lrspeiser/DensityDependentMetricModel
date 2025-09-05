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

# Optional: TAP client for live Gaia Archive queries (no credentials needed for public data)
try:
    import pyvo
    _HAS_PYVO = True
except Exception:
    _HAS_PYVO = False


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


def _preset_adql(name: str) -> str:
    name = name.upper().strip()
    if name == 'LMC':
        return (
            "SELECT source_id, ra, dec, parallax, pmra, pmdec, "
            "phot_g_mean_mag, bp_rp, radial_velocity, radial_velocity_error "
            "FROM gaiadr3.gaia_source "
            "WHERE 1=CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', 80.894, -69.756, 12))"
        )
    if name == 'SMC':
        return (
            "SELECT source_id, ra, dec, parallax, pmra, pmdec, "
            "phot_g_mean_mag, bp_rp, radial_velocity, radial_velocity_error "
            "FROM gaiadr3.gaia_source "
            "WHERE 1=CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', 13.186, -72.828, 7))"
        )
    raise ValueError(f'Unknown preset: {name}')


def _run_tap_query(adql: str) -> pd.DataFrame:
    if not _HAS_PYVO:
        raise RuntimeError('pyvo is required for API mode. Install pyvo to proceed.')
    svc = pyvo.dal.TAPService("https://gea.esac.esa.int/tap-server/tap")
    # Use synchronous query for simplicity; for large results consider async
    logging.info('Submitting TAP sync query…')
    res = svc.run_sync(adql)
    tab = res.to_table()
    df = tab.to_pandas()
    logging.info(f'TAP rows={len(df):,} cols={len(df.columns)}')
    return df


def main():
    ap = argparse.ArgumentParser(description='Gaia DR3 LMC/SMC: (A) convert local CSV/FITS to Parquet, or (B) fetch via TAP API and write Parquet')
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--input', help='Glob or file path(s) of Gaia outputs (CSV/FITS)')
    mode.add_argument('--api', action='store_true', help='Use Gaia TAP API to fetch data')
    ap.add_argument('--object', choices=['LMC', 'SMC'], required=True, help='Target cloud (also enables preset ADQL if --api and no --adql-file)')
    ap.add_argument('--adql-file', help='Path to a .sql ADQL file to run instead of preset')
    ap.add_argument('--out-dir', default='data/gaia_slices', help='Parquet output directory')
    ap.add_argument('--limit', type=int, default=None, help='Optional TOP N limiter injected into ADQL (API mode)')
    ap.add_argument('--debug', action='store_true')
    args = ap.parse_args()

    setup_logger(args.debug)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.api:
        # Build ADQL
        if args.adql_file:
            adql = Path(args.adql_file).read_text(encoding='utf-8')
        else:
            adql = _preset_adql(args.object)
        if args.limit and 'TOP' not in adql.upper():
            # Insert TOP N after SELECT for quick tests
            adql = adql.replace('SELECT', f'SELECT TOP {int(args.limit)}', 1)
        logging.info('Running TAP query (no credentials needed for public data)…')
        logging.debug(f'ADQL:\n{adql}')
        df = _run_tap_query(adql)
        df['cloud'] = args.object
        out_path = out_dir / f'gaia_{args.object.lower()}_tap.parquet'
        write_parquet(df, out_path)
        logging.info(f'Wrote {out_path}')
        return

    # Local conversion mode
    files: List[str] = []
    if any(ch in args.input or '' for ch in ['*', '?', '[']):
        files = glob.glob(args.input)
    else:
        files = [args.input]

    if not files:
        logging.error('No input files matched')
        return

    for fp in files:
        p = Path(fp)
        try:
            df = read_any(p)
            df['cloud'] = args.object
            logging.info(f"Read {p.name}: rows={len(df):,} cols={len(df.columns)}")
            out_path = out_dir / (p.stem + '.parquet')
            write_parquet(df, out_path)
            logging.info(f"Wrote {out_path}")
        except Exception as e:
            logging.warning(f"Skip {p}: {e}")


if __name__ == '__main__':
    main()
