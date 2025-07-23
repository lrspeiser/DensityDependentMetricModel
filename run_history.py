# run_history.py
import sqlite3
import json
import os
import datetime
import hashlib
import inspect
import sys
import subprocess
import logging
from pathlib import Path
from typing import Optional, Dict, Any

# Set up logging
logger = logging.getLogger(__name__)

# Database configuration
_DB_DIR = os.getenv("DDM_RUNLOG_DIR", ".")
_DB_PATH = os.path.join(_DB_DIR, "dynesty_runs.sqlite")

# Ensure database directory exists
Path(_DB_DIR).mkdir(parents=True, exist_ok=True)

TABLE_SQL = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    start_utc TEXT,
    end_utc TEXT,
    duration_s REAL,
    git_hash TEXT,
    cmdline TEXT,
    success INTEGER,
    logz REAL,
    logz_err REAL,
    eff REAL,
    rmse REAL,
    n_samples INTEGER,
    n_calls INTEGER,
    param_json TEXT,
    phys_check_pass INTEGER,
    phys_fail_reason TEXT
);
"""

def _connect():
    """Create database connection and ensure table exists."""
    try:
        conn = sqlite3.connect(_DB_PATH)
        conn.execute(TABLE_SQL)
        conn.commit()
        return conn
    except sqlite3.Error as e:
        logger.error(f"Database connection failed: {e}")
        raise

def _git_hash():
    """Get current git commit hash."""
    try:
        h = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        h = "unknown"
    return h

def start_record(args) -> str:
    """
    Start a new run record in the database.
    
    Parameters
    ----------
    args : object
        Arguments object with at least an output_dir attribute
        
    Returns
    -------
    str
        Unique run ID
    """
    # Generate unique run ID
    unique_string = f"{datetime.datetime.utcnow()}{getattr(args, 'output_dir', 'default')}"
    run_id = hashlib.sha1(unique_string.encode()).hexdigest()[:12]
    
    try:
        conn = _connect()
        
        # Check if run_id already exists (unlikely but possible)
        cur = conn.execute("SELECT COUNT(*) FROM runs WHERE run_id=?", (run_id,))
        if cur.fetchone()[0] > 0:
            # Add microseconds to make it unique
            unique_string += str(datetime.datetime.utcnow().microsecond)
            run_id = hashlib.sha1(unique_string.encode()).hexdigest()[:12]
        
        # Insert new record
        conn.execute(
            "INSERT OR REPLACE INTO runs(run_id, start_utc, cmdline, git_hash) VALUES(?, ?, ?, ?)",
            (run_id, datetime.datetime.utcnow().isoformat(),
             " ".join(map(str, sys.argv)), _git_hash())
        )
        conn.commit()
        conn.close()
        
        logger.info(f"Started run record: {run_id}")
        return run_id
        
    except Exception as e:
        logger.error(f"Failed to start run record: {e}")
        # Return a fallback run_id even if database fails
        return f"fallback_{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

def finalize_record(run_id: str, *, 
                    success: bool, 
                    logz: float, 
                    logz_err: float, 
                    eff: float, 
                    rmse: float,
                    n_samples: int, 
                    n_calls: int, 
                    param_stats: Dict[str, Any],
                    phys_ok: bool, 
                    phys_reason: str = "") -> bool:
    """
    Finalize a run record with results.
    
    Returns
    -------
    bool
        True if successfully updated, False otherwise
    """
    try:
        conn = _connect()
        end = datetime.datetime.utcnow()
        
        # Fetch start time for duration calculation
        cur = conn.execute("SELECT start_utc FROM runs WHERE run_id=?", (run_id,))
        result = cur.fetchone()
        
        if result is None:
            # No existing record found - create a minimal one
            logger.warning(f"No existing record found for run_id {run_id}, creating new record")
            
            # Insert a new record with estimated start time
            start = end - datetime.timedelta(seconds=60)  # Assume 1 minute runtime
            conn.execute(
                """INSERT INTO runs(run_id, start_utc, end_utc, duration_s, cmdline, git_hash,
                   success, logz, logz_err, eff, rmse, n_samples, n_calls, param_json,
                   phys_check_pass, phys_fail_reason)
                   VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (run_id, start.isoformat(), end.isoformat(), 60.0,
                 " ".join(map(str, sys.argv)), _git_hash(),
                 int(success), logz, logz_err, eff, rmse, n_samples, n_calls,
                 json.dumps(param_stats, default=float),
                 int(phys_ok), phys_reason)
            )
        else:
            # Update existing record
            start = datetime.datetime.fromisoformat(result[0])
            duration = (end - start).total_seconds()
            
            conn.execute(
                """UPDATE runs SET end_utc=?, duration_s=?, success=?, logz=?, logz_err=?,
                   eff=?, rmse=?, n_samples=?, n_calls=?, param_json=?, phys_check_pass=?,
                   phys_fail_reason=? WHERE run_id=?""",
                (end.isoformat(), duration, int(success),
                 logz, logz_err, eff, rmse, n_samples, n_calls,
                 json.dumps(param_stats, default=float),
                 int(phys_ok), phys_reason, run_id)
            )
        
        conn.commit()
        conn.close()
        
        logger.info(f"Finalized run record: {run_id} (success={success})")
        return True
        
    except Exception as e:
        logger.error(f"Failed to finalize run record {run_id}: {e}")
        return False

def get_run_summary(run_id: str) -> Optional[Dict[str, Any]]:
    """
    Get summary of a specific run.
    
    Parameters
    ----------
    run_id : str
        Run ID to query
        
    Returns
    -------
    dict or None
        Run summary if found, None otherwise
    """
    try:
        conn = _connect()
        cur = conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,))
        columns = [desc[0] for desc in cur.description]
        result = cur.fetchone()
        conn.close()
        
        if result:
            return dict(zip(columns, result))
        return None
        
    except Exception as e:
        logger.error(f"Failed to get run summary: {e}")
        return None

def list_recent_runs(n: int = 10) -> list:
    """
    List n most recent runs.
    
    Parameters
    ----------
    n : int
        Number of recent runs to return
        
    Returns
    -------
    list
        List of run summaries
    """
    try:
        conn = _connect()
        cur = conn.execute(
            "SELECT * FROM runs ORDER BY start_utc DESC LIMIT ?", (n,)
        )
        columns = [desc[0] for desc in cur.description]
        results = []
        for row in cur.fetchall():
            results.append(dict(zip(columns, row)))
        conn.close()
        return results
        
    except Exception as e:
        logger.error(f"Failed to list recent runs: {e}")
        return []

def cleanup_incomplete_runs():
    """Clean up runs that were started but never finalized."""
    try:
        conn = _connect()
        # Find runs with no end_utc that are older than 1 day
        cutoff = (datetime.datetime.utcnow() - datetime.timedelta(days=1)).isoformat()
        
        cur = conn.execute(
            "SELECT run_id FROM runs WHERE end_utc IS NULL AND start_utc < ?",
            (cutoff,)
        )
        
        incomplete_runs = [row[0] for row in cur.fetchall()]
        
        if incomplete_runs:
            logger.info(f"Cleaning up {len(incomplete_runs)} incomplete runs")
            
            for run_id in incomplete_runs:
                conn.execute(
                    """UPDATE runs SET success=0, end_utc=start_utc, duration_s=0,
                       phys_fail_reason='Run terminated abnormally' WHERE run_id=?""",
                    (run_id,)
                )
            
            conn.commit()
        
        conn.close()
        
    except Exception as e:
        logger.error(f"Failed to cleanup incomplete runs: {e}")

# Optional: Add a context manager for run tracking
class RunTracker:
    """Context manager for automatic run tracking."""
    
    def __init__(self, args):
        self.args = args
        self.run_id = None
        self.success = False
        self.results = {
            'logz': float('nan'),
            'logz_err': float('nan'),
            'eff': 0.0,
            'rmse': float('nan'),
            'n_samples': 0,
            'n_calls': 0,
            'param_stats': {},
            'phys_ok': False,
            'phys_reason': ''
        }
    
    def __enter__(self):
        self.run_id = start_record(self.args)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.success = True
        else:
            self.results['phys_reason'] = f"Exception: {exc_type.__name__}: {exc_val}"
        
        finalize_record(self.run_id, success=self.success, **self.results)
    
    def update_results(self, **kwargs):
        """Update results during the run."""
        self.results.update(kwargs)

# Example usage:
# with RunTracker(args) as tracker:
#     # Do your work
#     tracker.update_results(logz=-1000.0, n_samples=5000)
#     # If exception occurs, it will be caught and recorded