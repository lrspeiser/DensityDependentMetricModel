# utils/run_history.py
import sqlite3, json, os, datetime, hashlib, inspect, sys, subprocess

_DB_PATH = os.path.join(os.getenv("DDM_RUNLOG_DIR", "."), "dynesty_runs.sqlite")

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
    conn = sqlite3.connect(_DB_PATH)
    conn.execute(TABLE_SQL)
    return conn

def _git_hash():
    try:
        h = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                    stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        h = "unknown"
    return h

def start_record(args)->str:
    run_id = hashlib.sha1(f"{datetime.datetime.utcnow()}{args.output_dir}".encode()).hexdigest()[:12]
    conn=_connect()
    conn.execute("INSERT OR REPLACE INTO runs(run_id,start_utc,cmdline,git_hash) VALUES(?,?,?,?)",
                 (run_id, datetime.datetime.utcnow().isoformat(),
                  " ".join(map(str, sys.argv)), _git_hash()))
    conn.commit()
    conn.close()
    return run_id

def finalize_record(run_id, *, success, logz, logz_err, eff, rmse,
                    n_samples, n_calls, param_stats:dict,
                    phys_ok:bool, phys_reason:str=""):
    conn=_connect()
    end = datetime.datetime.utcnow()
    # fetch start for duration
    cur = conn.execute("SELECT start_utc FROM runs WHERE run_id=?", (run_id,))
    start = datetime.datetime.fromisoformat(cur.fetchone()[0])
    conn.execute("""UPDATE runs SET end_utc=?, duration_s=?, success=?, logz=?, logz_err=?,
                    eff=?, rmse=?, n_samples=?, n_calls=?, param_json=?, phys_check_pass=?,
                    phys_fail_reason=? WHERE run_id=?""",
                 (end.isoformat(), (end-start).total_seconds(), int(success),
                  logz, logz_err, eff, rmse, n_samples, n_calls,
                  json.dumps(param_stats, default=float),
                  int(phys_ok), phys_reason, run_id))
    conn.commit()
    conn.close()
