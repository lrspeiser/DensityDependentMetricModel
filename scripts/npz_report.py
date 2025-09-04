#!/usr/bin/env python3
import json
import numpy as np
import sys
from pathlib import Path

def weighted_quantiles(x, w, ps=(0.16,0.5,0.84)):
    x = np.asarray(x)
    w = np.asarray(w)
    order = np.argsort(x)
    xs = x[order]
    ws = w[order]
    cdf = np.cumsum(ws)
    tot = cdf[-1] if cdf.size else 1.0
    cdf = cdf / (tot if tot>0 else 1.0)
    return [float(np.interp(p, cdf, xs)) for p in ps]

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "usage: npz_report.py <path-to-npz>"}))
        sys.exit(1)
    npz_path = Path(sys.argv[1])
    d = np.load(npz_path, allow_pickle=True)
    out = {"file": str(npz_path)}
    # Evidence / likelihood
    if 'logz' in d:
        logz = d['logz']
        out['logZ_final'] = float(logz[-1]) if logz.size else None
    if 'logl' in d:
        logl = d['logl']
        out['logL_max'] = float(np.max(logl)) if logl.size else None
    # Best params
    names = d['param_names'].tolist() if 'param_names' in d else []
    names = [str(n) for n in names]
    if 'best_params' in d:
        vals = d['best_params']
        out['best_params'] = {n: float(v) for n, v in zip(names, vals)}
    # Reported metrics
    for k in ['chi2_report','chi_per_star_report','rmse_kms_report','v_infty_kms','M_b_BTFR_Msun','M_b_model_Msun','report_rmin','report_rmax']:
        if k in d:
            try:
                out[k] = float(d[k])
            except Exception:
                out[k] = None
    # Posteriors
    if 'samples' in d:
        samples = d['samples']
        weights = d['weights'] if 'weights' in d else np.ones(samples.shape[0], dtype=float)
        weights = np.asarray(weights, dtype=float)
        if weights.ndim != 1 or weights.shape[0] != samples.shape[0]:
            weights = np.ones(samples.shape[0], dtype=float)
        wsum = float(weights.sum())
        weights = weights / wsum if wsum > 0 else np.ones_like(weights)/weights.size
        qs = {}
        for j, n in enumerate(names):
            p16, p50, p84 = weighted_quantiles(samples[:, j], weights)
            qs[n] = {"p16": p16, "p50": p50, "p84": p84}
        out['posteriors'] = qs
    print(json.dumps(out))

if __name__ == '__main__':
    main()

