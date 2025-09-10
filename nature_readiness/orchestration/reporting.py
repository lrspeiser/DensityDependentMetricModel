"""Reporting helpers (code-based)
Utilities to render summary figures/tables into results/nature_readiness/.
"""
from __future__ import annotations
from typing import Dict, Any, List
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def plot_delta_ic_bar(out_path: Path, labels: List[str], delta_aic: List[float], delta_bic: List[float]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.bar(x - w/2, delta_aic, width=w, label='ΔAIC')
    ax.bar(x + w/2, delta_bic, width=w, label='ΔBIC')
    ax.set_xticks(x, labels)
    ax.axhline(0.0, color='k', ls=':')
    ax.set_ylabel('Δ Information Criterion')
    ax.set_title('Model comparison (lower is better)')
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(out_path, dpi=140); plt.close(fig)
