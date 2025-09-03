# Tidal Run Optimization Strategy

## Problem Diagnosis
Your current tidal_band run has been running for 2+ days with only 1.06% efficiency. This means:
- 99 out of 100 likelihood evaluations are being rejected
- The sampler is struggling with the posterior geometry
- You're burning compute time for minimal progress

## Why This Happened
1. **rslice + multi** combination is poor for anisotropic posteriors (your anisotropy = 1.8)
2. The prior box is too wide, forcing exploration of low-probability regions
3. After 120k samples, you've found the high-probability region but the sampler settings prevent efficient exploration

## The Solution: From-Best Restart

### Key Changes:
1. **`--resume_from_best`**: Start fresh run from your best-fit parameters
2. **`--best_param_spread 0.15`**: Tighten prior to ±15% around best (was exploring full range)
3. **`rwalk + balls`**: Better for moderate anisotropy (1.8 ratio from tuning snapshot)
4. **`--walks 50`**: More parallel chains for better mixing
5. **`nlive 2000`**: Balance between accuracy and speed

### Expected Improvements:
- Efficiency: 1% → 5-10% (5-10x speedup)
- Time to convergence: Days → Hours
- Better posterior sampling in high-probability region

## Evidence Status
Your run already shows **decisive evidence** for DDMM over GR:
- ΔlogZ = +964,092 (anything >10 is "decisive")
- This is one of the strongest DDMM preferences seen

The from-best restart will:
1. Refine parameter uncertainties
2. Improve posterior sampling
3. Reach proper convergence faster

## Alternative: Continue Current Run
If you prefer to keep the exact sampler state:
```bash
python runners\run_dynesty_cupy.py ^
  --xi tidal_band ^
  --resume_from runs\tidal_band_20250818_141227\resume_1_20250818_153320 ^
  --dlogz_target 0.001 ^
  --periodic_analysis --analysis_interval_min 30
```
But this will maintain the 1% efficiency and take many more days.

## Recommendation
**Use the from-best restart.** You've already explored the parameter space extensively (120k samples). Now you need efficient refinement around the best region, which the from-best restart provides.

The optimization script (`restart_tidal_optimized.bat`) implements all these improvements.
