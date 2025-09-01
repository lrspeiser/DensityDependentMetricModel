# Exponential Model Overflow Fix

## Problem
The exponential model was causing an `OverflowError` in the likelihood calculation at line 205:
```python
penalties -= 100.0 * (chi2_per_star - 2.0)**2
```

When `chi2_per_star` becomes very large (indicating a poor fit), squaring it results in a value that exceeds Python's float limits.

## Root Cause
The exponential model can produce extreme xi values that lead to:
1. Very poor velocity predictions
2. Huge chi-squared values per star
3. Overflow when computing the squared penalty

## Solution
We need to add numerical safeguards in the likelihood calculation to handle extreme values gracefully. The fix involves:

1. **Clamping penalties**: Limit the maximum penalty contribution to prevent overflow
2. **Adding numerical checks**: Ensure xi values stay within reasonable bounds  
3. **Safe computation**: Use logarithmic scales or clamped values for extreme cases

## Implementation
Created `fix_exponential_overflow.py` that:
- Adds bounds checking for chi2_per_star values
- Uses safe computation methods to prevent overflow
- Maintains the penalty structure but with numerical safeguards

## Testing
After applying the fix:
- The exponential model should run without overflow errors
- Poor parameter combinations will still be penalized but won't crash
- The sampler can explore the parameter space properly

## Related Files
- `run_dynesty_stellar_fit_cupy.py` - Main fitting script with likelihood calculation
- `run_full_analysis_parallel.py` - Parallel runner that now handles command-line arguments
- `core/density_metric_cupy.py` - Contains xi calculation functions
