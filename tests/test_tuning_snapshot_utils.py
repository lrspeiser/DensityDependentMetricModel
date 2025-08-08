import numpy as np
import math

# Tests for tuning snapshot helper utilities
# We import from runners.run_dynesty to validate numeric sanity of the
# weighted percentiles and PCA helpers used to build the snapshot.

from runners.run_dynesty import _weighted_percentiles, _top_pcs


def test_weighted_percentiles_monotonic_and_limits():
    rng = np.random.default_rng(42)
    x = rng.normal(loc=0.0, scale=1.0, size=1000)
    # Positive weights with some variability
    w = rng.random(1000)
    pct = _weighted_percentiles(x, w, (16, 50, 84))

    p16, p50, p84 = pct["p16"], pct["p50"], pct["p84"]

    # Monotonic: p16 <= p50 <= p84
    assert p16 <= p50 <= p84

    # Compare with unweighted percentiles for sanity (should be "close")
    q16, q50, q84 = np.percentile(x, [16, 50, 84])
    # Allow loose tolerance because weights can shift estimates
    assert math.isfinite(p16) and math.isfinite(p50) and math.isfinite(p84)
    assert abs(p50 - q50) < 0.25  # central tendency shouldn't drift too far


def test_top_pcs_shapes_and_ordering():
    rng = np.random.default_rng(0)
    n, d = 500, 4
    # Create a correlated Gaussian with known covariance eigen-structure
    true_eigs = np.array([4.0, 1.0, 0.5, 0.1])
    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    C = (Q * true_eigs) @ Q.T  # symmetric PD with eigenvalues true_eigs

    X = rng.multivariate_normal(np.zeros(d), C, size=n)
    w = rng.random(n)

    pcs = _top_pcs(X, w, k_max=3)

    # Expect at most 3 PCs returned
    assert 1 <= len(pcs) <= 3

    # Eigenvalues should be non-negative and in descending order
    eigvals = [pc["eigval"] for pc in pcs]
    assert all(ev >= 0 for ev in eigvals)
    assert all(eigvals[i] >= eigvals[i+1] for i in range(len(eigvals)-1))

    # Eigenvectors should have the correct dimensionality
    for pc in pcs:
        vec = pc["eigvec"]
        assert len(vec) == d

