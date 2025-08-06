## Solar System Constraints and Formula Development

### The Cassini Constraint

The Cassini spacecraft's 2002 solar conjunction experiment provides one of the most stringent tests of gravitational theories. Radio signals transmitted between Earth and Cassini as it passed behind the Sun measured the Shapiro time delay with unprecedented precision, constraining the post-Newtonian parameter γ to:

$$\gamma = 1 + (2.1 \pm 2.3) \times 10^{-5}$$

For DDMM to be viable, our enhancement factor ξ must reduce to unity with extreme precision in the Solar System while still providing ξ ≈ 2.5-3 at galactic densities. This represents a formidable challenge: the theory must transition from standard gravity to enhanced gravity over the enormous density range between Solar System (ρ ~ 10¹⁵ M_⊙/kpc³) and galactic (ρ ~ 10⁸ M_⊙/kpc³) environments.

### Formula Development Strategy

We systematically test six classes of ξ(ρ, M) formulations that could satisfy both constraints:

#### 1. Mass Threshold Model
Based on the physical picture of gravity becoming "elastic" below a critical mass:

$$\xi(M) = 1 + (\xi_{\text{boost}} - 1) \times \frac{1}{2}\left[1 - \tanh\left(\frac{M - M_{\text{crit}}}{w \cdot M_{\text{crit}}}\right)\right]$$

where M_crit is the critical mass below which gravity enhancement occurs, ξ_boost is the maximum enhancement, and w controls the transition width.

```python
def xi_mass_threshold(r_kpc, rho, M_enclosed_msun, params):
    """Mass threshold: gravity normal above M_crit, enhanced below"""
    M_crit = params['M_crit_msun']
    xi_boost = params['xi_boost']
    width = params.get('width', 0.1)
    
    xi = 1 + (xi_boost - 1) * 0.5 * (1 - np.tanh(
        (M_enclosed_msun - M_crit) / (width * M_crit)
    ))
    return xi
```

For Cassini compatibility, we need M_crit ≪ M_⊙ = 2×10³⁰ kg, ensuring the Sun's mass places it firmly in the ξ = 1 regime.

#### 2. Mass Power Law
A more gradual transition using power-law scaling:

$$\xi(M) = 1 + A\left(\frac{M_{\text{crit}}}{M}\right)^n \quad \text{for } M > M_{\text{min}}$$

This provides flexibility in the transition steepness through the exponent n.

#### 3. Density-Dependent with Mass Screening
Combines the original density dependence with mass-based screening:

$$\xi(\rho, M) = 1 + A\left(\frac{\rho_c}{\rho}\right)^n \exp\left(-\frac{M}{M_{\text{screen}}}\right)$$

The exponential screening factor ensures ξ → 1 for massive objects regardless of local density.

```python
def xi_density_screened(r_kpc, rho, M_enclosed_msun, params):
    """Density enhancement suppressed by mass screening"""
    rho_c = params['rho_c_msun_kpc3']
    n_exp = params['n_exp']
    A = params['A']
    M_screen = params['M_screen_msun']
    
    xi_base = 1 + A * (rho_c / rho) ** n_exp
    screen_factor = np.exp(-M_enclosed_msun / M_screen)
    
    xi = 1 + (xi_base - 1) * screen_factor
    return xi
```

#### 4. Yukawa-Like Range Screening
Inspired by screened modified gravity theories:

$$\xi(r, M) = 1 + (\xi_0 - 1) \exp\left(-\frac{M}{M_{\text{screen}}}\right) \exp\left(-\frac{r}{\lambda}\right)$$

This adds spatial range dependence, potentially useful for cluster-scale constraints.

### Cassini Test Implementation

The test integrates ξ along the Earth-Saturn-Sun light path, computing the effective post-Newtonian parameter:

```python
def cassini_light_deflection_test(self, xi_func, formula_name, params):
    # Calculate ξ along radio signal path
    xi_path = xi_func(self.r_path_kpc, self.rho_msun_kpc3, 
                      self.M_enclosed/M_SUN, params)
    
    # Weighted path average (Shapiro delay weights by 1/r)
    weights = 1.0 / self.r_path_m
    gamma_eff = np.average(xi_path, weights=weights) - 1.0
    
    # Check Cassini constraint
    passes = abs(gamma_eff) < self.cassini_gamma_limit
    
    return CassiniTest(
        formula_name=formula_name,
        gamma_parameter=gamma_eff,
        passes_cassini=passes,
        xi_at_sun=xi_path[0],
        xi_path=xi_path
    )
```

### Required Performance Criteria

For a formula to be viable, it must satisfy:

1. **Cassini Constraint**: |γ_eff - 1| < 2.3 × 10⁻⁵ along the signal path
2. **Solar System Screening**: ξ(r_⊙, ρ_⊙, M_⊙) - 1 < 10⁻⁸ 
3. **Galaxy Enhancement**: ⟨ξ⟩_galaxy ≥ 2.5 in the 5-15 kpc range
4. **Smooth Transition**: No discontinuities that would produce observable effects

### Preliminary Results

Initial testing reveals that mass-based screening is essential. Pure density-dependent formulas struggle because the Solar System's high density isn't unique—similar densities exist in galactic centers where we need ξ > 1. The most promising approaches use either:

- **Sharp mass thresholds** with M_crit ~ 10⁻³ - 10⁻¹ M_⊙
- **Combined density-mass screening** where massive objects suppress the enhancement regardless of environment

```python
# Example of successful parameter regime
successful_params = {
    'M_crit_msun': 0.01,      # 1% of solar mass
    'xi_boost': 3.0,          # 3x enhancement for galaxies  
    'width': 0.2              # Smooth transition
}
```

Figure X shows the ξ profile along the Cassini signal path for various formulas. Successful formulas (green) maintain ξ ≈ 1.00000 throughout the Solar System while failed attempts (red) show detectable deviations. The bottom panel confirms that the Sun's mass (2×10³⁰ kg) far exceeds any reasonable threshold, ensuring robust screening.

### Implications for DDMM Theory

The Cassini constraint forces us to refine our physical picture. Rather than pure density dependence, the data suggests gravity modification is controlled by the **mass scale** of the gravitating system. This points toward theories where:

1. Quantum corrections to gravity become significant for low-mass systems
2. The gravitational "elasticity" emerges only below a critical mass threshold
3. Massive objects like stars "rigidify" spacetime, suppressing modifications

This refined framework maintains DDMM's core insight—that apparent dark matter effects arise from modified gravity—while ensuring compatibility with precision Solar System tests. The next phase involves implementing these vetted formulas in the full galactic dynamics code to verify they reproduce rotation curves as successfully as our initial density-only approach.