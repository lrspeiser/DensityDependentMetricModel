
# Add to run_dynesty.py configuration
RG_INFORMED_PRIORS = {
    'n': (0.70, 0.90),  # Tight around 4γ
    'rho_c': (4.47e-20, 1.79e-19),
    'gamma': (0.190, 0.210)
}

# Likelihood augmentation
def log_likelihood_total(params):
    log_L = log_likelihood_gaia(params)  # Existing
    
    # Add multi-messenger constraints
    ddg = DensityDependentGravity(params['rho_c'], params['n'])
    mm = MultiMessengerConstraints(ddg)
    log_L += mm.combined_log_likelihood()
    
    # RG theory prior
    n_theory = 4 * params.get('gamma', 0.200)
    log_L += -0.5 * ((params['n'] - n_theory) / 0.05)**2
    
    return log_L
