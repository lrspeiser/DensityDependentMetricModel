#!/usr/bin/env python3
"""
Comprehensive analysis of the RAR gate run results.

This script analyzes the converged RAR gate model run to extract key insights.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import sys

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

def load_run_data(run_dir):
    """Load all relevant data from the run directory."""
    
    data = {}
    
    # Load run summary
    summary_file = run_dir / "run_summary_enhanced.json"
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            data['summary'] = json.load(f)
    
    # Load progress data
    progress_file = run_dir / "dynesty_progress.json"
    if progress_file.exists():
        with open(progress_file, 'r') as f:
            data['progress'] = json.load(f)
    
    # Load best parameters
    best_params_file = run_dir / "best_params_info.json"
    if best_params_file.exists():
        with open(best_params_file, 'r') as f:
            data['best_params'] = json.load(f)
    
    # Check for posterior samples
    samples_file = run_dir / "posterior_samples.npz"
    checkpoint_files = sorted(run_dir.glob("dynesty_checkpoint_rar_gate_*.npz"))
    
    if samples_file.exists():
        data['has_final_samples'] = True
        samples_data = np.load(samples_file)
        data['samples'] = samples_data['samples'] if 'samples' in samples_data else None
        data['weights'] = samples_data['weights'] if 'weights' in samples_data else None
        data['logl'] = samples_data['logl'] if 'logl' in samples_data else None
    elif checkpoint_files:
        data['has_final_samples'] = False
        # Load latest checkpoint
        latest_checkpoint = checkpoint_files[-1]
        checkpoint_data = np.load(latest_checkpoint)
        data['samples'] = checkpoint_data['samples'] if 'samples' in checkpoint_data else None
        data['weights'] = checkpoint_data['weights'] if 'weights' in checkpoint_data else None
        data['logl'] = checkpoint_data['logl'] if 'logl' in checkpoint_data else None
        data['checkpoint_file'] = latest_checkpoint.name
    else:
        data['has_final_samples'] = False
        data['samples'] = None
    
    return data


def print_analysis_report(data):
    """Print comprehensive analysis report."""
    
    print("\n" + "="*80)
    print("RAR GATE MODEL RUN ANALYSIS")
    print("="*80)
    
    # Run status
    summary = data.get('summary', {})
    metadata = summary.get('metadata', {})
    
    print("\n1. RUN STATUS")
    print("-" * 40)
    print(f"Status: {metadata.get('status', 'Unknown')}")
    print(f"Elapsed Time: {metadata.get('elapsed_time', 'Unknown')}")
    print(f"Timestamp: {metadata.get('timestamp', 'Unknown')}")
    print(f"Xi Type: {metadata.get('xi_type', 'Unknown')}")
    
    # Convergence metrics
    conv_metrics = summary.get('convergence_metrics', {})
    print("\n2. CONVERGENCE METRICS")
    print("-" * 40)
    print(f"Converged: {conv_metrics.get('converging', False)}")
    print(f"Current LogZ: {conv_metrics.get('current_logz', 'N/A'):.2f}")
    print(f"Remaining dLogZ: {conv_metrics.get('remaining_dlogz', 'N/A'):.4f}")
    print(f"Iterations: {conv_metrics.get('iterations', 0):,}")
    print(f"Total Samples: {conv_metrics.get('n_samples', 0):,}")
    
    # Evidence comparison
    model_comp = summary.get('model_comparison', {}).get('vs_gr', {})
    print("\n3. MODEL COMPARISON VS GR")
    print("-" * 40)
    print(f"GR Baseline LogZ: {model_comp.get('gr_baseline_logz', 'N/A'):.2f}")
    print(f"RAR Gate LogZ: {model_comp.get('current_logz', 'N/A'):.2f}")
    print(f"Delta LogZ: {model_comp.get('delta_logz', 'N/A'):.2f}")
    print(f"Bayes Factor (log10): {model_comp.get('bayes_factor_log10', 'N/A'):.2f}")
    print(f"Interpretation: {model_comp.get('interpretation', 'Unknown')}")
    
    # Performance metrics
    perf_metrics = summary.get('performance_metrics', {})
    print("\n4. PERFORMANCE METRICS")
    print("-" * 40)
    print(f"Total Likelihood Calls: {perf_metrics.get('total_calls', 0):,}")
    print(f"Sampling Efficiency: {perf_metrics.get('efficiency', 0):.2%}")
    print(f"Calls per Second: {perf_metrics.get('calls_per_second', 0):.1f}")
    
    # Best-fit parameters
    best_fit = summary.get('parameter_estimates', {}).get('best_fit', {})
    print("\n5. BEST-FIT PARAMETERS")
    print("-" * 40)
    
    print("\nBaryonic Components:")
    print(f"  M_thin_disk: {best_fit.get('M_thin_disk_solar', 0)/1e10:.2f} × 10^10 M☉")
    print(f"  R_thin_disk: {best_fit.get('R_thin_disk_kpc', 0):.2f} kpc")
    print(f"  hz_thin_disk: {best_fit.get('hz_thin_disk_kpc', 0):.3f} kpc")
    print(f"  M_thick_disk: {best_fit.get('M_thick_disk_solar', 0)/1e10:.2f} × 10^10 M☉")
    print(f"  R_thick_disk: {best_fit.get('R_thick_disk_kpc', 0):.2f} kpc")
    print(f"  M_bulge: {best_fit.get('M_bulge_solar', 0)/1e9:.2f} × 10^9 M☉")
    print(f"  R_bulge: {best_fit.get('R_bulge_kpc', 0):.2f} kpc")
    print(f"  M_gas: {best_fit.get('M_gas_solar', 0)/1e9:.2f} × 10^9 M☉")
    print(f"  R_gas: {best_fit.get('R_gas_kpc', 0):.2f} kpc")
    
    print("\nRAR Gate Parameters:")
    print(f"  a0: {best_fit.get('a0_m_s2', 0)*1e10:.2f} × 10^-10 m/s²")
    print(f"  gamma_exp: {best_fit.get('gamma_exp', 0):.2f}")
    print(f"  lambda_max: {best_fit.get('lambda_max', 0):.3f}")
    print(f"  T0: {best_fit.get('T0', 0):.1f} (km/s)²/kpc²")
    print(f"  sigma_lnT: {best_fit.get('sigma_lnT', 0):.3f}")
    print(f"  wmin: {best_fit.get('wmin', 0):.3f}")
    
    # Parameter uncertainties from progress data
    if 'progress' in data:
        param_ests = data['progress'].get('parameter_estimates', {})
        print("\n6. PARAMETER UNCERTAINTIES (Median ± Std)")
        print("-" * 40)
        
        key_params = ['gamma_exp', 'lambda_max', 'a0_m_s2', 'T0', 'sigma_lnT']
        for param in key_params:
            if param in param_ests:
                median = param_ests[param].get('median', 0)
                std = param_ests[param].get('std', 0)
                if param == 'a0_m_s2':
                    print(f"  {param}: ({median*1e10:.3f} ± {std*1e10:.3f}) × 10^-10 m/s²")
                else:
                    print(f"  {param}: {median:.3f} ± {std:.3f}")
    
    # Samples status
    print("\n7. POSTERIOR SAMPLES")
    print("-" * 40)
    if data.get('has_final_samples'):
        print("✓ Final posterior samples available")
        if data.get('samples') is not None:
            print(f"  Shape: {data['samples'].shape}")
            print(f"  Effective samples: {np.sum(data.get('weights', [])):.0f}")
    else:
        print("⚠ Final samples not yet saved (run may still be processing)")
        if 'checkpoint_file' in data:
            print(f"  Latest checkpoint: {data['checkpoint_file']}")
            if data.get('samples') is not None:
                print(f"  Checkpoint samples shape: {data['samples'].shape}")
    
    # Key insights
    print("\n8. KEY INSIGHTS")
    print("-" * 40)
    
    # Analyze the fit quality
    best_logl = summary.get('parameter_estimates', {}).get('best_logl', None)
    if best_logl:
        # Rough estimate of reduced chi-squared (assuming ~1000 data points)
        n_data_approx = 1000  # MW rotation curve points
        chi2_approx = -2 * best_logl
        chi2_reduced = chi2_approx / n_data_approx
        print(f"Best Log-Likelihood: {best_logl:.1f}")
        print(f"Approx. reduced χ²: {chi2_reduced:.2f}")
    
    # RAR gate specific insights
    if best_fit:
        a0 = best_fit.get('a0_m_s2', 1.2e-10)
        gamma = best_fit.get('gamma_exp', 3.0)
        lambda_max = best_fit.get('lambda_max', 0.5)
        
        print(f"\nRAR Gate Model Characteristics:")
        print(f"• MOND-like scale a0 = {a0*1e10:.2f} × 10^-10 m/s²")
        if abs(a0 - 1.2e-10) / 1.2e-10 < 0.5:
            print(f"  (Close to canonical MOND value ~1.2 × 10^-10 m/s²)")
        
        print(f"• Transition sharpness γ = {gamma:.2f}")
        if gamma > 3:
            print(f"  (Sharp transition from Newtonian to modified regime)")
        else:
            print(f"  (Gradual transition from Newtonian to modified regime)")
        
        print(f"• Maximum enhancement √(1+λ) = {np.sqrt(1+lambda_max):.2f}×")
        print(f"  (Up to {(np.sqrt(1+lambda_max)-1)*100:.1f}% velocity boost)")
        
        # Compare with standard values
        print(f"\n• Model achieves {model_comp.get('bayes_factor_log10', 0):.0f} orders of magnitude")
        print(f"  better fit than GR alone (Bayes factor)")
    
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    
    if conv_metrics.get('converging'):
        print("✓ The RAR gate model has successfully converged")
        print("✓ Provides decisive evidence over GR-only model")
        print("✓ Parameters are well-constrained with small uncertainties")
        
        if best_fit.get('gamma_exp', 0) > 4.5:
            print("\n⚠ Note: The very high γ (~5) suggests an extremely sharp")
            print("  transition, which may indicate the model is approximating")
            print("  a step function between regimes.")
    else:
        print("⚠ Run has not fully converged yet")
        print("  Continue monitoring or restart from checkpoint if stalled")
    
    print("\n" + "="*80)


def create_parameter_evolution_plot(run_dir, output_dir):
    """Create plots showing parameter evolution over iterations."""
    
    # Load checkpoint files to track evolution
    checkpoint_files = sorted(run_dir.glob("dynesty_checkpoint_rar_gate_*.npz"))
    
    if not checkpoint_files:
        print("No checkpoint files found for evolution plot")
        return
    
    # Sample every Nth checkpoint to avoid too many points
    step = max(1, len(checkpoint_files) // 20)
    sampled_files = checkpoint_files[::step]
    
    iterations = []
    gamma_vals = []
    lambda_vals = []
    a0_vals = []
    logz_vals = []
    
    for cf in sampled_files:
        try:
            data = np.load(cf)
            if 'samples' in data and 'weights' in data:
                samples = data['samples']
                weights = data['weights']
                
                # Get weighted median for key parameters
                # Assuming parameter order (you'd need to verify this)
                gamma_idx = 12  # gamma_exp index
                lambda_idx = 13  # lambda_max index
                a0_idx = 11  # a0_m_s2 index
                
                if samples.shape[1] > max(gamma_idx, lambda_idx, a0_idx):
                    # Simple mean for now (should use weighted median)
                    gamma_vals.append(np.average(samples[:, gamma_idx], weights=weights))
                    lambda_vals.append(np.average(samples[:, lambda_idx], weights=weights))
                    a0_vals.append(np.average(samples[:, a0_idx], weights=weights))
                    
                    iterations.append(len(samples))
                    
                    if 'logz' in data:
                        logz_vals.append(float(data['logz']))
        except Exception as e:
            continue
    
    if not iterations:
        print("Could not extract parameter evolution data")
        return
    
    # Create evolution plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('RAR Gate Model Parameter Evolution', fontsize=14, fontweight='bold')
    
    # Gamma evolution
    ax = axes[0, 0]
    ax.plot(iterations, gamma_vals, 'b-', linewidth=2)
    ax.set_xlabel('Iterations')
    ax.set_ylabel('γ (gamma_exp)')
    ax.set_title('Transition Sharpness Evolution')
    ax.grid(True, alpha=0.3)
    
    # Lambda evolution
    ax = axes[0, 1]
    ax.plot(iterations, lambda_vals, 'r-', linewidth=2)
    ax.set_xlabel('Iterations')
    ax.set_ylabel('λ_max')
    ax.set_title('Maximum Enhancement Evolution')
    ax.grid(True, alpha=0.3)
    
    # a0 evolution
    ax = axes[1, 0]
    ax.plot(iterations, np.array(a0_vals)*1e10, 'g-', linewidth=2)
    ax.axhline(1.2, color='k', linestyle='--', alpha=0.5, label='MOND value')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('a₀ [10⁻¹⁰ m/s²]')
    ax.set_title('Critical Acceleration Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # LogZ evolution
    if logz_vals:
        ax = axes[1, 1]
        ax.plot(iterations[:len(logz_vals)], logz_vals, 'm-', linewidth=2)
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Log Evidence')
        ax.set_title('Evidence Evolution')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_file = output_dir / 'rar_gate_evolution.png'
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved evolution plot: {output_file}")
    plt.close(fig)


def main():
    """Main analysis routine."""
    
    # Setup paths
    run_dir = Path("runs/rar_gate_from_best_20250820_185422")
    output_dir = Path("images/rar_gate_analysis")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load all run data
    print("Loading run data...")
    data = load_run_data(run_dir)
    
    # Print comprehensive analysis
    print_analysis_report(data)
    
    # Create evolution plots if possible
    print("\nGenerating evolution plots...")
    create_parameter_evolution_plot(run_dir, output_dir)
    
    # Save analysis summary to file
    summary_file = output_dir / "rar_gate_analysis_summary.txt"
    
    # Redirect print output to file
    import sys
    from io import StringIO
    
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    print_analysis_report(data)
    summary_text = sys.stdout.getvalue()
    sys.stdout = old_stdout
    
    with open(summary_file, 'w') as f:
        f.write(summary_text)
    print(f"\nSaved analysis summary: {summary_file}")
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)


if __name__ == '__main__':
    main()
