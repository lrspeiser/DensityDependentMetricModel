#!/usr/bin/env python3
"""
monitor_dashboard.py - Real-time monitoring dashboard for dynesty runs

This module provides:
- JSON-based progress tracking
- Command-line dashboard
- Web-based monitoring interface
- Email notifications for key events
- Real-time plotting of convergence

Author: Enhanced Monitoring System
Version: 1.0
"""

import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import argparse
import sys
import os
import pickle
import gzip
from typing import Dict, List, Optional, Tuple
import logging

# Optional imports for advanced features
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.backends.backend_pdf import PdfPages
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: matplotlib not available for plotting")

try:
    from flask import Flask, render_template, jsonify
    from flask_cors import CORS
    WEB_AVAILABLE = True
except ImportError:
    WEB_AVAILABLE = False

try:
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    EMAIL_AVAILABLE = True
except ImportError:
    EMAIL_AVAILABLE = False


class DynestyMonitor:
    """Main monitoring class for dynesty runs."""
    
    def __init__(self, output_dir: Path, config_file: Optional[Path] = None):
        self.output_dir = Path(output_dir)
        self.progress_file = self.output_dir / "progress.json"
        self.metrics_file = self.output_dir / "metrics_history.json"
        self.checkpoint_file = self.output_dir / "dynesty_checkpoint.pkl"
        
        # Load configuration
        self.config = self._load_config(config_file)
        
        # Initialize metrics history
        self.metrics_history = self._load_metrics_history()
        
    def _load_config(self, config_file: Optional[Path]) -> Dict:
        """Load monitoring configuration."""
        default_config = {
            "update_interval": 300,  # 5 minutes
            "email_notifications": False,
            "email_settings": {
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "sender": "",
                "password": "",
                "recipients": []
            },
            "notification_triggers": {
                "convergence_near": 0.05,  # Notify when dlogz < this
                "efficiency_low": 1.0,     # Notify when efficiency < this %
                "stuck_threshold": 3600,   # Notify if no progress for 1 hour
                "error_threshold": 5       # Notify after this many errors
            },
            "plotting": {
                "enabled": True,
                "update_interval": 1800,  # 30 minutes
                "metrics": ["logz", "dlogz", "efficiency", "parameters"]
            }
        }
        
        if config_file and config_file.exists():
            with open(config_file, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    def _load_metrics_history(self) -> List[Dict]:
        """Load historical metrics."""
        if self.metrics_file.exists():
            with open(self.metrics_file, 'r') as f:
                return json.load(f)
        return []
    
    def update_progress(self, sampler_state: Dict):
        """Update progress file with current state."""
        timestamp = datetime.now()
        
        # Extract key metrics
        progress = {
            "timestamp": timestamp.isoformat(),
            "elapsed_time": sampler_state.get("elapsed_time", 0),
            "n_samples": sampler_state.get("n_samples", 0),
            "n_calls": sampler_state.get("n_calls", 0),
            "efficiency": sampler_state.get("efficiency", 0),
            "logz": sampler_state.get("logz", -np.inf),
            "logz_err": sampler_state.get("logz_err", 0),
            "dlogz": sampler_state.get("dlogz", np.inf),
            "current_nlive": sampler_state.get("current_nlive", 0),
            "parameter_estimates": sampler_state.get("parameter_estimates", {}),
            "parameter_uncertainties": sampler_state.get("parameter_uncertainties", {}),
            "convergence_status": self._assess_convergence(sampler_state),
            "health_warnings": sampler_state.get("health_warnings", []),
            "estimated_completion": self._estimate_completion(sampler_state)
        }
        
        # Save current progress
        with open(self.progress_file, 'w') as f:
            json.dump(progress, f, indent=2)
        
        # Update metrics history
        self.metrics_history.append(progress)
        if len(self.metrics_history) > 10000:  # Keep last 10k entries
            self.metrics_history = self.metrics_history[-10000:]
        
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics_history, f)
        
        # Check notification triggers
        self._check_notifications(progress)
        
        # Update plots if enabled
        if self.config["plotting"]["enabled"] and PLOTTING_AVAILABLE:
            last_plot_time = getattr(self, '_last_plot_time', 0)
            if time.time() - last_plot_time > self.config["plotting"]["update_interval"]:
                self._update_plots()
                self._last_plot_time = time.time()
    
    def _assess_convergence(self, state: Dict) -> Dict:
        """Assess convergence status."""
        dlogz = state.get("dlogz", np.inf)
        efficiency = state.get("efficiency", 0)
        
        status = {
            "converged": dlogz < 0.01,
            "near_convergence": dlogz < 0.05,
            "dlogz": dlogz,
            "efficiency_ok": efficiency > 2.0,
            "stuck": False  # Will be determined by history
        }
        
        # Check if stuck
        if len(self.metrics_history) > 10:
            recent_logz = [m["logz"] for m in self.metrics_history[-10:]]
            if all(abs(recent_logz[i] - recent_logz[0]) < 0.01 for i in range(len(recent_logz))):
                status["stuck"] = True
        
        return status
    
    def _estimate_completion(self, state: Dict) -> Optional[str]:
        """Estimate time to completion based on convergence rate."""
        if len(self.metrics_history) < 10:
            return None
        
        # Get recent dlogz history
        recent_dlogz = [m["dlogz"] for m in self.metrics_history[-20:] if m["dlogz"] < np.inf]
        if len(recent_dlogz) < 5:
            return None
        
        # Estimate convergence rate
        times = np.arange(len(recent_dlogz))
        log_dlogz = np.log(recent_dlogz)
        
        try:
            # Fit exponential decay
            coeffs = np.polyfit(times, log_dlogz, 1)
            decay_rate = -coeffs[0]
            
            if decay_rate > 0:
                # Estimate time to reach dlogz = 0.01
                current_dlogz = recent_dlogz[-1]
                target_dlogz = 0.01
                
                if current_dlogz > target_dlogz:
                    steps_remaining = np.log(current_dlogz / target_dlogz) / decay_rate
                    time_per_step = self.config["update_interval"] / 60  # minutes
                    eta_minutes = steps_remaining * time_per_step
                    
                    eta = datetime.now() + timedelta(minutes=eta_minutes)
                    return eta.strftime("%Y-%m-%d %H:%M")
        except:
            pass
        
        return None
    
    def _check_notifications(self, progress: Dict):
        """Check if any notification triggers are met."""
        if not self.config["email_notifications"] or not EMAIL_AVAILABLE:
            return
        
        triggers = self.config["notification_triggers"]
        notifications = []
        
        # Check convergence
        if progress["dlogz"] < triggers["convergence_near"]:
            notifications.append(f"Near convergence: dlogz = {progress['dlogz']:.4f}")
        
        # Check efficiency
        if progress["efficiency"] < triggers["efficiency_low"]:
            notifications.append(f"Low efficiency: {progress['efficiency']:.2f}%")
        
        # Check if stuck
        if progress["convergence_status"]["stuck"]:
            notifications.append("Sampling appears stuck - no progress in last 10 updates")
        
        # Check health warnings
        if progress["health_warnings"]:
            notifications.append(f"Health warnings: {', '.join(progress['health_warnings'])}")
        
        if notifications:
            self._send_notification(notifications, progress)
    
    def _send_notification(self, messages: List[str], progress: Dict):
        """Send email notification."""
        if not EMAIL_AVAILABLE:
            return
        
        settings = self.config["email_settings"]
        if not all([settings["sender"], settings["password"], settings["recipients"]]):
            return
        
        # Create message
        msg = MIMEMultipart()
        msg['From'] = settings["sender"]
        msg['To'] = ", ".join(settings["recipients"])
        msg['Subject'] = f"Dynesty Monitor Alert - {self.output_dir.name}"
        
        body = f"""
Dynesty Monitoring Alert
========================

Time: {progress['timestamp']}
Elapsed: {progress['elapsed_time']:.1f} hours
Log(Z): {progress['logz']:.3f} ± {progress['logz_err']:.3f}
dlog(Z): {progress['dlogz']:.4f}
Efficiency: {progress['efficiency']:.2f}%

Alerts:
{chr(10).join(f"- {msg}" for msg in messages)}

Parameter Estimates:
{json.dumps(progress['parameter_estimates'], indent=2)}

Estimated Completion: {progress['estimated_completion'] or 'Unknown'}
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        try:
            server = smtplib.SMTP(settings["smtp_server"], settings["smtp_port"])
            server.starttls()
            server.login(settings["sender"], settings["password"])
            server.send_message(msg)
            server.quit()
        except Exception as e:
            print(f"Failed to send notification: {e}")
    
    def _update_plots(self):
        """Update monitoring plots."""
        if not PLOTTING_AVAILABLE or len(self.metrics_history) < 2:
            return
        
        # Prepare data
        times = [datetime.fromisoformat(m["timestamp"]) for m in self.metrics_history]
        logz = [m["logz"] for m in self.metrics_history if m["logz"] > -np.inf]
        dlogz = [m["dlogz"] for m in self.metrics_history if 0 < m["dlogz"] < np.inf]
        efficiency = [m["efficiency"] for m in self.metrics_history]
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Dynesty Progress - {self.output_dir.name}', fontsize=16)
        
        # Plot 1: Log(Z) evolution
        ax = axes[0, 0]
        if logz:
            ax.plot(times[-len(logz):], logz, 'b-', linewidth=2)
            ax.set_ylabel('log(Z)', fontsize=12)
            ax.set_title('Evidence Evolution')
            ax.grid(True, alpha=0.3)
        
        # Plot 2: dlog(Z) evolution (log scale)
        ax = axes[0, 1]
        if dlogz:
            ax.semilogy(times[-len(dlogz):], dlogz, 'r-', linewidth=2)
            ax.axhline(0.01, color='g', linestyle='--', label='Target')
            ax.set_ylabel('dlog(Z)', fontsize=12)
            ax.set_title('Convergence Progress')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot 3: Efficiency
        ax = axes[1, 0]
        ax.plot(times, efficiency, 'g-', linewidth=2)
        ax.axhline(2.0, color='r', linestyle='--', label='Good efficiency')
        ax.set_ylabel('Efficiency (%)', fontsize=12)
        ax.set_title('Sampling Efficiency')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Parameter evolution
        ax = axes[1, 1]
        param_history = {}
        for m in self.metrics_history[-100:]:  # Last 100 points
            for param, value in m["parameter_estimates"].items():
                if param not in param_history:
                    param_history[param] = []
                param_history[param].append(value)
        
        if param_history:
            for param, values in list(param_history.items())[:5]:  # Show top 5 params
                normalized_values = np.array(values) / np.median(values)
                ax.plot(times[-len(values):], normalized_values, label=param)
            ax.set_ylabel('Normalized Value', fontsize=12)
            ax.set_title('Parameter Evolution (normalized)')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        # Format x-axes
        for ax in axes.flat:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        # Save
        plot_file = self.output_dir / "progress_plots.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Also save as PDF with all history
        pdf_file = self.output_dir / "full_progress_report.pdf"
        self._create_full_report(pdf_file)
    
    def _create_full_report(self, pdf_file: Path):
        """Create comprehensive PDF report."""
        if not PLOTTING_AVAILABLE:
            return
        
        with PdfPages(pdf_file) as pdf:
            # Page 1: Overview
            self._create_overview_page(pdf)
            
            # Page 2: Detailed parameter plots
            self._create_parameter_page(pdf)
            
            # Page 3: Diagnostics
            self._create_diagnostics_page(pdf)
    
    def _create_overview_page(self, pdf):
        """Create overview page for PDF report."""
        fig = plt.figure(figsize=(11, 8.5))
        
        # Add text summary
        latest = self.metrics_history[-1] if self.metrics_history else {}
        
        summary_text = f"""
DYNESTY RUN SUMMARY
==================
Output Directory: {self.output_dir}
Last Update: {latest.get('timestamp', 'N/A')}
Total Runtime: {latest.get('elapsed_time', 0):.2f} hours
Total Samples: {latest.get('n_samples', 0):,}
Total Calls: {latest.get('n_calls', 0):,}
Efficiency: {latest.get('efficiency', 0):.2f}%

CONVERGENCE STATUS
==================
log(Z): {latest.get('logz', -np.inf):.3f} ± {latest.get('logz_err', 0):.3f}
dlog(Z): {latest.get('dlogz', np.inf):.4f}
Status: {'CONVERGED' if latest.get('dlogz', np.inf) < 0.01 else 'IN PROGRESS'}
Estimated Completion: {latest.get('estimated_completion', 'Unknown')}

HEALTH STATUS
=============
Warnings: {', '.join(latest.get('health_warnings', ['None']))}
"""
        
        plt.text(0.1, 0.9, summary_text, transform=fig.transFigure, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _create_parameter_page(self, pdf):
        """Create detailed parameter evolution plots."""
        # Implementation depends on specific parameters
        pass
    
    def _create_diagnostics_page(self, pdf):
        """Create diagnostic plots."""
        # Implementation for diagnostic plots
        pass
    
    def get_current_status(self) -> Dict:
        """Get current status for command-line display."""
        if not self.progress_file.exists():
            return {"status": "No progress file found"}
        
        with open(self.progress_file, 'r') as f:
            return json.load(f)
    
    def create_dashboard_string(self) -> str:
        """Create a formatted dashboard string for terminal display."""
        status = self.get_current_status()
        
        if "status" in status:
            return status["status"]
        
        # Format elapsed time
        elapsed_hours = status.get("elapsed_time", 0)
        elapsed_str = f"{int(elapsed_hours)}h {int((elapsed_hours % 1) * 60)}m"
        
        # Build dashboard
        dashboard = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                          DYNESTY MONITORING DASHBOARD                         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ Last Update: {status.get('timestamp', 'N/A'):<62} ║
║ Runtime: {elapsed_str:<69} ║
║ Output Dir: {str(self.output_dir):<65} ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                              SAMPLING STATISTICS                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ Samples: {status.get('n_samples', 0):>15,} │ Calls: {status.get('n_calls', 0):>15,} │ Efficiency: {status.get('efficiency', 0):>6.2f}% ║
║ Live Points: {status.get('current_nlive', 0):>11} │ log(Z): {status.get('logz', -np.inf):>15.3f} │ dlog(Z): {status.get('dlogz', np.inf):>10.4f} ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                              CONVERGENCE STATUS                               ║
╠══════════════════════════════════════════════════════════════════════════════╣
"""
        
        # Add convergence status
        conv_status = status.get('convergence_status', {})
        if conv_status.get('converged'):
            dashboard += "║ Status: ✅ CONVERGED                                                        ║\n"
        elif conv_status.get('near_convergence'):
            dashboard += "║ Status: 🟡 NEAR CONVERGENCE                                                ║\n"
        elif conv_status.get('stuck'):
            dashboard += "║ Status: ⚠️  STUCK - No progress detected                                    ║\n"
        else:
            dashboard += "║ Status: 🔄 IN PROGRESS                                                      ║\n"
        
        # Add ETA
        eta = status.get('estimated_completion')
        if eta:
            dashboard += f"║ Estimated Completion: {eta:<54} ║\n"
        
        # Add parameters
        params = status.get('parameter_estimates', {})
        if params:
            dashboard += "╠══════════════════════════════════════════════════════════════════════════════╣\n"
            dashboard += "║                           PARAMETER ESTIMATES                                 ║\n"
            dashboard += "╠══════════════════════════════════════════════════════════════════════════════╣\n"
            
            for param, value in list(params.items())[:8]:  # Show top 8 parameters
                param_str = f"{param}: {value:.3e}"
                dashboard += f"║ {param_str:<77} ║\n"
        
        # Add warnings
        warnings = status.get('health_warnings', [])
        if warnings:
            dashboard += "╠══════════════════════════════════════════════════════════════════════════════╣\n"
            dashboard += "║                              ⚠️  WARNINGS ⚠️                                   ║\n"
            dashboard += "╠══════════════════════════════════════════════════════════════════════════════╣\n"
            for warning in warnings[:5]:  # Show top 5 warnings
                dashboard += f"║ • {warning:<74} ║\n"
        
        dashboard += "╚══════════════════════════════════════════════════════════════════════════════╝"
        
        return dashboard


def create_web_dashboard(monitor: DynestyMonitor, port: int = 5000):
    """Create a web-based dashboard."""
    if not WEB_AVAILABLE:
        print("Flask not available. Install with: pip install flask flask-cors")
        return
    
    app = Flask(__name__)
    CORS(app)
    
    @app.route('/')
    def index():
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Dynesty Monitor</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .metric { display: inline-block; margin: 10px; padding: 15px; 
                         background: #f0f0f0; border-radius: 5px; }
                .metric h3 { margin: 0 0 10px 0; }
                .metric .value { font-size: 24px; font-weight: bold; }
                #plots { margin-top: 20px; }
            </style>
        </head>
        <body>
            <h1>Dynesty Monitoring Dashboard</h1>
            <div id="metrics"></div>
            <div id="plots"></div>
            <script>
                function updateDashboard() {
                    fetch('/api/status')
                        .then(response => response.json())
                        .then(data => {
                            // Update metrics
                            document.getElementById('metrics').innerHTML = `
                                <div class="metric">
                                    <h3>Runtime</h3>
                                    <div class="value">${data.elapsed_time.toFixed(1)}h</div>
                                </div>
                                <div class="metric">
                                    <h3>log(Z)</h3>
                                    <div class="value">${data.logz.toFixed(3)}</div>
                                </div>
                                <div class="metric">
                                    <h3>dlog(Z)</h3>
                                    <div class="value">${data.dlogz.toFixed(4)}</div>
                                </div>
                                <div class="metric">
                                    <h3>Efficiency</h3>
                                    <div class="value">${data.efficiency.toFixed(2)}%</div>
                                </div>
                            `;
                        });
                    
                    // Update plots
                    fetch('/api/history')
                        .then(response => response.json())
                        .then(data => {
                            // Create plots using Plotly
                            const times = data.map(d => d.timestamp);
                            const logz = data.map(d => d.logz);
                            
                            const trace = {
                                x: times,
                                y: logz,
                                type: 'scatter',
                                name: 'log(Z)'
                            };
                            
                            const layout = {
                                title: 'Evidence Evolution',
                                xaxis: { title: 'Time' },
                                yaxis: { title: 'log(Z)' }
                            };
                            
                            Plotly.newPlot('plots', [trace], layout);
                        });
                }
                
                // Update every 30 seconds
                updateDashboard();
                setInterval(updateDashboard, 30000);
            </script>
        </body>
        </html>
        """
    
    @app.route('/api/status')
    def api_status():
        return jsonify(monitor.get_current_status())
    
    @app.route('/api/history')
    def api_history():
        return jsonify(monitor.metrics_history[-100:])  # Last 100 points
    
    print(f"Starting web dashboard at http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)


def main():
    """Command-line interface for monitoring."""
    parser = argparse.ArgumentParser(description="Monitor dynesty runs")
    parser.add_argument('output_dir', help="Dynesty output directory")
    parser.add_argument('--config', help="Configuration file")
    parser.add_argument('--web', action='store_true', help="Start web dashboard")
    parser.add_argument('--port', type=int, default=5000, help="Web dashboard port")
    parser.add_argument('--watch', action='store_true', help="Continuous monitoring")
    parser.add_argument('--interval', type=int, default=30, help="Watch interval (seconds)")
    
    args = parser.parse_args()
    
    # Create monitor
    monitor = DynestyMonitor(Path(args.output_dir), Path(args.config) if args.config else None)
    
    if args.web:
        # Start web dashboard
        create_web_dashboard(monitor, args.port)
    elif args.watch:
        # Continuous monitoring
        print("Starting continuous monitoring. Press Ctrl+C to stop.")
        try:
            while True:
                os.system('clear' if os.name == 'posix' else 'cls')
                print(monitor.create_dashboard_string())
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")
    else:
        # Single status check
        print(monitor.create_dashboard_string())


if __name__ == "__main__":
    main()