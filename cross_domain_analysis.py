"""
Cross-Domain Lyapunov Analysis: Supernovae ↔ Neural Dynamics
=============================================================

This script demonstrates how Lyapunov exponents can serve as a 
bridge between astrophysical simulations (from The Well) and 
computational neuroscience models.

The key insight: Both systems exhibit similar mathematical structure
in their chaos signatures, enabling potential transfer learning.

Author: Claude (Anthropic)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp
from scipy.ndimage import gaussian_filter
from typing import Dict, Tuple, List
import warnings

# Import our Lyapunov toolkit
from lyapunov_well import (
    RosensteinMethod, 
    WolfMethod, 
    SpatialLyapunovField,
    FiniteTimeLyapunovExponent,
    plot_divergence_curve,
    plot_lyapunov_field
)


# ============================================================================
# NEURAL DYNAMICS MODELS
# ============================================================================

class FitzHughNagumo:
    """
    FitzHugh-Nagumo model - simplified Hodgkin-Huxley.
    
    This is a 2D reduction of neural dynamics that captures:
    - Excitability
    - Spike generation  
    - Refractory periods
    
    The PDE version models wave propagation in neural tissue,
    analogous to shock propagation in supernova simulations.
    """
    
    def __init__(
        self,
        a: float = 0.7,
        b: float = 0.8,
        tau: float = 12.5,
        I_ext: float = 0.5,
        D: float = 1.0  # Diffusion coefficient
    ):
        """
        Parameters
        ----------
        a, b : float
            Model parameters controlling excitability
        tau : float
            Time scale separation
        I_ext : float
            External input current
        D : float
            Diffusion coefficient for spatial coupling
        """
        self.a = a
        self.b = b
        self.tau = tau
        self.I_ext = I_ext
        self.D = D
        
    def ode_rhs(self, t: float, state: np.ndarray) -> np.ndarray:
        """Right-hand side for ODE (0D point model)."""
        v, w = state
        
        dv = v - v**3/3 - w + self.I_ext
        dw = (v + self.a - self.b * w) / self.tau
        
        return np.array([dv, dw])
    
    def simulate_0d(
        self, 
        t_span: Tuple[float, float],
        initial_state: np.ndarray = None,
        n_points: int = 10000
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate 0D (point) model."""
        if initial_state is None:
            initial_state = np.array([0.0, 0.0])
            
        t_eval = np.linspace(t_span[0], t_span[1], n_points)
        
        sol = solve_ivp(
            self.ode_rhs,
            t_span,
            initial_state,
            t_eval=t_eval,
            method='RK45'
        )
        
        return sol.t, sol.y.T
    
    def simulate_2d(
        self,
        nx: int = 64,
        ny: int = 64,
        n_timesteps: int = 500,
        dt: float = 0.1,
        dx: float = 1.0,
        initial_stimulus: str = 'center'
    ) -> np.ndarray:
        """
        Simulate 2D spatial FitzHugh-Nagumo (reaction-diffusion).
        
        This models wave propagation in neural tissue.
        """
        # Initialize fields
        v = np.zeros((nx, ny))
        w = np.zeros((nx, ny))
        
        # Initial stimulus
        if initial_stimulus == 'center':
            cx, cy = nx // 2, ny // 2
            v[cx-5:cx+5, cy-5:cy+5] = 1.0
        elif initial_stimulus == 'corner':
            v[:10, :10] = 1.0
        elif initial_stimulus == 'random':
            np.random.seed(42)
            v = 0.1 * np.random.randn(nx, ny)
            
        # Storage
        v_history = np.zeros((n_timesteps, nx, ny))
        
        # Laplacian stencil coefficient
        D_coeff = self.D * dt / dx**2
        
        for t in range(n_timesteps):
            v_history[t] = v
            
            # Compute Laplacian (5-point stencil with periodic BC)
            laplacian_v = (
                np.roll(v, 1, axis=0) + np.roll(v, -1, axis=0) +
                np.roll(v, 1, axis=1) + np.roll(v, -1, axis=1) -
                4 * v
            ) / dx**2
            
            # FHN dynamics
            dv = v - v**3/3 - w + self.I_ext + self.D * laplacian_v
            dw = (v + self.a - self.b * w) / self.tau
            
            # Euler step
            v = v + dt * dv
            w = w + dt * dw
            
        return v_history


class WilsonCowan:
    """
    Wilson-Cowan neural field model.
    
    Models interactions between excitatory and inhibitory
    neural populations - relevant for cortical dynamics.
    """
    
    def __init__(
        self,
        tau_e: float = 1.0,
        tau_i: float = 1.0,
        w_ee: float = 10.0,
        w_ei: float = -10.0,
        w_ie: float = 10.0,
        w_ii: float = -2.0,
        theta_e: float = 2.0,
        theta_i: float = 3.5,
        D_e: float = 0.5,
        D_i: float = 0.1
    ):
        self.tau_e = tau_e
        self.tau_i = tau_i
        self.w_ee = w_ee
        self.w_ei = w_ei
        self.w_ie = w_ie
        self.w_ii = w_ii
        self.theta_e = theta_e
        self.theta_i = theta_i
        self.D_e = D_e
        self.D_i = D_i
        
    @staticmethod
    def sigmoid(x: np.ndarray, theta: float, beta: float = 1.0) -> np.ndarray:
        """Sigmoidal activation function."""
        return 1.0 / (1.0 + np.exp(-beta * (x - theta)))
    
    def simulate_2d(
        self,
        nx: int = 64,
        ny: int = 64,
        n_timesteps: int = 500,
        dt: float = 0.05,
        dx: float = 1.0,
        external_input: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate 2D Wilson-Cowan field."""
        
        # Initialize
        E = 0.1 * np.random.rand(nx, ny)
        I = 0.05 * np.random.rand(nx, ny)
        
        E_history = np.zeros((n_timesteps, nx, ny))
        I_history = np.zeros((n_timesteps, nx, ny))
        
        for t in range(n_timesteps):
            E_history[t] = E
            I_history[t] = I
            
            # Laplacians
            lap_E = (
                np.roll(E, 1, axis=0) + np.roll(E, -1, axis=0) +
                np.roll(E, 1, axis=1) + np.roll(E, -1, axis=1) -
                4 * E
            ) / dx**2
            
            lap_I = (
                np.roll(I, 1, axis=0) + np.roll(I, -1, axis=0) +
                np.roll(I, 1, axis=1) + np.roll(I, -1, axis=1) -
                4 * I
            ) / dx**2
            
            # Inputs
            input_E = self.w_ee * E + self.w_ei * I + external_input
            input_I = self.w_ie * E + self.w_ii * I
            
            # Dynamics
            dE = (-E + self.sigmoid(input_E, self.theta_e)) / self.tau_e + self.D_e * lap_E
            dI = (-I + self.sigmoid(input_I, self.theta_i)) / self.tau_i + self.D_i * lap_I
            
            E = E + dt * dE
            I = I + dt * dI
            
            # Keep bounded
            E = np.clip(E, 0, 1)
            I = np.clip(I, 0, 1)
            
        return E_history, I_history


# ============================================================================
# SYNTHETIC SUPERNOVA-LIKE DYNAMICS
# ============================================================================

class SupernovaLikeField:
    """
    Generate synthetic fields with supernova-like dynamics:
    - Shock wave propagation
    - Turbulent mixing
    - Rayleigh-Taylor instabilities
    
    This allows testing without downloading The Well datasets.
    """
    
    def __init__(
        self,
        nx: int = 64,
        ny: int = 64,
        gamma: float = 5/3  # Adiabatic index
    ):
        self.nx = nx
        self.ny = ny
        self.gamma = gamma
        
    def generate_shock_propagation(
        self,
        n_timesteps: int = 200,
        shock_speed: float = 1.0,
        dt: float = 0.1
    ) -> np.ndarray:
        """Generate expanding shock wave."""
        fields = np.zeros((n_timesteps, self.nx, self.ny))
        
        x = np.linspace(-1, 1, self.nx)
        y = np.linspace(-1, 1, self.ny)
        X, Y = np.meshgrid(x, y, indexing='ij')
        R = np.sqrt(X**2 + Y**2)
        
        for t in range(n_timesteps):
            radius = 0.1 + shock_speed * t * dt
            
            # Shock profile
            shock = np.exp(-((R - radius) / 0.1)**2)
            
            # Add turbulent perturbations behind shock
            turbulence = np.zeros_like(R)
            if radius > 0.2:
                inner_mask = R < radius - 0.1
                np.random.seed(t)
                turbulence[inner_mask] = 0.3 * np.random.randn(inner_mask.sum())
                turbulence = gaussian_filter(turbulence, sigma=2)
                
            fields[t] = shock + turbulence
            
        return fields
    
    def generate_rayleigh_taylor(
        self,
        n_timesteps: int = 200,
        growth_rate: float = 0.1,
        n_modes: int = 5,
        dt: float = 0.1
    ) -> np.ndarray:
        """Generate Rayleigh-Taylor instability pattern."""
        fields = np.zeros((n_timesteps, self.nx, self.ny))
        
        x = np.linspace(0, 2*np.pi, self.nx)
        y = np.linspace(0, 2*np.pi, self.ny)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Random mode amplitudes
        np.random.seed(123)
        amplitudes = np.random.rand(n_modes)
        wavenumbers = np.arange(1, n_modes + 1)
        
        for t in range(n_timesteps):
            field = np.zeros((self.nx, self.ny))
            
            # Interface position with growing perturbations
            interface = self.ny // 2
            
            for k, (amp, wn) in enumerate(zip(amplitudes, wavenumbers)):
                # Exponential growth
                growth = amp * np.exp(growth_rate * wn * t * dt)
                growth = min(growth, self.ny / 4)  # Saturate
                
                perturbation = growth * np.sin(wn * X)
                
                for i in range(self.nx):
                    interface_y = interface + int(perturbation[i, 0])
                    interface_y = np.clip(interface_y, 0, self.ny - 1)
                    field[i, :interface_y] = 1.0
                    
            # Add small-scale turbulence
            field += 0.1 * gaussian_filter(np.random.randn(self.nx, self.ny), sigma=1)
            fields[t] = field
            
        return fields


# ============================================================================
# CROSS-DOMAIN ANALYSIS
# ============================================================================

class CrossDomainLyapunovAnalysis:
    """
    Compare Lyapunov signatures across different physical systems.
    
    The goal is to find universal patterns that might enable
    transfer learning between domains.
    """
    
    def __init__(self):
        self.results = {}
        
    def analyze_neural_model(
        self,
        model_type: str = 'fhn',
        **kwargs
    ) -> Dict:
        """Analyze neural field model."""
        print(f"\n{'='*50}")
        print(f"Analyzing Neural Model: {model_type.upper()}")
        print('='*50)
        
        if model_type == 'fhn':
            model = FitzHughNagumo(**kwargs.get('model_params', {}))
            fields = model.simulate_2d(
                n_timesteps=kwargs.get('n_timesteps', 300),
                dt=kwargs.get('dt', 0.1)
            )
        elif model_type == 'wilson_cowan':
            model = WilsonCowan(**kwargs.get('model_params', {}))
            fields, _ = model.simulate_2d(
                n_timesteps=kwargs.get('n_timesteps', 300),
                dt=kwargs.get('dt', 0.05)
            )
        else:
            raise ValueError(f"Unknown model: {model_type}")
            
        # Compute Lyapunov measures
        results = self._compute_lyapunov_measures(
            fields, 
            dt=kwargs.get('dt', 0.1),
            name=f"Neural ({model_type})"
        )
        
        self.results[f'neural_{model_type}'] = results
        return results
    
    def analyze_supernova_like(
        self,
        pattern_type: str = 'shock',
        **kwargs
    ) -> Dict:
        """Analyze supernova-like synthetic data."""
        print(f"\n{'='*50}")
        print(f"Analyzing Supernova-like: {pattern_type.upper()}")
        print('='*50)
        
        generator = SupernovaLikeField(
            nx=kwargs.get('nx', 64),
            ny=kwargs.get('ny', 64)
        )
        
        if pattern_type == 'shock':
            fields = generator.generate_shock_propagation(
                n_timesteps=kwargs.get('n_timesteps', 200)
            )
        elif pattern_type == 'rayleigh_taylor':
            fields = generator.generate_rayleigh_taylor(
                n_timesteps=kwargs.get('n_timesteps', 200)
            )
        else:
            raise ValueError(f"Unknown pattern: {pattern_type}")
            
        results = self._compute_lyapunov_measures(
            fields,
            dt=kwargs.get('dt', 0.1),
            name=f"Supernova-like ({pattern_type})"
        )
        
        self.results[f'supernova_{pattern_type}'] = results
        return results
    
    def _compute_lyapunov_measures(
        self,
        fields: np.ndarray,
        dt: float,
        name: str
    ) -> Dict:
        """Compute various Lyapunov measures."""
        results = {'name': name, 'field_shape': fields.shape}
        
        # 1. Global time series Lyapunov (from spatial average)
        print("\n1. Computing global Lyapunov exponent...")
        mean_series = fields.reshape(fields.shape[0], -1).mean(axis=1)
        
        try:
            estimator = RosensteinMethod(mean_series, dt=dt)
            lyap, time, divergence = estimator.compute(
                embedding_dim=5,
                tau=max(1, len(mean_series) // 100)
            )
            results['global_lyapunov'] = lyap
            results['divergence_time'] = time
            results['divergence_curve'] = divergence
            print(f"   Global λ = {lyap:.4f} ({'Chaotic' if lyap > 0 else 'Stable'})")
        except Exception as e:
            print(f"   Error: {e}")
            results['global_lyapunov'] = np.nan
            
        # 2. Spatial Lyapunov field
        print("2. Computing spatial Lyapunov field...")
        try:
            spatial_analyzer = SpatialLyapunovField(fields, dt=dt)
            lyap_field = spatial_analyzer.compute_local_lyapunov(
                window_size=3,
                time_window=min(20, fields.shape[0] // 5)
            )
            results['lyapunov_field'] = lyap_field
            results['spatial_mean_lyapunov'] = np.nanmean(lyap_field)
            results['spatial_max_lyapunov'] = np.nanmax(lyap_field)
            results['chaotic_fraction'] = np.mean(lyap_field > 0)
            print(f"   Spatial mean λ = {results['spatial_mean_lyapunov']:.4f}")
            print(f"   Chaotic fraction = {100*results['chaotic_fraction']:.1f}%")
        except Exception as e:
            print(f"   Error: {e}")
            results['lyapunov_field'] = None
            
        # 3. Compute spectrum statistics
        print("3. Computing spectral characteristics...")
        try:
            # FFT of final state
            spectrum = np.abs(np.fft.fft2(fields[-1]))
            spectrum = np.fft.fftshift(spectrum)
            
            # Radial average
            center = np.array(spectrum.shape) // 2
            y, x = np.ogrid[:spectrum.shape[0], :spectrum.shape[1]]
            r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
            r = r.astype(int)
            
            radial_spectrum = np.bincount(r.ravel(), spectrum.ravel()) / np.bincount(r.ravel())
            results['radial_spectrum'] = radial_spectrum
            
            # Estimate spectral slope (indicator of turbulence type)
            k = np.arange(1, len(radial_spectrum))
            valid = (radial_spectrum[1:] > 0) & (k < len(k)//2)
            if valid.sum() > 5:
                slope, _ = np.polyfit(
                    np.log(k[valid]), 
                    np.log(radial_spectrum[1:][valid]), 
                    1
                )
                results['spectral_slope'] = slope
                print(f"   Spectral slope = {slope:.2f}")
        except Exception as e:
            print(f"   Error: {e}")
            
        return results
    
    def compare_domains(self) -> None:
        """Compare Lyapunov signatures across analyzed domains."""
        print("\n" + "="*60)
        print("CROSS-DOMAIN COMPARISON")
        print("="*60)
        
        if len(self.results) < 2:
            print("Need at least 2 domains to compare. Run analyze_* methods first.")
            return
            
        # Comparison table
        print("\n{:<30} {:>12} {:>12} {:>12}".format(
            "Domain", "Global λ", "Spatial λ", "Chaos %"
        ))
        print("-" * 70)
        
        for key, res in self.results.items():
            global_lyap = res.get('global_lyapunov', np.nan)
            spatial_lyap = res.get('spatial_mean_lyapunov', np.nan)
            chaos_frac = res.get('chaotic_fraction', np.nan) * 100
            
            print("{:<30} {:>12.4f} {:>12.4f} {:>11.1f}%".format(
                res['name'],
                global_lyap,
                spatial_lyap,
                chaos_frac
            ))
            
        # Compute similarity metrics
        print("\n--- Similarity Analysis ---")
        
        keys = list(self.results.keys())
        for i, key1 in enumerate(keys):
            for key2 in keys[i+1:]:
                lyap1 = self.results[key1].get('global_lyapunov', np.nan)
                lyap2 = self.results[key2].get('global_lyapunov', np.nan)
                
                if not (np.isnan(lyap1) or np.isnan(lyap2)):
                    ratio = lyap1 / lyap2 if lyap2 != 0 else np.inf
                    print(f"{key1} vs {key2}: λ ratio = {ratio:.2f}")
                    
        # Check for transfer learning potential
        print("\n--- Transfer Learning Potential ---")
        
        chaos_states = {k: v.get('chaotic_fraction', 0) > 0.3 
                       for k, v in self.results.items()}
        
        if all(chaos_states.values()):
            print("✓ All domains show chaotic behavior")
            print("  → High potential for transfer learning of chaos signatures")
        elif any(chaos_states.values()):
            print("⚠ Mixed chaotic/stable behavior across domains")
            print("  → Transfer learning may require careful domain adaptation")
        else:
            print("✗ All domains appear stable")
            print("  → Consider different parameter regimes for chaos")
            
    def visualize_comparison(self) -> plt.Figure:
        """Create visualization comparing all analyzed domains."""
        n_domains = len(self.results)
        
        if n_domains == 0:
            print("No results to visualize.")
            return None
            
        fig, axes = plt.subplots(n_domains, 3, figsize=(15, 4*n_domains))
        
        if n_domains == 1:
            axes = axes.reshape(1, -1)
            
        for idx, (key, res) in enumerate(self.results.items()):
            # Column 1: Divergence curve
            ax1 = axes[idx, 0]
            if 'divergence_curve' in res and res['divergence_curve'] is not None:
                time = res.get('divergence_time', np.arange(len(res['divergence_curve'])))
                div = res['divergence_curve']
                valid = ~np.isnan(div)
                ax1.plot(time[valid], div[valid], 'b-', linewidth=2)
                
                lyap = res.get('global_lyapunov', 0)
                if not np.isnan(lyap):
                    fit_line = lyap * time + div[valid][0]
                    ax1.plot(time, fit_line, 'r--', linewidth=2, label=f'λ={lyap:.3f}')
                    ax1.legend()
                    
            ax1.set_xlabel('Time')
            ax1.set_ylabel('ln(divergence)')
            ax1.set_title(f"{res['name']}\nDivergence Curve")
            ax1.grid(True, alpha=0.3)
            
            # Column 2: Spatial Lyapunov field
            ax2 = axes[idx, 1]
            if 'lyapunov_field' in res and res['lyapunov_field'] is not None:
                lyap_field = res['lyapunov_field']
                vmax = np.nanmax(np.abs(lyap_field))
                im = ax2.imshow(lyap_field.T, origin='lower', cmap='RdBu_r',
                               vmin=-vmax, vmax=vmax)
                plt.colorbar(im, ax=ax2)
                ax2.set_title(f"Spatial λ Field\nMean={np.nanmean(lyap_field):.3f}")
            else:
                ax2.text(0.5, 0.5, 'Not computed', ha='center', va='center')
            ax2.set_xlabel('x')
            ax2.set_ylabel('y')
            
            # Column 3: Radial spectrum
            ax3 = axes[idx, 2]
            if 'radial_spectrum' in res:
                spectrum = res['radial_spectrum']
                k = np.arange(len(spectrum))
                ax3.loglog(k[1:], spectrum[1:], 'b-', linewidth=2)
                
                if 'spectral_slope' in res:
                    slope = res['spectral_slope']
                    k_fit = k[1:len(k)//2]
                    fit_line = spectrum[1] * (k_fit ** slope)
                    ax3.loglog(k_fit, fit_line, 'r--', 
                              label=f'slope={slope:.2f}')
                    ax3.legend()
                    
            ax3.set_xlabel('Wavenumber k')
            ax3.set_ylabel('Power')
            ax3.set_title('Energy Spectrum')
            ax3.grid(True, alpha=0.3)
            
        plt.tight_layout()
        return fig


# ============================================================================
# MAIN DEMONSTRATION
# ============================================================================

def main():
    """Run the cross-domain Lyapunov analysis demonstration."""
    
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║     CROSS-DOMAIN LYAPUNOV ANALYSIS: SUPERNOVAE ↔ NEURAL          ║
    ║                                                                   ║
    ║  Exploring universal chaos signatures across physical domains     ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # Initialize analyzer
    analyzer = CrossDomainLyapunovAnalysis()
    
    # Analyze supernova-like shock propagation
    print("\n" + "▶ " * 20)
    print("PART 1: SUPERNOVA-LIKE DYNAMICS")
    print("▶ " * 20)
    
    analyzer.analyze_supernova_like(
        pattern_type='shock',
        n_timesteps=150,
        dt=0.1
    )
    
    analyzer.analyze_supernova_like(
        pattern_type='rayleigh_taylor',
        n_timesteps=150,
        dt=0.1
    )
    
    # Analyze neural models
    print("\n" + "▶ " * 20)
    print("PART 2: NEURAL FIELD DYNAMICS")
    print("▶ " * 20)
    
    # FitzHugh-Nagumo with parameters that produce chaos
    analyzer.analyze_neural_model(
        model_type='fhn',
        model_params={'I_ext': 0.5, 'D': 1.0},
        n_timesteps=300,
        dt=0.1
    )
    
    # Wilson-Cowan
    analyzer.analyze_neural_model(
        model_type='wilson_cowan',
        n_timesteps=300,
        dt=0.05
    )
    
    # Cross-domain comparison
    print("\n" + "▶ " * 20)
    print("PART 3: CROSS-DOMAIN COMPARISON")
    print("▶ " * 20)
    
    analyzer.compare_domains()
    
    # Visualize
    print("\n" + "▶ " * 20)
    print("PART 4: VISUALIZATION")
    print("▶ " * 20)
    
    fig = analyzer.visualize_comparison()
    
    if fig is not None:
        fig.savefig('cross_domain_lyapunov_comparison.png', dpi=150, bbox_inches='tight')
        print("\nFigure saved to: cross_domain_lyapunov_comparison.png")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY: KEY INSIGHTS FOR TRANSFER LEARNING")
    print("="*60)
    print("""
    1. MATHEMATICAL BRIDGE:
       Both supernova MHD and neural field dynamics are governed by
       reaction-diffusion PDEs with similar mathematical structure.
       
    2. CHAOS SIGNATURES:
       Positive Lyapunov exponents indicate chaos in both domains.
       The magnitude and spatial distribution of λ reveals:
       - Shock fronts (supernova) ↔ Wave fronts (neural)
       - Turbulent mixing ↔ Neural avalanches
       
    3. TRANSFER LEARNING HYPOTHESIS:
       A neural network trained to predict:
       - λ(x,t) in supernova simulations
       Could potentially transfer to predicting:
       - Seizure onset zones in EEG
       - Spreading cortical depression patterns
       
    4. NEXT STEPS:
       - Train FNO/U-Net on The Well supernova data
       - Fine-tune on neural field simulations
       - Evaluate transfer to real EEG/MEG data
    """)
    
    return analyzer


if __name__ == "__main__":
    analyzer = main()
