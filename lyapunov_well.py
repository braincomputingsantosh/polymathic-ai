"""
Lyapunov Exponent Computation for The Well Datasets
====================================================

This module provides tools for computing Lyapunov exponents from 
spatiotemporal PDE simulation data, specifically designed to work
with Polymathic AI's "The Well" dataset collection.

Methods implemented:
1. Direct method (for known dynamical systems)
2. Rosenstein method (for time series)
3. Wolf method (classic algorithm)
4. Spatial Lyapunov exponents (for PDEs)
5. Finite-Time Lyapunov Exponents (FTLE) for flow fields

"""

import numpy as np
from scipy.spatial import KDTree
from scipy.ndimage import sobel, gaussian_filter
from typing import Tuple, Optional, Dict, List, Union
import warnings


# ============================================================================
# CORE LYAPUNOV COMPUTATION METHODS
# ============================================================================

class LyapunovEstimator:
    """
    Base class for Lyapunov exponent estimation from time series data.
    
    The Lyapunov exponent λ measures the rate of separation of infinitesimally 
    close trajectories:
        |δZ(t)| ≈ e^{λt} |δZ(0)|
    
    For chaotic systems: λ > 0
    For stable systems: λ < 0
    For edge of chaos: λ ≈ 0
    """
    
    def __init__(self, data: np.ndarray, dt: float = 1.0):
        """
        Parameters
        ----------
        data : np.ndarray
            Time series data. Shape can be:
            - (n_timesteps,) for scalar time series
            - (n_timesteps, n_features) for multivariate
            - (n_timesteps, nx, ny, ...) for spatial fields
        dt : float
            Time step between samples
        """
        self.data = np.asarray(data)
        self.dt = dt
        self.n_timesteps = data.shape[0]
        
    def embed_time_series(
        self, 
        embedding_dim: int = 3, 
        tau: int = 1
    ) -> np.ndarray:
        """
        Create delay embedding of time series (Takens embedding).
        
        For a scalar time series x(t), creates vectors:
        [x(t), x(t-τ), x(t-2τ), ..., x(t-(m-1)τ)]
        
        Parameters
        ----------
        embedding_dim : int
            Embedding dimension m
        tau : int
            Time delay in samples
            
        Returns
        -------
        embedded : np.ndarray
            Shape (n_vectors, embedding_dim)
        """
        if self.data.ndim > 1:
            # Flatten spatial dimensions for embedding
            flat_data = self.data.reshape(self.n_timesteps, -1)
            # Use first principal component or mean
            series = flat_data.mean(axis=1)
        else:
            series = self.data
            
        n_vectors = len(series) - (embedding_dim - 1) * tau
        embedded = np.zeros((n_vectors, embedding_dim))
        
        for i in range(embedding_dim):
            embedded[:, i] = series[i * tau : i * tau + n_vectors]
            
        return embedded


class RosensteinMethod(LyapunovEstimator):
    """
    Rosenstein et al. (1993) method for largest Lyapunov exponent.
    
    This is one of the most reliable methods for experimental time series.
    It tracks the divergence of nearest neighbors in reconstructed phase space.
    
    Reference:
    Rosenstein, M.T., Collins, J.J., De Luca, C.J. (1993). 
    "A practical method for calculating largest Lyapunov exponents 
    from small data sets." Physica D, 65(1-2), 117-134.
    """
    
    def compute(
        self,
        embedding_dim: int = 5,
        tau: int = 1,
        min_separation: int = None,
        max_iter: int = None
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Compute the largest Lyapunov exponent using Rosenstein method.
        
        Parameters
        ----------
        embedding_dim : int
            Embedding dimension for phase space reconstruction
        tau : int
            Time delay for embedding
        min_separation : int
            Minimum temporal separation between neighbors (to avoid 
            selecting temporally correlated points). Default: tau * embedding_dim
        max_iter : int
            Maximum iterations to track divergence. Default: n_vectors // 10
            
        Returns
        -------
        lyapunov_exp : float
            Estimated largest Lyapunov exponent
        time : np.ndarray
            Time values for divergence curve
        divergence : np.ndarray
            Mean log divergence at each time step
        """
        # Embed the time series
        embedded = self.embed_time_series(embedding_dim, tau)
        n_vectors = len(embedded)
        
        if min_separation is None:
            min_separation = tau * embedding_dim
        if max_iter is None:
            max_iter = n_vectors // 10
            
        # Build KD-tree for nearest neighbor search
        tree = KDTree(embedded)
        
        # For each point, find nearest neighbor (excluding temporally close points)
        divergence_curves = []
        
        for i in range(n_vectors - max_iter):
            # Find k nearest neighbors
            distances, indices = tree.query(embedded[i], k=n_vectors)
            
            # Find the nearest neighbor with sufficient temporal separation
            for j, idx in enumerate(indices[1:], 1):  # Skip self (index 0)
                if abs(idx - i) >= min_separation:
                    nearest_idx = idx
                    initial_dist = distances[j]
                    break
            else:
                continue  # No valid neighbor found
                
            if initial_dist == 0:
                continue
                
            # Track divergence over time
            curve = np.zeros(max_iter)
            for k in range(max_iter):
                if i + k < n_vectors and nearest_idx + k < n_vectors:
                    dist = np.linalg.norm(embedded[i + k] - embedded[nearest_idx + k])
                    curve[k] = np.log(dist) if dist > 0 else np.nan
                else:
                    curve[k] = np.nan
                    
            divergence_curves.append(curve)
            
        if len(divergence_curves) == 0:
            raise ValueError("Could not find valid neighbor pairs")
            
        # Average divergence curves
        divergence_curves = np.array(divergence_curves)
        mean_divergence = np.nanmean(divergence_curves, axis=0)
        time = np.arange(max_iter) * self.dt
        
        # Fit line to linear region to get Lyapunov exponent
        # Use first 10-20% of points where growth is typically linear
        fit_end = max(10, max_iter // 5)
        valid_mask = ~np.isnan(mean_divergence[:fit_end])
        
        if valid_mask.sum() < 2:
            raise ValueError("Insufficient valid points for fitting")
            
        coeffs = np.polyfit(
            time[:fit_end][valid_mask], 
            mean_divergence[:fit_end][valid_mask], 
            1
        )
        lyapunov_exp = coeffs[0]
        
        return lyapunov_exp, time, mean_divergence


class WolfMethod(LyapunovEstimator):
    """
    Wolf et al. (1985) method for Lyapunov exponent estimation.
    
    Classic algorithm that tracks fiducial trajectory and replaces
    nearest neighbor when separation becomes too large.
    
    Reference:
    Wolf, A., Swift, J.B., Swinney, H.L., Vastano, J.A. (1985).
    "Determining Lyapunov exponents from a time series."
    Physica D, 16(3), 285-317.
    """
    
    def compute(
        self,
        embedding_dim: int = 5,
        tau: int = 1,
        min_separation: int = None,
        max_distance: float = None,
        min_distance: float = None
    ) -> Tuple[float, List[float]]:
        """
        Compute Lyapunov exponent using Wolf algorithm.
        
        Parameters
        ----------
        embedding_dim : int
            Embedding dimension
        tau : int
            Time delay
        min_separation : int
            Minimum temporal separation for neighbors
        max_distance : float
            Maximum distance before replacement. Default: 10% of attractor size
        min_distance : float
            Minimum initial distance. Default: 1% of attractor size
            
        Returns
        -------
        lyapunov_exp : float
            Estimated largest Lyapunov exponent
        exponent_evolution : list
            Running estimate of exponent over time
        """
        embedded = self.embed_time_series(embedding_dim, tau)
        n_vectors = len(embedded)
        
        if min_separation is None:
            min_separation = tau * embedding_dim
            
        # Estimate attractor size
        attractor_size = np.std(embedded) * 2
        if max_distance is None:
            max_distance = 0.1 * attractor_size
        if min_distance is None:
            min_distance = 0.01 * attractor_size
            
        tree = KDTree(embedded)
        
        # Initialize
        total_expansion = 0.0
        n_replacements = 0
        exponent_evolution = []
        
        # Start with first point
        current_idx = 0
        
        # Find initial nearest neighbor
        distances, indices = tree.query(embedded[current_idx], k=n_vectors)
        neighbor_idx = None
        for j, idx in enumerate(indices[1:], 1):
            if abs(idx - current_idx) >= min_separation:
                if min_distance < distances[j] < max_distance:
                    neighbor_idx = idx
                    initial_dist = distances[j]
                    break
                    
        if neighbor_idx is None:
            raise ValueError("Could not find suitable initial neighbor")
            
        # Iterate through trajectory
        while current_idx < n_vectors - 1 and neighbor_idx < n_vectors - 1:
            # Evolve both points
            current_idx += 1
            neighbor_idx += 1
            
            # Compute new distance
            new_dist = np.linalg.norm(embedded[current_idx] - embedded[neighbor_idx])
            
            if new_dist > 0 and initial_dist > 0:
                # Accumulate log expansion
                expansion = np.log(new_dist / initial_dist)
                total_expansion += expansion
                n_replacements += 1
                
                # Record running estimate
                elapsed_time = n_replacements * self.dt
                if elapsed_time > 0:
                    exponent_evolution.append(total_expansion / elapsed_time)
                    
            # Check if replacement needed
            if new_dist > max_distance or new_dist == 0:
                # Find new neighbor close to current evolved direction
                direction = embedded[current_idx] - embedded[neighbor_idx]
                if np.linalg.norm(direction) > 0:
                    direction = direction / np.linalg.norm(direction)
                    
                distances, indices = tree.query(embedded[current_idx], k=n_vectors)
                
                best_idx = None
                best_angle = np.pi  # worst case
                
                for j, idx in enumerate(indices[1:], 1):
                    if abs(idx - current_idx) >= min_separation:
                        if min_distance < distances[j] < max_distance:
                            # Check angle alignment
                            new_direction = embedded[idx] - embedded[current_idx]
                            if np.linalg.norm(new_direction) > 0:
                                new_direction = new_direction / np.linalg.norm(new_direction)
                                angle = np.arccos(np.clip(np.dot(direction, new_direction), -1, 1))
                                if angle < best_angle:
                                    best_angle = angle
                                    best_idx = idx
                                    initial_dist = distances[j]
                                    
                if best_idx is not None:
                    neighbor_idx = best_idx
                else:
                    break  # No suitable replacement found
                    
        if n_replacements == 0:
            raise ValueError("No valid expansions recorded")
            
        lyapunov_exp = total_expansion / (n_replacements * self.dt)
        
        return lyapunov_exp, exponent_evolution


# ============================================================================
# SPATIAL LYAPUNOV EXPONENTS FOR PDE FIELDS
# ============================================================================

class SpatialLyapunovField:
    """
    Compute spatially-resolved Lyapunov exponents for PDE simulation data.
    
    This is particularly relevant for The Well datasets, which contain
    spatiotemporal fields from physics simulations.
    """
    
    def __init__(
        self, 
        field_sequence: np.ndarray,
        dt: float = 1.0,
        dx: float = 1.0
    ):
        """
        Parameters
        ----------
        field_sequence : np.ndarray
            Shape (n_timesteps, nx, ny) or (n_timesteps, nx, ny, nz)
            The spatiotemporal field data
        dt : float
            Time step
        dx : float
            Spatial grid spacing
        """
        self.fields = field_sequence
        self.dt = dt
        self.dx = dx
        self.n_timesteps = field_sequence.shape[0]
        self.spatial_shape = field_sequence.shape[1:]
        
    def compute_local_lyapunov(
        self,
        window_size: int = 5,
        time_window: int = 10
    ) -> np.ndarray:
        """
        Compute local Lyapunov exponents at each spatial point.
        
        Uses local time series at each grid point and estimates
        divergence rate from neighboring trajectories.
        
        Parameters
        ----------
        window_size : int
            Spatial window for local neighborhood
        time_window : int
            Number of time steps for divergence estimation
            
        Returns
        -------
        lyapunov_field : np.ndarray
            Same shape as spatial_shape, containing local λ estimates
        """
        lyapunov_field = np.zeros(self.spatial_shape)
        
        # Pad for boundary handling
        pad_width = window_size // 2
        
        if len(self.spatial_shape) == 2:
            nx, ny = self.spatial_shape
            
            for i in range(nx):
                for j in range(ny):
                    # Extract local time series
                    local_series = self.fields[:, i, j]
                    
                    # Find spatial neighbors
                    i_min = max(0, i - pad_width)
                    i_max = min(nx, i + pad_width + 1)
                    j_min = max(0, j - pad_width)
                    j_max = min(ny, j + pad_width + 1)
                    
                    # Compute divergence from neighbors
                    divergences = []
                    for ni in range(i_min, i_max):
                        for nj in range(j_min, j_max):
                            if ni == i and nj == j:
                                continue
                            neighbor_series = self.fields[:, ni, nj]
                            
                            # Track divergence over time_window
                            for t0 in range(0, self.n_timesteps - time_window, time_window):
                                initial_diff = abs(local_series[t0] - neighbor_series[t0])
                                if initial_diff > 1e-10:
                                    final_diff = abs(local_series[t0 + time_window] - 
                                                   neighbor_series[t0 + time_window])
                                    if final_diff > 1e-10:
                                        div = np.log(final_diff / initial_diff) / (time_window * self.dt)
                                        divergences.append(div)
                                        
                    if divergences:
                        lyapunov_field[i, j] = np.median(divergences)
                        
        elif len(self.spatial_shape) == 3:
            nx, ny, nz = self.spatial_shape
            # Similar logic for 3D fields...
            # (abbreviated for space, same principle applies)
            warnings.warn("3D local Lyapunov not fully implemented, using slice average")
            for k in range(nz):
                lyapunov_field[:, :, k] = self.compute_local_lyapunov_2d(
                    self.fields[:, :, :, k], window_size, time_window
                )
                
        return lyapunov_field
    
    def compute_local_lyapunov_2d(
        self, 
        fields_2d: np.ndarray,
        window_size: int,
        time_window: int
    ) -> np.ndarray:
        """Helper for 2D slices."""
        temp_analyzer = SpatialLyapunovField(fields_2d, self.dt, self.dx)
        return temp_analyzer.compute_local_lyapunov(window_size, time_window)


class FiniteTimeLyapunovExponent:
    """
    Compute Finite-Time Lyapunov Exponents (FTLE) for flow fields.
    
    FTLE measures the maximum stretching rate of material elements
    in a flow over a finite time interval. Ridges of FTLE reveal
    Lagrangian Coherent Structures (LCS).
    
    Particularly useful for:
    - Supernova shock front detection
    - Neural wave propagation boundaries
    - Turbulent mixing regions
    """
    
    def __init__(
        self,
        velocity_fields: np.ndarray,
        dt: float = 1.0,
        dx: float = 1.0
    ):
        """
        Parameters
        ----------
        velocity_fields : np.ndarray
            Shape (n_timesteps, n_components, nx, ny, ...)
            Velocity field time series. n_components = 2 for 2D, 3 for 3D.
        dt : float
            Time step
        dx : float
            Grid spacing
        """
        self.velocity = velocity_fields
        self.dt = dt
        self.dx = dx
        self.n_timesteps = velocity_fields.shape[0]
        self.n_components = velocity_fields.shape[1]
        self.grid_shape = velocity_fields.shape[2:]
        
    def compute_ftle(
        self,
        integration_time: int = 10,
        direction: int = 1
    ) -> np.ndarray:
        """
        Compute FTLE field.
        
        Parameters
        ----------
        integration_time : int
            Number of time steps to integrate
        direction : int
            1 for forward-time FTLE (repelling LCS)
            -1 for backward-time FTLE (attracting LCS)
            
        Returns
        -------
        ftle : np.ndarray
            FTLE field with same spatial shape as velocity
        """
        if self.n_components == 2:
            return self._compute_ftle_2d(integration_time, direction)
        else:
            raise NotImplementedError("Only 2D FTLE implemented")
            
    def _compute_ftle_2d(
        self,
        integration_time: int,
        direction: int
    ) -> np.ndarray:
        """Compute 2D FTLE using flow map gradient."""
        nx, ny = self.grid_shape
        
        # Initialize particle positions on grid
        x_init, y_init = np.meshgrid(
            np.arange(nx) * self.dx,
            np.arange(ny) * self.dx,
            indexing='ij'
        )
        
        # Advect particles
        x_final, y_final = self._advect_particles(
            x_init, y_init, integration_time, direction
        )
        
        # Compute flow map gradient (deformation gradient tensor)
        # F = ∂φ/∂X where φ is the flow map
        dx_dx = np.gradient(x_final, self.dx, axis=0)
        dx_dy = np.gradient(x_final, self.dx, axis=1)
        dy_dx = np.gradient(y_final, self.dx, axis=0)
        dy_dy = np.gradient(y_final, self.dx, axis=1)
        
        # Compute Cauchy-Green strain tensor C = F^T F
        # and find maximum eigenvalue
        ftle = np.zeros((nx, ny))
        
        for i in range(nx):
            for j in range(ny):
                F = np.array([
                    [dx_dx[i, j], dx_dy[i, j]],
                    [dy_dx[i, j], dy_dy[i, j]]
                ])
                C = F.T @ F
                eigenvalues = np.linalg.eigvalsh(C)
                lambda_max = max(eigenvalues)
                
                if lambda_max > 0:
                    T = integration_time * self.dt
                    ftle[i, j] = np.log(np.sqrt(lambda_max)) / abs(T)
                    
        return ftle
    
    def _advect_particles(
        self,
        x0: np.ndarray,
        y0: np.ndarray,
        n_steps: int,
        direction: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Advect particles through velocity field using RK4.
        """
        x = x0.copy()
        y = y0.copy()
        
        nx, ny = self.grid_shape
        
        # Time stepping direction
        if direction > 0:
            time_indices = range(0, min(n_steps, self.n_timesteps - 1))
        else:
            time_indices = range(self.n_timesteps - 1, 
                                max(self.n_timesteps - 1 - n_steps, 0), -1)
            
        for t_idx in time_indices:
            # Get velocity at current positions (with interpolation)
            u, v = self._interpolate_velocity(x, y, t_idx)
            
            # RK4 integration
            dt = direction * self.dt
            
            k1_x, k1_y = u, v
            
            x_mid = x + 0.5 * dt * k1_x
            y_mid = y + 0.5 * dt * k1_y
            k2_x, k2_y = self._interpolate_velocity(x_mid, y_mid, t_idx)
            
            x_mid = x + 0.5 * dt * k2_x
            y_mid = y + 0.5 * dt * k2_y
            k3_x, k3_y = self._interpolate_velocity(x_mid, y_mid, t_idx)
            
            x_end = x + dt * k3_x
            y_end = y + dt * k3_y
            k4_x, k4_y = self._interpolate_velocity(x_end, y_end, t_idx)
            
            x = x + dt * (k1_x + 2*k2_x + 2*k3_x + k4_x) / 6
            y = y + dt * (k1_y + 2*k2_y + 2*k3_y + k4_y) / 6
            
            # Boundary handling (periodic or clip)
            x = np.clip(x, 0, (nx - 1) * self.dx)
            y = np.clip(y, 0, (ny - 1) * self.dx)
            
        return x, y
    
    def _interpolate_velocity(
        self,
        x: np.ndarray,
        y: np.ndarray,
        t_idx: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Bilinear interpolation of velocity field."""
        from scipy.interpolate import RegularGridInterpolator
        
        nx, ny = self.grid_shape
        x_grid = np.arange(nx) * self.dx
        y_grid = np.arange(ny) * self.dx
        
        u_interp = RegularGridInterpolator(
            (x_grid, y_grid), 
            self.velocity[t_idx, 0],
            bounds_error=False,
            fill_value=0
        )
        v_interp = RegularGridInterpolator(
            (x_grid, y_grid),
            self.velocity[t_idx, 1],
            bounds_error=False,
            fill_value=0
        )
        
        points = np.stack([x.ravel(), y.ravel()], axis=-1)
        u = u_interp(points).reshape(x.shape)
        v = v_interp(points).reshape(x.shape)
        
        return u, v


# ============================================================================
# THE WELL DATASET INTEGRATION
# ============================================================================

class WellLyapunovAnalyzer:
    """
    High-level interface for computing Lyapunov exponents from The Well datasets.
    
    Handles data loading, preprocessing, and analysis for different dataset types.
    """
    
    def __init__(
        self,
        dataset_name: str,
        base_path: str = "hf://datasets/polymathic-ai/",
        split: str = "train"
    ):
        """
        Parameters
        ----------
        dataset_name : str
            Name of The Well dataset (e.g., 'active_matter', 'supernova_explosion_64')
        base_path : str
            Base path for The Well datasets
        split : str
            Dataset split ('train', 'valid', 'test')
        """
        self.dataset_name = dataset_name
        self.base_path = base_path
        self.split = split
        self.dataset = None
        self.metadata = None
        
    def load_dataset(self, use_normalization: bool = False):
        """Load the dataset from The Well."""
        try:
            from the_well.data import WellDataset
            
            self.dataset = WellDataset(
                well_base_path=self.base_path,
                well_dataset_name=self.dataset_name,
                well_split_name=self.split,
                use_normalization=use_normalization
            )
            print(f"Loaded {self.dataset_name} with {len(self.dataset)} samples")
            
        except ImportError:
            raise ImportError(
                "The Well package not installed. Install with: pip install the_well"
            )
            
    def analyze_sample(
        self,
        sample_idx: int = 0,
        field_name: str = None,
        method: str = 'rosenstein',
        **kwargs
    ) -> Dict:
        """
        Compute Lyapunov analysis for a single sample.
        
        Parameters
        ----------
        sample_idx : int
            Index of sample in dataset
        field_name : str
            Name of field to analyze (if None, uses first available)
        method : str
            'rosenstein', 'wolf', 'spatial', or 'ftle'
        **kwargs
            Additional arguments passed to the specific method
            
        Returns
        -------
        results : dict
            Dictionary containing Lyapunov estimates and diagnostics
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
            
        sample = self.dataset[sample_idx]
        
        # Extract field data
        if 'input_fields' in sample:
            fields = sample['input_fields'].numpy()
        else:
            # Handle different data formats
            fields = list(sample.values())[0]
            if hasattr(fields, 'numpy'):
                fields = fields.numpy()
                
        # Get temporal dimension
        # The Well typically uses shape (time, channels, x, y, ...)
        if fields.ndim >= 3:
            n_time = fields.shape[0]
            
            if field_name is None:
                # Use first channel
                field_data = fields[:, 0] if fields.ndim > 3 else fields
            else:
                # Try to find named field
                field_data = fields[:, 0]  # Simplified
        else:
            field_data = fields
            
        # Get dt from metadata if available
        dt = kwargs.pop('dt', 1.0)
        
        results = {
            'sample_idx': sample_idx,
            'dataset': self.dataset_name,
            'method': method,
            'field_shape': field_data.shape
        }
        
        if method == 'rosenstein':
            # Flatten spatial dimensions for time series analysis
            flat_series = field_data.reshape(field_data.shape[0], -1).mean(axis=1)
            estimator = RosensteinMethod(flat_series, dt=dt)
            
            embedding_dim = kwargs.get('embedding_dim', 5)
            tau = kwargs.get('tau', 1)
            
            try:
                lyap, time, divergence = estimator.compute(
                    embedding_dim=embedding_dim,
                    tau=tau
                )
                results['lyapunov_exponent'] = lyap
                results['divergence_time'] = time
                results['divergence_curve'] = divergence
                results['is_chaotic'] = lyap > 0
            except Exception as e:
                results['error'] = str(e)
                
        elif method == 'wolf':
            flat_series = field_data.reshape(field_data.shape[0], -1).mean(axis=1)
            estimator = WolfMethod(flat_series, dt=dt)
            
            try:
                lyap, evolution = estimator.compute(**kwargs)
                results['lyapunov_exponent'] = lyap
                results['exponent_evolution'] = evolution
                results['is_chaotic'] = lyap > 0
            except Exception as e:
                results['error'] = str(e)
                
        elif method == 'spatial':
            analyzer = SpatialLyapunovField(field_data, dt=dt)
            
            try:
                lyap_field = analyzer.compute_local_lyapunov(**kwargs)
                results['lyapunov_field'] = lyap_field
                results['mean_lyapunov'] = np.nanmean(lyap_field)
                results['max_lyapunov'] = np.nanmax(lyap_field)
                results['chaotic_fraction'] = np.mean(lyap_field > 0)
            except Exception as e:
                results['error'] = str(e)
                
        elif method == 'ftle':
            # FTLE requires velocity field
            # Estimate velocity from scalar field time derivative
            if field_data.ndim >= 3:
                # Compute pseudo-velocity from gradient
                velocity = self._estimate_velocity_field(field_data)
                analyzer = FiniteTimeLyapunovExponent(velocity, dt=dt)
                
                try:
                    ftle = analyzer.compute_ftle(**kwargs)
                    results['ftle_field'] = ftle
                    results['mean_ftle'] = np.mean(ftle)
                    results['max_ftle'] = np.max(ftle)
                except Exception as e:
                    results['error'] = str(e)
            else:
                results['error'] = "FTLE requires 2D or 3D spatial data"
                
        return results
    
    def _estimate_velocity_field(self, scalar_field: np.ndarray) -> np.ndarray:
        """
        Estimate velocity field from scalar field using optical flow principle.
        
        For a conserved quantity: ∂φ/∂t + u·∇φ = 0
        So: u ≈ -∂φ/∂t / |∇φ|² * ∇φ
        """
        n_time = scalar_field.shape[0]
        spatial_shape = scalar_field.shape[1:]
        n_dims = len(spatial_shape)
        
        velocity = np.zeros((n_time, n_dims) + spatial_shape)
        
        for t in range(n_time - 1):
            # Time derivative
            dphi_dt = scalar_field[t + 1] - scalar_field[t]
            
            # Spatial gradients
            gradients = np.array(np.gradient(scalar_field[t]))
            grad_mag_sq = np.sum(gradients**2, axis=0) + 1e-10
            
            # Estimate velocity components
            for d in range(n_dims):
                velocity[t, d] = -dphi_dt * gradients[d] / grad_mag_sq
                
        # Handle last timestep
        velocity[-1] = velocity[-2]
        
        return velocity
    
    def batch_analysis(
        self,
        n_samples: int = 10,
        method: str = 'rosenstein',
        **kwargs
    ) -> List[Dict]:
        """
        Analyze multiple samples and compute statistics.
        
        Returns
        -------
        results : list
            List of result dictionaries for each sample
        """
        results = []
        
        for i in range(min(n_samples, len(self.dataset))):
            print(f"Analyzing sample {i+1}/{n_samples}...")
            result = self.analyze_sample(i, method=method, **kwargs)
            results.append(result)
            
        # Compute aggregate statistics
        lyap_values = [r.get('lyapunov_exponent', np.nan) for r in results]
        valid_values = [v for v in lyap_values if not np.isnan(v)]
        
        if valid_values:
            print(f"\n=== Aggregate Statistics ===")
            print(f"Mean λ: {np.mean(valid_values):.4f}")
            print(f"Std λ: {np.std(valid_values):.4f}")
            print(f"Chaotic samples: {sum(v > 0 for v in valid_values)}/{len(valid_values)}")
            
        return results


# ============================================================================
# VISUALIZATION UTILITIES
# ============================================================================

def plot_divergence_curve(
    time: np.ndarray,
    divergence: np.ndarray,
    lyapunov_exp: float,
    title: str = "Lyapunov Exponent Estimation"
):
    """Plot divergence curve with fitted line."""
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot divergence curve
    valid_mask = ~np.isnan(divergence)
    ax.plot(time[valid_mask], divergence[valid_mask], 'b-', linewidth=2, label='Mean divergence')
    
    # Plot fitted line
    fit_line = lyapunov_exp * time + divergence[valid_mask][0]
    ax.plot(time, fit_line, 'r--', linewidth=2, 
            label=f'Fit: λ = {lyapunov_exp:.4f}')
    
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('ln(divergence)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add chaos indicator
    chaos_text = "CHAOTIC" if lyapunov_exp > 0 else "STABLE"
    color = 'red' if lyapunov_exp > 0 else 'green'
    ax.text(0.95, 0.95, chaos_text, transform=ax.transAxes,
            fontsize=14, fontweight='bold', color=color,
            ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    return fig, ax


def plot_lyapunov_field(
    lyap_field: np.ndarray,
    title: str = "Local Lyapunov Exponent Field",
    cmap: str = 'RdBu_r'
):
    """Plot spatial Lyapunov exponent field."""
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Center colormap at zero
    vmax = np.nanmax(np.abs(lyap_field))
    
    im = ax.imshow(lyap_field.T, origin='lower', cmap=cmap,
                   vmin=-vmax, vmax=vmax)
    
    plt.colorbar(im, ax=ax, label='λ (local Lyapunov exponent)')
    
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    # Add statistics
    stats_text = f"Mean: {np.nanmean(lyap_field):.3f}\n"
    stats_text += f"Max: {np.nanmax(lyap_field):.3f}\n"
    stats_text += f"Chaotic: {100*np.mean(lyap_field > 0):.1f}%"
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    return fig, ax


def plot_ftle_field(
    ftle: np.ndarray,
    title: str = "Finite-Time Lyapunov Exponent (FTLE)"
):
    """Plot FTLE field showing Lagrangian Coherent Structures."""
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(ftle.T, origin='lower', cmap='hot')
    plt.colorbar(im, ax=ax, label='FTLE')
    
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    return fig, ax


# ============================================================================
# EXAMPLE USAGE AND DEMOS
# ============================================================================

def demo_with_lorenz():
    """
    Demonstrate Lyapunov computation on the Lorenz system.
    Known largest Lyapunov exponent ≈ 0.906
    """
    from scipy.integrate import odeint
    
    print("=" * 60)
    print("DEMO: Lyapunov Exponent of Lorenz System")
    print("Expected λ ≈ 0.906")
    print("=" * 60)
    
    # Lorenz system
    def lorenz(state, t, sigma=10, rho=28, beta=8/3):
        x, y, z = state
        return [
            sigma * (y - x),
            x * (rho - z) - y,
            x * y - beta * z
        ]
    
    # Generate trajectory
    t = np.linspace(0, 100, 10000)
    dt = t[1] - t[0]
    initial_state = [1.0, 1.0, 1.0]
    trajectory = odeint(lorenz, initial_state, t)
    
    # Use x-component as time series
    x_series = trajectory[:, 0]
    
    # Rosenstein method
    print("\n--- Rosenstein Method ---")
    estimator = RosensteinMethod(x_series, dt=dt)
    lyap, time, divergence = estimator.compute(embedding_dim=5, tau=10)
    print(f"Estimated λ: {lyap:.4f}")
    print(f"Error: {abs(lyap - 0.906):.4f}")
    
    # Wolf method
    print("\n--- Wolf Method ---")
    estimator = WolfMethod(x_series, dt=dt)
    lyap_wolf, _ = estimator.compute(embedding_dim=5, tau=10)
    print(f"Estimated λ: {lyap_wolf:.4f}")
    print(f"Error: {abs(lyap_wolf - 0.906):.4f}")
    
    return lyap, time, divergence


def demo_synthetic_pde():
    """
    Demonstrate spatial Lyapunov analysis on synthetic turbulent field.
    """
    print("\n" + "=" * 60)
    print("DEMO: Spatial Lyapunov on Synthetic Turbulent Field")
    print("=" * 60)
    
    # Generate synthetic turbulent field (simplified)
    np.random.seed(42)
    nx, ny = 64, 64
    n_time = 100
    
    # Create field with some chaotic and stable regions
    x = np.linspace(0, 2*np.pi, nx)
    y = np.linspace(0, 2*np.pi, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    fields = np.zeros((n_time, nx, ny))
    
    for t in range(n_time):
        # Chaotic region (center)
        chaotic = 0.5 * np.sin(X + 0.1*t) * np.cos(Y - 0.1*t)
        chaotic += 0.3 * np.random.randn(nx, ny) * np.exp(-((X-np.pi)**2 + (Y-np.pi)**2)/2)
        
        # Stable wave region (edges)
        stable = np.sin(X - 0.05*t) * np.sin(Y)
        
        # Blend
        mask = np.exp(-((X-np.pi)**2 + (Y-np.pi)**2)/3)
        fields[t] = mask * chaotic + (1-mask) * stable
        
    # Analyze
    analyzer = SpatialLyapunovField(fields, dt=0.1, dx=2*np.pi/nx)
    lyap_field = analyzer.compute_local_lyapunov(window_size=3, time_window=10)
    
    print(f"Mean local λ: {np.nanmean(lyap_field):.4f}")
    print(f"Max local λ: {np.nanmax(lyap_field):.4f}")
    print(f"Chaotic fraction: {100*np.mean(lyap_field > 0):.1f}%")
    
    return fields, lyap_field


if __name__ == "__main__":
    # Run demos
    print("\n" + "=" * 60)
    print("LYAPUNOV EXPONENT TOOLKIT FOR THE WELL")
    print("=" * 60)
    
    # Demo 1: Lorenz system validation
    lyap, time, divergence = demo_with_lorenz()
    
    # Demo 2: Synthetic PDE
    fields, lyap_field = demo_synthetic_pde()
    
    print("\n" + "=" * 60)
    print("TO USE WITH THE WELL:")
    print("=" * 60)
    print("""
    from lyapunov_well import WellLyapunovAnalyzer
    
    # Initialize analyzer
    analyzer = WellLyapunovAnalyzer(
        dataset_name='supernova_explosion_64',
        base_path='hf://datasets/polymathic-ai/'
    )
    
    # Load data
    analyzer.load_dataset()
    
    # Analyze single sample
    results = analyzer.analyze_sample(
        sample_idx=0,
        method='rosenstein',
        embedding_dim=5,
        tau=2
    )
    
    print(f"Lyapunov exponent: {results['lyapunov_exponent']:.4f}")
    print(f"Is chaotic: {results['is_chaotic']}")
    
    # Batch analysis
    all_results = analyzer.batch_analysis(n_samples=10, method='spatial')
    """)
