"""
Extended diagnostics module for analyzing network training and performance
"""

import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Any
import logging
from scipy import stats
import matplotlib.pyplot as plt
from collections import defaultdict

logger = logging.getLogger(__name__)

def default_temporal_stats() -> Dict:
    """Return default temporal statistics structure with zero values"""
    return {
        'mean_isi': 0.0,
        'isi_cv': 0.0,
        'sync_index': 0.0,
        'isi_stats': {
            'min': 0.0,
            'max': 0.0,
            'median': 0.0,
            'n_intervals': 0
        },
        'firing_rate_stats': {
            'mean': 0.0,
            'std': 0.0,
            'max': 0.0,
            'min': 0.0
        }
    }

def calculate_synchronization(neuron_isis: Dict[int, np.ndarray]) -> float:
    """Calculate synchronization index based on ISI cross-correlation
    
    Args:
        neuron_isis: Dictionary mapping neuron IDs to their ISI arrays
        
    Returns:
        float: Synchronization index between 0 and 1
    """
    try:
        if len(neuron_isis) < 2:
            return 0.0
            
        # Convert ISIs to instantaneous rates
        rates = {}
        for neuron_id, isis in neuron_isis.items():
            if len(isis) > 0:
                rates[neuron_id] = 1.0 / np.mean(isis)
            
        if len(rates) < 2:
            return 0.0
            
        # Calculate correlation coefficient matrix
        rate_values = np.array(list(rates.values()))
        corr_matrix = np.corrcoef(rate_values)
        
        # Average upper triangle of correlation matrix (excluding diagonal)
        mask = np.triu(np.ones_like(corr_matrix), k=1)
        sync_index = np.mean(corr_matrix[mask > 0])
        
        return float(sync_index) if not np.isnan(sync_index) else 0.0
    except Exception as e:
        logger.error(f"Error calculating synchronization: {str(e)}")
        return 0.0

def safe_plot_data(data: Any, title: str, ax: Optional[plt.Axes] = None) -> Optional[plt.Figure]:
    """Safely plot data with proper error handling
    
    Args:
        data: Data to plot
        title: Plot title
        ax: Optional matplotlib axes to plot on
        
    Returns:
        Optional[plt.Figure]: Figure object if plotting successful, None otherwise
    """
    if isinstance(data, (float, np.float64, int, np.int64)):
        logger.info(f"Skipping plot for scalar value {title}: {data}")
        return None
        
    if not isinstance(data, (list, np.ndarray)) or len(data) == 0:
        logger.warning(f"Invalid data for plotting {title}")
        return None
        
    try:
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        else:
            fig = ax.figure
            
        ax.plot(data)
        ax.set_title(title)
        ax.grid(True)
        return fig
    except Exception as e:
        logger.error(f"Error plotting {title}: {str(e)}")
        return None

def analyze_temporal_patterns(spike_monitors, time_window: float = None) -> Dict:
    """Analyze temporal characteristics of neural activity
    
    Args:
        spike_monitors: Brian2 spike monitors
        time_window: Optional time window for analysis (in seconds)
        
    Returns:
        Dict containing temporal statistics
    """
    try:
        # Get spike times and indices
        if not isinstance(spike_monitors, dict) or 'Ae' not in spike_monitors:
            raise ValueError("Invalid spike monitors")
        
        # Try different attribute names for spike times
        if hasattr(spike_monitors['Ae'], 't'):
            spike_times = np.atleast_1d(np.array(spike_monitors['Ae'].t))
            logger.info("Using 't' attribute for spike times")
        elif hasattr(spike_monitors['Ae'], 't_'):
            spike_times = np.atleast_1d(np.array(spike_monitors['Ae'].t_))
            logger.info("Using 't_' attribute for spike times")
        else:
            raise ValueError("Cannot find spike times in monitor (no 't' or 't_' attribute)")
            
        # Get spike indices
        if hasattr(spike_monitors['Ae'], 'i'):
            spike_indices = np.atleast_1d(np.array(spike_monitors['Ae'].i))
            logger.info("Using 'i' attribute for spike indices")
        elif hasattr(spike_monitors['Ae'], 'i_'):
            spike_indices = np.atleast_1d(np.array(spike_monitors['Ae'].i_))
            logger.info("Using 'i_' attribute for spike indices")
        else:
            raise ValueError("Cannot find spike indices in monitor (no 'i' or 'i_' attribute)")
        
        logger.info(f"Temporal analysis: processing {len(spike_times)} spikes from {len(np.unique(spike_indices))} unique neurons")
        logger.info(f"Temporal analysis input shapes: times={spike_times.shape}, indices={spike_indices.shape}")
        logger.info(f"Temporal analysis data types: times={spike_times.dtype}, indices={spike_indices.dtype}")
        
        if len(spike_times) == 0:
            return {
                'mean_isi': 0.0,
                'isi_cv': 0.0,
                'sync_index': 0.0,
                'firing_rate_stats': {
                    'mean': 0.0,
                    'std': 0.0,
                    'max': 0.0,
                    'min': 0.0
                }
            }
        
        # Calculate ISIs and firing rates for each neuron
        unique_indices = np.unique(spike_indices)
        n_neurons = len(unique_indices)
        isis_all = []
        firing_rates = []
        
        # Use actual time window or maximum spike time
        actual_time_window = float(time_window) if time_window else float(np.max(spike_times))
        logger.info(f"Using time window of {actual_time_window:.2f} seconds for temporal analysis")
        
        # Calculate ISIs and firing rates per neuron
        total_spikes = 0
        active_neurons = 0
        
        for i in unique_indices:
            neuron_spikes = spike_times[spike_indices == i]
            n_spikes = len(neuron_spikes)
            total_spikes += n_spikes
            
            if n_spikes > 0:
                active_neurons += 1
                firing_rates.append(n_spikes / actual_time_window)
                
                if n_spikes > 1:
                    # Calculate ISIs only for neurons with multiple spikes
                    isis = np.diff(neuron_spikes)
                    if len(isis) > 0:  # Extra check to be safe
                        isis_all.extend(isis.tolist())  # Convert to list for safe extension
                        
        logger.info(f"Temporal analysis: {active_neurons} active neurons, {total_spikes} total spikes")
        if isis_all:
            logger.info(f"ISI statistics: {len(isis_all)} intervals, range: [{min(isis_all):.3f}, {max(isis_all):.3f}]")
        
        # Calculate temporal statistics
        if isis_all:
            mean_isi = np.mean(isis_all)
            isi_cv = np.std(isis_all) / mean_isi if mean_isi > 0 else 0
        else:
            mean_isi = 0
            isi_cv = 0
            
        # Calculate synchronization index using ISIs
        neuron_isis = {}
        for i in unique_indices:
            neuron_spikes = spike_times[spike_indices == i]
            if len(neuron_spikes) > 1:
                neuron_isis[i] = np.diff(neuron_spikes)
        
        sync_index = calculate_synchronization(neuron_isis)
            
        # Calculate firing rate statistics
        fr_stats = {
            'mean': float(np.mean(firing_rates)) if firing_rates else 0.0,
            'std': float(np.std(firing_rates)) if firing_rates else 0.0,
            'max': float(np.max(firing_rates)) if firing_rates else 0.0,
            'min': float(np.min(firing_rates)) if firing_rates else 0.0
        }
            
        return {
            'mean_isi': float(mean_isi),
            'isi_cv': float(isi_cv),
            'sync_index': float(sync_index),
            'firing_rate_stats': fr_stats
        }
    except Exception as e:
        logger.warning(f"Error in temporal pattern analysis: {str(e)}")
        return {
            'mean_isi': 0.0,
            'isi_cv': 0.0,
            'sync_index': 0.0,
            'firing_rate_stats': {
                'mean': 0.0,
                'std': 0.0,
                'max': 0.0,
                'min': 0.0
            }
        }

def analyze_learning_dynamics(connections, previous_weights=None) -> Dict:
    """Analyze learning dynamics and weight changes
    
    Args:
        connections: Current network connections
        previous_weights: Optional previous weights for change analysis
        
    Returns:
        Dict containing learning dynamics statistics
    """
    try:
        if not isinstance(connections, dict) or 'XeAe' not in connections:
            raise ValueError("Invalid connections dictionary or missing XeAe connection")
            
        # Get current weights
        current_weights = np.array(connections['XeAe'].w)
        if current_weights.size == 0:
            raise ValueError("Empty weight matrix")
            
        logger.info(f"Learning dynamics: weight array shape={current_weights.shape}, dtype={current_weights.dtype}")
        logger.info(f"Weight range: [{np.min(current_weights):.6f}, {np.max(current_weights):.6f}]")
        logger.info(f"Weight distribution: mean={np.mean(current_weights):.6f}, std={np.std(current_weights):.6f}")
        
        # Basic weight statistics
        weight_stats = {
            'mean': float(np.mean(current_weights)),
            'std': float(np.std(current_weights)),
            'min': float(np.min(current_weights)),
            'max': float(np.max(current_weights))
        }
        
        # Weight distribution analysis
        hist, bins = np.histogram(current_weights, bins=50, density=True)
        weight_distribution = {
            'histogram': hist.tolist(),
            'bins': bins.tolist(),
            'skewness': float(stats.skew(current_weights)),
            'kurtosis': float(stats.kurtosis(current_weights))
        }
        
        # Weight change analysis if previous weights available
        if previous_weights is not None:
            weight_changes = current_weights - previous_weights
            change_stats = {
                'mean_change': float(np.mean(np.abs(weight_changes))),
                'max_change': float(np.max(np.abs(weight_changes))),
                'significant_changes': float(np.mean(np.abs(weight_changes) > 0.01) * 100)
            }
        else:
            change_stats = {
                'mean_change': 0.0,
                'max_change': 0.0,
                'significant_changes': 0.0
            }
            
        return {
            'weight_stats': weight_stats,
            'weight_distribution': weight_distribution,
            'change_stats': change_stats
        }
    except Exception as e:
        logger.warning(f"Error in learning dynamics analysis: {str(e)}")
        return {
            'weight_stats': {
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0
            },
            'weight_distribution': {
                'histogram': [],
                'bins': [],
                'skewness': 0.0,
                'kurtosis': 0.0
            },
            'change_stats': {
                'mean_change': 0.0,
                'max_change': 0.0,
                'significant_changes': 0.0
            }
        }

def analyze_specialization(connections, neuron_groups, n_classes: int = 10) -> Dict:
    """Analyze neuron specialization and receptive fields
    
    Analyzes the specialization of neurons by examining their receptive fields,
    weight patterns, and response properties.
    
    Args:
        connections: Network connections dictionary containing weight matrices
        neuron_groups: Dictionary of neuron groups containing thresholds and states
        n_classes: Number of output classes (default: 10 for MNIST)
        
    Returns:
        Dict containing:
            - rf_overlap: Measure of receptive field overlap between neurons
            - selectivity_stats: Statistics about neuronal selectivity
            - threshold_stats: Statistics about neuronal thresholds
            - specialization_index: Overall measure of neuronal specialization
    """
    try:
        if not isinstance(connections, dict) or 'XeAe' not in connections:
            raise ValueError("Invalid connections dictionary or missing XeAe connection")
            
        if not isinstance(neuron_groups, dict) or 'Ae' not in neuron_groups:
            raise ValueError("Invalid neuron_groups dictionary or missing Ae group")
            
        # Get weights and reshape if needed
        weights = np.array(connections['XeAe'].w)
        logger.info(f"Specialization analysis: initial weight shape={weights.shape}, dtype={weights.dtype}")
        
        if weights.size == 0:
            raise ValueError("Empty weight matrix")
            
        n_input = 784  # MNIST input size
        n_neurons = 400  # Number of excitatory neurons
        
        # Reshape weights if they are flattened
        if len(weights.shape) == 1:
            try:
                if weights.size != n_input * n_neurons:
                    raise ValueError(f"Weight array size {weights.size} doesn't match expected size {n_input * n_neurons}")
                weights = weights.reshape(n_input, n_neurons)
                logger.info(f"Reshaped weights to shape={weights.shape}")
            except Exception as e:
                logger.error(f"Failed to reshape weights: {str(e)}")
                raise
        
        if weights.shape != (n_input, n_neurons):
            raise ValueError(f"Unexpected weight shape after reshape: {weights.shape}, expected ({n_input}, {n_neurons})")
            
        logger.info(f"Weight matrix properties: min={weights.min():.6f}, max={weights.max():.6f}, mean={weights.mean():.6f}")
        
        # Reshape weights to 28x28 receptive fields
        receptive_fields = []
        for i in range(n_neurons):
            if weights.shape[0] == n_input:
                rf = weights[:, i].reshape(28, 28)
            else:
                rf = weights[i, :].reshape(28, 28)
            receptive_fields.append(rf)
        receptive_fields = np.array(receptive_fields)
        
        # Calculate RF overlap
        rf_correlations = []
        for i in range(n_neurons):
            for j in range(i+1, n_neurons):
                corr = np.corrcoef(receptive_fields[i].flat, receptive_fields[j].flat)[0,1]
                if not np.isnan(corr):
                    rf_correlations.append(abs(corr))
        
        rf_overlap = np.mean(rf_correlations) if rf_correlations else 0
        
        # Calculate selectivity (using L2 norm of weights as simple proxy)
        selectivity = np.linalg.norm(weights, axis=0) if len(weights.shape) > 1 else [np.linalg.norm(weights)]
        selectivity = selectivity / np.max(selectivity)  # Normalize
        
        # Get neuron thresholds
        thresholds = np.array(neuron_groups['Ae'].theta_)
        
        return {
            'rf_overlap': float(rf_overlap),
            'selectivity_stats': {
                'mean': float(np.mean(selectivity)),
                'std': float(np.std(selectivity)),
                'max': float(np.max(selectivity)),
                'min': float(np.min(selectivity))
            },
            'threshold_stats': {
                'mean': float(np.mean(thresholds)),
                'std': float(np.std(thresholds)),
                'max': float(np.max(thresholds)),
                'min': float(np.min(thresholds))
            }
        }
    except Exception as e:
        logger.warning(f"Error in specialization analysis: {str(e)}")
        return {
            'rf_overlap': 0.0,
            'selectivity_stats': {
                'mean': 0.0,
                'std': 0.0,
                'max': 0.0,
                'min': 0.0
            },
            'threshold_stats': {
                'mean': 0.0,
                'std': 0.0,
                'max': 0.0,
                'min': 0.0
            }
        }

def calculate_efficiency_metrics(spike_monitors, accuracy: float = None, n_samples: int = None) -> Dict:
    """Calculate network efficiency metrics
    
    Args:
        spike_monitors: Brian2 spike monitors
        accuracy: Optional classification accuracy
        n_samples: Number of samples processed
        
    Returns:
        Dict containing efficiency metrics
    """
    try:
        if not isinstance(spike_monitors, dict) or 'Ae' not in spike_monitors:
            raise ValueError("Invalid spike monitors dictionary or missing Ae monitor")
            
        # Try different attribute names for spike times
        if hasattr(spike_monitors['Ae'], 't'):
            spike_times = np.atleast_1d(np.array(spike_monitors['Ae'].t))
            logger.info("Using 't' attribute for spike times in efficiency calculation")
        elif hasattr(spike_monitors['Ae'], 't_'):
            spike_times = np.atleast_1d(np.array(spike_monitors['Ae'].t_))
            logger.info("Using 't_' attribute for spike times in efficiency calculation")
        else:
            raise ValueError("Cannot find spike times in monitor (no 't' or 't_' attribute)")
            
        logger.info(f"Efficiency metrics: spike times shape={spike_times.shape}, dtype={spike_times.dtype}")
        logger.info(f"Processing {len(spike_times)} total spikes")
        
        if accuracy is not None:
            logger.info(f"Network accuracy: {accuracy:.4f}")
            
        if n_samples is not None:
            logger.info(f"Number of processed samples: {n_samples}")
        
        # Calculate total spikes
        total_spikes = len(spike_times)
        
        # Calculate spikes per sample
        spikes_per_sample = total_spikes / n_samples if n_samples else 0
        
        # Calculate efficiency (accuracy per spike)
        if accuracy is not None and total_spikes > 0:
            spike_efficiency = accuracy / total_spikes
        else:
            spike_efficiency = 0
            
        return {
            'total_spikes': int(total_spikes),
            'spikes_per_sample': float(spikes_per_sample),
            'spike_efficiency': float(spike_efficiency)
        }
    except Exception as e:
        logger.warning(f"Error in efficiency metrics calculation: {str(e)}")
        return {
            'total_spikes': 0,
            'spikes_per_sample': 0.0,
            'spike_efficiency': 0.0
        }

def plot_metrics(data, title: str, ax=None):
    """Safely plot metrics data
    
    Args:
        data: Data to plot
        title: Plot title
        ax: Optional matplotlib axis
    """
    if isinstance(data, (float, np.float64, int, np.int64)):
        logger.warning(f"Skipping plot for scalar value: {data} (type: {type(data)})")
        return
        
    if not isinstance(data, (list, np.ndarray)) or len(data) == 0:
        logger.warning(f"Invalid data for plotting: {type(data)}, length: {len(data) if hasattr(data, '__len__') else 'N/A'}")
        return
        
    try:
        if ax is None:
            ax = plt.gca()
        ax.plot(data)
        ax.set_title(title)
    except Exception as e:
        logger.error(f"Error plotting {title}: {str(e)}")

def extended_diagnostic_report(
    connections, 
    spike_monitors, 
    neuron_groups, 
    previous_weights=None,
    accuracy=None,
    n_samples=None,
    time_window=None
) -> Dict:
    """Generate extended diagnostic report
    
    Args:
        connections: Network connections
        spike_monitors: Brian2 spike monitors
        neuron_groups: Neuron groups
        previous_weights: Optional previous weights for change analysis
        accuracy: Optional classification accuracy
        n_samples: Number of samples processed
        time_window: Optional time window for analysis
        
    Returns:
        Dict containing all diagnostic metrics
    """
    temporal_stats = analyze_temporal_patterns(spike_monitors, time_window)
    learning_stats = analyze_learning_dynamics(connections, previous_weights)
    specialization_stats = analyze_specialization(connections, neuron_groups)
    efficiency_stats = calculate_efficiency_metrics(spike_monitors, accuracy, n_samples)
    
    # Log detailed report
    logger.info("\n=== Extended Diagnostic Report ===\n")
    
    logger.info("Network Activity Efficiency:")
    logger.info(f"- Total spikes: {efficiency_stats['total_spikes']}")
    logger.info(f"- Spikes per sample: {efficiency_stats['spikes_per_sample']:.2f}")
    logger.info(f"- Energy efficiency: {efficiency_stats['spike_efficiency']:.4f}")
    
    logger.info("\nTemporal Characteristics:")
    logger.info(f"- Mean ISI: {temporal_stats['mean_isi']:.2f} s")
    logger.info(f"- ISI CV: {temporal_stats['isi_cv']:.2f}")
    logger.info(f"- Network synchronization: {temporal_stats['sync_index']:.2f}")
    logger.info(f"- Mean firing rate: {temporal_stats['firing_rate_stats']['mean']:.2f} Hz")
    
    logger.info("\nLearning Dynamics:")
    logger.info(f"- Mean weight: {learning_stats['weight_stats']['mean']:.4f}")
    logger.info(f"- Weight std: {learning_stats['weight_stats']['std']:.4f}")
    logger.info(f"- Weight changes: {learning_stats['change_stats']['significant_changes']:.1f}%")
    
    logger.info("\nNeuron Specialization:")
    logger.info(f"- RF overlap: {specialization_stats['rf_overlap']:.2f}")
    logger.info(f"- Mean selectivity: {specialization_stats['selectivity_stats']['mean']:.2f}")
    logger.info(f"- Mean threshold: {specialization_stats['threshold_stats']['mean']:.4f}")
    
    return {
        'temporal_stats': temporal_stats,
        'learning_stats': learning_stats,
        'specialization_stats': specialization_stats,
        'efficiency_stats': efficiency_stats
    }