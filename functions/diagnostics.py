import numpy as np
import logging
from typing import Dict, Any
from brian2 import ms, second  # Import Brian2 units

logger = logging.getLogger(__name__)

def analyze_neuron_weights(connections: Dict[str, Any], neuron_indices: np.ndarray, connection_name: str) -> Dict[str, float]:
    """Analyze input or output weights for specific neurons.
    
    Args:
        connections: Dictionary of network connections
        neuron_indices: Indices of neurons to analyze
        connection_name: Name of the connection to analyze
        
    Returns:
        Dictionary with weight statistics
    """
    if connection_name not in connections:
        return {}
        
    conn = connections[connection_name]
    weights = np.array(conn.w)
    indices = np.array(conn.i if 'Ae' in connection_name else conn.j)
    
    # Get weights connected to specified neurons
    mask = np.isin(indices, neuron_indices)
    relevant_weights = weights[mask]
    
    if len(relevant_weights) == 0:
        return {
            'min': 0,
            'max': 0,
            'mean': 0,
            'std': 0,
            'count': 0
        }
    
    return {
        'min': float(np.min(relevant_weights)),
        'max': float(np.max(relevant_weights)),
        'mean': float(np.mean(relevant_weights)),
        'std': float(np.std(relevant_weights)),
        'count': len(relevant_weights)
    }

def analyze_activation_times(spike_monitor: Any, neuron_indices: np.ndarray) -> Dict[str, float]:
    """Analyze when neurons become active during simulation.
    
    Args:
        spike_monitor: Brian2 spike monitor
        neuron_indices: Indices of neurons to analyze
        
    Returns:
        Dictionary with timing statistics
    """
    if len(neuron_indices) == 0:
        return {}
        
    # Convert spike times to milliseconds, handling Brian2 units properly
    spike_times = np.array(spike_monitor.t/ms, dtype=float)
    spike_indices = np.array(spike_monitor.i, dtype=int)
    
    first_spikes = {}
    last_spikes = {}
    
    for idx in neuron_indices:
        neuron_spikes = spike_times[spike_indices == idx]
        if len(neuron_spikes) > 0:
            first_spikes[idx] = float(neuron_spikes[0])
            last_spikes[idx] = float(neuron_spikes[-1])
    
    if not first_spikes:
        return {}
        
    return {
        'earliest_spike': float(min(first_spikes.values())),
        'latest_spike': float(max(last_spikes.values())),
        'mean_first_spike': float(np.mean(list(first_spikes.values()))),
        'mean_last_spike': float(np.mean(list(last_spikes.values())))
    }

def diagnostic_report(
    connections: Dict[str, Any],
    spike_monitors: Dict[str, Any],
    save_conns: list,
    stdp_params: Dict[str, Any],
    neuron_groups: Dict[str, Any] = None
) -> None:
    """Generate diagnostic report about network state and parameters.
    
    Args:
        connections: Dictionary of network connections
        spike_monitors: Dictionary of spike monitors
        save_conns: List of connection names to analyze
        stdp_params: Dictionary of STDP parameters
        neuron_groups: Optional dictionary of neuron groups for theta analysis
    """
    logger.info("\n=== Diagnostic Report ===")
    
    # STDP parameters
    logger.info("\nSTDP Parameters:")
    for param, value in stdp_params.items():
        logger.info(f"{param}: {value}")
    
    # Weight statistics
    logger.info("\nWeight Statistics:")
    for connName in save_conns:
        if connName in connections:
            conn = connections[connName]
            weights = conn.w
            logger.info(f"\n{connName}:")
            logger.info(f"Min: {float(np.min(weights)):.6f}")
            logger.info(f"Max: {float(np.max(weights)):.6f}")
            logger.info(f"Mean: {float(np.mean(weights)):.6f}")
            logger.info(f"Std: {float(np.std(weights)):.6f}")
            
            # Additional weight analysis
            n_zero = np.sum(weights == 0)
            n_max = np.sum(weights >= stdp_params['wmax_ee'] * 0.95)  # weights close to max
            total = len(weights)
            logger.info(f"Zero weights: {n_zero}/{total} ({n_zero/total*100:.1f}%)")
            logger.info(f"Near-max weights: {n_max}/{total} ({n_max/total*100:.1f}%)")
    
    # Neuron activation statistics
    if 'Ae' in spike_monitors:
        spike_counts = spike_monitors['Ae'].count
        logger.info("\nNeuron Activation Statistics:")
        logger.info(f"Min spikes: {int(np.min(spike_counts))}")
        logger.info(f"Max spikes: {int(np.max(spike_counts))}")
        logger.info(f"Mean spikes: {float(np.mean(spike_counts)):.2f}")
        n_silent = np.sum(spike_counts == 0)
        n_total = len(spike_counts)
        logger.info(f"Silent neurons: {n_silent}/{n_total} ({n_silent/n_total*100:.1f}%)")
        
        # Find silent and active neurons
        silent_neurons = np.where(spike_counts == 0)[0]
        active_neurons = np.where(spike_counts > 0)[0]
        
        # Analyze timing of active neurons
        if len(active_neurons) > 0:
            timing_stats = analyze_activation_times(spike_monitors['Ae'], active_neurons)
            if timing_stats:
                logger.info("\nTiming Statistics (Active Neurons):")
                logger.info(f"Earliest spike: {timing_stats['earliest_spike']:.1f} ms")
                logger.info(f"Latest spike: {timing_stats['latest_spike']:.1f} ms")
                logger.info(f"Mean first spike: {timing_stats['mean_first_spike']:.1f} ms")
                logger.info(f"Mean last spike: {timing_stats['mean_last_spike']:.1f} ms")
        
        # Analyze weights for silent neurons
        if len(silent_neurons) > 0:
            logger.info("\nSilent Neurons Analysis:")
            # Input weights
            input_stats = analyze_neuron_weights(connections, silent_neurons, 'XeAe')
            if input_stats:
                logger.info("\nInput weights (XeAe) for silent neurons:")
                logger.info(f"Min: {input_stats['min']:.6f}")
                logger.info(f"Max: {input_stats['max']:.6f}")
                logger.info(f"Mean: {input_stats['mean']:.6f}")
                logger.info(f"Std: {input_stats['std']:.6f}")
                logger.info(f"Number of connections: {input_stats['count']}")
            
            # Output weights
            output_stats = analyze_neuron_weights(connections, silent_neurons, 'AeAi')
            if output_stats:
                logger.info("\nOutput weights (AeAi) for silent neurons:")
                logger.info(f"Min: {output_stats['min']:.6f}")
                logger.info(f"Max: {output_stats['max']:.6f}")
                logger.info(f"Mean: {output_stats['mean']:.6f}")
                logger.info(f"Std: {output_stats['std']:.6f}")
                logger.info(f"Number of connections: {output_stats['count']}")
        
        # Analyze theta values if available
        if neuron_groups and 'Ae' in neuron_groups:
            theta_values = np.array(neuron_groups['Ae'].theta_)
            if len(theta_values) > 0:
                logger.info("\nTheta Analysis:")
                if len(silent_neurons) > 0:
                    silent_theta = theta_values[silent_neurons]
                    logger.info("\nSilent neurons theta:")
                    logger.info(f"Min: {float(np.min(silent_theta)):.6f}")
                    logger.info(f"Max: {float(np.max(silent_theta)):.6f}")
                    logger.info(f"Mean: {float(np.mean(silent_theta)):.6f}")
                    logger.info(f"Std: {float(np.std(silent_theta)):.6f}")
                
                if len(active_neurons) > 0:
                    active_theta = theta_values[active_neurons]
                    logger.info("\nActive neurons theta:")
                    logger.info(f"Min: {float(np.min(active_theta)):.6f}")
                    logger.info(f"Max: {float(np.max(active_theta)):.6f}")
                    logger.info(f"Mean: {float(np.mean(active_theta)):.6f}")
                    logger.info(f"Std: {float(np.std(active_theta)):.6f}")
        
        # Activity distribution
        logger.info("\nActivity Distribution:")
        spike_ranges = [(0, 0), (1, 10), (11, 50), (51, 100), (101, np.inf)]
        for low, high in spike_ranges:
            if high == np.inf:
                count = np.sum(spike_counts > low)
                logger.info(f"Neurons with >{low} spikes: {count} ({count/n_total*100:.1f}%)")
            else:
                count = np.sum((spike_counts > low) & (spike_counts <= high))
                logger.info(f"Neurons with {low}-{high} spikes: {count} ({count/n_total*100:.1f}%)")
        
        # Top active neurons
        logger.info("\nMost Active Neurons:")
        if len(active_neurons) > 0:
            top_neurons = np.argsort(spike_counts)[-10:]
            top_counts = spike_counts[top_neurons]
            for n, c in zip(top_neurons[::-1], top_counts[::-1]):
                logger.info(f"Neuron {n}: {int(c)} spikes")