import numpy as np
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def diagnostic_report(
    connections: Dict[str, Any],
    spike_monitors: Dict[str, Any],
    save_conns: list,
    stdp_params: Dict[str, Any]
) -> None:
    """Generate diagnostic report about network state and parameters.
    
    Args:
        connections: Dictionary of network connections
        spike_monitors: Dictionary of spike monitors
        save_conns: List of connection names to analyze
        stdp_params: Dictionary of STDP parameters
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
        
        # Additional spike analysis
        active_neurons = np.where(spike_counts > 0)[0]
        logger.info(f"\nActive neurons: {len(active_neurons)} out of {n_total}")
        if len(active_neurons) > 0:
            top_neurons = np.argsort(spike_counts)[-10:]
            top_counts = spike_counts[top_neurons]
            for n, c in zip(top_neurons[::-1], top_counts[::-1]):
                logger.info(f"Neuron {n}: {int(c)} spikes")