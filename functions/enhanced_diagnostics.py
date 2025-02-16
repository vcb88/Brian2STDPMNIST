import numpy as np
import logging
from typing import Dict, Any, List, Tuple
from brian2 import ms, second
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy import stats

logger = logging.getLogger(__name__)

class EnhancedDiagnostics:
    def __init__(
        self,
        connections: Dict[str, Any],
        spike_monitors: Dict[str, Any],
        save_conns: list,
        stdp_params: Dict[str, Any],
        neuron_groups: Dict[str, Any] = None,
        record_history: bool = True
    ):
        self.connections = connections
        self.spike_monitors = spike_monitors
        self.save_conns = save_conns
        self.stdp_params = stdp_params
        self.neuron_groups = neuron_groups
        self.record_history = record_history
        
        # History containers
        self.weight_history = defaultdict(list)
        self.spike_history = defaultdict(list)
        self.theta_history = defaultdict(list)
        
    def record_state(self) -> None:
        """Record current network state for temporal analysis"""
        if not self.record_history:
            return
            
        # Record weights
        for conn_name in self.save_conns:
            if conn_name in self.connections:
                self.weight_history[conn_name].append(
                    np.array(self.connections[conn_name].w)
                )
        
        # Record spikes
        for monitor_name, monitor in self.spike_monitors.items():
            self.spike_history[monitor_name].append({
                'times': np.array(monitor.t/ms),
                'indices': np.array(monitor.i)
            })
            
        # Record theta values
        if self.neuron_groups and 'Ae' in self.neuron_groups:
            self.theta_history['Ae'].append(
                np.array(self.neuron_groups['Ae'].theta_)
            )

    def analyze_stdp_efficiency(self) -> Dict[str, Any]:
        """Analyze STDP learning efficiency"""
        results = {}
        
        for conn_name in self.save_conns:
            if conn_name not in self.connections or not self.weight_history[conn_name]:
                continue
                
            weights = self.weight_history[conn_name]
            
            # Weight change analysis
            if len(weights) > 1:
                weight_changes = np.diff(weights, axis=0)
                results[conn_name] = {
                    'mean_change': float(np.mean(weight_changes)),
                    'std_change': float(np.std(weight_changes)),
                    'max_change': float(np.max(np.abs(weight_changes))),
                    'change_distribution': np.histogram(weight_changes, bins=20)[0].tolist()
                }
                
                # Weight stability score (lower is more stable)
                stability = np.mean(np.abs(weight_changes)) / np.mean(weights[-1])
                results[conn_name]['stability_score'] = float(stability)
                
        return results

    def analyze_temporal_patterns(self) -> Dict[str, Any]:
        """Analyze temporal firing patterns"""
        results = {}
        
        for monitor_name, history in self.spike_history.items():
            if not history:
                continue
                
            # Last recording
            last_spikes = history[-1]
            spike_times = last_spikes['times']
            spike_indices = last_spikes['indices']
            
            # Inter-spike intervals
            isis = []
            for neuron in np.unique(spike_indices):
                neuron_times = spike_times[spike_indices == neuron]
                if len(neuron_times) > 1:
                    isis.extend(np.diff(neuron_times))
            
            if isis:
                results[monitor_name] = {
                    'mean_isi': float(np.mean(isis)),
                    'std_isi': float(np.std(isis)),
                    'cv_isi': float(np.std(isis) / np.mean(isis)),  # Coefficient of variation
                    'isi_distribution': np.histogram(isis, bins=20)[0].tolist()
                }
                
                # Bursting analysis
                short_isis = np.sum(np.array(isis) < 10)  # ISIs < 10ms
                results[monitor_name]['burst_ratio'] = float(short_isis / len(isis))
                
        return results

    def analyze_network_stability(self) -> Dict[str, Any]:
        """Analyze overall network stability"""
        results = {
            'weight_stability': {},
            'firing_stability': {},
            'theta_stability': {}
        }
        
        # Weight stability analysis
        for conn_name, history in self.weight_history.items():
            if len(history) > 1:
                weight_vars = [np.var(w) for w in history]
                weight_means = [np.mean(w) for w in history]
                
                results['weight_stability'][conn_name] = {
                    'variance_trend': float(np.mean(np.diff(weight_vars))),
                    'mean_trend': float(np.mean(np.diff(weight_means))),
                    'final_variance': float(weight_vars[-1]),
                    'final_mean': float(weight_means[-1])
                }
        
        # Firing rate stability
        for monitor_name, history in self.spike_history.items():
            if not history:
                continue
                
            rates = []
            for record in history:
                unique_neurons = np.unique(record['indices'])
                rates.append(len(unique_neurons))
            
            if rates:
                results['firing_stability'][monitor_name] = {
                    'rate_variance': float(np.var(rates)),
                    'rate_trend': float(np.mean(np.diff(rates))),
                    'final_rate': float(rates[-1])
                }
        
        # Theta stability
        for group_name, history in self.theta_history.items():
            if len(history) > 1:
                theta_vars = [np.var(t) for t in history]
                theta_means = [np.mean(t) for t in history]
                
                results['theta_stability'][group_name] = {
                    'variance_trend': float(np.mean(np.diff(theta_vars))),
                    'mean_trend': float(np.mean(np.diff(theta_means))),
                    'final_variance': float(theta_vars[-1]),
                    'final_mean': float(theta_means[-1])
                }
        
        return results

    def analyze_spatial_patterns(self) -> Dict[str, Any]:
        """Analyze spatial patterns of activity"""
        results = {}
        
        if 'Ae' not in self.spike_monitors:
            return results
            
        spike_counts = self.spike_monitors['Ae'].count
        n_neurons = len(spike_counts)
        
        # Spatial correlation analysis
        correlation_matrix = np.zeros((n_neurons, n_neurons))
        for i in range(n_neurons):
            for j in range(i+1, n_neurons):
                correlation_matrix[i,j] = correlation_matrix[j,i] = \
                    float(stats.pearsonr(spike_counts[i:i+1], spike_counts[j:j+1])[0])
        
        results['spatial'] = {
            'mean_correlation': float(np.mean(correlation_matrix)),
            'correlation_std': float(np.std(correlation_matrix)),
            'max_correlation': float(np.max(correlation_matrix)),
            'correlation_distribution': np.histogram(correlation_matrix.flatten(), bins=20)[0].tolist()
        }
        
        return results

    def generate_enhanced_report(self) -> None:
        """Generate comprehensive diagnostic report"""
        logger.info("\n=== Enhanced Diagnostic Report ===")
        
        # STDP efficiency analysis
        stdp_results = self.analyze_stdp_efficiency()
        if stdp_results:
            logger.info("\nSTDP Learning Efficiency:")
            for conn_name, stats in stdp_results.items():
                logger.info(f"\n{conn_name}:")
                logger.info(f"Mean weight change: {stats['mean_change']:.6f}")
                logger.info(f"Max weight change: {stats['max_change']:.6f}")
                logger.info(f"Weight stability score: {stats['stability_score']:.6f}")
        
        # Temporal patterns analysis
        temporal_results = self.analyze_temporal_patterns()
        if temporal_results:
            logger.info("\nTemporal Firing Patterns:")
            for monitor_name, stats in temporal_results.items():
                logger.info(f"\n{monitor_name}:")
                logger.info(f"Mean ISI: {stats['mean_isi']:.2f} ms")
                logger.info(f"ISI CV: {stats['cv_isi']:.2f}")
                logger.info(f"Burst ratio: {stats['burst_ratio']:.2f}")
        
        # Network stability analysis
        stability_results = self.analyze_network_stability()
        if stability_results.get('weight_stability'):
            logger.info("\nNetwork Stability:")
            for conn_name, stats in stability_results['weight_stability'].items():
                logger.info(f"\n{conn_name} weights:")
                logger.info(f"Variance trend: {stats['variance_trend']:.6f}")
                logger.info(f"Mean trend: {stats['mean_trend']:.6f}")
        
        # Spatial patterns analysis
        spatial_results = self.analyze_spatial_patterns()
        if spatial_results:
            logger.info("\nSpatial Activity Patterns:")
            spatial_stats = spatial_results['spatial']
            logger.info(f"Mean correlation: {spatial_stats['mean_correlation']:.3f}")
            logger.info(f"Max correlation: {spatial_stats['max_correlation']:.3f}")
            
        # Original diagnostic report
        logger.info("\n=== Original Diagnostic Report ===")
        from . import diagnostics
        diagnostics.diagnostic_report(
            self.connections,
            self.spike_monitors,
            self.save_conns,
            self.stdp_params,
            self.neuron_groups
        )