"""
Module for analyzing network training state independently from recognition mechanisms
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import Dict, List, Tuple

class TrainingAnalyzer:
    def __init__(self, connections, neuron_groups):
        self.connections = connections
        self.neuron_groups = neuron_groups
        
    def analyze_weights(self) -> Dict:
        """Analyze weight distributions and patterns"""
        XeAe_weights = self.connections['XeAe'].w
        
        # Basic statistics
        basic_stats = {
            'mean': float(np.mean(XeAe_weights)),
            'std': float(np.std(XeAe_weights)),
            'min': float(np.min(XeAe_weights)),
            'max': float(np.max(XeAe_weights))
        }
        
        # Distribution analysis
        hist, bins = np.histogram(XeAe_weights, bins=50, density=True)
        distribution = {
            'histogram': hist.tolist(),
            'bins': bins.tolist(),
            'skewness': float(stats.skew(XeAe_weights)),
            'kurtosis': float(stats.kurtosis(XeAe_weights))
        }
        
        # Weight organization
        organization = self._analyze_weight_organization(XeAe_weights)
        
        return {
            'basic_stats': basic_stats,
            'distribution': distribution,
            'organization': organization
        }
    
    def analyze_receptive_fields(self) -> Dict:
        """Analyze receptive fields of neurons"""
        weights = self.get_2d_input_weights()
        
        # Receptive field properties
        rf_properties = {
            'distinctiveness': self._calculate_rf_distinctiveness(weights),
            'overlap': self._calculate_rf_overlap(weights),
            'structure': self._analyze_rf_structure(weights)
        }
        
        # Feature selectivity
        selectivity = {
            'orientation': self._analyze_orientation_selectivity(weights),
            'position': self._analyze_position_selectivity(weights),
            'feature_maps': self._extract_feature_maps(weights)
        }
        
        return {
            'rf_properties': rf_properties,
            'selectivity': selectivity
        }
    
    def analyze_specialization(self) -> Dict:
        """Analyze neuron specialization"""
        # Specialization metrics
        metrics = {
            'selectivity': self._calculate_neuron_selectivity(),
            'response_patterns': self._analyze_response_patterns(),
            'category_preference': self._analyze_category_preference()
        }
        
        # Population analysis
        population = {
            'diversity': self._calculate_population_diversity(),
            'coverage': self._analyze_feature_coverage(),
            'redundancy': self._calculate_representation_redundancy()
        }
        
        return {
            'metrics': metrics,
            'population': population
        }
    
    def get_training_score(self) -> float:
        """Calculate overall training score"""
        # Weight quality (30% of score)
        weight_score = self._calculate_weight_quality()
        
        # Receptive field quality (40% of score)
        rf_score = self._calculate_rf_quality()
        
        # Specialization quality (30% of score)
        spec_score = self._calculate_specialization_quality()
        
        # Weighted average
        total_score = 0.3 * weight_score + 0.4 * rf_score + 0.3 * spec_score
        
        return float(total_score)
    
    def visualize_training_state(self, save_path: str = None):
        """Generate comprehensive visualization of training state"""
        fig = plt.figure(figsize=(15, 10))
        
        # Weight distribution
        ax1 = fig.add_subplot(231)
        self._plot_weight_distribution(ax1)
        
        # Receptive fields
        ax2 = fig.add_subplot(232)
        self._plot_receptive_fields(ax2)
        
        # Specialization
        ax3 = fig.add_subplot(233)
        self._plot_specialization(ax3)
        
        # Feature maps
        ax4 = fig.add_subplot(234)
        self._plot_feature_maps(ax4)
        
        # Category preference
        ax5 = fig.add_subplot(235)
        self._plot_category_preference(ax5)
        
        # Training metrics
        ax6 = fig.add_subplot(236)
        self._plot_training_metrics(ax6)
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    # Private helper methods
    def _calculate_weight_quality(self) -> float:
        """Calculate weight quality score (0-1)"""
        weights = self.connections['XeAe'].w
        
        # Check weight distribution
        dist_score = self._evaluate_weight_distribution(weights)
        
        # Check weight structure
        struct_score = self._evaluate_weight_structure(weights)
        
        # Check weight stability
        stab_score = self._evaluate_weight_stability(weights)
        
        return (dist_score + struct_score + stab_score) / 3
    
    def _calculate_rf_quality(self) -> float:
        """Calculate receptive field quality score (0-1)"""
        rfs = self.get_2d_input_weights()
        
        # Check RF distinctiveness
        distinct_score = self._evaluate_rf_distinctiveness(rfs)
        
        # Check RF structure
        struct_score = self._evaluate_rf_structure(rfs)
        
        # Check feature representation
        feature_score = self._evaluate_feature_representation(rfs)
        
        return (distinct_score + struct_score + feature_score) / 3
    
    def _calculate_specialization_quality(self) -> float:
        """Calculate specialization quality score (0-1)"""
        # Check neuron selectivity
        select_score = self._evaluate_neuron_selectivity()
        
        # Check population diversity
        div_score = self._evaluate_population_diversity()
        
        # Check category representation
        cat_score = self._evaluate_category_representation()
        
        return (select_score + div_score + cat_score) / 3

    def _evaluate_weight_distribution(self, weights) -> float:
        """Evaluate quality of weight distribution (0-1)"""
        # Calculate optimal weight parameters based on literature
        optimal_mean = 0.099492  # from well-trained network
        optimal_std = 0.055222
        
        # Get current parameters
        current_mean = np.mean(weights)
        current_std = np.std(weights)
        
        # Calculate scores
        mean_score = 1.0 - min(abs(current_mean - optimal_mean) / optimal_mean, 1.0)
        std_score = 1.0 - min(abs(current_std - optimal_std) / optimal_std, 1.0)
        
        return (mean_score + std_score) / 2

    def _evaluate_weight_structure(self, weights) -> float:
        """Evaluate structural organization of weights (0-1)"""
        # Check for dead weights
        dead_weights = np.sum(weights < 0.001) / len(weights)
        dead_score = 1.0 - dead_weights
        
        # Check for saturated weights
        saturated = np.sum(weights > 0.9) / len(weights)
        saturation_score = 1.0 - saturated
        
        # Check weight variance across neurons
        variance_score = self._evaluate_weight_variance(weights)
        
        return (dead_score + saturation_score + variance_score) / 3

    def _evaluate_weight_stability(self, weights) -> float:
        """Evaluate weight stability (0-1)"""
        # This would ideally compare with previous state
        # For now, use heuristics based on weight distribution
        
        # Check for extreme values
        extremes = np.sum((weights < 0.001) | (weights > 0.9)) / len(weights)
        extreme_score = 1.0 - extremes
        
        # Check distribution shape
        _, bins = np.histogram(weights, bins=50)
        bin_widths = np.diff(bins)
        uniformity = 1.0 - np.std(bin_widths) / np.mean(bin_widths)
        
        return (extreme_score + uniformity) / 2

    def _evaluate_rf_distinctiveness(self, rfs) -> float:
        """Evaluate distinctiveness of receptive fields (0-1)"""
        # Calculate pairwise correlations
        correlations = []
        for i in range(len(rfs)):
            for j in range(i+1, len(rfs)):
                corr = np.corrcoef(rfs[i].flat, rfs[j].flat)[0,1]
                correlations.append(abs(corr))
        
        # Convert to score (lower correlation is better)
        mean_correlation = np.mean(correlations)
        return 1.0 - mean_correlation

    def _evaluate_rf_structure(self, rfs) -> float:
        """Evaluate structure of receptive fields (0-1)"""
        scores = []
        for rf in rfs:
            # Check for spatial organization
            spatial_score = self._evaluate_spatial_organization(rf)
            
            # Check for feature representation
            feature_score = self._evaluate_feature_representation(rf)
            
            scores.append((spatial_score + feature_score) / 2)
        
        return np.mean(scores)

    def _evaluate_neuron_selectivity(self) -> float:
        """Evaluate selectivity of neurons (0-1)"""
        # This would ideally use actual response data
        # For now, use weight patterns as proxy
        weights = self.connections['XeAe'].w
        
        selectivity_scores = []
        for neuron_weights in weights.T:  # per neuron
            # Calculate weight concentration
            sorted_weights = np.sort(neuron_weights)[::-1]
            top_20_percent = sorted_weights[:len(sorted_weights)//5]
            selectivity = np.sum(top_20_percent) / np.sum(sorted_weights)
            selectivity_scores.append(selectivity)
        
        return np.mean(selectivity_scores)

    @staticmethod
    def _evaluate_spatial_organization(rf) -> float:
        """Evaluate spatial organization of a receptive field (0-1)"""
        # Calculate gradient structure
        gradient_x = np.gradient(rf)[0]
        gradient_y = np.gradient(rf)[1]
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        
        # Strong gradients indicate good spatial organization
        return min(np.mean(gradient_magnitude) * 5, 1.0)

    @staticmethod
    def _evaluate_feature_representation(rf) -> float:
        """Evaluate quality of feature representation (0-1)"""
        # Calculate local contrast
        local_contrast = np.std(rf) / np.mean(rf) if np.mean(rf) > 0 else 0
        
        # Calculate spatial frequency content
        freq_content = np.fft.fft2(rf)
        freq_magnitude = np.abs(freq_content)
        freq_score = np.sum(freq_magnitude[1:]) / np.sum(freq_magnitude)
        
        return (min(local_contrast, 1.0) + min(freq_score, 1.0)) / 2

def analyze_training(connections, neuron_groups, save_path=None):
    """
    Analyze training state and save results
    
    Args:
        connections: Network connections
        neuron_groups: Network neuron groups
        save_path: Optional path to save analysis results
    
    Returns:
        Dict containing analysis results
    """
    analyzer = TrainingAnalyzer(connections, neuron_groups)
    
    # Get all analyses
    results = {
        'weights': analyzer.analyze_weights(),
        'receptive_fields': analyzer.analyze_receptive_fields(),
        'specialization': analyzer.analyze_specialization(),
        'training_score': analyzer.get_training_score()
    }
    
    # Generate visualizations
    if save_path:
        # Create timestamp-based filename
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        viz_path = f"{save_path}/training_state_{timestamp}.png"
        analyzer.visualize_training_state(viz_path)
        
        # Save numerical results
        import json
        results_path = f"{save_path}/training_analysis_{timestamp}.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
    
    return results