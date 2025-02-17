"""
Module for analyzing network training state independently from recognition mechanisms
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import Dict

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
            'distinctiveness': self._evaluate_rf_distinctiveness(weights),
            'structure': self._evaluate_rf_structure(weights)
        }
        
        return {
            'rf_properties': rf_properties
        }
        
    def get_2d_input_weights(self) -> np.ndarray:
        """Get input weights reshaped into 2D form (for MNIST 28x28 images)
        
        Returns:
            Array of shape (n_neurons, 28, 28) containing receptive fields,
            or (1, 28, 28) if reshaping is not possible
        """
        try:
            weights = self.connections['XeAe'].w
            if not isinstance(weights, np.ndarray):
                weights = np.array(weights)
                
            # Handle different input shapes
            if len(weights.shape) == 1:
                # Single vector of weights
                if weights.size == 784:  # 28*28
                    return weights.reshape(1, 28, 28)
                else:
                    return np.zeros((1, 28, 28))  # Return empty RF
            elif len(weights.shape) == 2:
                # Matrix of weights
                if weights.shape[0] == 784:  # Input dimension is correct
                    n_neurons = weights.shape[1]
                    reshaped = []
                    for i in range(n_neurons):
                        neuron_weights = weights[:, i].reshape(28, 28)
                        reshaped.append(neuron_weights)
                    return np.array(reshaped)
                elif weights.shape[1] == 784:  # Transposed matrix
                    n_neurons = weights.shape[0]
                    reshaped = []
                    for i in range(n_neurons):
                        neuron_weights = weights[i, :].reshape(28, 28)
                        reshaped.append(neuron_weights)
                    return np.array(reshaped)
                else:
                    # Invalid dimensions, return empty RF
                    return np.zeros((1, 28, 28))
            else:
                # More than 2 dimensions or empty
                return np.zeros((1, 28, 28))
        except (AttributeError, ValueError, IndexError) as e:
            print(f"Warning: Could not reshape weights to 2D form: {str(e)}")
            return np.zeros((1, 28, 28))
    
    def analyze_specialization(self) -> Dict:
        """Analyze neuron specialization"""
        return {
            'selectivity': self._evaluate_neuron_selectivity()
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
        
        # Weight metrics
        ax3 = fig.add_subplot(233)
        self._plot_weight_metrics(ax3)
        
        # Specialization metrics
        ax4 = fig.add_subplot(234)
        self._plot_specialization_metrics(ax4)
        
        # Weight organization
        ax5 = fig.add_subplot(235)
        self._plot_weight_organization(ax5)
        
        # Training metrics
        ax6 = fig.add_subplot(236)
        self._plot_training_metrics(ax6)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_weight_distribution(self, ax):
        """Plot weight distribution histogram"""
        weights = self.connections['XeAe'].w
        ax.hist(weights.flatten(), bins=50, density=True)
        ax.set_title('Weight Distribution')
        ax.set_xlabel('Weight Value')
        ax.set_ylabel('Density')
        
    def _plot_receptive_fields(self, ax):
        """Plot sample of receptive fields"""
        weights = self.get_2d_input_weights()
        n_samples = min(9, len(weights))
        indices = np.random.choice(len(weights), n_samples, replace=False)
        
        for idx, i in enumerate(indices):
            if idx >= n_samples:
                break
            plt.subplot(3, 3, idx + 1)
            plt.imshow(weights[i], cmap='viridis')
            plt.axis('off')
        plt.suptitle('Sample Receptive Fields')
        
    def _plot_weight_metrics(self, ax):
        """Plot weight quality metrics"""
        metrics = [
            ('Distribution', self._evaluate_weight_distribution(self.connections['XeAe'].w)),
            ('Structure', self._evaluate_weight_structure(self.connections['XeAe'].w)),
            ('Stability', self._evaluate_weight_stability(self.connections['XeAe'].w))
        ]
        
        x = np.arange(len(metrics))
        values = [m[1] for m in metrics]
        ax.bar(x, values)
        ax.set_xticks(x)
        ax.set_xticklabels([m[0] for m in metrics])
        ax.set_ylim(0, 1)
        ax.set_title('Weight Quality Metrics')
        
    def _plot_specialization_metrics(self, ax):
        """Plot neuron specialization metrics"""
        selectivity = self._evaluate_neuron_selectivity()
        
        ax.bar(['Selectivity'], [selectivity])
        ax.set_ylim(0, 1)
        ax.set_title('Neuron Specialization')
        
    def _plot_weight_organization(self, ax):
        """Plot weight organization metrics"""
        org = self._analyze_weight_organization(self.connections['XeAe'].w)
        metrics = ['Sparsity', 'Clustering', 'Symmetry', 'Topology']
        values = [org[k.lower()] for k in metrics]
        
        x = np.arange(len(metrics))
        ax.bar(x, values)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        ax.set_title('Weight Organization')
        
    def _plot_training_metrics(self, ax):
        """Plot overall training metrics"""
        metrics = [
            ('Weight', self._calculate_weight_quality()),
            ('RF', self._calculate_rf_quality()),
            ('Spec', self._calculate_specialization_quality())
        ]
        
        x = np.arange(len(metrics))
        values = [m[1] for m in metrics]
        ax.bar(x, values)
        ax.set_xticks(x)
        ax.set_xticklabels([m[0] for m in metrics])
        ax.set_ylim(0, 1)
        ax.set_title('Training Quality Metrics')
    
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
        try:
            # Convert weights to numpy array for analysis
            weights = np.array(self.connections['XeAe'].w)
            if weights.size == 0:
                return 0.0
                
            # Reshape weights if needed
            if len(weights.shape) == 1:
                weights = weights.reshape(-1, 1)
                
            # Calculate neuron selectivity (how specialized each neuron is)
            selectivity_score = self._evaluate_neuron_selectivity()
            
            # Calculate population sparsity (how distributed the activity is)
            sparsity_score = self._calculate_population_sparsity(weights)
            
            # Calculate response diversity (how different neurons are from each other)
            diversity_score = self._calculate_response_diversity(weights)
            
            # Combine scores
            scores = [s for s in [selectivity_score, sparsity_score, diversity_score] if not np.isnan(s)]
            return float(np.mean(scores) if scores else 0.0)
            
        except (ValueError, RuntimeWarning, AttributeError) as e:
            print(f"Warning: Error in specialization quality calculation: {str(e)}")
            return 0.0
            
    def _calculate_population_sparsity(self, weights) -> float:
        """Calculate population sparsity score (0-1)
        
        Higher score indicates better distributed activity across the population
        """
        try:
            if weights.size == 0:
                return 0.0
                
            # Calculate mean activity per neuron
            neuron_activity = np.mean(np.abs(weights), axis=0)
            
            if np.sum(neuron_activity) < 1e-10:
                return 0.0
                
            # Calculate participation ratio
            normalized_activity = neuron_activity / np.sum(neuron_activity)
            entropy = -np.sum(normalized_activity * np.log2(normalized_activity + 1e-10))
            max_entropy = np.log2(len(normalized_activity))
            
            return float(entropy / max_entropy if max_entropy > 0 else 0.0)
            
        except (ValueError, RuntimeWarning) as e:
            print(f"Warning: Error in population sparsity calculation: {str(e)}")
            return 0.0
            
    def _calculate_response_diversity(self, weights) -> float:
        """Calculate response diversity score (0-1)
        
        Higher score indicates more diverse response patterns across neurons
        """
        try:
            if weights.shape[1] < 2:  # Need at least 2 neurons for diversity
                return 0.0
                
            # Calculate pairwise correlations between neurons
            correlations = []
            n_neurons = min(1000, weights.shape[1])  # Limit computation for large populations
            indices = np.random.choice(weights.shape[1], n_neurons, replace=False)
            
            for i in range(len(indices)):
                for j in range(i+1, len(indices)):
                    corr = np.corrcoef(weights[:, indices[i]], weights[:, indices[j]])[0,1]
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
                        
            if not correlations:
                return 0.0
                
            # Convert mean correlation to diversity score
            # Lower correlation means higher diversity
            mean_correlation = np.mean(correlations)
            return float(1.0 - mean_correlation)
            
        except (ValueError, RuntimeWarning) as e:
            print(f"Warning: Error in response diversity calculation: {str(e)}")
            return 0.0

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
        
    def _evaluate_weight_variance(self, weights) -> float:
        """Evaluate weight variance across neurons
        
        Returns a score between 0 and 1, where 1 indicates optimal variance
        """
        # Convert to 2D if needed
        weights_2d = weights.reshape(-1, weights.shape[-1]) if len(weights.shape) > 2 else weights
        
        # Calculate variance per neuron
        neuron_variances = np.var(weights_2d, axis=0)
        
        # Calculate coefficient of variation (CV) of variances
        cv = np.std(neuron_variances) / np.mean(neuron_variances) if np.mean(neuron_variances) > 0 else 0
        
        # Score based on CV (lower CV is better, but we want some variation)
        optimal_cv = 0.5  # We expect some variation between neurons
        score = 1.0 - min(abs(cv - optimal_cv), 1.0)
        
        return float(score)

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

    def _analyze_weight_organization(self, weights) -> Dict:
        """Analyze weight organization patterns
        
        Args:
            weights: Network weights to analyze
            
        Returns:
            Dict containing organization metrics
        """
        # Convert weights to 2D array if needed
        if isinstance(weights, np.ndarray):
            weights_arr = weights
        else:
            weights_arr = np.array(weights)
            
        if len(weights_arr.shape) == 1:
            weights_2d = weights_arr.reshape(-1, 1)
        else:
            weights_2d = weights_arr
            
        # Calculate weight sparsity
        sparsity = np.mean(weights_2d < 0.01)
        
        # Calculate weight clustering
        clustering = self._calculate_weight_clustering(weights_2d)
        
        # Calculate weight symmetry
        symmetry = self._calculate_weight_symmetry(weights_2d)
        
        # Calculate topological organization
        topology = self._calculate_topological_organization(weights_2d)
        
        return {
            'sparsity': float(sparsity),
            'clustering': float(clustering),
            'symmetry': float(symmetry),
            'topology': float(topology)
        }
        
    def _calculate_weight_clustering(self, weights_2d) -> float:
        """Calculate weight clustering coefficient"""
        # Simple clustering metric based on weight correlation
        if weights_2d.shape[1] < 2:  # Not enough columns for correlation
            return 0.0
            
        correlations = []
        n_samples = min(1000, weights_2d.shape[1])  # Limit computation for large matrices
        try:
            indices = np.random.choice(weights_2d.shape[1], min(n_samples, weights_2d.shape[1]), replace=False)
            
            for i in range(len(indices)):
                for j in range(i+1, len(indices)):
                    corr = np.corrcoef(weights_2d[:, indices[i]], weights_2d[:, indices[j]])[0,1]
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
        except (ValueError, IndexError):
            # If something goes wrong with the sampling, try a simpler approach
            if weights_2d.shape[1] >= 2:
                corr = np.corrcoef(weights_2d[:, 0], weights_2d[:, 1])[0,1]
                if not np.isnan(corr):
                    correlations.append(abs(corr))
                    
        return float(np.mean(correlations) if correlations else 0.0)
        
    def _calculate_weight_symmetry(self, weights_2d) -> float:
        """Calculate weight matrix symmetry"""
        try:
            if weights_2d.shape[0] != weights_2d.shape[1]:
                # For non-square matrices, calculate correlation between input and output patterns
                if weights_2d.shape[1] > 1:
                    input_pattern = np.mean(weights_2d, axis=1)
                    output_pattern = np.mean(weights_2d, axis=0)
                    if len(input_pattern) > len(output_pattern):
                        input_pattern = input_pattern[:len(output_pattern)]
                    else:
                        output_pattern = output_pattern[:len(input_pattern)]
                    corr = np.corrcoef(input_pattern, output_pattern)[0,1]
                    return float(abs(corr)) if not np.isnan(corr) else 0.0
                return 0.0
                
            # For square matrices, calculate traditional symmetry
            diff = np.abs(weights_2d - weights_2d.T)
            symmetry = 1.0 - (np.sum(diff) / (weights_2d.shape[0] * weights_2d.shape[1]))
            return float(symmetry)
        except (ValueError, IndexError):
            return 0.0
        
    def _calculate_topological_organization(self, weights_2d) -> float:
        """Calculate topological organization of weights"""
        try:
            # Ensure we have at least 2 points for gradient calculation
            if weights_2d.shape[1] < 2:
                return 0.0
                
            # Calculate gradients along both dimensions if possible
            gradients_x = np.gradient(weights_2d, axis=1)
            if weights_2d.shape[0] > 1:
                gradients_y = np.gradient(weights_2d, axis=0)
                # Combine both gradients
                gradient_magnitude = np.sqrt(gradients_x**2 + gradients_y**2)
            else:
                gradient_magnitude = np.abs(gradients_x)
                
            # Calculate smoothness metric
            smoothness = 1.0 - np.mean(gradient_magnitude)
            return float(min(max(smoothness, 0.0), 1.0))  # Ensure result is between 0 and 1
        except (ValueError, IndexError):
            return 0.0

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
        try:
            # Convert weights to numpy array
            weights = np.array(self.connections['XeAe'].w)
            if len(weights.shape) < 2:
                # If weights are 1D, reshape to 2D
                if weights.size == 0:
                    return 0.0
                weights = weights.reshape(-1, 1)
            
            selectivity_scores = []
            # Iterate over neurons (columns)
            for i in range(weights.shape[1] if len(weights.shape) > 1 else 1):
                # Get weights for current neuron
                neuron_weights = weights[:, i] if len(weights.shape) > 1 else weights
                
                # Calculate weight concentration
                if len(neuron_weights) == 0 or np.sum(neuron_weights) == 0:
                    continue
                    
                sorted_weights = np.sort(neuron_weights)[::-1]
                top_size = max(1, len(sorted_weights)//5)  # At least 1 weight
                top_weights = sorted_weights[:top_size]
                
                total_weight = np.sum(sorted_weights)
                if total_weight > 0:  # Avoid division by zero
                    selectivity = np.sum(top_weights) / total_weight
                    selectivity_scores.append(selectivity)
            
            return float(np.mean(selectivity_scores) if selectivity_scores else 0.0)
        except (AttributeError, ValueError, IndexError) as e:
            print(f"Warning: Could not evaluate neuron selectivity: {str(e)}")
            return 0.0

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
        try:
            if rf.size == 0:
                return 0.0
                
            # Calculate local contrast
            rf_mean = np.mean(rf)
            rf_std = np.std(rf)
            local_contrast = rf_std / rf_mean if abs(rf_mean) > 1e-10 else 0.0
            
            # Calculate spatial frequency content
            freq_content = np.fft.fft2(rf)
            freq_magnitude = np.abs(freq_content)
            freq_sum = np.sum(freq_magnitude)
            
            if freq_sum > 1e-10:  # Avoid division by very small numbers
                freq_score = np.sum(freq_magnitude[1:]) / freq_sum
            else:
                freq_score = 0.0
            
            # Normalize scores to [0, 1]
            contrast_score = min(local_contrast, 1.0) if not np.isnan(local_contrast) else 0.0
            freq_score = min(freq_score, 1.0) if not np.isnan(freq_score) else 0.0
            
            return (contrast_score + freq_score) / 2
        except (ValueError, RuntimeWarning) as e:
            print(f"Warning: Error in feature representation evaluation: {str(e)}")
            return 0.0

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