#!/usr/bin/env python3
"""
Weight visualization tool for analyzing neural network weights.
Updated for Python 3 compatibility and modern practices.
"""

import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm

# Configure logging
logger = logging.getLogger(__name__)

class WeightVisualizer:
    """Class for visualizing neural network weights."""
    
    def __init__(self, 
                 cmap_name: str = 'hot_r',
                 fig_size: int = 18,
                 ending: str = '') -> None:
        """Initialize the visualizer.
        
        Args:
            cmap_name: Name of the colormap to use
            fig_size: Size of the figure in inches
            ending: Optional suffix for weight file names
        """
        self.cmap = plt.colormaps[cmap_name]
        self.fig_size = fig_size
        self.ending = ending
        
        # Network dimensions
        self.n_input = 784  # 28x28 MNIST images
        self.n_e = 400      # Number of excitatory neurons
        
        # Store weight matrices
        self.weight_values: Dict[str, NDArray] = {}
        
        # Define custom colors
        self.colors = {
            'bright_grey': '#f4f4f4',
            'red': '#ff0000',
            'green': '#00ff00',
            'black': '#000000'
        }
        
        # Create custom colormap
        self.custom_cmap = plt.LinearSegmentedColormap.from_list(
            'own2',
            [self.colors['bright_grey'], self.colors['black']]
        )

    def compute_pop_vector(self, pop_array: NDArray) -> float:
        """Compute population vector from array.
        
        Args:
            pop_array: 1D array of population activity
            
        Returns:
            float: Normalized angle of the population vector
        """
        size = len(pop_array)
        complex_unit_roots = np.array([
            np.exp(1j * (2*np.pi/size) * cur_pos) 
            for cur_pos in range(size)
        ])
        cur_pos = (np.angle(np.sum(pop_array * complex_unit_roots)) % (2*np.pi)) / (2*np.pi)
        return cur_pos

    def get_2d_input_weights(self) -> NDArray:
        """Convert 1D input weights to 2D representation for visualization.
        
        Returns:
            ndarray: 2D array of rearranged weights
            
        Raises:
            KeyError: If XA_values are not loaded
        """
        if 'XA' not in self.weight_values:
            raise KeyError("XA weight values not loaded")
            
        weight_matrix = self.weight_values['XA']
        n_e_sqrt = int(np.sqrt(self.n_e))
        n_in_sqrt = int(np.sqrt(self.n_input))
        num_values = n_e_sqrt * n_in_sqrt
        
        rearranged_weights = np.zeros((num_values, num_values))
        
        for i in range(n_e_sqrt):
            for j in range(n_e_sqrt):
                start_i = i * n_in_sqrt
                end_i = (i + 1) * n_in_sqrt
                start_j = j * n_in_sqrt
                end_j = (j + 1) * n_in_sqrt
                
                rearranged_weights[start_i:end_i, start_j:end_j] = \
                    weight_matrix[:, i + j*n_e_sqrt].reshape((n_in_sqrt, n_in_sqrt))
        
        return rearranged_weights

    def plot_2d_input_weights(self, save_path: Optional[Path] = None) -> Tuple[plt.Figure, plt.Axes]:
        """Plot 2D visualization of input weights.
        
        Args:
            save_path: Optional path to save the plot
            
        Returns:
            tuple: (figure object, axes object)
        """
        weights = self.get_2d_input_weights()
        
        fig, ax = plt.subplots(figsize=(self.fig_size, self.fig_size))
        im = ax.imshow(weights, interpolation="nearest", vmin=0, cmap=self.cmap)
        plt.colorbar(im)
        ax.set_title('XeAe Connection Weights')
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Saved weight plot to {save_path}")
        
        return fig, ax

    def load_weights(self, data_path: Union[str, Path]) -> None:
        """Load and process weight files.
        
        Args:
            data_path: Path to the directory containing weight files
            
        Raises:
            FileNotFoundError: If weight files are not found
        """
        data_path = Path(data_path)
        readout_names = ['XeAe' + self.ending]
        
        for name in tqdm(readout_names, desc="Loading weights"):
            try:
                weight_file = data_path / f"{name}.npy"
                if not weight_file.exists():
                    raise FileNotFoundError(f"Weight file not found: {weight_file}")
                
                readout = np.load(weight_file)
                
                # Initialize weight matrix
                if name == 'XeAe' + self.ending:
                    value_arr = np.nan * np.ones((self.n_input, self.n_e))
                else:
                    value_arr = np.nan * np.ones((self.n_e, self.n_e))
                
                # Process connection parameters
                for conn in readout:
                    src, tgt = int(conn[0]), int(conn[1])
                    value = float(conn[2])
                    
                    if np.isnan(value_arr[src, tgt]):
                        value_arr[src, tgt] = value
                    else:
                        value_arr[src, tgt] += value
                
                # Store processed weights
                self.weight_values[name] = np.asarray(value_arr)
                
                # Create visualization
                fig = plt.figure()
                plt.pcolor(value_arr)
                plt.colorbar()
                plt.title(name)
                plt.savefig(data_path / f"{name}_plot.png")
                plt.close(fig)
                
                logger.info(f"Processed weight file: {name}")
                
            except Exception as e:
                logger.error(f"Error processing {name}: {e}")
                raise

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Neural network weight visualization tool')
    parser.add_argument('--data-path', type=str, default='./weights',
                       help='Path to weight files directory')
    parser.add_argument('--cmap', type=str, default='hot_r',
                       help='Matplotlib colormap name')
    parser.add_argument('--fig-size', type=int, default=18,
                       help='Figure size in inches')
    parser.add_argument('--save-dir', type=str, default='./plots',
                       help='Directory to save plots')
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Create save directory
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize visualizer
        visualizer = WeightVisualizer(
            cmap_name=args.cmap,
            fig_size=args.fig_size
        )
        
        # Load and process weights
        visualizer.load_weights(args.data_path)
        
        # Create and save plots
        fig, _ = visualizer.plot_2d_input_weights(
            save_path=save_dir / "input_weights_2d.png"
        )
        plt.close(fig)
        
        logger.info("Weight visualization completed successfully")
        
    except Exception as e:
        logger.error(f"Error during visualization: {e}")
        raise

if __name__ == "__main__":
    main()