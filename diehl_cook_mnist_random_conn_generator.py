#!/usr/bin/env python3
'''
Created on 15.12.2014
@author: Peter U. Diehl
Updated for Python 3 compatibility and modern practices
'''

import logging
import os
from pathlib import Path
from typing import List, Dict, Tuple, Union, Optional
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

def random_delay(min_delay: float, max_delay: float) -> float:
    """Generate random delay between min_delay and max_delay.
    
    Args:
        min_delay: Minimum delay value in milliseconds
        max_delay: Maximum delay value in milliseconds
        
    Returns:
        float: Random delay value in milliseconds
        
    Raises:
        ValueError: If min_delay is greater than or equal to max_delay
    """
    if min_delay >= max_delay:
        raise ValueError("min_delay must be less than max_delay")
    return np.random.rand() * (max_delay - min_delay) + min_delay

def compute_pop_vector(pop_array: NDArray) -> float:
    """Compute population vector from array.
    
    Args:
        pop_array: 1D array of population activity
        
    Returns:
        float: Normalized angle of the population vector (0 to 1)
    """
    size = len(pop_array)
    complex_unit_roots = np.array([np.exp(1j*(2*np.pi/size)*cur_pos) for cur_pos in range(size)])
    cur_pos = (np.angle(np.sum(pop_array * complex_unit_roots)) % (2*np.pi)) / (2*np.pi)
    return cur_pos

def sparsen_matrix(base_matrix: NDArray, 
                  p_conn: float) -> Tuple[NDArray, List[Tuple[int, int, float]]]:
    """Create sparse matrix with connection probability p_conn.
    
    Args:
        base_matrix: Input dense matrix to be sparsified
        p_conn: Connection probability (0 to 1)
        
    Returns:
        Tuple containing:
            - Sparse weight matrix
            - List of connections (source, target, weight)
            
    Raises:
        ValueError: If p_conn is not between 0 and 1
    """
    if not 0 <= p_conn <= 1:
        raise ValueError("Connection probability must be between 0 and 1")
        
    weight_matrix = np.zeros(base_matrix.shape)
    num_weights = 0
    num_target_weights = int(base_matrix.shape[0] * base_matrix.shape[1] * p_conn)
    weight_list = [(0, 0, 0.0)] * num_target_weights  # Pre-allocate list
    
    while num_weights < num_target_weights:
        idx = (np.int32(np.random.rand()*base_matrix.shape[0]), 
               np.int32(np.random.rand()*base_matrix.shape[1]))
        if not weight_matrix[idx]:
            weight_matrix[idx] = base_matrix[idx]
            weight_list[num_weights] = (idx[0], idx[1], float(base_matrix[idx]))
            num_weights += 1
            
    return weight_matrix, weight_list

def create_weights(data_path: Optional[str] = None) -> None:
    """Create and save random connection weights for the neural network.
    
    Args:
        data_path: Optional path to save weight files. Defaults to './weights/random/'
        
    Raises:
        OSError: If unable to create output directory
    """
    # Network parameters
    n_input = 784  # Input layer neurons (28x28 MNIST images)
    n_e = 400      # Excitatory neurons
    n_i = n_e      # Inhibitory neurons
    
    # Set up paths
    if data_path is None:
        data_path = './weights/random/'
    data_path = Path(data_path)
    try:
        data_path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create directory {data_path}: {e}")
        raise
    
    # Weight parameters with documentation
    weight: Dict[str, float] = {
        'ee_input': 0.3,   # Input -> Excitatory connection strength
        'ei_input': 0.2,   # Input -> Inhibitory connection strength
        'ee': 0.1,         # Excitatory -> Excitatory connection strength
        'ei': 10.4,        # Excitatory -> Inhibitory connection strength
        'ie': 17.0,        # Inhibitory -> Excitatory connection strength
        'ii': 0.4          # Inhibitory -> Inhibitory connection strength
    }
    
    # Connection probabilities with documentation
    p_conn: Dict[str, float] = {
        'ee_input': 1.0,   # Input -> Excitatory (fully connected)
        'ei_input': 0.1,   # Input -> Inhibitory (sparse)
        'ee': 1.0,         # Excitatory -> Excitatory (fully connected)
        'ei': 0.0025,      # Excitatory -> Inhibitory (very sparse)
        'ie': 0.9,         # Inhibitory -> Excitatory (dense)
        'ii': 0.1          # Inhibitory -> Inhibitory (sparse)
    }
    
    try:
        # Create input to excitatory connections (XeAe)
        logger.info('Creating random connection matrices')
        for name in ['XeAe']:
            weight_matrix = (np.random.random((n_input, n_e)) + 0.01) * weight['ee_input']
            if p_conn['ee_input'] < 1.0:
                weight_matrix, weight_list = sparsen_matrix(weight_matrix, p_conn['ee_input'])
            else:
                weight_list = [(i, j, float(weight_matrix[i,j])) 
                              for j in range(n_e) 
                              for i in range(n_input)]
            np.save(data_path / f"{name}", weight_list)
            logger.info(f'Saved connection matrix: {name}')
        
        # Create input to inhibitory connections (XeAi)
        logger.info('Creating E->I connection matrices (input)')
        for name in ['XeAi']:
            weight_matrix = np.random.random((n_input, n_i)) * weight['ei_input']
            weight_matrix, weight_list = sparsen_matrix(weight_matrix, p_conn['ei_input'])
            np.save(data_path / f"{name}", weight_list)
            logger.info(f'Saved connection matrix: {name}')
        
        # Create excitatory to inhibitory connections (AeAi)
        logger.info('Creating E->I connection matrices (recurrent)')
        for name in ['AeAi']:
            if n_e == n_i:
                # One-to-one connections
                weight_list = [(i, i, weight['ei']) for i in range(n_e)]
            else:
                weight_matrix = np.random.random((n_e, n_i)) * weight['ei']
                weight_matrix, weight_list = sparsen_matrix(weight_matrix, p_conn['ei'])
            np.save(data_path / f"{name}", weight_list)
            logger.info(f'Saved connection matrix: {name}')
        
        # Create inhibitory to excitatory connections (AiAe)
        logger.info('Creating I->E connection matrices')
        for name in ['AiAe']:
            if n_e == n_i:
                # All-to-all except self-connections
                weight_matrix = np.ones((n_i, n_e)) * weight['ie']
                for i in range(n_i):
                    weight_matrix[i,i] = 0
                weight_list = [(i, j, float(weight_matrix[i,j])) 
                              for i in range(n_i) 
                              for j in range(n_e)]
            else:
                weight_matrix = np.random.random((n_i, n_e)) * weight['ie']
                weight_matrix, weight_list = sparsen_matrix(weight_matrix, p_conn['ie'])
            np.save(data_path / f"{name}", weight_list)
            logger.info(f'Saved connection matrix: {name}')
            
    except Exception as e:
        logger.error(f"Failed to create weight matrices: {e}")
        raise

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    create_weights()