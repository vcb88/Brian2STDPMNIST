#!/usr/bin/env python3
'''
Created on 15.12.2014
@author: Peter U. Diehl
Updated for Python 3 compatibility and modern practices
'''

import numpy as np
import matplotlib.pyplot as plt
import time
import os.path
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from brian2 import *

from functions.data import get_labeled_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_recognized_number_ranking(assignments: np.ndarray, 
                                spike_rates: np.ndarray) -> np.ndarray:
    """Get ranking of recognized numbers based on neuron assignments and spike rates.
    
    Args:
        assignments: Array of neuron assignments to digits
        spike_rates: Array of neuron spike rates
        
    Returns:
        Array of digits sorted by likelihood (most likely first)
    """
    summed_rates = [0] * 10
    num_assignments = [0] * 10
    for i in range(10):
        num_assignments[i] = len(np.where(assignments == i)[0])
        if num_assignments[i] > 0:
            summed_rates[i] = np.sum(spike_rates[assignments == i]) / num_assignments[i]
    return np.argsort(summed_rates)[::-1]

def get_new_assignments(result_monitor: np.ndarray, 
                       input_numbers: np.ndarray) -> np.ndarray:
    """Assign neurons to digits based on their response patterns.
    
    Args:
        result_monitor: Array of neuron responses
        input_numbers: Array of input digit labels
        
    Returns:
        Array of neuron assignments to digits
    """
    logger.info(f'Result monitor shape: {result_monitor.shape}')
    assignments = np.ones(n_e) * -1  # initialize them as not assigned
    input_nums = np.asarray(input_numbers)
    maximum_rate = [0] * n_e    
    
    for j in range(10):
        num_inputs = len(np.where(input_nums == j)[0])
        if num_inputs > 0:
            rate = np.sum(result_monitor[input_nums == j], axis=0) / num_inputs
            for i in range(n_e):
                if rate[i] > maximum_rate[i]:
                    maximum_rate[i] = rate[i]
                    assignments[i] = j
    return assignments

# Configuration
MNIST_data_path = './mnist/'
data_path = './activity/'
training_ending = '10000'
testing_ending = '10000'
start_time_training = 0
end_time_training = int(training_ending)
start_time_testing = 0
end_time_testing = int(testing_ending)

n_e = 400  # Number of excitatory neurons
n_input = 784  # Number of input neurons (28x28 MNIST images)
ending = ''

logger.info('Loading MNIST data...')
training = get_labeled_data(MNIST_data_path + 'training')
testing = get_labeled_data(MNIST_data_path + 'testing', bTrain=False)

logger.info('Loading results...')
training_result_monitor = np.load(data_path + 'resultPopVecs' + training_ending + ending + '.npy')
training_input_numbers = np.load(data_path + 'inputNumbers' + training_ending + '.npy')
testing_result_monitor = np.load(data_path + 'resultPopVecs' + testing_ending + '.npy')
testing_input_numbers = np.load(data_path + 'inputNumbers' + testing_ending + '.npy')
logger.info(f'Training result monitor shape: {training_result_monitor.shape}')

def main():
    """Main evaluation function."""
    logger.info('Getting assignments...')
    assignments = get_new_assignments(
        training_result_monitor[start_time_training:end_time_training],
        training_input_numbers[start_time_training:end_time_training]
    )
    logger.info('Assignments obtained')
    
    counter = 0
    num_tests = end_time_testing // 10000
    sum_accuracy = [0] * num_tests
    
    while counter < num_tests:
        end_time = min(end_time_testing, 10000*(counter+1))
        start_time = 10000*counter
        test_results = np.zeros((10, end_time-start_time))
        
        logger.info('Calculating accuracy for sum...')
        for i in range(end_time - start_time):
            test_results[:,i] = get_recognized_number_ranking(
                assignments,
                testing_result_monitor[i+start_time,:]
            )
            
        difference = test_results[0,:] - testing_input_numbers[start_time:end_time]
        correct = len(np.where(difference == 0)[0])
        incorrect = np.where(difference != 0)[0]
        sum_accuracy[counter] = correct/float(end_time-start_time) * 100
        
        logger.info(f'Sum response - accuracy: {sum_accuracy[counter]:.2f}%, '
                   f'num incorrect: {len(incorrect)}')
        counter += 1
        
    logger.info(f'Sum response - accuracy: mean={np.mean(sum_accuracy):.2f}%, '
                f'std={np.std(sum_accuracy):.2f}%')
    
    # Create plots directory if it doesn't exist
    plots_dir = Path('plots')
    plots_dir.mkdir(exist_ok=True)
    
    # Plot accuracy distribution
    plt.figure(figsize=(10, 5))
    plt.hist(sum_accuracy, bins=20, edgecolor='black')
    plt.title('Accuracy Distribution Across Test Batches')
    plt.xlabel('Accuracy (%)')
    plt.ylabel('Count')
    plt.savefig(plots_dir / 'accuracy_distribution.png')
    plt.close()
    
    logger.info('Evaluation completed successfully')

if __name__ == "__main__":
    main()
