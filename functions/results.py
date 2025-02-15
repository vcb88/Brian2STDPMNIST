import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)

def analyze_results(input_numbers_file: str, result_vectors_file: str) -> Tuple[float, List[int], List[int]]:
    """
    Analyze the results of the network simulation.
    
    Args:
        input_numbers_file: Path to the .npy file containing input numbers
        result_vectors_file: Path to the .npy file containing result vectors
        
    Returns:
        Tuple containing:
        - accuracy: Classification accuracy
        - correct_indices: Indices of correctly classified examples
        - incorrect_indices: Indices of incorrectly classified examples
    """
    input_numbers = np.load(input_numbers_file)
    result_vecs = np.load(result_vectors_file)
    
    # Get the predicted numbers (maximum activation)
    predicted_numbers = np.argmax(result_vecs, axis=1)
    
    # Calculate accuracy
    correct = (predicted_numbers == input_numbers)
    accuracy = np.mean(correct) * 100
    
    # Get indices of correct and incorrect classifications
    correct_indices = np.where(correct)[0]
    incorrect_indices = np.where(~correct)[0]
    
    logger.info(f'Classification accuracy: {accuracy:.2f}%')
    logger.info(f'Correctly classified: {len(correct_indices)} examples')
    logger.info(f'Incorrectly classified: {len(incorrect_indices)} examples')
    
    return accuracy, correct_indices.tolist(), incorrect_indices.tolist()

def plot_confusion_matrix(input_numbers_file: str, result_vectors_file: str, save_path: str = None):
    """
    Plot and optionally save the confusion matrix.
    
    Args:
        input_numbers_file: Path to the .npy file containing input numbers
        result_vectors_file: Path to the .npy file containing result vectors
        save_path: Optional path to save the plot
    """
    input_numbers = np.load(input_numbers_file)
    result_vecs = np.load(result_vectors_file)
    predicted_numbers = np.argmax(result_vecs, axis=1)
    
    # Create confusion matrix
    confusion = np.zeros((10, 10), dtype=int)
    for true, pred in zip(input_numbers, predicted_numbers):
        confusion[true, pred] += 1
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.imshow(confusion, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    # Add labels
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(np.arange(10))
    plt.yticks(np.arange(10))
    
    # Add numbers to cells
    for i in range(10):
        for j in range(10):
            plt.text(j, i, confusion[i, j],
                    ha="center", va="center")
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f'Confusion matrix saved to {save_path}')
    
    plt.close()

def plot_accuracy_per_digit(input_numbers_file: str, result_vectors_file: str, save_path: str = None):
    """
    Plot and optionally save the accuracy per digit.
    
    Args:
        input_numbers_file: Path to the .npy file containing input numbers
        result_vectors_file: Path to the .npy file containing result vectors
        save_path: Optional path to save the plot
    """
    input_numbers = np.load(input_numbers_file)
    result_vecs = np.load(result_vectors_file)
    predicted_numbers = np.argmax(result_vecs, axis=1)
    
    accuracies = []
    counts = []
    
    # Calculate accuracy for each digit
    for digit in range(10):
        mask = (input_numbers == digit)
        count = np.sum(mask)
        if count > 0:
            accuracy = np.mean(predicted_numbers[mask] == digit) * 100
        else:
            accuracy = 0
        accuracies.append(accuracy)
        counts.append(count)
    
    # Plot
    plt.figure(figsize=(12, 6))
    
    # Bar plot for accuracies
    plt.subplot(1, 2, 1)
    plt.bar(range(10), accuracies)
    plt.title('Accuracy per Digit')
    plt.xlabel('Digit')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 100)
    
    # Bar plot for counts
    plt.subplot(1, 2, 2)
    plt.bar(range(10), counts)
    plt.title('Number of Examples per Digit')
    plt.xlabel('Digit')
    plt.ylabel('Count')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f'Accuracy per digit plot saved to {save_path}')
    
    plt.close()