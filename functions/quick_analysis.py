import numpy as np
import termplotlib as tpl
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

def get_predictions(result_vecs: np.ndarray) -> np.ndarray:
    """Convert network activations to digit predictions."""
    max_activations = np.argmax(result_vecs, axis=1)
    predicted_numbers = max_activations // 40
    return np.clip(predicted_numbers, 0, 9)

def quick_analyze(input_numbers_file: str, result_vectors_file: str) -> Tuple[float, dict]:
    """
    Perform quick analysis of test results and display them in terminal.
    
    Args:
        input_numbers_file: Path to the .npy file containing input numbers
        result_vectors_file: Path to the .npy file containing result vectors
        
    Returns:
        Tuple containing:
        - accuracy: Overall classification accuracy
        - per_digit_acc: Dictionary with per-digit accuracy
    """
    input_numbers = np.load(input_numbers_file)
    result_vecs = np.load(result_vectors_file)
    predicted_numbers = get_predictions(result_vecs)
    
    # Calculate overall accuracy
    correct = (predicted_numbers == input_numbers)
    accuracy = np.mean(correct) * 100
    
    # Calculate per-digit accuracy
    per_digit_acc = {}
    per_digit_counts = {}
    for digit in range(10):
        mask = (input_numbers == digit)
        count = np.sum(mask)
        if count > 0:
            acc = np.mean(predicted_numbers[mask] == digit) * 100
        else:
            acc = 0
        per_digit_acc[digit] = acc
        per_digit_counts[digit] = count

    # Print results
    print("\n=== Test Results ===")
    print(f"Overall accuracy: {accuracy:.1f}%")
    print(f"Total examples: {len(input_numbers)}")
    print(f"Correct predictions: {np.sum(correct)}")
    
    # Create ASCII bar chart for per-digit accuracy
    print("\nAccuracy by digit:")
    fig = tpl.figure()
    digits = list(range(10))
    accs = [per_digit_acc[d] for d in digits]
    counts = [per_digit_counts[d] for d in digits]
    
    # Accuracy plot
    fig.barh(accs, [str(d) for d in digits])
    fig.show()
    
    # Print detailed per-digit stats
    print("\nDetailed per-digit statistics:")
    for digit in range(10):
        print(f"Digit {digit}: {per_digit_acc[digit]:5.1f}% ({per_digit_counts[digit]} examples)")
    
    # Find best and worst performing digits
    best_digit = max(per_digit_acc.items(), key=lambda x: x[1])
    worst_digit = min(per_digit_acc.items(), key=lambda x: x[1])
    
    print(f"\nBest performing digit: {best_digit[0]} ({best_digit[1]:.1f}%)")
    print(f"Worst performing digit: {worst_digit[0]} ({worst_digit[1]:.1f}%)")
    
    return accuracy, per_digit_acc