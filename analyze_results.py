import os
import logging
from functions.results import analyze_results, plot_confusion_matrix, plot_accuracy_per_digit

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # Create results directory if it doesn't exist
    if not os.path.exists('results'):
        os.makedirs('results')
        logger.info("Created results directory")

    # Analyze the results
    accuracy, correct_indices, incorrect_indices = analyze_results(
        'activity/inputNumbers10000.npy',
        'activity/resultPopVecs10000.npy'
    )

    # Plot confusion matrix
    plot_confusion_matrix(
        'activity/inputNumbers10000.npy',
        'activity/resultPopVecs10000.npy',
        'results/confusion_matrix.png'
    )

    # Plot accuracy per digit
    plot_accuracy_per_digit(
        'activity/inputNumbers10000.npy',
        'activity/resultPopVecs10000.npy',
        'results/accuracy_per_digit.png'
    )

if __name__ == "__main__":
    main()