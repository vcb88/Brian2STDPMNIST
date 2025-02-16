'''
A variety of functions for managing the loading of MNIST data.
'''
import random
import os
import pickle
import logging
import numpy as np
from struct import unpack
from tqdm import tqdm

logger = logging.getLogger(__name__)

def get_labeled_data(picklename, bTrain = True, MNIST_data_path='./mnist'):
    """ Read input-vector (image) and target class (label, 0-9) and return
        it as list of tuples.
        picklename: Path to the output pickle file.
        bTrain: True if training data, else False for test data.
        MNIST_data_path: Directory containing the MNIST files.
    """
    pickle_path = os.path.join(MNIST_data_path, '{}.pickle'.format(os.path.basename(picklename)))
    if os.path.isfile(pickle_path):
        data = pickle.load(open(pickle_path, mode='rb'))
    else:
        # Open the images with gzip in read binary mode
        if bTrain:
            images = open(os.path.join(MNIST_data_path, 'train-images-idx3-ubyte'), mode='rb')
            labels = open(os.path.join(MNIST_data_path, 'train-labels-idx1-ubyte'), mode='rb')
        else:
            images = open(os.path.join(MNIST_data_path, 't10k-images-idx3-ubyte'), mode='rb')
            labels = open(os.path.join(MNIST_data_path, 't10k-labels-idx1-ubyte'), mode='rb')
        # Get metadata for images
        images.read(4)  # skip the magic_number
        number_of_images = unpack('>I', images.read(4))[0]
        rows = unpack('>I', images.read(4))[0]
        cols = unpack('>I', images.read(4))[0]
        # Get metadata for labels
        labels.read(4)  # skip the magic_number
        N = unpack('>I', labels.read(4))[0]

        if number_of_images != N:
            raise ValueError('The number of labels did not match the number of images')
        # Get the data
        x = np.zeros((N, rows, cols), dtype=np.uint8)  # Initialize numpy array
        y = np.zeros((N, 1), dtype=np.uint8)  # Initialize numpy array
        logger.info('Unpacking %s images...', 'training' if bTrain else 'test')
        for i in tqdm(range(N)):
            x[i] = [[unpack('>B', images.read(1))[0] for unused_col in range(cols)]  for unused_row in range(rows) ]
            y[i] = unpack('>B', labels.read(1))[0]

        data = {'x': x, 'y': y, 'rows': rows, 'cols': cols}
        pickle.dump(data, open("{}.pickle".format(picklename), "wb"))
    return data

def get_data_subset(data, size, random_subset=False):
    """Get a subset of the data.
    
    Args:
        data: Dictionary containing 'x' and 'y' arrays
        size: Number of examples to include
        random_subset: If True, randomly select examples, otherwise take first N
        
    Returns:
        Dictionary containing the subset of data
    """
    total_size = len(data['y'])
    size = min(size, total_size)  # Ensure we don't request more than available
    
    if random_subset:
        # Create list of indices and shuffle it
        indices = list(range(total_size))
        random.shuffle(indices)
        selected_indices = indices[:size]
    else:
        # Take first N examples
        selected_indices = list(range(size))
    
    return {
        'x': data['x'][selected_indices],
        'y': data['y'][selected_indices],
        'rows': data['rows'],
        'cols': data['cols']
    }