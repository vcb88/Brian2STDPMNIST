#!/usr/bin/env python3
"""
Script to download and prepare MNIST dataset for the STDP learning project.
"""

import os
import gzip
import urllib.request
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# MNIST dataset URLs
MNIST_URLS = {
    'train_images': 'https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz',
    'train_labels': 'https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz',
    'test_images': 'https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz',
    'test_labels': 'https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz'
}

def ensure_dir(directory):
    """Ensure that a directory exists."""
    Path(directory).mkdir(parents=True, exist_ok=True)

def download_file(url, filename):
    """Download a file from URL and save it to filename."""
    try:
        logger.info(f"Downloading {url} to {filename}")
        urllib.request.urlretrieve(url, filename)
        return True
    except Exception as e:
        logger.error(f"Error downloading {url}: {str(e)}")
        return False

def extract_gz(gz_path, output_path):
    """Extract a .gz file to the specified output path."""
    try:
        logger.info(f"Extracting {gz_path} to {output_path}")
        with gzip.open(gz_path, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                f_out.write(f_in.read())
        return True
    except Exception as e:
        logger.error(f"Error extracting {gz_path}: {str(e)}")
        return False

def prepare_mnist(data_dir='mnist'):
    """Download and prepare MNIST dataset."""
    ensure_dir(data_dir)
    
    # Download and extract each file
    for name, url in MNIST_URLS.items():
        base_name = os.path.basename(url)
        gz_path = os.path.join(data_dir, base_name)
        final_path = os.path.join(data_dir, os.path.splitext(base_name)[0])
        
        # Download if not exists
        if not os.path.exists(gz_path):
            if not download_file(url, gz_path):
                continue
        
        # Extract if not exists
        if not os.path.exists(final_path):
            if not extract_gz(gz_path, final_path):
                continue
            
        # Clean up gz file
        try:
            os.remove(gz_path)
        except Exception as e:
            logger.warning(f"Could not remove {gz_path}: {str(e)}")
    
    logger.info("MNIST dataset preparation completed!")

def verify_files(data_dir='mnist'):
    """Verify that all required files exist and have correct sizes."""
    required_files = {
        'train-images-idx3-ubyte': 47040016,
        'train-labels-idx1-ubyte': 60008,
        't10k-images-idx3-ubyte': 7840016,
        't10k-labels-idx1-ubyte': 10008
    }
    
    all_valid = True
    for filename, expected_size in required_files.items():
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            logger.error(f"Missing file: {filepath}")
            all_valid = False
            continue
            
        actual_size = os.path.getsize(filepath)
        if actual_size != expected_size:
            logger.error(f"Invalid file size for {filepath}. Expected: {expected_size}, Got: {actual_size}")
            all_valid = False
    
    return all_valid

def show_status(data_dir='mnist'):
    """Show the status of dataset files."""
    required_files = {
        'train-images-idx3-ubyte': 47040016,
        'train-labels-idx1-ubyte': 60008,
        't10k-images-idx3-ubyte': 7840016,
        't10k-labels-idx1-ubyte': 10008
    }
    
    logger.info("Dataset Status:")
    all_valid = True
    for filename, expected_size in required_files.items():
        filepath = os.path.join(data_dir, filename)
        status = "✓" if os.path.exists(filepath) else "✗"
        size_info = ""
        if os.path.exists(filepath):
            actual_size = os.path.getsize(filepath)
            if actual_size == expected_size:
                size_info = f"(Size: {actual_size} bytes - Valid)"
            else:
                size_info = f"(Size: {actual_size} bytes - Expected: {expected_size} bytes)"
                status = "!"
                all_valid = False
        logger.info(f"{status} {filename} {size_info}")
    
    return all_valid

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='MNIST Dataset preparation script')
    parser.add_argument('--download-only', action='store_true', help='Only download the dataset files')
    parser.add_argument('--prepare-only', action='store_true', help='Only prepare (extract) the dataset files')
    parser.add_argument('--status', action='store_true', help='Show dataset status')
    parser.add_argument('--data-dir', default='mnist', help='Directory for dataset files')
    
    args = parser.parse_args()
    
    if args.status:
        show_status(args.data_dir)
    elif args.download_only:
        logger.info("Downloading dataset files...")
        ensure_dir(args.data_dir)
        for name, url in MNIST_URLS.items():
            base_name = os.path.basename(url)
            gz_path = os.path.join(args.data_dir, base_name)
            if not os.path.exists(gz_path):
                download_file(url, gz_path)
    elif args.prepare_only:
        logger.info("Preparing dataset files...")
        ensure_dir(args.data_dir)
        for name, url in MNIST_URLS.items():
            base_name = os.path.basename(url)
            gz_path = os.path.join(args.data_dir, base_name)
            final_path = os.path.join(args.data_dir, os.path.splitext(base_name)[0])
            if os.path.exists(gz_path) and not os.path.exists(final_path):
                extract_gz(gz_path, final_path)
    else:
        prepare_mnist(args.data_dir)
        if verify_files(args.data_dir):
            logger.info("Dataset verification successful!")
        else:
            logger.error("Dataset verification failed!")