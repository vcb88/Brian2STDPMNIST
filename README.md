# "Unsupervised Learning of Digit recognition using STDP" in Brian2

This is a modified version of the source code for the paper:

‘Unsupervised Learning of Digit Recognition Using Spike-Timing-Dependent Plasticity’, Diehl and Cook, (2015).

Original code: Peter U. Diehl (https://github.com/peter-u-diehl/stdp-mnist)

Updated for Brian2: zxzhijia (https://github.com/zxzhijia/Brian2STDPMNIST)

Updated for Python3: sdpenguin

## Prerequisites

You have two options to run this project:

### Option 1: Local Installation

1. Brian2 
2. MNIST datasets, which can be downloaded from http://yann.lecun.com/exdb/mnist/. 
   * The data set includes four gz files. Extract them after you downloaded them.

### Option 2: Docker Environment (Recommended)

1. Docker and Docker Compose installed on your system
2. Run `make docker-build` to build the container
3. Run `make docker-run` to start the environment

## Running Tests

### Quick Start with Docker

1. Start with a clean environment:
   ```bash
   make reset-and-test
   ```
   This will reset the environment, prepare the dataset, and run full tests.

### Available Test Commands

The project provides several options for running tests:

1. Full Testing:
   ```bash
   make container-test
   ```
   Runs tests on the complete test dataset.

2. Quick Testing (1000 examples):
   ```bash
   make container-test-quick
   ```
   Runs tests on a fixed subset of 1000 examples.

3. Random Testing (1000 examples):
   ```bash
   make container-test-random
   ```
   Runs tests on a random subset of 1000 examples.

4. Custom Size Testing:
   ```bash
   make container-test-size SIZE=500
   ```
   Runs tests on the first N examples (specified by SIZE).

5. Random Size Testing:
   ```bash
   make container-test-size-random SIZE=500
   ```
   Runs tests on a random subset of N examples (specified by SIZE).

6. Fully Custom Testing:
   ```bash
   make container-test-custom TEST_ARGS="--test-size 500 --random-subset --verbose"
   ```
   Allows full customization of test parameters.

### Test Parameters

- `SIZE`: Number of examples to test (required for -size commands)
- `TEST_ARGS`: Custom arguments for test configuration (for container-test-custom)
  Available options:
  - `--test-size N`: Number of examples to test
  - `--random-subset`: Use random subset of examples
  - `--verbose`: Enable verbose output
  - `--data-dir PATH`: Custom data directory path

### Dataset Management

1. Download MNIST dataset:
   ```bash
   make dataset-download
   ```

2. Prepare dataset:
   ```bash
   make dataset-prepare
   ```

3. Check dataset status:
   ```bash
   make dataset-status
   ```

4. Clean dataset:
   ```bash
   make dataset-clean
   ```

## Testing with pretrained weights:

1. Run the main file "diehl_cook_spiking_mnist_brian2.py". It might take hours depending on your computer 
2. After the previous step is finished, evaluate it by running "diehl_cook_mnist_evaluation.py".

## Training a new network:

1. Modify the main file "diehl_cook_spiking_mnist_brian2.py" by changing line 214 to "test_mode=False" and run the code. 
2. The trained weights will be stored in the folder "weights", which can be used to test the performance.
3. In order to test your training, change line 214 back to "test_mode=True". 
4. Run the "diehl_cook_spiking_mnist_brian2.py" code to get the results. 
