# "Unsupervised Learning of Digit recognition using STDP" in Brian2

This is a modified version of the source code for the paper:

‘Unsupervised Learning of Digit Recognition Using Spike-Timing-Dependent Plasticity’, Diehl and Cook, (2015).

Original code: Peter U. Diehl (https://github.com/peter-u-diehl/stdp-mnist)

Updated for Brian2: zxzhijia (https://github.com/zxzhijia/Brian2STDPMNIST)

Updated for Python3: sdpenguin

## Prerequisites

You have two options to run this project:

### Option 1: Local Installation

1. Python 3.8 or higher
2. Required Python packages:
   * Brian2 (2.5.0.1 or higher)
   * NumPy (1.26.0 or higher)
   * Matplotlib (3.8.0 or higher)
   * tqdm (for progress visualization)
   * typing_extensions (for Python <3.9)

3. MNIST datasets, which can be downloaded from http://yann.lecun.com/exdb/mnist/. 
   * The data set includes four gz files. Extract them after you downloaded them.

You can install the required packages using pip:
```bash
pip install -r requirements.txt
```

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

### Performance Optimization

The simulation supports several performance optimization features:

1. **Default Runtime with Cython** (default):
   ```bash
   make container-train-custom EPOCHS=3 TRAIN_SIZE=60000
   ```
   - Uses Cython-optimized code
   - Good balance of setup time and performance
   - Recommended for quick experiments

2. **Multi-threaded Runtime**:
   ```bash
   make container-train-custom EPOCHS=3 TRAIN_SIZE=60000 NUM_THREADS=8
   ```
   - Uses OpenMP for parallel processing
   - Better performance on multi-core systems
   - Recommended for longer training sessions

3. **C++ Standalone with OpenMP** (fastest):
   ```bash
   make container-train-custom EPOCHS=3 TRAIN_SIZE=60000 NUM_THREADS=8 DEVICE=cpp_standalone
   ```
   - Compiles simulation to optimized C++ code
   - Uses OpenMP for maximum parallelization
   - Longer initial compilation time
   - Best performance for long simulations

#### Performance Parameters

- `EPOCHS`: Number of training epochs (default: 3)
- `TRAIN_SIZE`: Number of training examples per epoch (default: 60000, max: 60000)
- `NUM_THREADS`: Number of OpenMP threads for parallel processing (default: 8)
- `DEVICE`: Simulation device (`runtime` or `cpp_standalone`, default: runtime)

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
