# "Unsupervised Learning of Digit recognition using STDP" in Brian2

This is a modified version of the source code for the paper:

‘Unsupervised Learning of Digit Recognition Using Spike-Timing-Dependent Plasticity’, Diehl and Cook, (2015).

Original code: Peter U. Diehl (https://github.com/peter-u-diehl/stdp-mnist)
Updated for Brian2: zxzhijia (https://github.com/zxzhijia/Brian2STDPMNIST)
Updated for Python3: sdpenguin

## Prerequisites

1. Brian2 
2. MNIST datasets, which can be downloaded from http://yann.lecun.com/exdb/mnist/. 
   * The data set includes four gz files. Extract them after you downloaded them.

## Testing with pretrained weights:

1. Run the main file "Diehl&Cook_spiking_MNIST_Brian2.py". It might take hours depending on your computer 
2. After the previous step is finished, evaluate it by running "Diehl&Cook_MNIST_evaluation.py".

## Training a new network:

1. Modify the main file "Diehl&Cook_spiking_MNIST_Brian2.py" by changing line 214 to "test_mode=False" and run the code. 
2. The trained weights will be stored in the folder "weights", which can be used to test the performance.
3. In order to test your training, change line 214 back to "test_mode=True". 
4. Run the "Diehl&Cook_spiking_MNIST_Brian2.py" code to get the results. 
