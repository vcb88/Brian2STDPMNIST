# Network Architecture

The network implements a spiking neural network (SNN) architecture for unsupervised learning of digit recognition. The architecture is based on the model described in Diehl & Cook (2015).

## Overview

![Network Architecture](../images/network_architecture.png)

The network consists of three main layers:
1. Input Layer (784 neurons)
2. Excitatory Layer (400 neurons)
3. Inhibitory Layer (400 neurons)

## Layer Details

### Input Layer
- Size: 784 neurons (28x28 MNIST image)
- Type: Poisson spike generator
- Function: Converts pixel intensities to spike rates
- Connectivity: Projects to excitatory layer

### Excitatory Layer
- Size: 400 neurons (20x20 array)
- Type: Leaky integrate-and-fire (LIF) neurons
- Function: Learning and classification
- Plasticity: STDP on input-excitatory synapses
- Connections:
  - Receives input from input layer (plastic)
  - Receives inhibition from inhibitory layer
  - Projects to inhibitory layer

### Inhibitory Layer
- Size: 400 neurons
- Type: LIF neurons
- Function: Lateral inhibition
- Connections:
  - Receives input from excitatory layer
  - Projects back to excitatory layer

## Connectivity Patterns

1. Input → Excitatory (XeAe)
   - All-to-all connectivity
   - Plastic synapses (STDP)
   - Initial weights: Random uniform

2. Excitatory → Inhibitory (AeAi)
   - One-to-one connectivity
   - Fixed synapses
   - Strong weights

3. Inhibitory → Excitatory (AiAe)
   - All-to-all connectivity
   - Fixed synapses
   - Implements WTA mechanism

## Winner-Take-All (WTA) Circuit

The network implements a soft Winner-Take-All mechanism through:
1. Strong excitatory → inhibitory connections
2. Widespread inhibitory feedback
3. Adaptive thresholds in excitatory neurons

This causes excitatory neurons to:
- Compete for input
- Specialize on different input patterns
- Form a sparse representation

## Learning Mechanism

1. Input patterns cause spike timing patterns
2. STDP strengthens synapses between coactive neurons
3. WTA ensures different neurons learn different patterns
4. Adaptive thresholds prevent single-neuron dominance

## Dimensionality

All critical dimensions in the network:
- Input: 784 (28x28)
- Excitatory neurons: 400 (20x20)
- Inhibitory neurons: 400
- Time step: 0.1 ms
- Simulation time per example: 350 ms

## Key Features

1. Unsupervised Learning
   - No labels needed during training
   - Self-organizing through STDP and WTA

2. Sparse Coding
   - WTA ensures sparse activity
   - Each digit activates few neurons

3. Competitive Learning
   - Neurons compete through inhibition
   - Leads to specialization

4. Local Learning Rule
   - STDP is purely local
   - Biologically plausible