# Neuron Model

This document describes the neuron models used in the network, their equations, parameters, and implementation details.

## Excitatory Neurons

### Basic Model
The excitatory neurons are implemented as adaptive Leaky Integrate-and-Fire (LIF) neurons with the following dynamics:

```python
dv/dt = ((v_rest_e - v) + (I_synE + I_synI) / nS) / (100*ms)  : volt (unless refractory)
I_synE = ge * nS * -v                                          : amp
I_synI = gi * nS * (-100.*mV-v)                               : amp
dge/dt = -ge/(1.0*ms)                                         : 1
dgi/dt = -gi/(2.0*ms)                                         : 1
dtheta/dt = -theta / (tc_theta)                               : volt
dtimer/dt = 0.1                                               : second
```

### Parameters
- Rest potential (v_rest_e): -65 mV
- Reset potential (v_reset_e): -65 mV
- Threshold (v_thresh_e): -52 mV
- Refractory period (refrac_e): 5 ms
- Membrane time constant (τm): 100 ms
- Excitatory synaptic time constant (τe): 1 ms
- Inhibitory synaptic time constant (τi): 2 ms
- Threshold adaptation time constant (tc_theta): 1e7 ms
- Threshold increment (theta_plus_e): 0.05 mV

### Adaptive Threshold
The neuron implements an adaptive threshold mechanism:
1. Threshold increases by theta_plus_e after each spike
2. Slowly decays back to base threshold
3. Helps prevent single-neuron dominance
4. Encourages distributed learning

## Inhibitory Neurons

### Basic Model
Inhibitory neurons use a simpler LIF model:

```python
dv/dt = ((v_rest_i - v) + (I_synE + I_synI) / nS) / (10*ms)  : volt (unless refractory)
I_synE = ge * nS * -v                                         : amp
I_synI = gi * nS * (-85.*mV-v)                               : amp
dge/dt = -ge/(1.0*ms)                                         : 1
dgi/dt = -gi/(2.0*ms)                                         : 1
```

### Parameters
- Rest potential (v_rest_i): -60 mV
- Reset potential (v_reset_i): -45 mV
- Threshold (v_thresh_i): -40 mV
- Refractory period (refrac_i): 2 ms
- Membrane time constant (τm): 10 ms
- Excitatory synaptic time constant (τe): 1 ms
- Inhibitory synaptic time constant (τi): 2 ms

## Input Layer

### Poisson Neurons
Input layer neurons are implemented as Poisson spike generators:
- Firing rate proportional to pixel intensity
- No internal dynamics
- Rate = pixel_intensity * input_intensity

## Key Implementation Details

### Voltage Dynamics
1. Membrane Potential
   - Exponential decay to rest
   - Synaptic current integration
   - Reset after spike

2. Synaptic Currents
   - Instantaneous conductance changes
   - Exponential conductance decay
   - Separate E and I conductances

### Spike Generation
1. Excitatory Neurons
   - Adaptive threshold
   - Refractory period
   - Voltage reset + threshold increase

2. Inhibitory Neurons
   - Fixed threshold
   - Simple reset
   - Shorter refractory period

### Numerical Implementation
1. Integration Method
   - Euler method
   - Time step: 0.1 ms
   - State variables updated each step

2. Units
   - Voltage: millivolts (mV)
   - Current: amperes (A)
   - Time: milliseconds (ms)
   - Conductance: dimensionless (scaled by nS)

## Brian2 Implementation Notes

### Equations Setup
```python
neuron_eqs_e = '''
    dv/dt = ((v_rest_e - v) + (I_synE+I_synI) / nS) / (100*ms)  : volt
    I_synE = ge * nS * -v                                        : amp
    I_synI = gi * nS * (-100.*mV-v)                             : amp
    dge/dt = -ge/(1.0*ms)                                       : 1
    dgi/dt = -gi/(2.0*ms)                                       : 1
    dtheta/dt = -theta / (tc_theta)                             : volt
    dtimer/dt = 0.1                                             : second
'''
```

### Neuron Group Creation
```python
neuron_groups['e'] = b2.NeuronGroup(
    n_e * len(population_names),
    neuron_eqs_e,
    threshold='v>(theta - offset + v_thresh_e) and (timer>refrac_e)',
    refractory='refrac_e',
    reset='v = v_reset_e; theta += theta_plus_e; timer = 0*ms',
    method='euler'
)
```

### State Variable Access
- Direct access to voltage: `neuron_groups['e'].v`
- Threshold monitoring: `neuron_groups['e'].theta`
- Conductance updates: `neuron_groups['e'].ge += w`