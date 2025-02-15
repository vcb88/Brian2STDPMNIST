# Spike-Timing-Dependent Plasticity (STDP)

This document describes the implementation of STDP in the network, including equations, parameters, and key features.

## STDP Model

### Basic Equations

The STDP implementation uses a triplet-based model with the following equations:

```python
# State variables
post2before          : 1            # Previous post-synaptic state
dpre/dt = -pre/(tc_pre_ee)         : 1 (event-driven)
dpost1/dt = -post1/(tc_post_1_ee)  : 1 (event-driven)
dpost2/dt = -post2/(tc_post_2_ee)  : 1 (event-driven)

# Pre-synaptic event
pre = 1.
w = clip(w + nu_ee_pre * post1, 0, wmax_ee)

# Post-synaptic event
post2before = post2
w = clip(w + nu_ee_post * pre * post2before, 0, wmax_ee)
post1 = 1.
post2 = 1.
```

### Parameters

```python
tc_pre_ee = 20*ms       # Pre-synaptic time constant
tc_post_1_ee = 20*ms    # Post-synaptic time constant 1
tc_post_2_ee = 40*ms    # Post-synaptic time constant 2
nu_ee_pre = 0.0001      # Learning rate (pre-synaptic)
nu_ee_post = 0.01       # Learning rate (post-synaptic)
wmax_ee = 1.0           # Maximum weight
```

## Implementation Details

### Weight Updates

1. Pre-synaptic Spike
   - Triggers immediate weight update
   - Update depends on post1 trace
   - pre trace set to 1

2. Post-synaptic Spike
   - Saves current post2 state
   - Updates weight based on pre trace
   - Resets post1 and post2 traces

### Weight Normalization

```python
def normalize_weights():
    for connName in connections:
        if connName[1] == 'e' and connName[3] == 'e':
            connection = get_connection_matrix(connections[connName])
            colSums = np.sum(connection, axis=0)
            colFactors = weight['ee_input']/colSums
            connection *= colFactors
            update_connection_weights(connections[connName], connection)
```

Features:
- Maintains total input strength
- Prevents runaway potentiation
- Applied periodically during training

## Plasticity Windows

### Time Constants
- Pre-synaptic: 20 ms
- Post-synaptic 1: 20 ms
- Post-synaptic 2: 40 ms

### Learning Rates
- Pre-synaptic: 0.0001 (small)
- Post-synaptic: 0.01 (large)
- Asymmetric to favor causality

## Key Features

### 1. Event-Driven Updates
- Updates only on spikes
- Computationally efficient
- Accurate timing preservation

### 2. Triplet Mechanism
- Uses two post-synaptic traces
- More biologically realistic
- Better handles high frequencies

### 3. Weight Bounds
- Hard minimum (0)
- Hard maximum (wmax_ee)
- Prevents unbounded growth

### 4. Normalization
- Maintains total input strength
- Encourages competition
- Stabilizes learning

## Brian2 Implementation

### Synapse Model
```python
model = '''
    w                                      : 1
    post2before                            : 1
    dpre/dt = -pre/(tc_pre_ee)            : 1 (event-driven)
    dpost1/dt = -post1/(tc_post_1_ee)     : 1 (event-driven)
    dpost2/dt = -post2/(tc_post_2_ee)     : 1 (event-driven)
'''
```

### Pre-synaptic Event
```python
on_pre = '''
    ge_post += w
    pre = 1.
    w = clip(w + nu_ee_pre * post1, 0, wmax_ee)
'''
```

### Post-synaptic Event
```python
on_post = '''
    post2before = post2
    w = clip(w + nu_ee_post * pre * post2before, 0, wmax_ee)
    post1 = 1.
    post2 = 1.
'''
```

## Usage in Network

### Connection Creation
```python
connections[name] = Synapses(
    source_group,
    target_group,
    model=model,
    on_pre=on_pre,
    on_post=on_post
)
```

### Weight Initialization
```python
connections[name].w = 'rand() * 0.3'
```

### Monitoring
```python
StateMonitor(
    connections[name],
    ['w', 'pre', 'post1', 'post2'],
    record=True
)
```

## Common Issues and Solutions

### 1. Weight Saturation
- Problem: Weights hit bounds too quickly
- Solution: Adjust learning rates
- Monitor: Check weight distributions

### 2. Learning Stability
- Problem: Unstable weight changes
- Solution: Use normalization
- Monitor: Track total weights

### 3. Timing Precision
- Problem: Missed updates
- Solution: Use event-driven
- Monitor: Spike time differences