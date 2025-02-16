# Parameter Optimization Experiments

## Baseline (Original Parameters)
Date: 2025-02-16

### Parameters
```python
# Temporal parameters
single_example_time = 0.35 * second
resting_time = 0.15 * second

# Neuron parameters
v_rest_e = -65. * mV
v_reset_e = -65. * mV
v_thresh_e = -52. * mV
refrac_e = 5. * ms

# Threshold adaptation
tc_theta = 1e7 * ms
theta_plus_e = 0.05 * mV
theta_init = 20.0 * mV
```

### Results
- Silent neurons: 35.5% (142/400)
- Spike statistics:
  * Mean spikes: 7.45
  * Max spikes: 81
  * Distribution:
    - 1-10 spikes: 30.8%
    - 11-50 spikes: 21.8%
    - 51-100 spikes: 0.8%
    - >100 spikes: 0.0%
- Timing:
  * Mean first spike: 34.38s
  * Mean last spike: 78.68s
- Weight statistics:
  * Min: 0.001728
  * Max: 0.241996
  * Mean: 0.099496
  * Std: 0.056040
- Theta statistics:
  * Silent neurons: 19.786mV (std: 0.0)
  * Active neurons: 20.361mV (std: 0.623)

### Notes
- Baseline performance after reverting all experimental changes
- Acceptable number of silent neurons (<40%)
- Good activity distribution without over-active neurons
- Room for improvement in reducing silent neurons and earlier activation

## Experiment Plan

1. Threshold Parameters (Priority High):
   - Vary v_thresh_e: -53mV to -51mV
   - Vary theta_plus_e: 0.03mV to 0.07mV
   - Test different initial theta values

2. Temporal Parameters (Priority Medium):
   - Vary resting_time: 0.13s to 0.17s
   - Vary refrac_e: 4ms to 6ms

3. Reset Parameters (Priority Low):
   - Vary v_reset_e: -66mV to -64mV
   - Test different rest potential values

## Guidelines
1. Change only one parameter at a time
2. Document:
   - Parameter changed and rationale
   - Full test results
   - Observations and hypotheses
   - Decision for next step

3. Success Metrics:
   - Primary: % of silent neurons
   - Secondary:
     * Mean spike count
     * Activity distribution
     * Timing of first/last spikes
     * Weight distribution
     * Theta adaptation

4. Failure Conditions:
   - Silent neurons > 40%
   - Extreme activity (>100 spikes per neuron)
   - Very late first spikes (>40s)
   - Unstable learning (rapid weight changes)
   - Loss of neuron specialization

## Experiment #1: Lower Activation Threshold
Date: TBD

### Hypothesis
Lowering v_thresh_e slightly (from -52mV to -52.5mV) might:
- Reduce number of silent neurons by making activation easier
- Lead to earlier first spikes
- Maintain stable learning due to small change

### Parameter Change
```python
v_thresh_e = -52.5 * mV  # Changed from -52.0 * mV
```

Would you like to proceed with this first experiment?