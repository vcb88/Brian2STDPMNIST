# Experiment B1.2: Reduced STDP Time Constant

## Background
Following experiments:
- A2: Baseline (tc_post_2_ee = 40ms) - optimal performance
- B1: Increased to 60ms - significant degradation
- B1.1: Testing 45ms - [pending results]

## Current Configuration
- nu_ee_pre: 0.00005 (fixed from A2)
- nu_ee_post: 0.01
- tc_pre_ee: 20 ms
- tc_post_1_ee: 20 ms
- tc_post_2_ee: 40 ms → 35 ms

## Proposed Change
Reduction in post-synaptic time constant:
```python
tc_post_2_ee = 35*b2.ms  # decrease from 40ms
```

## Rationale
1. Testing sensitivity in opposite direction:
   - Previous increases showed negative impact
   - Shorter integration window might:
     * Reduce temporal interference
     * Allow faster network response
     * Improve temporal precision

2. Maintaining proportions:
   - tc_post_2_ee still > tc_post_1_ee
   - Conservative change (-5ms)
   - Preserves basic STDP mechanics

## Success Criteria
1. Primary metrics (compared to A2 baseline):
   - Zero silent neurons
   - Mean spikes 15-25
   - >90% neurons in optimal range

2. Secondary metrics:
   - Weight stability preserved
   - Temporal characteristics maintained/improved
   - No overactive neurons

## Testing Strategy
1. Initial test (200 samples):
   - Quick validation
   - Activity distribution check
   - Basic performance metrics

2. If promising:
   - Full test (600 samples)
   - Comprehensive analysis
   - A2 baseline comparison

## Monitoring Focus
1. Network activity:
   - Spike distribution
   - Temporal patterns
   - Activity onset timing

2. Learning dynamics:
   - Weight evolution
   - Synaptic stability
   - Response characteristics

## Next Steps
1. If improved performance:
   - Document enhancements
   - Consider testing 32-33ms range

2. If similar to baseline:
   - Document stability
   - Consider other optimization paths

3. If degraded performance:
   - Return to 40ms baseline
   - Document timing sensitivity
   - Explore other parameters