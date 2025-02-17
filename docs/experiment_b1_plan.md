# Experiment B1: STDP Time Constants Optimization

## Current Optimal Configuration
- nu_ee_pre: 0.00005 (fixed from A2)
- nu_ee_post: 0.01
- tc_pre_ee: 20 ms
- tc_post_1_ee: 20 ms
- tc_post_2_ee: 40 ms

## Proposed Changes
Change tc_post_2_ee from 40ms to 60ms while maintaining other parameters:
```python
tc_pre_ee = 20*b2.ms        # unchanged
tc_post_1_ee = 20*b2.ms     # unchanged
tc_post_2_ee = 60*b2.ms     # increased from 40ms
```

## Rationale
1. Longer tc_post_2_ee will:
   - Extend the influence of post-synaptic spikes
   - Potentially improve temporal integration
   - Allow for better long-term synaptic modifications

2. Maintaining tc_pre_ee and tc_post_1_ee ensures:
   - Stability of immediate spike interactions
   - Consistency with proven basic STDP mechanics
   - Controlled experimental conditions

## Expected Effects
1. Learning Dynamics:
   - More gradual weight changes
   - Better temporal integration
   - Potentially improved pattern recognition

2. Network Activity:
   - Possibly reduced temporal variance
   - More consistent neuron activation
   - Better distributed network activity

3. Synaptic Strength:
   - More stable weight evolution
   - Better capture of temporal patterns
   - Potentially more robust weight distribution

## Success Criteria
1. Maintain or improve:
   - Current 0% inactive neurons
   - 97%+ neurons in optimal activity range
   - Weight stability characteristics

2. Potential improvements in:
   - Temporal consistency
   - Learning stability
   - Activity distribution

## Testing Strategy
1. Initial test with 200 samples to:
   - Verify basic functionality
   - Check for any immediate issues
   - Assess general trends

2. If promising, test with 600 samples to:
   - Confirm improvements
   - Validate stability
   - Evaluate long-term effects

## Monitoring Focus
1. Temporal characteristics:
   - Spike timing patterns
   - Activity window changes
   - First spike timing

2. Weight dynamics:
   - Distribution changes
   - Stability metrics
   - Learning progression

3. Activity patterns:
   - Neuron participation
   - Spike count distribution
   - Temporal consistency