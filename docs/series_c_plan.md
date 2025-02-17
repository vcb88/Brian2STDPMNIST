# Series C: Pre-synaptic STDP Time Constant Optimization

## Current Configuration
Baseline parameters after B1 series optimization:
- tc_pre_ee: 20ms (target for optimization)
- tc_post_1_ee: 20ms
- tc_post_2_ee: 32ms (optimized)
- nu_ee_pre: 0.00005
- nu_ee_post: 0.01

## Series Objectives
1. Optimize tc_pre_ee for:
   - Better pattern recognition
   - Improved temporal precision
   - Enhanced learning stability

2. Study interaction with:
   - Optimized tc_post_2_ee (32ms)
   - Learning rates
   - Network stability

## Planned Experiments

### C1 (tc_pre_ee = 17ms)
Testing shorter integration window:
- Potentially faster pattern recognition
- More precise temporal coding
- Reduced synaptic trace overlap

### C2 (tc_pre_ee = 24ms)
Testing longer integration window:
- Extended pattern recognition
- Better temporal integration
- Enhanced feature extraction

## Testing Strategy
- Large sample size (600) for reliable results
- Comprehensive metric collection
- Focus on pattern recognition quality

## Success Metrics

### Primary Metrics:
1. Network Activity:
   - Maintain 0% silent neurons
   - Mean spikes ~20
   - >98% in optimal range (11-50 spikes)

2. Learning Quality:
   - Weight distribution stability
   - Temporal response characteristics
   - Pattern recognition efficiency

### Secondary Metrics:
1. Temporal Characteristics:
   - First spike timing
   - Activity window length
   - Response consistency

2. Network Stability:
   - Theta values distribution
   - Weight evolution patterns
   - Learning convergence

## Analysis Focus
1. Pattern Recognition:
   - Temporal precision
   - Feature extraction quality
   - Response reliability

2. Network Dynamics:
   - Interaction with tc_post_2_ee
   - Synaptic plasticity efficiency
   - Learning stability

## Expected Outcomes

### For C1 (17ms):
Potential benefits:
- Faster response times
- More precise temporal coding
- Better pattern discrimination

Potential risks:
- Insufficient integration time
- Missed temporal patterns
- Reduced learning stability

### For C2 (24ms):
Potential benefits:
- Better pattern integration
- Enhanced feature extraction
- More stable learning

Potential risks:
- Temporal smearing
- Slower response times
- Pattern overlap

## Decision Points
1. If both experiments show degradation:
   - Maintain 20ms as optimal
   - Document parameter sensitivity
   - Move to next optimization series

2. If improvement found:
   - Consider additional fine-tuning
   - Test intermediate values
   - Document optimization principles

3. If mixed results:
   - Analyze trade-offs
   - Consider application-specific optimization
   - Document context-dependent effects