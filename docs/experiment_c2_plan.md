# Experiment C2: Extended Pre-synaptic STDP Time Constant

## Configuration
- tc_pre_ee: 24ms (increased from 20ms baseline)
- Sample size: 600 (large-scale validation)
- Other parameters:
  * tc_post_1_ee: 20ms
  * tc_post_2_ee: 32ms (optimized)
  * nu_ee_pre: 0.00005
  * nu_ee_post: 0.01

## Rationale
1. Testing longer integration window:
   - Enhanced pattern integration
   - Better temporal feature extraction
   - More robust learning

2. Interaction with optimized tc_post_2_ee:
   - Modified pre/post synaptic balance
   - Extended temporal integration
   - Improved pattern recognition

## Success Criteria
### Primary Metrics (vs Baseline)
- Maintain 0% silent neurons
- Mean spikes ~20
- >98% neurons in optimal range
- Stable weight distribution

### Secondary Metrics
- Pattern recognition quality
- Temporal integration efficiency
- Learning stability metrics

## Monitoring Focus
1. Learning Quality:
   - Pattern extraction
   - Feature integration
   - Response stability

2. Network Dynamics:
   - Weight evolution
   - Activity patterns
   - Temporal characteristics

## Risk Assessment
### Potential Benefits
- Better pattern integration
- More stable learning
- Enhanced feature extraction

### Potential Risks
- Temporal smearing
- Slower response
- Reduced precision

## Decision Points
1. If improved:
   - Consider testing 26-28ms
   - Document improvements
   - Analyze mechanisms

2. If degraded:
   - Return to baseline
   - Document limitations
   - Complete series C