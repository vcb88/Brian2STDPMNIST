# Experiment C1: Reduced Pre-synaptic STDP Time Constant

## Configuration
- tc_pre_ee: 17ms (decreased from 20ms baseline)
- Sample size: 600 (large-scale validation)
- Other parameters:
  * tc_post_1_ee: 20ms
  * tc_post_2_ee: 32ms (optimized)
  * nu_ee_pre: 0.00005
  * nu_ee_post: 0.01

## Rationale
1. Testing shorter integration window:
   - Potentially improved temporal precision
   - Faster synaptic response
   - More specific pattern detection

2. Interaction with optimized tc_post_2_ee:
   - New balance of pre/post synaptic timing
   - Modified learning dynamics
   - Enhanced temporal coding

## Success Criteria
### Primary Metrics (vs Baseline)
- Maintain 0% silent neurons
- Mean spikes ~20
- >98% neurons in optimal range
- Stable weight distribution

### Secondary Metrics
- First spike timing
- Activity window characteristics
- Learning stability indicators

## Monitoring Focus
1. Pattern Recognition:
   - Temporal precision
   - Response speed
   - Feature detection quality

2. Network Stability:
   - Weight evolution
   - Theta dynamics
   - Activity distribution

## Risk Assessment
### Potential Benefits
- Faster pattern recognition
- More precise temporal coding
- Better discrimination ability

### Potential Risks
- Too short integration window
- Missed temporal patterns
- Learning instability

## Decision Points
1. If improved:
   - Consider testing 15-16ms
   - Document benefits
   - Analyze mechanism

2. If degraded:
   - Return to baseline
   - Document limitations
   - Proceed to C2