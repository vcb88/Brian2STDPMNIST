# Experiment B1.3: Further STDP Time Constant Optimization

## Background
Previous experiments showed clear trend:
- B1 (60ms): Poor performance, 7.5% silent
- B1.1 (45ms): Poor performance, 6.2% silent
- B1.2 (35ms): Excellent performance, exceeded baseline

## Current Configuration
- nu_ee_pre: 0.00005 (fixed from A2)
- nu_ee_post: 0.01
- tc_pre_ee: 20 ms
- tc_post_1_ee: 20 ms
- tc_post_2_ee: 35 ms → 32 ms

## Proposed Change
Further reduction in post-synaptic time constant:
```python
tc_post_2_ee = 32*b2.ms  # decreased from 35ms
```

## Rationale
1. Performance trend suggests potential optimization:
   - Each reduction improved performance
   - 35ms showed better results than baseline
   - 32ms might find better temporal precision

2. Hypothesis:
   - Shorter integration window may:
     * Further improve temporal precision
     * Enhance pattern recognition
     * Maintain network stability

## Success Criteria
1. Primary metrics (compared to B1.2):
   - Maintain 0% silent neurons
   - Mean spikes ≥ 20
   - ≥97% neurons in optimal range

2. Secondary metrics:
   - Maintain or improve min spikes (>8)
   - Stable weight distribution
   - Efficient temporal characteristics

## Testing Strategy
1. Full-scale validation:
   - Sample size: 600
   - Comprehensive metrics collection
   - Detailed temporal analysis

2. Focus areas:
   - Activity distribution
   - Temporal precision
   - Learning stability

## Monitoring Focus
1. Critical indicators:
   - Early activity onset
   - Spike distribution
   - Weight evolution

2. Stability metrics:
   - Theta values
   - Weight statistics
   - Temporal patterns

## Next Steps
1. If improved:
   - Consider testing 30ms
   - Document optimization principle
   - Update network guidelines

2. If degraded:
   - Lock 35ms as optimal
   - Document parameter sensitivity
   - Move to other optimizations

3. If similar:
   - Accept 35ms as simpler value
   - Document optimization bounds
   - Proceed with other parameters