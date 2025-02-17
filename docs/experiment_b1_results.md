# Experiment B1: Extended Post-synaptic Time Window

## Configuration
- tc_post_2_ee increased to 60ms (from 40ms baseline)
- Other parameters unchanged:
  * nu_ee_pre: 0.00005
  * nu_ee_post: 0.01
  * tc_pre_ee: 20ms
  * tc_post_1_ee: 20ms

## Results Summary

### Network Activity
- Silent neurons: 30/400 (7.5%) ← Critical degradation from 0%
- Mean spikes: 7.18 (down from 20.21)
- Max spikes: 30 (down from 37)
- Min spikes: 0 (down from 7)

### Activity Distribution
- Low activity (1-10 spikes): 63.0% (up from 0.8%)
- Optimal range (11-50 spikes): 17.5% (down from 97.0%)
- High activity (>50 spikes): 0% (unchanged)

### Weight Statistics (XeAe)
- Minimum: 0.001837 (stable)
- Maximum: 0.239628 (slight increase)
- Mean: 0.099495 (stable)
- Standard deviation: 0.055840 (slight increase)

### Temporal Characteristics
- Earliest spike: 84.1 ms (unchanged)
- Latest spike: 106849.9 ms (down from 335311.1 ms)
- Mean first spike: 33711.8 ms (improved from 37144.6 ms)
- Mean last spike: 77714.7 ms (down from 303866.2 ms)

## Analysis

### Critical Issues
1. Network Silencing:
   - 7.5% silent neurons is unacceptable
   - Indicates severe suppression of activity
   - Complete loss of functionality in these neurons

2. Activity Collapse:
   - Mean activity dropped by 64.5%
   - Most neurons (63.0%) in low activity range
   - Only 17.5% neurons maintain optimal activity

3. Temporal Window Compression:
   - Activity window shortened by ~68%
   - May indicate premature termination of processing

### Positive Aspects
1. Weight Stability:
   - Mean weights remained stable
   - No pathological weight growth

2. Early Response:
   - Maintained quick initial response
   - Improved mean first spike timing

## Conclusions

1. Performance Impact:
   - Severe degradation of network functionality
   - Loss of optimal activity distribution
   - Critical increase in silent neurons

2. Mechanistic Analysis:
   - Extended tc_post_2_ee (60ms) appears to:
     * Over-suppress neuron activity
     * Create excessive inhibitory effects
     * Disrupt balance of excitation/inhibition

3. Learning Dynamics:
   - Despite stable weights, learning is compromised
   - Temporal integration window may be too long
   - Network unable to maintain sustained activity

## Recommendations

1. Immediate Actions:
   - Abandon 60ms setting
   - Test more conservative adjustment (45ms in B1.1)
   - Prepare fallback to 35ms if needed (B1.2)

2. Future Investigations:
   - Consider interaction with other time constants
   - Investigate balance with inhibitory mechanisms
   - Study temporal integration patterns

## Next Steps
1. Proceed with Experiment B1.1:
   - Test tc_post_2_ee = 45ms
   - Focus on activity restoration
   - Monitor silent neuron count

2. Prepare contingency:
   - If 45ms shows improvement but not optimal:
     * Consider testing 42-43ms range
   - If 45ms shows degradation:
     * Proceed with 35ms test
     * Consider returning to baseline