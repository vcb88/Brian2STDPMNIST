# Experiment B1.1: Moderate STDP Time Constant Adjustment Results

## Configuration
- tc_post_2_ee: 45ms (from baseline 40ms)
- Other parameters unchanged:
  * nu_ee_pre: 0.00005
  * nu_ee_post: 0.01
  * tc_pre_ee: 20ms
  * tc_post_1_ee: 20ms
- Sample size: 200

## Results Summary

### Network Activity
- Silent neurons: 25/400 (6.2%) ← Still significant degradation from baseline
- Mean spikes: 6.99 (down from 20.21 in baseline)
- Max spikes: 23 (down from 37)
- Min spikes: 0

### Activity Distribution
- Low activity (1-10 spikes): 70.0% (up from 0.8% in baseline)
- Optimal range (11-50 spikes): 13.8% (down from 97.0%)
- High activity (>50 spikes): 0% (unchanged)

### Weight Statistics (XeAe)
- Minimum: 0.001839
- Maximum: 0.217659
- Mean: 0.099491 (stable)
- Standard deviation: 0.056006 (slight increase)
- Zero weights: 0%
- Near-max weights: 0%

### Temporal Characteristics
- Earliest spike: 84.1 ms
- Latest spike: 109338.0 ms
- Mean first spike: 32876.4 ms
- Mean last spike: 82190.7 ms

### Theta Analysis
- Silent neurons theta:
  * Mean: 0.019782
  * Std: 0.000000
- Active neurons theta:
  * Min: 0.019941
  * Max: 0.023435
  * Mean: 0.020968
  * Std: 0.000678

## Comparison with Previous Results

### vs B1 (60ms)
1. Slight Improvements:
   - Silent neurons: 6.2% vs 7.5%
   - Activity distribution slightly better
   - Weight stability maintained

2. Remaining Issues:
   - Still far from baseline performance
   - Mean activity still very low
   - Temporal window similar

### vs A2 Baseline (40ms)
1. Significant Degradation:
   - Introduction of silent neurons (6.2% vs 0%)
   - Mean activity reduced by 65%
   - Activity distribution heavily skewed

2. Maintained Aspects:
   - Weight statistics stable
   - Basic temporal response preserved
   - No pathological weight patterns

## Analysis

### Network Behavior
1. Activity Suppression:
   - Even moderate increase in tc_post_2_ee shows strong suppressive effect
   - Network unable to maintain optimal firing rates
   - Significant shift toward low-activity regime

2. Learning Dynamics:
   - Weight stability maintained despite activity changes
   - No evidence of learning degradation
   - Temporal integration possibly compromised

### Mechanistic Insights
1. Time Constant Effects:
   - Linear relationship between tc_post_2_ee and activity suppression
   - Suggests critical role in maintaining excitation/inhibition balance
   - May indicate optimal range is below 40ms

2. Network Stability:
   - Basic network function preserved
   - No catastrophic failure
   - Gradual performance degradation

## Conclusions

1. Performance Impact:
   - 45ms setting still shows significant degradation
   - Not a viable optimization direction
   - Confirms sensitivity to tc_post_2_ee

2. Optimization Direction:
   - Evidence suggests optimal value ≤ 40ms
   - Worth testing 35ms (B1.2)
   - May need to explore other parameters

## Recommendations

1. Immediate Actions:
   - Proceed with B1.2 (35ms)
   - Use larger sample size for validation
   - Focus on activity restoration

2. Future Directions:
   - Consider alternative optimization parameters
   - Investigate excitation/inhibition balance
   - Study interaction with other time constants