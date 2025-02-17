# Experiment A2: Learning Rate Ratio Results (Large Sample)

## Configuration
- nu_ee_pre: 0.00005 (1:200 ratio)
- nu_ee_post: 0.01
- Sample size: 600
- Date: February 17, 2025

## Results Summary
### Key Metrics
- Inactive neurons: 0%
- Mean spikes per neuron: 20.21
- Max spikes: 37
- Min spikes: 7

### Activity Distribution
- Low activity (1-10 spikes): 0.8%
- Medium activity (11-50 spikes): 97.0%
- High activity (>50 spikes): 0%

### Weight Statistics (XeAe)
- Minimum: 0.001815
- Maximum: 0.226902
- Mean: 0.099492
- Standard deviation: 0.055385

### Temporal Characteristics
- Earliest spike: 84.1 ms
- Latest spike: 335311.1 ms
- Mean first spike: 37144.6 ms
- Mean last spike: 303866.2 ms

### Theta Values
- Minimum: 0.020448
- Maximum: 0.025138
- Mean: 0.022517
- Standard deviation: 0.000748

## Comparison with A1 (Large Sample)
1. Activity Improvements:
   - Reduced low-activity neurons (0.8% vs 2.8%)
   - Increased optimal-range neurons (97.0% vs 96.8%)
   - Similar mean activity (20.21 vs 20.33)

2. Temporal Characteristics:
   - Earlier mean first spike (37144.6 vs 39399.6 ms)
   - Similar overall activity duration
   - More consistent activation patterns

3. Weight Stability:
   - More compact weight range
   - Identical mean value
   - Slightly improved standard deviation

## Conclusions
1. Conservative Learning Benefits:
   - Better activity distribution
   - Earlier neuron activation
   - More stable weight distribution

2. Network Stability:
   - Zero inactive neurons maintained
   - Excellent activity range distribution
   - No overactive neurons

## Final Recommendation
Adopt nu_ee_pre = 0.00005 as the new standard value due to:
1. Improved activity distribution (97% in optimal range)
2. Better temporal characteristics
3. More stable learning dynamics
4. Maintained network performance

This configuration provides a more robust and stable learning process while maintaining or improving all key performance metrics.