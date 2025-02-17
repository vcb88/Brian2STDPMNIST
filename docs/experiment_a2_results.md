# Experiment A2: Learning Rate Ratio Results (Small Sample)

## Configuration
- nu_ee_pre: 0.00005 (1:200 ratio)
- nu_ee_post: 0.01
- Sample size: 200
- Date: February 17, 2025

## Results Summary
### Key Metrics
- Inactive neurons: 5.5% (22/400)
- Mean spikes per neuron: 7.38
- Max spikes: 21 (decreased from 28 in A1)

### Activity Distribution
- Low activity (1-10 spikes): 67.5%
- Medium activity (11-50 spikes): 16.8%
- High activity (>50 spikes): 0%

### Weight Statistics (XeAe)
- Minimum: 0.001858
- Maximum: 0.217770
- Mean: 0.099497
- Standard deviation: 0.055981

### Temporal Characteristics
- Earliest spike: 84.1 ms
- Latest spike: 108845.6 ms
- Mean first spike: 31395.6 ms
- Mean last spike: 79740.5 ms

### Theta Values
#### Silent Neurons
- Mean: 0.019783
- Standard deviation: 0.000000

#### Active Neurons
- Minimum: 0.019942
- Maximum: 0.023120
- Mean: 0.021027
- Standard deviation: 0.000666

## Comparison with A1
1. Activity Metrics:
   - Same inactive neuron percentage (5.5%)
   - Higher mean spike count (7.38 vs 7.21)
   - Lower maximum spikes (21 vs 28)

2. Temporal Improvements:
   - Earlier mean first spike (31395.6 vs 33352.0 ms)
   - More compact activity window
   - Better temporal consistency

3. Weight Stability:
   - Narrower weight range
   - Similar mean and standard deviation
   - More conservative weight changes

## Analysis
1. Positive Changes:
   - More stable maximum activity
   - Earlier neuron activation
   - More compact temporal characteristics
   - Conservative weight distribution

2. Neutral/Negative Aspects:
   - No improvement in inactive neurons
   - Slight decrease in medium activity neurons

## Conclusions
1. More conservative learning shows potential benefits:
   - Better temporal characteristics
   - More stable activity patterns
   - Maintained weight stability

2. Areas for Further Investigation:
   - Scale testing with larger sample size
   - Long-term stability analysis
   - Impact on learning convergence