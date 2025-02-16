# Experiment A2: Learning Rate Ratio Optimization

## Configuration Changes
- nu_ee_pre: 0.00005 (decreased from 0.0002)
- nu_ee_post: 0.01 (unchanged)
- New ratio: 1:200

## Previous Results (Experiment A1, 200 samples)
### Network Activity
- Inactive neurons: 5.5% (22/400)
- Mean spikes: 7.21
- Max spikes: 28

### Activity Distribution
- Low activity (1-10 spikes): 68.0%
- Medium activity (11-50 spikes): 17.5%
- High activity (>50 spikes): 0%

### Weights
- Min: 0.001839
- Max: 0.221166
- Mean: 0.099493
- Std: 0.055952

## Experiment A2 Goals
1. Investigate effects of weaker pre-synaptic learning
2. Test network stability with larger learning rate difference
3. Potentially improve weight stability
4. Maintain or improve neuron activity distribution

## Expected Effects
1. More conservative weight changes
2. Potentially slower learning
3. Possibly better final weight distribution
4. Greater emphasis on post-synaptic plasticity

## Success Criteria
1. Maintain or reduce inactive neurons (< 5.5%)
2. Keep stable weight distribution
3. Maintain temporal characteristics
4. Avoid activity deterioration

## Monitoring Focus
1. Weight stability metrics
2. Learning convergence rate
3. Temporal spike patterns
4. Activity distribution changes

## Risk Assessment
### Potential Benefits
- More stable weights
- Better final convergence
- More selective weight changes

### Potential Risks
- Slower learning
- Possible increase in inactive neurons
- Reduced network plasticity

## Implementation Note
This experiment represents a more conservative approach to weight updates during pre-synaptic events, which might lead to more stable but slower learning. The increased ratio (1:200) emphasizes post-synaptic plasticity effects.