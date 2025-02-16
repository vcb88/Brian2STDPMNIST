# Learning Rates Optimization Experiments

## Background
Current configuration:
- nu_ee_pre: 0.0001 (pre-synaptic learning rate)
- nu_ee_post: 0.01 (post-synaptic learning rate)
- Ratio: 1:100

Base performance metrics (600 samples):
- Inactive neurons: 0%
- Mean spikes: 20.48
- Activity distribution: 99% in optimal range (11-50 spikes)
- Weight range: [0.001815, 0.226687]
- Mean weight: 0.099490

## Experiment Series A: Rate Ratio Optimization

### Experiment A1
Configuration:
- nu_ee_pre: 0.0002 (increased 2x)
- nu_ee_post: 0.01 (unchanged)
- New ratio: 1:50

Goals:
1. Investigate effect of stronger pre-synaptic learning
2. Potentially improve learning speed
3. Maintain network stability
4. Preserve optimal activity distribution

Expected effects:
1. Faster weight convergence
2. More balanced pre/post synaptic influence
3. Possibly more dynamic weight changes

Success criteria:
1. Maintain 0% inactive neurons
2. Keep >95% neurons in optimal activity range
3. Stable weight distribution
4. No degradation in temporal characteristics

### Future Experiments (Planned)
A2: Ratio 1:200
- nu_ee_pre: 0.00005
- nu_ee_post: 0.01

Testing strategy:
1. Initial validation on 200 samples
2. Full verification on 600 samples
3. Continuous monitoring of stability metrics