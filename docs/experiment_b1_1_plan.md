# Experiment B1.1: Conservative STDP Time Constant Adjustment

## Background
Previous experiment (B1) with tc_post_2_ee = 60ms showed significant degradation:
- 7.5% silent neurons (from 0% in A2)
- Mean spikes dropped to 7.18 (from 20.21)
- Only 17.5% neurons in optimal range (from 97.0%)

## Current Configuration
- nu_ee_pre: 0.00005 (fixed from A2)
- nu_ee_post: 0.01
- tc_pre_ee: 20 ms
- tc_post_1_ee: 20 ms
- tc_post_2_ee: 40 ms → 45 ms

## Proposed Change
More conservative adjustment:
```python
tc_post_2_ee = 45*b2.ms  # moderate increase from 40ms
```

## Rationale
1. Previous increase to 60ms was too aggressive:
   - Caused excessive activity suppression
   - Created potential excitation/inhibition imbalance
   - Led to significant performance degradation

2. Conservative approach (45ms):
   - Smaller step change (+5ms instead of +20ms)
   - Maintains closer balance with other time constants
   - Lower risk of disrupting network dynamics

## Success Criteria
1. Primary metrics (compared to A2 baseline):
   - No silent neurons (critical)
   - Mean spikes > 15
   - >90% neurons in optimal range (11-50 spikes)

2. Secondary metrics:
   - Weight stability (mean ~0.099)
   - Early spike timing preserved
   - Balanced activity distribution

## Testing Strategy
1. Initial test (200 samples):
   - Quick validation of network stability
   - Check for silent neurons
   - Basic activity distribution

2. If promising:
   - Full test (600 samples)
   - Detailed performance analysis
   - Comparison with A2 baseline

## Monitoring Focus
1. Critical indicators:
   - Number of silent neurons
   - Activity distribution
   - Mean spike count

2. Stability metrics:
   - Weight distribution
   - Temporal characteristics
   - Theta values

## Next Steps
1. If successful (meets all criteria):
   - Document improvements
   - Consider further optimization

2. If partially successful:
   - Analyze specific improvements/degradations
   - Consider testing 42-43ms range

3. If unsuccessful:
   - Proceed with 35ms test (B1.2)
   - Document failure modes
   - Analyze system sensitivity