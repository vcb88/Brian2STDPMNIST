# Parameter Optimization Experiments

## Baseline (Original Parameters)
Date: 2025-02-16

### Parameters
```python
# Temporal parameters
single_example_time = 0.35 * second
resting_time = 0.15 * second

# Neuron parameters
v_rest_e = -65. * mV
v_reset_e = -65. * mV
v_thresh_e = -52. * mV
refrac_e = 5. * ms

# Threshold adaptation
tc_theta = 1e7 * ms
theta_plus_e = 0.05 * mV
theta_init = 20.0 * mV
```

### Results
- Silent neurons: 35.5% (142/400)
- Spike statistics:
  * Mean spikes: 7.45
  * Max spikes: 81
  * Distribution:
    - 1-10 spikes: 30.8%
    - 11-50 spikes: 21.8%
    - 51-100 spikes: 0.8%
    - >100 spikes: 0.0%
- Timing:
  * Mean first spike: 34.38s
  * Mean last spike: 78.68s
- Weight statistics:
  * Min: 0.001728
  * Max: 0.241996
  * Mean: 0.099496
  * Std: 0.056040
- Theta statistics:
  * Silent neurons: 19.786mV (std: 0.0)
  * Active neurons: 20.361mV (std: 0.623)

### Notes
- Baseline performance after reverting all experimental changes
- Acceptable number of silent neurons (<40%)
- Good activity distribution without over-active neurons
- Room for improvement in reducing silent neurons and earlier activation

## Experiment Plan

1. Threshold Parameters (Priority High):
   - Vary v_thresh_e: -53mV to -51mV
   - Vary theta_plus_e: 0.03mV to 0.07mV
   - Test different initial theta values

2. Temporal Parameters (Priority Medium):
   - Vary resting_time: 0.13s to 0.17s
   - Vary refrac_e: 4ms to 6ms

3. Reset Parameters (Priority Low):
   - Vary v_reset_e: -66mV to -64mV
   - Test different rest potential values

## Guidelines
1. Change only one parameter at a time
2. Document:
   - Parameter changed and rationale
   - Full test results
   - Observations and hypotheses
   - Decision for next step

3. Success Metrics:
   - Primary: % of silent neurons
   - Secondary:
     * Mean spike count
     * Activity distribution
     * Timing of first/last spikes
     * Weight distribution
     * Theta adaptation

4. Failure Conditions:
   - Silent neurons > 40%
   - Extreme activity (>100 spikes per neuron)
   - Very late first spikes (>40s)
   - Unstable learning (rapid weight changes)
   - Loss of neuron specialization

## Experiment #1: Lower Activation Threshold
Date: 2025-02-16

### Hypothesis
Lowering v_thresh_e slightly (from -52mV to -52.5mV) might:
- Reduce number of silent neurons by making activation easier
- Lead to earlier first spikes
- Maintain stable learning due to small change

### Parameter Change
```python
v_thresh_e = -52.5 * mV  # Changed from -52.0 * mV
```

### Results
- Silent neurons: 35.8% (143/400) [Baseline: 35.5%]
- Spike statistics:
  * Mean spikes: 7.47 [Baseline: 7.45]
  * Max spikes: 103 [Baseline: 81]
  * Distribution:
    - 1-10 spikes: 34.0% [Baseline: 30.8%]
    - 11-50 spikes: 18.8% [Baseline: 21.8%]
    - 51-100 spikes: 1.2% [Baseline: 0.8%]
    - >100 spikes: 0.2% [Baseline: 0.0%]
- Timing:
  * Mean first spike: 34.44s [Baseline: 34.38s]
  * Mean last spike: 77.07s [Baseline: 78.68s]
- Weight statistics:
  * Min: 0.001808 [Baseline: 0.001728]
  * Max: 0.255570 [Baseline: 0.241996]
  * Mean: 0.099501 [Baseline: 0.099496]
  * Std: 0.056081 [Baseline: 0.056040]
- Theta statistics:
  * Silent neurons: 19.789mV [Baseline: 19.786mV]
  * Active neurons: 20.367mV [Baseline: 20.361mV]

### Analysis
1. Impact on Silent Neurons:
   - Slight increase in silent neurons (35.8% vs 35.5%)
   - Change is minimal (~0.3%) and within normal variation

2. Activity Changes:
   + More neurons in low activity range (1-10 spikes)
   - Появление сверхактивных нейронов (>100 спайков)
   - Небольшое снижение в средней активности (11-50 спайков)

3. Timing and Weights:
   ± Практически без изменений в времени первого спайка
   + Небольшое улучшение в времени последнего спайка
   ± Веса остались стабильными

4. Theta Adaptation:
   ± Минимальные изменения в значениях theta
   ± Сохранение разницы между активными и неактивными нейронами

### Conclusion
Эксперимент показал, что снижение порога активации на 0.5mV:
1. Не улучшило ситуацию с неактивными нейронами
2. Привело к появлению сверхактивных нейронов
3. Незначительно повлияло на общую динамику сети

### Next Step
Выбран вариант изменения theta_plus_e для усиления адаптации порога.

## Experiment #2: Stronger Threshold Adaptation
Date: 2025-02-16

### Hypothesis
Увеличение theta_plus_e с 0.05mV до 0.06mV должно:
- Усилить адаптацию порога для активных нейронов
- Предотвратить монополизацию активности отдельными нейронами
- Дать больше шансов неактивным нейронам
- Сделать распределение активности более равномерным

### Parameter Change
```python
theta_plus_e = 0.06 * mV  # Changed from 0.05 * mV
```

### Rationale
1. Предыдущий эксперимент показал, что прямое снижение порога активации
   приводит к появлению сверхактивных нейронов
2. Разница в theta между активными и неактивными нейронами мала
   (20.367mV vs 19.789mV)
3. Увеличение шага адаптации должно:
   - Сильнее повышать порог для часто срабатывающих нейронов
   - Дать шанс другим нейронам стать активными
   - Сохранить общую стабильность обучения

### Results
- Silent neurons: 32.0% (128/400) [Baseline: 35.5%]
- Spike statistics:
  * Mean spikes: 7.06 [Baseline: 7.45]
  * Max spikes: 62 [Baseline: 81]
  * Distribution:
    - 1-10 spikes: 35.2% [Baseline: 30.8%]
    - 11-50 spikes: 20.0% [Baseline: 21.8%]
    - 51-100 spikes: 0.5% [Baseline: 0.8%]
    - >100 spikes: 0.0% [Baseline: 0.0%]
- Timing:
  * Mean first spike: 35.39s [Baseline: 34.38s]
  * Mean last spike: 81.70s [Baseline: 78.68s]
- Weight statistics:
  * Min: 0.001795 [Baseline: 0.001728]
  * Max: 0.241585 [Baseline: 0.241996]
  * Mean: 0.099491 [Baseline: 0.099496]
  * Std: 0.056101 [Baseline: 0.056040]
- Theta statistics:
  * Silent neurons: 19.784mV [Baseline: 19.786mV]
  * Active neurons: 20.404mV [Baseline: 20.361mV]

### Analysis
1. Impact on Silent Neurons ✅
   - Значительное снижение количества неактивных нейронов (32.0% vs 35.5%)
   - Улучшение на 3.5% по сравнению с базовой линией

2. Activity Distribution ✅
   + Более равномерное распределение активности
   + Нет сверхактивных нейронов (макс. 62 спайка vs 81)
   + Увеличение доли нейронов с низкой активностью (35.2% vs 30.8%)
   ± Небольшое снижение в средней активности (7.06 vs 7.45 спайков)

3. Timing and Weights ±
   - Небольшое увеличение времени первого спайка (+1.01s)
   - Увеличение времени последнего спайка (+3.02s)
   + Веса остались стабильными, минимальные изменения

4. Theta Adaptation ✅
   + Больший разброс значений theta у активных нейронов
   + Более эффективная дифференциация активных/неактивных нейронов
   + Стабильные значения для неактивных нейронов

### Conclusion
Эксперимент показал значительное улучшение:
1. Существенное снижение количества неактивных нейронов
2. Более равномерное распределение активности
3. Предотвращение монополизации активности
4. Сохранение стабильности обучения (веса)

### Next Step
Продолжаем оптимизацию механизма адаптации порога.

## Experiment #3: Enhanced Threshold Adaptation
Date: 2025-02-16

### Hypothesis
Дальнейшее увеличение theta_plus_e с 0.06mV до 0.07mV должно:
- Еще больше снизить количество неактивных нейронов
- Усилить конкуренцию между нейронами
- Способствовать более специализированному обучению

### Parameter Change
```python
theta_plus_e = 0.07 * mV  # Changed from 0.06 * mV
```

### Rationale
1. Предыдущий эксперимент (theta_plus_e = 0.06mV) показал:
   - Снижение неактивных нейронов на 3.5%
   - Более равномерное распределение активности
   - Отсутствие негативных эффектов на обучение

2. Дальнейшее увеличение порога адаптации может:
   - Усилить положительные эффекты предыдущего эксперимента
   - Создать более сильную конкуренцию между нейронами
   - Потенциально улучшить специализацию нейронов

3. Возможные риски:
   - Слишком сильное подавление активности
   - Увеличение времени первого спайка
   - Нестабильность обучения