"""
MNIST classification test script using trained SNN.
Updated for Python 3 compatibility.
"""

import logging
import os
import time
from pathlib import Path

import numpy as np
from brian2 import *

from functions.data import get_labeled_data

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Paths
MNIST_data_path = Path('./mnist/')

def get_recognized_number_ranking(assignments, spike_rates):
    summed_rates = [0] * 10
    num_assignments = [0] * 10
    for i in range(10):
        num_assignments[i] = len(np.where(assignments == i)[0])
        if num_assignments[i] > 0:
            summed_rates[i] = np.sum(spike_rates[assignments == i]) / num_assignments[i]
    return np.argsort(summed_rates)[::-1]

def get_new_assignments(result_monitor, input_numbers):
    assignments = np.zeros(n_e)
    input_nums = np.asarray(input_numbers)
    maximum_rate = [0] * n_e
    for j in range(10):
        num_assignments = len(np.where(input_nums == j)[0])
        if num_assignments > 0:
            rate = np.sum(result_monitor[input_nums == j], axis = 0) / num_assignments
        for i in range(n_e):
            if rate[i] > maximum_rate[i]:
                maximum_rate[i] = rate[i]
                assignments[i] = j
    return assignments

# Load test data
logger.info('Loading test data...')
testing = get_labeled_data(MNIST_data_path / 'testing', bTrain=False)

# Parameters
test_mode = True
np.random.seed(0)
data_path = Path('./')
weight_path = data_path / 'weights'
num_examples = 10000
use_testing_set = True
record_spikes = True
ee_STDP_on = False
update_interval = num_examples

n_input = 784
n_e = 400
n_i = n_e
single_example_time = 0.35 * second
resting_time = 0.15 * second
runtime = num_examples * (single_example_time + resting_time)
v_rest_e = -65. * mV 
v_rest_i = -60. * mV
v_reset_e = -65. * mV
v_reset_i = -45. * mV
v_thresh_e = -52. * mV
v_thresh_i = -40. * mV
refrac_e = 5. * ms
refrac_i = 2. * ms

weight = {}
delay = {}
input_population_names = ['X']
population_names = ['A']
input_connection_names = ['XA']
save_conns = ['XeAe']
input_conn_names = ['ee_input']
recurrent_conn_names = ['ei', 'ie']
weight['ee_input'] = 78.
delay['ee_input'] = (0*ms,10*ms)
delay['ei_input'] = (0*ms,5*ms)
input_intensity = 2.

print('Setting up network...')
neuron_eqs_e = '''
    dv/dt = ((v_rest_e - v) + (I_synE+I_synI) / nS) / (100*ms)  : volt (unless refractory)
    I_synE = ge * nS * -v                           : amp
    I_synI = gi * nS * (-100.*mV-v)                : amp
    dge/dt = -ge/(1.0*ms)                          : 1
    dgi/dt = -gi/(2.0*ms)                          : 1
    '''
neuron_eqs_e += '\n  theta      :volt'
neuron_eqs_e += '\n  dtimer/dt = 0.1  : second'

neuron_eqs_i = '''
    dv/dt = ((v_rest_i - v) + (I_synE+I_synI) / nS) / (10*ms)  : volt (unless refractory)
    I_synE = ge * nS * -v                           : amp
    I_synI = gi * nS * (-85.*mV-v)                 : amp
    dge/dt = -ge/(1.0*ms)                          : 1
    dgi/dt = -gi/(2.0*ms)                          : 1
    '''

# Set up the network
neuron_groups = {}
input_groups = {}
connections = {}
rate_monitors = {}
spike_monitors = {}
spike_counters = {}
result_monitor = np.zeros((update_interval, n_e))

scr_e = 'v = v_reset_e; timer = 0*ms'
offset = 20.0*mV
v_thresh_e_str = '(v>(theta - offset + v_thresh_e)) and (timer>refrac_e)'
v_thresh_i_str = 'v>v_thresh_i'
v_reset_i_str = 'v=v_reset_i'

neuron_groups['e'] = NeuronGroup(n_e*len(population_names), neuron_eqs_e, 
                                threshold=v_thresh_e_str, refractory=refrac_e, 
                                reset=scr_e, method='euler')
neuron_groups['i'] = NeuronGroup(n_i*len(population_names), neuron_eqs_i, 
                                threshold=v_thresh_i_str, refractory=refrac_i, 
                                reset=v_reset_i_str, method='euler')

# Create network population and recurrent connections
for subgroup_n, name in enumerate(population_names):
    logger.info(f'Creating neuron group {name}')
    
    neuron_groups[name+'e'] = neuron_groups['e'][subgroup_n*n_e:(subgroup_n+1)*n_e]
    neuron_groups[name+'i'] = neuron_groups['i'][subgroup_n*n_i:(subgroup_n+1)*n_e]
    
    neuron_groups[name+'e'].v = v_rest_e - 40. * mV
    neuron_groups[name+'i'].v = v_rest_i - 40. * mV
    neuron_groups['e'].theta = np.load(weight_path / f'theta_{name}.npy') * volt
    
    logger.info('Creating recurrent connections')
    for conn_type in recurrent_conn_names:
        connName = name+conn_type[0]+name+conn_type[1]
        weightMatrix = np.load(weight_path / '../random' / f'{connName}.npy')
        model = 'w : 1'
        pre = f'g{conn_type[0]}_post += w'
        post = ''
        connections[connName] = Synapses(neuron_groups[connName[0:2]], 
                                       neuron_groups[connName[2:4]],
                                       model=model, on_pre=pre, on_post=post)
        connections[connName].connect(True)
        connections[connName].w = weightMatrix[connections[connName].i, 
                                             connections[connName].j]
    
    logger.info(f'Creating monitors for {name}')
    rate_monitors[name+'e'] = PopulationRateMonitor(neuron_groups[name+'e'])
    rate_monitors[name+'i'] = PopulationRateMonitor(neuron_groups[name+'i'])
    spike_counters[name+'e'] = SpikeMonitor(neuron_groups[name+'e'])
    
    if record_spikes:
        spike_monitors[name+'e'] = SpikeMonitor(neuron_groups[name+'e'])
        spike_monitors[name+'i'] = SpikeMonitor(neuron_groups[name+'i'])

# Create input population and connections from input populations
for i,name in enumerate(input_population_names):
    input_groups[name+'e'] = PoissonGroup(n_input, 0*Hz)
    rate_monitors[name+'e'] = PopulationRateMonitor(input_groups[name+'e'])

for name in input_connection_names:
    logger.info(f'Creating connections between {name[0]} and {name[1]}')
    for connType in input_conn_names:
        connName = name[0] + connType[0] + name[1] + connType[1]
        weightMatrix = np.load(weight_path / f'{connName}.npy')
        model = 'w : 1'
        pre = f'g{connType[0]}_post += w'
        post = ''
        
        connections[connName] = Synapses(input_groups[connName[0:2]], 
                                       neuron_groups[connName[2:4]],
                                       model=model, on_pre=pre, on_post=post)
        minDelay = delay[connType][0]
        maxDelay = delay[connType][1]
        deltaDelay = maxDelay - minDelay
        connections[connName].connect(True)
        connections[connName].delay = 'minDelay + rand() * deltaDelay'
        connections[connName].w = weightMatrix[connections[connName].i, 
                                             connections[connName].j]

logger.info('Setting up simulation...')
net = Network()
for obj_list in [neuron_groups, input_groups, connections, rate_monitors,
        spike_monitors, spike_counters]:
    for key in obj_list:
        net.add(obj_list[key])

previous_spike_count = np.zeros(n_e)
assignments = np.zeros(n_e)
input_numbers = [0] * num_examples
outputNumbers = np.zeros((num_examples, 10))

# Initialize variables
j = 0
num_retries = 0
num_failures = 0
num_successes = 0

logger.info('Starting testing...')
start_time = time.time()
while j < (int(num_examples)):
    if use_testing_set:
        spike_rates = testing['x'][j%10000,:,:].reshape((n_input)) / 8. * input_intensity
    input_groups['Xe'].rates = spike_rates * Hz
    
    net.run(single_example_time, report='text')
    
    current_spike_count = np.asarray(spike_counters['Ae'].count[:]) - previous_spike_count
    previous_spike_count = np.copy(np.asarray(spike_counters['Ae'].count[:]))
    
    if np.sum(current_spike_count) < 5:
        num_retries += 1
        input_intensity += 1
        j -= 1
        logger.warning(f'Number of spikes: {sum(current_spike_count)}')
        logger.warning(f'Retrying with intensity: {input_intensity}')
    else:
        num_successes += 1
        if j % update_interval == 0 and j > 0:
            assignments = get_new_assignments(result_monitor[:], input_numbers[j-update_interval : j])
        result_monitor[j%update_interval,:] = current_spike_count
        input_numbers[j] = testing['y'][j%10000][0]
        outputNumbers[j,:] = get_recognized_number_ranking(assignments, 
                                                         result_monitor[j%update_interval,:])
        
        if j % 100 == 0 and j > 0:
            recognition_rate = np.mean(outputNumbers[j-100:j,0] == input_numbers[j-100:j]) * 100
            logger.info(f'Example {j}')
            logger.info(f'Current recognition rate: {recognition_rate:.2f}%')
    
    j += 1

end_time = time.time()
logger.info('Testing completed')
success_rate = (num_successes/(num_successes+num_failures)) * 100
recognition_rate = np.mean(outputNumbers[:,0] == input_numbers) * 100
duration = end_time - start_time

logger.info(f'Success rate: {success_rate:.2f}%')
logger.info(f'Average recognition rate: {recognition_rate:.2f}%')
logger.info(f'Total time: {duration:.2f} seconds')