'''
Created on 15.12.2014

@author: Peter U. Diehl
Updated for command-line usage by vcb88
'''

import argparse
import logging
import numpy as np
import matplotlib.cm as cmap
import time
import os.path
import scipy
from brian2 import *
import os
import brian2 as b2
from brian2tools import *
import matplotlib.pyplot as plt
from tqdm import tqdm
import datetime

from functions.data import get_labeled_data, get_data_subset
from functions.quick_analysis import quick_analyze
from functions.diagnostics import diagnostic_report

# Parse command line arguments
parser = argparse.ArgumentParser(description='STDP-based MNIST Classification')
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--train', action='store_true', help='Train the network')
group.add_argument('--test', action='store_true', help='Test the network')
parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
parser.add_argument('--data-dir', default='./mnist/', help='Directory containing MNIST data')
parser.add_argument('--save-interval', type=int, default=10000, help='Interval for saving weights')
parser.add_argument('--test-size', type=int, default=10000, help='Number of examples to use for testing (default: 10000)')
parser.add_argument('--random-subset', action='store_true', help='Use random subset of test data instead of first N examples')
parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs (default: 3)')
parser.add_argument('--train-size', type=int, default=60000, help='Number of training examples per epoch (default: 60000, max: 60000)')
parser.add_argument('--num-threads', type=int, default=8, help='Number of threads for parallel processing (default: 8)')
parser.add_argument('--device', choices=['runtime', 'cpp_standalone'], default='runtime', help='Brian2 device to use (default: runtime)')
args = parser.parse_args()

# Configure Brian2 preferences for performance
prefs.devices.cpp_standalone.openmp_threads = args.num_threads
prefs.codegen.target = 'cython'  # Use cython for faster code generation

# Configure logging
logging.basicConfig(
    level=logging.INFO if args.verbose else logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set device
if args.device == 'cpp_standalone':
    set_device('cpp_standalone', directory=None)  # None means use temp directory
    logger.info(f'Using cpp_standalone device with {args.num_threads} OpenMP threads')
else:
    logger.info('Using runtime device with Cython optimization')

# specify the location of the MNIST data
MNIST_data_path = args.data_dir
test_mode = args.test  # Set test_mode based on command line argument

#------------------------------------------------------------------------------
# functions
#------------------------------------------------------------------------------

def create_initial_weight_matrix(n_src, n_tgt, fileName):
    logger.info(f'Creating initial weight matrix for {fileName}')
    if 'XeAe' in fileName:
        weight_matrix = np.random.random((n_src, n_tgt)) * 0.3
    elif 'AeAi' in fileName:
        weight_matrix = np.ones((n_src, n_tgt)) * 10.4
    elif 'AiAe' in fileName:
        weight_matrix = np.ones((n_src, n_tgt)) * 17.0
    else:
        weight_matrix = np.random.random((n_src, n_tgt)) * 0.3
    
    indices = np.nonzero(weight_matrix)
    values = weight_matrix[indices]
    sparse_weights = list(zip(indices[0], indices[1], values))
    
    os.makedirs(os.path.dirname(fileName), exist_ok=True)
    np.save(fileName, sparse_weights)
    logger.info(f'Saved initial weights to {fileName}')
    return weight_matrix

def get_matrix_from_file(fileName):
    logger.info(f'Loading weight matrix from: {fileName}')
    offset = len(ending) + 4
    if fileName[-4-offset] == 'X':
        n_src = n_input
    else:
        if fileName[-3-offset]=='e':
            n_src = n_e
        else:
            n_src = n_i
    if fileName[-1-offset]=='e':
        n_tgt = n_e
    else:
        n_tgt = n_i
    try:
        readout = np.load(fileName)
        logger.info(f'Loaded weights from {fileName}, shape: {readout.shape}')
        value_arr = np.zeros((n_src, n_tgt))
        if not readout.shape == (0,):
            value_arr[np.int32(readout[:,0]), np.int32(readout[:,1])] = readout[:,2]
        return value_arr
    except FileNotFoundError:
        logger.info(f'Weight file {fileName} not found, creating initial weights')
        return create_initial_weight_matrix(n_src, n_tgt, fileName)


def save_connections(ending = ''):
    logger.info('Saving connections')
    for connName in save_conns:
        conn = connections[connName]
        # Convert to numpy arrays and then to list to ensure proper serialization
        connListSparse = list(zip(
            np.array(conn.i).astype(int),
            np.array(conn.j).astype(int),
            np.array(conn.w).astype(float)
        ))
        # Ensure directory exists
        os.makedirs(os.path.dirname(data_path + 'weights/' + connName + ending), exist_ok=True)
        np.save(data_path + 'weights/' + connName + ending, connListSparse)

def save_theta(ending = ''):
    logger.info('Saving theta values')
    for pop_name in population_names:
        # Convert Brian2 quantity to float array
        theta_values = np.array(neuron_groups[pop_name + 'e'].theta_).astype(float)
        # Ensure directory exists
        os.makedirs(os.path.dirname(data_path + 'weights/theta_' + pop_name + ending), exist_ok=True)
        np.save(data_path + 'weights/theta_' + pop_name + ending, theta_values)

def normalize_weights():
    for connName in connections:
        if connName[1] == 'e' and connName[3] == 'e':
            len_source = len(connections[connName].source)
            len_target = len(connections[connName].target)
            connection = np.zeros((len_source, len_target))
            connection[connections[connName].i, connections[connName].j] = connections[connName].w
            temp_conn = np.copy(connection)
            colSums = np.sum(temp_conn, axis = 0)
            colFactors = weight['ee_input']/colSums
            for j in range(n_e):#
                temp_conn[:,j] *= colFactors[j]
            connections[connName].w = temp_conn[connections[connName].i, connections[connName].j]

def get_2d_input_weights():
    name = 'XeAe'
    weight_matrix = np.zeros((n_input, n_e))
    n_e_sqrt = int(np.sqrt(n_e))
    n_in_sqrt = int(np.sqrt(n_input))
    num_values_col = n_e_sqrt*n_in_sqrt
    num_values_row = num_values_col
    rearranged_weights = np.zeros((num_values_col, num_values_row))
    connMatrix = np.zeros((n_input, n_e))
    connMatrix[connections[name].i, connections[name].j] = connections[name].w
    weight_matrix = np.copy(connMatrix)

    for i in range(n_e_sqrt):
        for j in range(n_e_sqrt):
                rearranged_weights[i*n_in_sqrt : (i+1)*n_in_sqrt, j*n_in_sqrt : (j+1)*n_in_sqrt] = \
                    weight_matrix[:, i + j*n_e_sqrt].reshape((n_in_sqrt, n_in_sqrt))
    return rearranged_weights


def plot_2d_input_weights():
    name = 'XeAe'
    weights = get_2d_input_weights()
    fig = b2.figure(fig_num, figsize = (18, 18))
    im2 = b2.imshow(weights, interpolation = "nearest", vmin = 0, vmax = wmax_ee, cmap = plt.colormaps['hot_r'])
    b2.colorbar(im2)
    b2.title('weights of connection' + name)
    fig.canvas.draw()
    return im2, fig

def update_2d_input_weights(im, fig):
    weights = get_2d_input_weights()
    im.set_array(weights)
    fig.canvas.draw()
    return im

def get_current_performance(performance, current_example_num):
    current_evaluation = int(current_example_num/update_interval)
    start_num = current_example_num - update_interval
    end_num = current_example_num
    difference = outputNumbers[start_num:end_num, 0] - input_numbers[start_num:end_num]
    correct = len(np.where(difference == 0)[0])
    performance[current_evaluation] = correct / float(update_interval) * 100
    return performance

def plot_performance(fig_num):
    num_evaluations = int(num_examples/update_interval)
    time_steps = range(0, num_evaluations)
    performance = np.zeros(num_evaluations)
    fig = b2.figure(fig_num, figsize = (5, 5))
    fig_num += 1
    ax = fig.add_subplot(111)
    im2, = ax.plot(time_steps, performance) #my_cmap
    b2.ylim(ymax = 100)
    b2.title('Classification performance')
    fig.canvas.draw()
    return im2, performance, fig_num, fig

def update_performance_plot(im, performance, current_example_num, fig):
    performance = get_current_performance(performance, current_example_num)
    im.set_ydata(performance)
    fig.canvas.draw()
    return im, performance

def get_recognized_number_ranking(assignments, spike_rates):
    summed_rates = [0] * 10
    num_assignments = [0] * 10
    for i in range(10):
        num_assignments[i] = len(np.where(assignments == i)[0])
        if num_assignments[i] > 0:
            summed_rates[i] = np.sum(spike_rates[assignments == i]) / num_assignments[i]
    return np.argsort(summed_rates)[::-1]

def save_results(accuracy_fig, confusion_fig):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    accuracy_path = f'results/accuracy_per_digit_{timestamp}.png'
    confusion_path = f'results/confusion_matrix_{timestamp}.png'
    
    logger.info(f'Saving accuracy plot to: {accuracy_path}')
    accuracy_fig.savefig(accuracy_path)
    
    logger.info(f'Saving confusion matrix to: {confusion_path}')
    confusion_fig.savefig(confusion_path)
    
    # Create symbolic links for latest results
    latest_accuracy = 'results/accuracy_per_digit.png'
    latest_confusion = 'results/confusion_matrix.png'
    
    if os.path.exists(latest_accuracy):
        os.remove(latest_accuracy)
    if os.path.exists(latest_confusion):
        os.remove(latest_confusion)
        
    os.symlink(os.path.basename(accuracy_path), latest_accuracy)
    os.symlink(os.path.basename(confusion_path), latest_confusion)
    
    logger.info('Results saved successfully')

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


#------------------------------------------------------------------------------
# load MNIST
#------------------------------------------------------------------------------
start = time.time()
training = get_labeled_data('training', bTrain=True, MNIST_data_path=MNIST_data_path)
end = time.time()
logger.info('Time needed to load training set: %.2fs', end - start)

start = time.time()
full_testing = get_labeled_data('testing', bTrain=False, MNIST_data_path=MNIST_data_path)
if test_mode and (args.test_size < 10000 or args.random_subset):
    testing = get_data_subset(full_testing, args.test_size, args.random_subset)
    logger.info(f'Using {len(testing["y"])} test examples' + (' (randomly selected)' if args.random_subset else ''))
else:
    testing = full_testing
end = time.time()
logger.info(f'Time needed to load test set: {end - start:.2f}s')


#------------------------------------------------------------------------------
# set parameters and equations
#------------------------------------------------------------------------------
# test_mode is now set from command line arguments

np.random.seed(0)
data_path = './' # TODO: This should be a parameter
if test_mode:
    weight_path = data_path + 'weights/random/'
    num_examples = args.test_size
    use_testing_set = True
    do_plot_performance = False
    record_spikes = True
    ee_STDP_on = False
    logger.info(f'Testing on {num_examples} examples' + (' (random subset)' if args.random_subset else ''))
    update_interval = num_examples
else:
    weight_path = data_path + 'weights/random/'
    train_size = min(args.train_size, 60000)  # Limit to maximum available examples
    num_examples = train_size * args.epochs
    use_testing_set = False
    do_plot_performance = True
    logger.info(f'Training on {train_size} examples for {args.epochs} epochs (total {num_examples} iterations)')
    if num_examples <= 60000:
        record_spikes = True
    else:
        record_spikes = True
    ee_STDP_on = True


ending = ''
n_input = 784
n_e = 400
n_i = n_e
# Time constants for example presentation and network reset
single_example_time = 0.35 * b2.second
resting_time = 0.15 * b2.second
runtime = num_examples * (single_example_time + resting_time)
if num_examples <= 10000:
    update_interval = num_examples
    weight_update_interval = 20
else:
    update_interval = 10000
    weight_update_interval = 100
if num_examples <= 60000:
    save_connections_interval = 10000
else:
    save_connections_interval = 10000
    update_interval = 10000

# Neuronal parameters
v_rest_e = -65. * b2.mV
v_rest_i = -60. * b2.mV
v_reset_e = -65. * b2.mV
v_reset_i = -45. * b2.mV
v_thresh_e = -52. * b2.mV
v_thresh_i = -40. * b2.mV
refrac_e = 5. * b2.ms
refrac_i = 2. * b2.ms

weight = {}
delay = {}
input_population_names = ['X']
population_names = ['A']
input_connection_names = ['XA']
save_conns = ['XeAe']
input_conn_names = ['ee_input']
recurrent_conn_names = ['ei', 'ie']
weight['ee_input'] = 78.
delay['ee_input'] = (0*b2.ms,10*b2.ms)
delay['ei_input'] = (0*b2.ms,5*b2.ms)
input_intensity = 2.
start_input_intensity = input_intensity

tc_pre_ee = 20*b2.ms
tc_post_1_ee = 20*b2.ms
tc_post_2_ee = 45*b2.ms  # moderate increase from 40ms for B1.1 experiment
nu_ee_pre =  0.00005     # pre-synaptic learning rate (decreased for more conservative learning)
nu_ee_post = 0.01       # post-synaptic learning rate (1:200 ratio with pre-synaptic)
wmax_ee = 1.0
exp_ee_pre = 0.2
exp_ee_post = exp_ee_pre
STDP_offset = 0.4

# Threshold adaptation parameters
if test_mode:
    scr_e = 'v = v_reset_e; timer = 0*ms'
else:
    # Threshold adaptation parameters
    tc_theta = 1e7 * b2.ms
    theta_plus_e = 0.16 * b2.mV  # Further increased from 0.15mV for final optimization verification
    
    # Simple reset with theta modification
    scr_e = '''
    v = v_reset_e
    theta += theta_plus_e
    timer = 0*ms
    '''
offset = 20.0*b2.mV
# Original threshold condition
v_thresh_e_str = '(v>(theta - offset + v_thresh_e)) and (timer>refrac_e)'
v_thresh_i_str = 'v>v_thresh_i'
v_reset_i_str = 'v=v_reset_i'


neuron_eqs_e = '''
        dv/dt = ((v_rest_e - v) + (I_synE+I_synI) / nS) / (100*ms)  : volt (unless refractory)
        I_synE = ge * nS *         -v                           : amp
        I_synI = gi * nS * (-100.*mV-v)                          : amp
        dge/dt = -ge/(1.0*ms)                                   : 1
        dgi/dt = -gi/(2.0*ms)                                  : 1
        '''
if test_mode:
    neuron_eqs_e += '\n  theta      :volt'
else:
    neuron_eqs_e += '\n  dtheta/dt = -theta / (tc_theta)  : volt'
neuron_eqs_e += '\n  dtimer/dt = 0.1  : second'

neuron_eqs_i = '''
        dv/dt = ((v_rest_i - v) + (I_synE+I_synI) / nS) / (10*ms)  : volt (unless refractory)
        I_synE = ge * nS *         -v                           : amp
        I_synI = gi * nS * (-85.*mV-v)                          : amp
        dge/dt = -ge/(1.0*ms)                                   : 1
        dgi/dt = -gi/(2.0*ms)                                  : 1
        '''
eqs_stdp_ee = '''
                post2before                            : 1
                dpre/dt   =   -pre/(tc_pre_ee)         : 1 (event-driven)
                dpost1/dt  = -post1/(tc_post_1_ee)     : 1 (event-driven)
                dpost2/dt  = -post2/(tc_post_2_ee)     : 1 (event-driven)
            '''
eqs_stdp_pre_ee = 'pre = 1.; w = clip(w + nu_ee_pre * post1, 0, wmax_ee)'
eqs_stdp_post_ee = 'post2before = post2; w = clip(w + nu_ee_post * pre * post2before, 0, wmax_ee); post1 = 1.; post2 = 1.'

b2.ion()
fig_num = 1
neuron_groups = {}
input_groups = {}
connections = {}
rate_monitors = {}
spike_monitors = {}
spike_counters = {}
result_monitor = np.zeros((update_interval,n_e))

neuron_groups['e'] = b2.NeuronGroup(n_e*len(population_names), neuron_eqs_e, threshold= v_thresh_e_str, refractory= refrac_e, reset= scr_e, method='euler')
neuron_groups['i'] = b2.NeuronGroup(n_i*len(population_names), neuron_eqs_i, threshold= v_thresh_i_str, refractory= refrac_i, reset= v_reset_i_str, method='euler')


#------------------------------------------------------------------------------
# create network population and recurrent connections
#------------------------------------------------------------------------------
for subgroup_n, name in enumerate(population_names):
    print('create neuron group', name)

    neuron_groups[name+'e'] = neuron_groups['e'][subgroup_n*n_e:(subgroup_n+1)*n_e]
    neuron_groups[name+'i'] = neuron_groups['i'][subgroup_n*n_i:(subgroup_n+1)*n_e]

    neuron_groups[name+'e'].v = v_rest_e - 40. * b2.mV
    neuron_groups[name+'i'].v = v_rest_i - 40. * b2.mV
    if test_mode or weight_path[-8:] == 'weights/':
        theta_path = data_path + 'weights/' + 'theta_' + name + ending + '.npy'
        logger.info(f'Loading theta weights from: {theta_path}')
        neuron_groups['e'].theta = np.load(theta_path) * b2.volt
    else:
        # Original theta initialization
        neuron_groups['e'].theta = np.ones((n_e)) * 20.0*b2.mV

    print('create recurrent connections')
    for conn_type in recurrent_conn_names:
        connName = name+conn_type[0]+name+conn_type[1]
        weightMatrix = get_matrix_from_file(weight_path + connName + ending + '.npy')
        model = 'w : 1'
        pre = 'g%s_post += w' % conn_type[0]
        post = ''
        if ee_STDP_on:
            if 'ee' in recurrent_conn_names:
                model += eqs_stdp_ee
                pre += '; ' + eqs_stdp_pre_ee
                post = eqs_stdp_post_ee
        connections[connName] = b2.Synapses(neuron_groups[connName[0:2]], neuron_groups[connName[2:4]],
                                                    model=model, on_pre=pre, on_post=post)
        connections[connName].connect(True) # all-to-all connection
        connections[connName].w = weightMatrix[connections[connName].i, connections[connName].j]

    print('create monitors for', name)
    rate_monitors[name+'e'] = b2.PopulationRateMonitor(neuron_groups[name+'e'])
    rate_monitors[name+'i'] = b2.PopulationRateMonitor(neuron_groups[name+'i'])
    spike_counters[name+'e'] = b2.SpikeMonitor(neuron_groups[name+'e'])

    if record_spikes:
        spike_monitors[name+'e'] = b2.SpikeMonitor(neuron_groups[name+'e'])
        spike_monitors[name+'i'] = b2.SpikeMonitor(neuron_groups[name+'i'])


#------------------------------------------------------------------------------
# create input population and connections from input populations
#------------------------------------------------------------------------------
pop_values = [0,0,0]
for i,name in enumerate(input_population_names):
    input_groups[name+'e'] = b2.PoissonGroup(n_input, 0*Hz)
    rate_monitors[name+'e'] = b2.PopulationRateMonitor(input_groups[name+'e'])

for name in input_connection_names:
    print('create connections between', name[0], 'and', name[1])
    for connType in input_conn_names:
        connName = name[0] + connType[0] + name[1] + connType[1]
        weightMatrix = get_matrix_from_file(weight_path + connName + ending + '.npy')
        model = 'w : 1'
        pre = 'g%s_post += w' % connType[0]
        post = ''
        if ee_STDP_on:
            print('create STDP for connection', name[0]+'e'+name[1]+'e')
            model += eqs_stdp_ee
            pre += '; ' + eqs_stdp_pre_ee
            post = eqs_stdp_post_ee

        connections[connName] = b2.Synapses(input_groups[connName[0:2]], neuron_groups[connName[2:4]],
                                                    model=model, on_pre=pre, on_post=post)
        minDelay = delay[connType][0]
        maxDelay = delay[connType][1]
        deltaDelay = maxDelay - minDelay
        # TODO: test this
        connections[connName].connect(True) # all-to-all connection
        connections[connName].delay = 'minDelay + rand() * deltaDelay'
        connections[connName].w = weightMatrix[connections[connName].i, connections[connName].j]


#------------------------------------------------------------------------------
# run the simulation and set inputs
#------------------------------------------------------------------------------

logger.info(f'Simulation started at {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
net = Network()
for obj_list in [neuron_groups, input_groups, connections, rate_monitors,
        spike_monitors, spike_counters]:
    for key in obj_list:
        net.add(obj_list[key])
logger.info(f'Network built with {len(neuron_groups)} neuron groups, {len(connections)} connections')

test_start_time = time.time()
logger.info(f'Starting test run at {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
logger.info(f'Testing {num_examples} examples')

previous_spike_count = np.zeros(n_e)
assignments = np.zeros(n_e)
input_numbers = [0] * num_examples
outputNumbers = np.zeros((num_examples, 10))
if not test_mode:
    input_weight_monitor, fig_weights = plot_2d_input_weights()
    fig_num += 1
if do_plot_performance:
    performance_monitor, performance, fig_num, fig_performance = plot_performance(fig_num)
for i,name in enumerate(input_population_names):
    input_groups[name+'e'].rates = 0 * Hz
net.run(0*second)
j = 0
while j < (int(num_examples)):
    test_iteration_start = datetime.datetime.now()
    logger.info(f'Starting test {j+1}/{num_examples} at {test_iteration_start.strftime("%Y-%m-%d %H:%M:%S")}')
    if test_mode:
        if use_testing_set:
            spike_rates = testing['x'][j%10000,:,:].reshape((n_input)) / 8. *  input_intensity
        else:
            spike_rates = training['x'][j%60000,:,:].reshape((n_input)) / 8. *  input_intensity
    else:
        normalize_weights()
        spike_rates = training['x'][j%60000,:,:].reshape((n_input)) / 8. *  input_intensity
    input_groups['Xe'].rates = spike_rates * Hz
#     print('run number:', j+1, 'of', int(num_examples))
    net.run(single_example_time, report='text')

    if j % update_interval == 0 and j > 0:
        assignments = get_new_assignments(result_monitor[:], input_numbers[j-update_interval : j])
    if j % weight_update_interval == 0 and not test_mode:
        update_2d_input_weights(input_weight_monitor, fig_weights)
    if j % save_connections_interval == 0 and j > 0 and not test_mode:
        save_connections(str(j))
        save_theta(str(j))

    current_spike_count = np.asarray(spike_counters['Ae'].count[:]) - previous_spike_count
    previous_spike_count = np.copy(spike_counters['Ae'].count[:])
    if np.sum(current_spike_count) < 5:
        input_intensity += 1
        for i,name in enumerate(input_population_names):
            input_groups[name+'e'].rates = 0 * Hz
        net.run(resting_time)
    else:
        result_monitor[j%update_interval,:] = current_spike_count
        if test_mode and use_testing_set:
            input_numbers[j] = testing['y'][j%10000][0]
        else:
            input_numbers[j] = training['y'][j%60000][0]
        outputNumbers[j,:] = get_recognized_number_ranking(assignments, result_monitor[j%update_interval,:])
        if j % 100 == 0 and j > 0:
            logger.info('Runs completed: %d of %d', j, int(num_examples))
        if j % update_interval == 0 and j > 0:
            if do_plot_performance:
                unused, performance = update_performance_plot(performance_monitor, performance, j, fig_performance)
                logger.info('Classification performance: %s', 
                             performance[:int(j/update_interval)+1])
        for i,name in enumerate(input_population_names):
            input_groups[name+'e'].rates = 0 * Hz
        net.run(resting_time)
        input_intensity = start_input_intensity
        test_iteration_end = datetime.datetime.now()
        test_iteration_duration = (test_iteration_end - test_iteration_start).total_seconds()
        logger.info(f'Test {j+1} completed in {test_iteration_duration:.2f} seconds')
        j += 1


#------------------------------------------------------------------------------
# save and analyze results
#------------------------------------------------------------------------------
test_duration = time.time() - test_start_time
logger.info(f'Test run completed in {test_duration:.2f} seconds')

logger.info('Saving and analyzing results')

# Run diagnostic report
stdp_params = {
    'tc_pre_ee': tc_pre_ee,
    'tc_post_1_ee': tc_post_1_ee,
    'tc_post_2_ee': tc_post_2_ee,
    'nu_ee_pre': nu_ee_pre,
    'nu_ee_post': nu_ee_post,
    'wmax_ee': wmax_ee
}
# Save results and perform diagnostics
if test_mode:
    # Save test results
    np.save(data_path + 'activity/resultPopVecs' + str(num_examples), result_monitor)
    np.save(data_path + 'activity/inputNumbers' + str(num_examples), input_numbers)
    
    # Run analysis
    logger.info('Performing quick analysis of test results...')
    accuracy, per_digit_acc = quick_analyze(
        data_path + 'activity/inputNumbers' + str(num_examples) + '.npy',
        data_path + 'activity/resultPopVecs' + str(num_examples) + '.npy'
    )
else:
    # Save training results
    save_theta()
    save_connections()

# Always run diagnostics
diagnostic_report(connections, spike_monitors, save_conns, stdp_params, neuron_groups)


#------------------------------------------------------------------------------
# plot results
#------------------------------------------------------------------------------
if rate_monitors:
    b2.figure(fig_num)
    fig_num += 1
    for i, name in enumerate(rate_monitors):
        b2.subplot(len(rate_monitors), 1, 1+i)
        b2.plot(rate_monitors[name].t/b2.second, rate_monitors[name].rate, '.')
        b2.title('Rates of population ' + name)

if spike_monitors:
    b2.figure(fig_num)
    fig_num += 1
    for i, name in enumerate(spike_monitors):
        b2.subplot(len(spike_monitors), 1, 1+i)
        b2.plot(spike_monitors[name].t/b2.ms, spike_monitors[name].i, '.')
        b2.title('Spikes of population ' + name)

if spike_counters:
    b2.figure(fig_num)
    fig_num += 1
    b2.plot(spike_monitors['Ae'].count[:])
    b2.title('Spike count of population Ae')


plot_2d_input_weights()

# Plot connection weights
# Plot connection weights
plt.figure(5, figsize=(10, 12))
plt.subplot(3,1,1)
try:
    w = np.array(connections['XeAe'].w)
    plt.imshow(w.reshape(n_input, n_e), aspect='auto', cmap='viridis')
    plt.colorbar(label='Weight')
    plt.title('XeAe weights')
except Exception as e:
    logger.warning(f"Failed to plot XeAe weights: {e}")

plt.subplot(3,1,2)
try:
    w = np.array(connections['AeAi'].w)
    plt.imshow(w.reshape(n_e, n_i), aspect='auto', cmap='viridis')
    plt.colorbar(label='Weight')
    plt.title('AeAi weights')
except Exception as e:
    logger.warning(f"Failed to plot AeAi weights: {e}")

plt.subplot(3,1,3)
try:
    w = np.array(connections['AiAe'].w)
    plt.imshow(w.reshape(n_i, n_e), aspect='auto', cmap='viridis')
    plt.colorbar(label='Weight')
    plt.title('AiAe weights')
except Exception as e:
    logger.warning(f"Failed to plot AiAe weights: {e}")
plt.tight_layout()

# Plot connection delays
plt.figure(6, figsize=(10, 12))
plt.subplot(3,1,1)
try:
    d = np.array(connections['XeAe'].delay)
    plt.imshow(d.reshape(n_input, n_e), aspect='auto', cmap='viridis')
    plt.colorbar(label='Delay (ms)')
    plt.title('XeAe delays')
except Exception as e:
    logger.warning(f"Failed to plot XeAe delays: {e}")

plt.subplot(3,1,2)
try:
    d = np.array(connections['AeAi'].delay)
    plt.imshow(d.reshape(n_e, n_i), aspect='auto', cmap='viridis')
    plt.colorbar(label='Delay (ms)')
    plt.title('AeAi delays')
except Exception as e:
    logger.warning(f"Failed to plot AeAi delays: {e}")

plt.subplot(3,1,3)
try:
    d = np.array(connections['AiAe'].delay)
    plt.imshow(d.reshape(n_i, n_e), aspect='auto', cmap='viridis')
    plt.colorbar(label='Delay (ms)')
    plt.title('AiAe delays')
except Exception as e:
    logger.warning(f"Failed to plot AiAe delays: {e}")
plt.tight_layout()

# Show all plots
b2.ioff()
b2.show()

