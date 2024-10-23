import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import argparse
import imageio
import os
import config
import time
from glob import glob
import concurrent.futures
import multiprocessing as mp
from queue import Empty
import signal
import sys
import utils
import pathlib
import copy

from jax.config import config as jax_config
jax_config.update("jax_enable_x64", config.enable_double_precision)
jax_config.update("jax_debug_nans", False) # warning: nan checking is expensive

# Memory preallocation must be disabled to allow jax use in threads.
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'False'
    
conv_layout = ('NHWC', 'HWIO', 'NHWC')

# ==================================================
# Handles for general utility objects & processes

data_manager = None
visualiser = None
process_pool_executor = None
visualisation_process = None
visualisation_queue = None
    
# Initialiser method for pool processes. Imports jax and sets up the SIGINT handler.
# importing jax in the initialiser is a somewhat hacky solution to allow jax use in pool processes.
# Each subprocess must import jax before the main process makes its first jax call.
# see: https://github.com/tensorflow/tensorflow/issues/8220
# Signal handler for ignoring SIGINT is added to allow clean exit on Control-C.
# Pool processes typically die on Control-C, which causes the process pool executor to hang when
# trying to exit. Letting the processes survive Control-C allows the executor to clean the pool up
# gracefully when exiting.
def process_initialiser():
    print('Initialising pool process with PID:', os.getpid())
    import jax
    signal.signal(signal.SIGINT,signal.SIG_IGN)

# ==================================================


# Dummy future object.
# Used instead of real futures when multiprocessing is disabled.
class Dummy_future:
    def __init__(self,data): self.data = data
    def result(self): return self.data
    def done(self): return True


# Convenience method for submitting tasks to the process pool if multiprocessing is enabled.
# If multiprocessing is disabled, we just run the task on the current process and return a dummy future.
def submit_pool_task(*args):
    if config.enable_multiprocessing:
        return process_pool_executor.submit(*args)
    return Dummy_future(args[0](*args[1:]))


def dense_layer_params(rng_key,nI,nO):
    """Generates weights and biases for a single dense layer.
    Weights are initialised using uniform Glorot initialisation.
    Biases are initialised to 0.
    
    Args:
        rng_key: JAX RNG key
        nI: The layer's input neuron count.
        nO: The layer's output neuron count.
        
    Returns:
        A tuple of JAX DeviceArrays holding weights and biases.
    """
    print('dense layer params with I/O', nI, nO)
    r = jnp.sqrt(6.0/(nI+nO))
    return (jax.random.uniform(rng_key,(nI,nO),minval=-r,maxval=r),jnp.zeros(nO))


# Same as dense_layer_params, with support for multiple M modules.
def dense_layer_paramsM(rng_key,nI,nO):
    print('dense layer params with I/O', nI, nO)
    r = jnp.sqrt(6.0/(nI+nO))
    return (jax.random.uniform(rng_key,(config.number_of_unique_M_modules,nI,nO),minval=-r,maxval=r),jnp.zeros((config.number_of_unique_M_modules,nO)))


def conv_layer_params(rng_key,input_shape,kernel_shape):
    """Generates weights and biases for a single convolution layer of arbitrary dimensionality.
    Weights are initialised using uniform Glorot initialisation.
    Biases are initialised to 0.
    
    Args:
        rng_key: JAX RNG key
        input_shape: Shape of the layer's input array.
        kernel_shape: Shape of the convolution kernel.
        
    Returns:
        A tuple of JAX DeviceArrays holding weights and biases.
    """
    print('conv layer params with shapes', input_shape, kernel_shape)
    n = jnp.prod(jnp.array(kernel_shape[:-1]))+kernel_shape[-1] # number of inputs + number of outputs
    r = jnp.sqrt(6.0/n) # range for Glorot uniform initialisation
    return (jax.random.uniform(rng_key,kernel_shape,minval=-r,maxval=r),jnp.zeros(kernel_shape[-1]))


# traces through the architecture and builds up the parameter list (weights & biases)
def init_module_params(rng_key,module):
    """Generates weights and biases for all layers of a module.
    
    Args:
        rng_key: JAX RNG key
        module: One of 'E', 'M', 'D'.
        
    Returns:
        A list containing the (weights,biases) tuples for every connection layer in the module.
    """
    shape = config.architectures[module][0]['shape']
    if type(shape) is int: shape = [shape]
    shape = np.array(shape)
    keys = jax.random.split(rng_key,len(config.architectures[module])-1)
    params = []
    
    for i, layer in enumerate(config.architectures[module][1:]):
        count = layer['count'] if 'count' in layer else 1
        for j in range(count):
            
            # dense layer case
            if layer['type'] == 'dense':
                if module == 'M':
                    params.append(dense_layer_paramsM(keys[i],np.prod(shape),layer['nO']))
                else:
                    params.append(dense_layer_params(keys[i],np.prod(shape),layer['nO']))
                shape = np.array([layer['nO']])
                if module == 'M':
                    shape[0] += config.state_extra_dims+config.hidden_dims+config.action_dims_at_nn_input
            
            # convolution layer case
            if layer['type'] == 'conv':
                # reshape if given (must be given when going from a dense layer into a convolution layer)
                if 'reshape_input' in layer:
                    shape = np.array(layer['reshape_input'])
                # upscale if given (using upscale&conv instead of transposed conv to avoid artefacts)
                if 'upscale' in layer:
                    shape[:-1] = np.array(shape[:-1])*np.array(layer['upscale'])
                # get the kernel weights & biases
                params.append(conv_layer_params(keys[i],shape,layer['kernel']))
                # apply striding if given
                if 'strides' in layer:
                    shape[:-1] = (shape[:-1]/np.array(layer['strides'])).astype(int)
                shape[-1] = layer['kernel'][-1]
            
    print('layer parameters for module', module, ':', len(params))
    return params


@jax.jit
def local_response_normalisation(act):
    sqr_sum = jnp.sum(act**2,axis=-1,keepdims=True)
    bias, alpha, beta = 1, 1, .5
    return act / (bias + alpha * sqr_sum) ** beta

# E module
@jax.jit
def E(params,state):
    """Applies the E (Encoder) module of the pEM*D architecture.
    
    Args:
        params: List of (weights,biases) tuples for all layers of the E module.
        state: The state to be encoded. Format should match config.state_dims_in.
        
    Returns:
        Latent representation of the input state.
    """
    activation = state.reshape(config.state_dims_in)
    for i, layer in enumerate(config.architectures['E'][1:]):
        
        w, b = params[i]
        
        if layer['type'] == 'conv':
            if 'reshape_input' in layer:
                activation = jnp.reshape(activation,layer['reshape_input'])
            if 'upscale' in layer:
                for axis, repeats in enumerate(layer['upscale']):
                    activation = jnp.repeat(activation,repeats,axis)
            if 'strides' in layer:
                strides = layer['strides']
            else:
                strides = [1]*(activation.ndim-2)
            activation = jax.lax.conv_general_dilated(activation[None],w,strides,'SAME',dimension_numbers=conv_layout)[0]
            activation = config.activation_function_hidden(activation+b)
        
        if layer['type'] == 'dense':
            activation = activation.reshape(-1)
            activation = config.activation_function_hidden(jnp.dot(activation,w)+b)
            
        if config.use_local_response_normalisation_ED:
            if i < len(params)-1:
                activation = local_response_normalisation(activation)
                
    if config.use_local_response_normalisation_latents:
        activation = local_response_normalisation(activation)
        
    return activation


# M module
@jax.jit
def M(params,state,state_extra,hidden,action,M_index,memory=None,full_state=None):
    """Applies the M (Manipulation) module of the pEM*D architecture.
    
    Args:
        params: List of (weights,biases) tuples for all layers of the M module.
        state: Latent representation of the pre-manipulation state.
        state_extra: Any additional state variables that are deterministically calculated outside the net.
        hidden: Hidden variable estimate, represented as an array of length config.hidden_dims.
        action: A single action, represented as an array of length config.action_dims.
        M_index: Index of the M-submodule to use for performing prediction in latent space.
        memory: Pre-manipulation memory state. If None, the zero vector is used.
        full_state: A full representation of the pre-manipulation state (to be passed on to data_manager.apply_action_modification).
                    Can be omitted if data_manager.apply_action_modification is not used.
        
    Returns:
        - The latent representation of the post-manipulation state.
        - The post-manipulation memory trace.
        - The post-manipulation auxilliary outputs.
        
    """
    if memory is None:
        memory = jnp.zeros(config.M_memory_bandwidth)
    
    if full_state is not None:
        action = data_manager.apply_action_modification(full_state,action)
    
    stored = [state for skip in config.skips]
    activation = jnp.concatenate((state,hidden,state_extra,memory,action),axis=0)
    for i, (w, b) in enumerate(params):
        raw_activation = jnp.dot(activation, w[M_index]) + b[M_index]
        activation = config.activation_function_hidden(raw_activation)
        
        n_incoming = 1 if (i==0 or not config.scale_by_incoming_branches) else max(1,sum([i%skip==0 for skip in config.skips]))
        for s, skip in enumerate(config.skips[::-1]):
            if (i+1) % skip == 0:
                activation = activation.at[:config.M_state_bandwidth].add(stored[s]/n_incoming)
                if config.only_one_incoming_skip_per_layer:
                    break
                
        for s, skip in enumerate(config.skips):
            if (i+1) % skip == 0:
                stored[s] = activation[:config.M_state_bandwidth]
        
        if i < len(params)-1:
            activation = jnp.concatenate((activation,hidden,state_extra,action),axis=0)
        
    state_out = activation[:config.M_state_bandwidth]
    memory_out = activation[config.M_state_bandwidth:config.M_state_bandwidth+config.M_memory_bandwidth]
    aux_out = config.activation_function_out(raw_activation[config.M_state_bandwidth+config.M_memory_bandwidth:])

    if config.use_local_response_normalisation_latents and not config.differential_prediction:
        state_out = local_response_normalisation(state_out)
        
    return state_out, memory_out, aux_out


# M* - multiple applications of M module
@jax.jit
def M_star(params,state,state_extra_sequence,hidden,action_sequence,M_index_sequence,memory=None):
    """Applies multiple instances of the M (Manipulation) module of the pEM*D architecture in sequence.
    
    Args:
        params: List of (weights,biases) tuples for all layers of the M module.
        state: Latent representation of the pre-manipulation state.
        state_extra_sequence: A sequence of n state_extra values (see M), represented as an array of size (n,config.state_extra_dims).
        hidden: Hidden variable estimate, represented as an array of length config.hidden_dims.
        action_sequence: A sequence of n actions, represented as an array of size (n,config.action_dims).
        M_index_sequence: A sequence of M-submodule indices (one per action) to use for performing prediction in latent space. Size (n).
        
    Returns:
        A list of tuples consisting of:
        - The latent representation of the post-manipulation state.
        - The latent representation of the state differential between pre- and post-manipulation state (None if prediction is not differential).
        - The post-manipulation memory trace.
        - The post-manipulation auxilliary outputs.
    """
    output_sequence = []
    for action, state_extra, M_index in zip(action_sequence,state_extra_sequence,M_index_sequence):
        
        if config.differential_prediction:
            differential, memory, aux = M(params,state,state_extra,hidden,action,M_index,memory)
            state += differential
            if config.use_local_response_normalisation_latents:
                state = local_response_normalisation(state)
            output_sequence.append((state,differential,memory,aux))
        else:
            state, memory, aux = M(params,state,state_extra,hidden,action,M_index,memory)
            output_sequence.append((state,None,memory,aux))

    return output_sequence


# (MD)* - multiple applications of M,D modules
@jax.jit
def MD_star(params_M,params_D,state,state_extra_sequence,hidden,action_sequence,M_index_sequence,memory=None,full_state=None):
    """Applies multiple instances of the M (Manipulation) module of the pEM*D architecture in sequence, 
       following each pass through M with a pass through D to obtain a full representation of each predicted state.
    
    Args:
        params: List of (weights,biases) tuples for all layers of the M module.
        state: Latent representation of the pre-manipulation state.
        state_extra_sequence: A sequence of n state_extra values (see M), represented as an array of size (n,config.state_extra_dims).
        hidden: Hidden variable estimate, represented as an array of length config.hidden_dims.
        action_sequence: A sequence of n actions, represented as an array of size (n,config.action_dims).
        M_index_sequence: A sequence of M-submodule indices (one per action) to use for performing prediction in latent space. Size (n).
        
    Returns:
        A list of tuples consisting of:
        - The latent representation of the post-manipulation state.
        - The latent representation of the state differential between pre- and post-manipulation state (None if prediction is not differential).
        - The post-manipulation memory trace.
        - The post-manipulation auxilliary outputs.
        - The full representation of the post-manipulation state.
    """
    full_predicted_state = full_state
    
    output_sequence = []
    for i, (action, state_extra, M_index) in enumerate(zip(action_sequence,state_extra_sequence,M_index_sequence)):
        if config.differential_prediction:
            differential, memory, aux = M(params_M,state,state_extra,hidden,action,M_index,memory,full_predicted_state)
            state += differential
            if config.use_local_response_normalisation_latents:
                state = local_response_normalisation(state)
            full_predicted_diff = D(params_D,state)
            r = full_predicted_diff[...,-1:]
            full_predicted_state = (1.0-r)*full_predicted_state+r*full_predicted_diff[...,:config.state_dims_out[-1]]
            output_sequence.append((state,differential,memory,aux,full_predicted_state))
        else:
            state, memory, aux = M(params,state,state_extra,hidden,action,M_index,memory,full_predicted_state)
            full_predicted_state = D(state)
            output_sequence.append((state,None,memory,aux,full_predicted_state))
    
    return output_sequence

# D module
@jax.jit
def D(params,state):
    """Applies the D (Decoder) module of the pEM*D architecture.
    
    Args:
        params: List of (weights,biases) tuples for all layers of the D module.
        state: The latent state to be decoded.
        
    Returns:
        Decoded (i.e. full representation of) state.
    """
    activation = state
    for i, layer in enumerate(config.architectures['D'][1:]):
        w, b = params[i]
        if layer['type'] == 'conv':
            if 'reshape_input' in layer:
                activation = jnp.reshape(activation,layer['reshape_input'])
            if 'upscale' in layer:
                for axis, repeats in enumerate(layer['upscale']):
                    activation = jnp.repeat(activation,repeats,axis)
            if 'strides' in layer:
                strides = layer['strides']
            else:
                strides = [1]*(activation.ndim-1)
            activation = jax.lax.conv_general_dilated(activation[None],w,strides,'SAME',dimension_numbers=conv_layout)[0]
            activation = config.activation_function_hidden(activation+b)
        
        if layer['type'] == 'dense':
            activation = activation.reshape(-1)
            activation = jnp.dot(activation,w)+b
        
        if i < len(params)-1:
            activation = config.activation_function_hidden(activation)
            if config.use_local_response_normalisation_ED:
                activation = local_response_normalisation(activation)
    
    # reshape to output state format
    activation = activation.reshape(config.nn_state_dims_out)
    
    activation = config.activation_function_out(activation)
    
    return activation

# auto-batched application of D to batch over steps
batched_D = jax.vmap(D,in_axes=(None,0))


# forward pass through full pEM*D net
@jax.jit
def _predict_sequence(params,state_in,state_extra_sequence,hidden,action_sequence,M_index_sequence,memory=None):
    """Predicts the state sequence for a given action sequence performed from a given state, by applying the full pEM*D architecture.
    
    Args:
        params: List of (weights,biases) tuples for all layers of the M module.
        state: Latent representation of the pre-manipulation state.
        state_extra_sequence: A sequence of n state_extra values (see M), represented as an array of size (n,config.state_extra_dims).
        hidden: Hidden variable estimate, represented as an array of length config.hidden_dims.
        action_sequence: A sequence of n actions, represented as an array of size (n,config.action_dims).
        M_index_sequence: A sequence of M-submodule indices (one per action) to use for performing prediction in latent space. Size (n).
        
    Returns:
        - The sequence of latent states (initial state and predicted states).
        - The sequence of auxilliary outputs.
        - The sequence of memory traces.
    """
    
    latent_state_in = E(params[0],state_in)
    prediction_seq = MD_star(params[1],params[2],latent_state_in,state_extra_sequence,hidden,action_sequence,M_index_sequence,memory,state_in)
    
    latent_state_seq = jnp.array([items[0] for items in prediction_seq])
    latent_diff_seq = jnp.array([items[0] for items in prediction_seq])
    memory_seq = jnp.array([items[2] for items in prediction_seq])
    aux_seq = jnp.array([items[3] for items in prediction_seq])
    state_seq = jnp.array([items[4] for items in prediction_seq])

    return state_seq, aux_seq, memory_seq


def predict_sequence(params,state_in,state_extra_sequence,hidden,action_sequence,M_index_sequence,memory,return_memory=False):
    if M_index_sequence is None:
        M_index_sequence = jnp.zeros(action_sequence.shape[0],dtype=int)
    if hidden is None:
        hidden = jnp.zeros(config.hidden_dims)
    if state_extra_sequence is None:
        state_extra_sequence = jnp.zeros((action_sequence.shape[0],config.state_extra_dims))
    state_seq, aux_seq, memory_seq = _predict_sequence(params,state_in,state_extra_sequence,hidden,action_sequence,M_index_sequence,memory)
    if return_memory:
        return state_seq, aux_seq, memory_seq
    else:
        return state_seq, aux_seq

# auto-batched sequence prediction
_batched_predict_sequence = jax.vmap(predict_sequence,in_axes=(None,0,0,0,0,0,0,None))

def batched_predict_sequence(params,state_in,state_extra_sequence,hidden,action_sequence,M_index_sequence,memory=None,return_memory=False):
    n = state_in.shape[0]
    if memory is None:
        memory = np.zeros((n,config.M_memory_bandwidth))
    if M_index_sequence is None:
        M_index_sequence = jnp.zeros((n,action_sequence.shape[1]),dtype=int)
    if hidden is None:
        hidden = jnp.zeros((n,config.hidden_dims))
    if state_extra_sequence is None:
        state_extra_sequence = jnp.zeros((n,action_sequence.shape[1],config.state_extra_dims))
    return _batched_predict_sequence(params,state_in,state_extra_sequence,hidden,action_sequence,M_index_sequence,memory,return_memory)
    


# Set up loss functions
@jax.jit
def mse_loss(predicted,target):
    losses = (predicted-target)**2
    while losses.ndim>1:
        losses = jnp.mean(losses,axis=1)
    return jnp.mean(losses), losses

# check for custom losses
import importlib
data_manager_module = '.'.join(config.data_manager_class.split('.')[:-1])
data_manager_class = config.data_manager_class.split('.')[-1]
dm_module = importlib.import_module(data_manager_module)
dm_class = getattr(dm_module,data_manager_class)

training_loss_state = dm_class.state_loss_training if config.custom_state_loss_for_training else mse_loss
training_loss_aux = dm_class.aux_loss_training if config.custom_aux_loss_for_training else mse_loss


# network parameter update for training
@jax.jit
def training_update(params,learning_rate,grad_sets):
    for m in range(3):
        for i, (w,b) in enumerate(params[m]):
            dw = jnp.sign(jnp.sum(jnp.sign(jnp.array([gg[i][0] for gg in grad_sets[m]])),axis=0))
            db = jnp.sign(jnp.sum(jnp.sign(jnp.array([gg[i][1] for gg in grad_sets[m]])),axis=0))
            params[m][i] = (w-learning_rate*dw, b-learning_rate*db)
    return params


# Prune batch for jit-compiled code.
# Prunes a batch to leave only the elements necessary for NN processing.
# This removes elements that would cause jax to throw errors in jit-compiled code.
def _jax_batch(batch):
    keep = ('states',
            'goal_state',
            'current_state',
            'past_states',
            'state_extras',
            'past_state_extras',
            'hidden',
            'actions',
            'past_actions',
            'M_indices',
            'past_M_indices',
            'auxes',
            'n_steps',
            'past_n_steps',
            'current_step',
            'u_means',
            'u_stds',
            'fA_data',
            'fA_GT')
    jax_batch = {key:item for key, item in batch.items() if key in keep}
    return jax_batch


if config.aux_dims:
    
    # If the net has auxilliary outputs batch processing is split into two stages. This is a work-around to
    # obtain grads for two losses while avoiding forward propagation with the same batch twice. Stage 1 first
    # calls stage 2. Stage 2 runs forward prediction once, obtaining predictions for the state and aux elements.
    # It obtains the loss and gradient for the state output. Loss, gradient, and predictions are passed back to
    # stage 1, which calculate loss and gradient for the aux element, using the prediction obtained in stage 2.
    
    # sub-stage 2 for process_batch
    @jax.jit
    def process_batch_for_training_stage2(params,batch):
        batch_predicted_state, batch_predicted_aux = \
            batched_predict_sequence(params,batch['states'][:,0],batch['state_extras'],batch['hidden'],batch['actions'],batch['M_indices'])
        loss_state, loss_list_state = training_loss_state(batch_predicted_state,batch['states'][:,1:])
        return loss_state, (batch_predicted_state, batch_predicted_aux)

    # sub-stage 1 for process_batch
    @jax.jit
    def process_batch_for_training_stage1(params,batch):
        (loss_state, (batch_predicted_state,batch_predicted_aux)), grads_state = \
            jax.value_and_grad(process_batch_for_training_stage2,has_aux=True)(params,batch)
        loss_aux, loss_list_aux = training_loss_aux(batch_predicted_aux,batch['auxes'])
        return loss_aux, (loss_state, batch_predicted_state, batch_predicted_aux, grads_state)

    @jax.jit
    def _process_batch_for_training(params,batch):
        (loss_aux, (loss_state, batch_predicted_state, batch_predicted_aux, grads_state)), grads_aux = \
            jax.value_and_grad(process_batch_for_training_stage1,has_aux=True)(params,batch)
        return (loss_state,loss_aux), (batch_predicted_state,batch_predicted_aux), (grads_state,grads_aux)
else:
    @jax.jit
    def process_batch_for_training_main(params,batch):
        batch_predicted_state, batch_predicted_aux = \
            batched_predict_sequence(params,batch['states'][:,0],batch['state_extras'],batch['hidden'],batch['actions'],batch['M_indices'])
        loss_state, loss_list_state = training_loss_state(batch_predicted_state,batch['states'][:,1:])
        return loss_state, (batch_predicted_state, batch_predicted_aux)
    
    @jax.jit
    def _process_batch_for_training(params,batch):
        (loss_state, (batch_predicted_state, batch_predicted_aux)), grads_state = \
            jax.value_and_grad(process_batch_for_training_main,has_aux=True)(params,batch)
        loss_aux, batch_predicted_aux, grads_aux = None, None, None
        return (loss_state,loss_aux), (batch_predicted_state,batch_predicted_aux), (grads_state,grads_aux)

process_batch_for_training = lambda params, batch: _process_batch_for_training(params,_jax_batch(batch))


@jax.jit
def calculate_epistemic_uncertainty(predicted_states,batch,uncertainty_step_weights=1.0):
    u_means = batch['u_means']
    u_stds = batch['u_stds']
    sub = jnp.abs(predicted_states[0][...,:3]-predicted_states[1][...,:3])
    sub_confidence_list = sub.mean((2,3))
    sub_confidence_list = (sub_confidence_list-u_means[None,:])/(u_stds[None,:]+config.std_epsilon)
    sub_confidence_list = config.conf_function(sub_confidence_list)*uncertainty_step_weights
    return sub_confidence_list.mean(1)

# -------------------------------------------

def fetch_meta_data_from_batch(batch,i,s):
    if 'meta_data' not in batch:
        return None
    if len(batch['meta_data'][i]) == 1:
        return batch['meta_data'][i][0]
    return batch['meta_data'][i][s]


def draw_training_image(iteration,batch,predicted,scores):
    
    print('Drawing visual for', batch['subset'], 'batch...')

    state_predictions, aux_predictions = predicted
    n_examples, n_steps = batch['states'].shape[:2]

    column = []
    for i in range(min(config.n_examples_in_training_image,n_examples)):

        # draw panel row for ground truth sequence
        panel_row = []
        for s in range(n_steps):
            state = batch['states'][i,s].copy()
            state_extras = batch['state_extras'][i].copy()
            hidden = batch['hidden'][i].copy()
            action = batch['actions'][i,s].copy() if s<n_steps-1 else None
            M_index = batch['M_indices'][i,s].copy() if s<n_steps-1 else None
            aux = batch['auxes'][i,s-1].copy() if s>0 else None
            meta_data = fetch_meta_data_from_batch(batch,i,s)
            panel = visualiser.draw_panel(False,state,state_extras,hidden,action,M_index,aux,meta_data)
            panel = np.pad(panel,((config.image_panel_separation//2,config.image_panel_separation//2),(0,0),(0,0)),mode='constant')
            panel_row.append(panel)
            
        panel_row = np.concatenate(panel_row,axis=0)
        panel_row = np.pad(panel_row,((0,0),(0,config.image_panel_separation),(0,0)),mode='constant')
        column.append(panel_row)
    
        # draw panel row for predicted sequence
        panel_row = []
        for s in range(n_steps):
            state = np.array(state_predictions[i,s-1]) if s>0 else batch['states'][i,0]
            state_extras = batch['state_extras'][i].copy()
            hidden = batch['hidden'][i].copy()
            action = batch['actions'][i,s].copy() if s<n_steps-1 else None
            M_index = batch['M_indices'][i,s].copy() if s<n_steps-1 else None
            if config.aux_dims:
                aux = np.array(aux_predictions[i,s-1]) if s>0 else None
            else:
                aux = None
            score_list = scores[i,s-1] if s>0 else None
            meta_data = fetch_meta_data_from_batch(batch,i,s)
            panel = visualiser.draw_panel(s>0,state,state_extras,hidden,action,M_index,aux,meta_data,score_list)
            panel = np.pad(panel,((config.image_panel_separation//2,config.image_panel_separation//2),(0,0),(0,0)),mode='constant')
            panel_row.append(panel)
            
        panel_row = np.concatenate(panel_row,axis=0)
        panel_row = np.pad(panel_row,((0,0),(0,config.image_example_separation),(0,0)),mode='constant')
        column.append(panel_row)

    im = np.concatenate(column,axis=1).swapaxes(0,1)
    
    # multiply by 255 if colour range is [0,1]
    if im.max() <= 1.0:
        im *= 255
    
    # clip to valid range
    im = np.clip(im,0,255)
    
    # save image
    imageio.imwrite(args.run_name+'/training_images/'+batch['subset']+'/'+str(iteration).zfill(7)+'.png',np.uint8(im))


def batch_process(rng_key,subset,n_examples,augmentation=True):
    batch = data_manager.make_batch(rng_key,subset,n_examples,augmentation)
    batch = utils.check_and_complete_batch(batch,subset,n_examples)
    return batch


def request_batch(rng_key,subset,n_examples=1,augmentation=True):
    future_batch = submit_pool_task(batch_process,rng_key,subset,n_examples,augmentation)
    return future_batch


def fetch_batch(key,dataset,future_batches,augmentation=True):
    future_batches[dataset].append(request_batch(key,dataset,config.n_batch,augmentation))
    batch_future = future_batches[dataset].pop(0)
    done = batch_future.done()
    return batch_future.result(), done
    
    
# save current state of a training run
def save_training_state(iteration,params,learning_rate,best_validation_loss,stale_count,initial_rng_key,current_rng_key,best=False):
    
    if best:
        marker = '(best)'
        n_keep = 0
    else:
        marker = ''
        n_keep = config.n_old_run_files_to_keep
    
    # find old files for removal
    if n_keep >= 0:
        old_run_state_files = sorted(glob(args.run_name+'/networks/'+marker+'iteration_[0-9]*.npz'))
    
    print('')
    if best:
        print('Saving new validation-best:')
    else:
        print('Saving run state:')
    print('    Iteration:', iteration)
    print('    Learning rate:', learning_rate)
    print('    Best validation loss:', best_validation_loss)
    print('    Stale count:', stale_count)
    print('    Initial RNG key:', initial_rng_key)
    print('    Current RNG key:', current_rng_key)
    file_path = args.run_name+'/networks/'+marker+'iteration_'+str(iteration).zfill(7)
    np.savez(file_path,
             iteration=iteration,
             params=params,
             learning_rate=learning_rate,
             best_validation_loss=best_validation_loss,
             stale_count=stale_count,
             initial_rng_key=initial_rng_key,
             current_rng_key=current_rng_key,
             )
    print('Run state saved to:', file_path)
    
    if n_keep >= 0:
        if n_keep > 0:
            n_files_to_remove = max(0,len(old_run_state_files)-n_keep)
            old_run_state_files = old_run_state_files[:n_files_to_remove]
        for old_file_path in old_run_state_files:
            print('Discarding old state file:', old_file_path)
            os.remove(old_file_path)
            
    print('')
    
    
def init_net(rng_key):
    print('Initialising blank network.')
    rng_key, key = jax.random.split(rng_key)
    params_E = init_module_params(key,'E')
    rng_key, key = jax.random.split(rng_key)
    params_M = init_module_params(key,'M')
    rng_key, key = jax.random.split(rng_key)
    params_D = init_module_params(key,'D')
    params = [params_E, params_M, params_D]
    return params
    
    
def load_or_init_training_state(run_name,iteration=None,allow_new=True,validation_best=False):
    
    marker = '(best)' if validation_best else ''
    if iteration is None:
        # if no iteration is given, find all existing files for the run
        run_files = sorted(glob(run_name+'/networks/'+marker+'iteration_[0-9]*.npz'))
        # if any files are found, use the file with the highest iteration number. Otherwise, 
        if len(run_files):
            file_path = run_files[-1]
        else:
            print('No state files found for run ', run_name, '.', sep='')
            if not allow_new:
                print('*'*60)
                print("ERROR: No pre-trained network found for run name:", run_name, '.\n',
                      "Planning mode requires a pre-trained network.", sep="")
                print('*'*60)
                raise ValueError('Network file not found')
            initial_rng_key = utils.init_rng()
            rng_key, key = jax.random.split(initial_rng_key)
            params = init_net(key)
            return initial_rng_key, rng_key, 0, params, config.initial_learning_rate, np.inf, 0, time.time()
            
    else:
        file_path = run_name+'/networks/'+marker+'iteration_'+str(iteration).zfill(7)+'.npz'
        if not os.path.exists(file_path):
            print('*'*60)
            print('ERROR: No run state file found for run ', run_name, ', iteration ', iteration, '.', sep='')
            print('*'*60)
            raise ValueError('Network file not found')
    print('Restoring state from file:', file_path)
    file_name = file_path.split('/')[-1]
    iteration = int(''.join(c for c in file_name if c.isdigit()))
    print('iteration:',iteration)
    print('loading network parameters from:', file_path)
    run_state = np.load(file_path,allow_pickle=True)
    params = [[(jnp.array(w),jnp.array(b)) for (w,b) in module] for module in run_state['params']]
    learning_rate = run_state['learning_rate']
    best_validation_loss = run_state['best_validation_loss']
    stale_count = run_state['stale_count']
    initial_rng_key = run_state['initial_rng_key']
    current_rng_key = run_state['current_rng_key']
    
    modification_time = pathlib.Path(file_path).stat().st_mtime
    
    #TODO: return less stuff here for non-training modes
    return initial_rng_key, current_rng_key, iteration, params, learning_rate, best_validation_loss, stale_count, modification_time


def score_prediction_batch(predictions,batch):
    scores = np.array(jax.vmap(jax.vmap(data_manager.calculate_scores))(predictions[0],batch['states'][:,1:],predictions[1],batch['auxes']))
    return scores.transpose((1,2,0))


# Opens a log file.
# If continuing an existing run, we open the existing log file and truncate it at the current iteration.
def open_log_file(name,iteration,custom=False):
    path = args.run_name+'/'+name
    if iteration>0:
        print('attempting to open log file:',path)
        log_file = open(path,'r+',buffering=1)
        line = log_file.readline()
        while len(line):
            line = log_file.readline()
            i = int(line.split(' ')[0])
            if i >= iteration:
                log_file.truncate(log_file.tell()) # truncating with tell() avoids unexpected truncation behaviour
                log_file.seek(0,2)
                break
    else:
        log_file = open(path,'w+',buffering=1)
        if custom:
            log_file.write('iteration'+''.join([' train/'+s for s in config.custom_scores]+[' test/'+s for s in config.custom_scores])+'\n')
        else:
            log_file.write('iteration train/state train/aux test/state test/aux validation learning_rate\n')
    return log_file


def set_up_process_pool(n_processes):
    if not config.enable_multiprocessing: return
    global process_pool_executor
    print('Main process (PID: ', os.getpid(), ') sets up process pool.',sep='')
    process_pool_executor = concurrent.futures.ProcessPoolExecutor(max_workers=n_processes,initializer=process_initialiser)
    # Call pool with a dummy task to force the initialisation (jax import must happen before the maizn thread calls jax).
    process_pool_executor.map(lambda x:x,[None]*n_processes,chunksize=1)
    print('process pool initialised with', config.n_batch_generation_processes, 'processes.')


def clear_queue(queue):
    try:
        while True:
            queue.get_nowait()
    except Empty:
        pass


def set_up_visualisation_process(n_steps):
    global visualisation_process
    global visualisation_queue
    
    # If there was a visualisation process running, we shut it down.
    # Relaunching the process rebuilds the visualisation window.
    # This is useful because the number of steps can vary per plan.
    if visualisation_process is not None:
        visualisation_queue.put(('shutdown',))
        clear_queue(visualisation_queue)
    
    # Set up communication queue and process
    visualisation_queue = mp.Queue()
    visualisation_process = Visualisation_process(visualisation_queue,n_steps)
    visualisation_process.start()

# ==================================================


# Trains the network
def run_training(args):
    
    global visualiser
    visualiser = utils.get_visualiser()
    set_up_process_pool(config.n_batch_generation_processes)
    print('pool setup ok')
    
    # initialise network params (blank or some file)
    initial_rng_key, rng_key, from_iteration, params, learning_rate, best_validation_loss, stale_count, modification_time = load_or_init_training_state(args.run_name,args.iteration,allow_new=True)
    print('network initialisation ok')
    
    if args.lr is not None:
        learning_rate = args.lr
        print('learning rate forced to:', learning_rate)

    if from_iteration == 0:
        # make run directories
        print('Setting up directories...')
        utils.makedir(args.run_name)
        utils.makedir(args.run_name+'/source')
        utils.makedir(args.run_name+'/networks')
        utils.makedir(args.run_name+'/training_images/')
        utils.makedir(args.run_name+'/training_images/train')
        utils.makedir(args.run_name+'/training_images/test')
        print('directory setup ok')
        print('copying source files...')
        utils.copy_source_files(args.run_name+'/source')
        print('source files copied')
        print()
        
    # open train log files
    print('set up log files...')
    loss_log = open_log_file('loss_log.csv',from_iteration)
    if config.custom_scores:
        score_log = open_log_file('score_log.csv',from_iteration,True)
    print('log files ok')
        
    # Request initial train and test batches
    future_batches = {'train':[],'test':[]}
    for dataset in ('train','test'):
        n = config.n_batch_generation_processes if dataset == 'train' else 1
        for i in range(n):
            rng_key, key = jax.random.split(rng_key)
            future_batches[dataset].append(request_batch(key,dataset,config.n_batch))

    print('JAX is running on', jax.lib.xla_bridge.get_backend().platform)

    iteration = from_iteration+1
    batch_time_cost_sum = 0
    update_time_cost_sum = 0
    n_late_batches = 0
    
    # Start main training loop (using while loop instead of for loop to allow manipulation of the iteration variable)
    while iteration < config.n_training_iterations+1:
        loss_log_string = str(iteration).zfill(7)
        rng_key, key = jax.random.split(rng_key)
        t = time.time()
        rng_key, key = jax.random.split(rng_key)
        train_batch, was_done = fetch_batch(key,'train',future_batches)
        
        if not was_done:
            n_late_batches += 1
        
        batch_time_cost = time.time()-t
        batch_time_cost_sum += batch_time_cost
        
        t = time.time()
        losses, predicted, grads = process_batch_for_training(params,train_batch)
        
        loss_state, loss_aux = losses
        if np.isnan(loss_state) or (loss_aux is not None and np.isnan(loss_aux)):
            print('*'*60)
            print('ERROR: NaN value in loss calculation.')
            print('NaN in state loss:', np.isnan(loss_state))
            if config.aux_dims:
                print('NaN in aux loss:', np.isnan(loss_aux))
            print('*'*60)
            iteration -= 1
            continue
        
        grads_state, grads_aux = grads
        if config.aux_dims:
            params = training_update(params,learning_rate,((grads_state[0],grads_aux[0]),
                                                           (grads_state[1],grads_aux[1]),
                                                           (grads_state[2],)))
        else:
            params = training_update(params,learning_rate,((grads_state[0],),
                                                           (grads_state[1],),
                                                           (grads_state[2],)))
        update_time_cost = time.time()-t
        update_time_cost_sum += update_time_cost
        
        loss_log_string += ' '+str(loss_state)+' '+str(loss_aux)
        
        if (iteration%config.training_print_interval == 0) or (iteration == 1):
            batch_predicted_state, batch_predicted_aux = predicted
            print('\n', iteration)
            print('[TRAIN LOSS] state:', loss_state, 'loss aux:', loss_aux)
            print('Batch waiting time:  ', np.round(batch_time_cost,4), 
                  '( mean:', np.round(batch_time_cost_sum/config.training_print_interval,5), 
                  '/ late rate:', np.round(n_late_batches/config.training_print_interval,5),')')
            print('Training update time:', np.round(update_time_cost,4),
                  '( mean:', np.round(update_time_cost_sum/config.training_print_interval,5),')')
            print('Batch step count:', train_batch['n_steps'])
            batch_time_cost_sum = 0
            update_time_cost_sum = 0
            n_late_batches = 0
            
        if iteration%config.training_test_interval == 0:
            rng_key, key = jax.random.split(rng_key)
            test_batch, _ = fetch_batch(key,'test',future_batches,augmentation=False)
            test_losses, test_predicted, _grads = process_batch_for_training(params,test_batch)
            test_loss_state, test_loss_aux = test_losses
            loss_log_string += ' '+str(test_loss_state)+' '+str(test_loss_aux)
            
            print('[TEST LOSS] state:', test_loss_state, 'loss aux:', test_loss_aux)
            print('Batch step count:', test_batch['n_steps'])
            
            if config.custom_scores:
                scores_train = score_prediction_batch(predicted,train_batch)
                scores_test = score_prediction_batch(test_predicted,test_batch)
                print()
                
                seq_mean_scores_train = scores_train.mean((0,1))
                seq_mean_scores_test = scores_test.mean((0,1))
                final_mean_scores_train = scores_train[:,-1].mean((0))
                final_mean_scores_test = scores_test[:,-1].mean((0))
                print('Custom scores:')
                print('  [TRAIN SCORES]',train_batch['n_steps'],'step(s):')
                print('    full sequence:\n', *['      '+name+': '+str(score)+'\n' for (name,score) in zip(config.custom_scores,seq_mean_scores_train)],end='')
                print('    final outcome:\n', *['      '+name+': '+str(score)+'\n' for (name,score) in zip(config.custom_scores,final_mean_scores_train)],end='')
                print('  [TEST SCORES]',test_batch['n_steps'],'step(s):')
                print('    full sequence:\n', *['      '+name+': '+str(score)+'\n' for (name,score) in zip(config.custom_scores,seq_mean_scores_test)],end='')
                print('    final outcome:\n', *['      '+name+': '+str(score)+'\n' for (name,score) in zip(config.custom_scores,final_mean_scores_test)],end='')
                print()
                
                string_train = ''.join([' '+str(s) for s in final_mean_scores_train])
                string_test = ''.join([' '+str(s) for s in final_mean_scores_test])
                score_log_string = str(iteration).zfill(7)+string_train+string_test
                # write score log string
                score_log.write(score_log_string+'\n')
                # forced flushing
                score_log.flush()
                os.fsync(score_log.fileno())
            else:
                scores_train = None
                scores_test = None
            
            # draw training image for train batch
            draw_training_image(iteration,train_batch,predicted,scores_train)
            # draw training image for test batch
            draw_training_image(iteration,test_batch,test_predicted,scores_test)
            
        else:
            loss_log_string += ' - -' # 'no value' indicator
            
        if iteration%config.training_validation_interval == 0:
            # evaluate prediction performance on the validation set
            set_to_use = 'validation'
            print('Evaluating performance on', set_to_use, 'set...')
            total_validation_loss = 0
            n_examples = data_manager.get_example_count(set_to_use)
            times = []
            for i_example in range(n_examples):
                utils.print_progress(i_example,n_examples)
                example = utils.request_example(set_to_use,i_example)
                select_steps = lambda item, from_step: item[:,from_step:] if hasattr(item, "ndim") and item.ndim>=2 else item
                for from_step in range(example['n_steps']):
                    sub_example = {key: select_steps(item,from_step) for (key, item) in example.items()}
                    
                    t = time.time()
                    validation_losses, validation_predicted, _grads = process_batch_for_training(params,sub_example)
                    times.append(time.time()-t)
                    validation_loss_state, validation_loss_aux = validation_losses
                    total_validation_loss += validation_loss_state
                
            utils.print_progress()
            print('  Validation loss:',total_validation_loss)
            
            learning_rate_string = '-'
            if total_validation_loss < best_validation_loss:
                print('  Validation loss improved:',best_validation_loss,'-->',total_validation_loss)
                best_validation_loss = total_validation_loss
                print('  Stale count:', stale_count, '-->', 0)
                stale_count = 0
                save_training_state(iteration,params,learning_rate,best_validation_loss,stale_count,initial_rng_key,rng_key,best=True)
            else:
                print('  No improvement (best: ',best_validation_loss,')',sep='')
                print('  Stale count:', stale_count, '-->', stale_count+1)
                stale_count += 1
                print('  Stale count:',stale_count,'/',config.stale_count_limit)
                if stale_count >= config.stale_count_limit:
                    print('  Reducing learning rate:',learning_rate,'-->',learning_rate*config.learning_rate_reduction_factor)
                    learning_rate *= config.learning_rate_reduction_factor
                    print('  Stale count:', stale_count, '-->', 0)
                    stale_count = 0
                    learning_rate_string = str(learning_rate)
                    
            loss_log_string += ' '+str(total_validation_loss)+' '+learning_rate_string
        else:
            loss_log_string += ' - -' # 'no value' indicator
            
        if iteration%config.training_save_interval == 0:
            save_training_state(iteration,params,learning_rate,best_validation_loss,stale_count,initial_rng_key,rng_key)
        
        # write loss log string
        loss_log.write(loss_log_string+'\n')
        # forced flushing
        loss_log.flush()
        os.fsync(loss_log.fileno())
        iteration += 1


class NN(object):
    
    def __init__(self,run_name,iteration=None,validation_best=False):
        
        global data_manager
        data_manager = utils.get_data_manager(False,False) # arguments will be ignored if a manager exists already
        
        self.run_name = run_name
        self.dual_mode = False
                
        print('loading net from run:', run_name)
        self.params = load_or_init_training_state(run_name,allow_new=False,iteration=iteration,validation_best=validation_best)[3]
        
        
    def predict(self,state_in,action_sequence,state_extra_sequence=None,M_index_sequence=None,hidden=None,memory=None,return_memory=False,net_index=None):
        
        if state_in.shape == config.state_dims_in:
            predicted_states, predicted_auxes = predict_sequence(self.params,state_in,state_extra_sequence,hidden,action_sequence,M_index_sequence,memory,return_memory)
            return np.array(predicted_states), np.array(predicted_auxes)
        if state_in.shape[1:] == config.state_dims_in:
            predicted_states, predicted_auxes = batched_predict_sequence(self.params,state_in,state_extra_sequence,hidden,action_sequence,M_index_sequence,memory,return_memory)
            return np.array(predicted_states), np.array(predicted_auxes)
        print('ERROR: bad state input shape:', state_in.shape)
        return None, None
            
        
    def shutdown(self):
        if visualisation_process is not None:
            visualisation_queue.put(('shutdown',))
            clear_queue(visualisation_queue)
            visualisation_queue.close()
        

if __name__ == '__main__':

    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('run_name',type=str,
                        help='Run name. If no run by the given name exists, a new run is started.')
    parser.add_argument('-i','--iteration',type=int,default=None,
                        help='Iteration from which to continue training (defaults to highest iteration found).')
    parser.add_argument('-r','--lr',type=float,default=None,
                        help='Initial training rate (defaults to highest iteration found).')
    args = parser.parse_args()
    
    print('Run name:', args.run_name)
    
    data_manager = utils.get_data_manager(True,True)

    try:
        run_training(args)
    except KeyboardInterrupt:
        print('\n')
        print('*'*60)
        print('Training terminated by user.')
        print('*'*60)
    
    print('\nShutting down process pool...')
    process_pool_executor.shutdown(wait=True,cancel_futures=True)
    print('All pool processes ended.')
    data_manager.shutdown()
    
