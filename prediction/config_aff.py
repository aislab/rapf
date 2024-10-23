import jax.numpy as jnp

# Seed for RNG. Use None for random seeding.
# Two runs with the same seed and settings should produce the same result.
rng_seed = None

# Whether to include the pixel values (RGBD) of the pixel at the affordance coordinates in the net's action input.
aff_pixel = 1

# The number of unique M modules to initialise and train.
# You can use separate M modules for qualitatively different action types.
# When the number of unique M modules is larger than 1, each batch/example should specify the M_indices element.
# M_indices should specify the M module to be used for each step in each example, as an integer index.
number_of_unique_M_modules = 3

# Data manager and visualiser classes to use.
data_manager_class = 'data_manager_aff.Data_manager_aff'
visualiser_class = 'visualiser_aff.Visualiser_aff'

# List of additional files to copy to /source folder in run folders.
# Note: 'pEMD.py','data_manager.py','config.py','utils.py' are copied by default.
source_files_to_copy = ['config_aff.py','data_manager_aff.py','visualiser_aff.py']

# Toggles double precision (note: double precision is significantly slower).
enable_double_precision = False

# Sets whether to use custom loss functions.
# Custom loss functions are assumed to be implemented in the data_manager module.
custom_state_loss_for_training = True
custom_aux_loss_for_training = False

# Sets whether to predict future states in full or to predict each future state's difference w.r.t. the preceding state.
differential_prediction = 1

### Domain definitions

# Number of dimensions in the action representation used by the dataset.
action_dims = 6 if (number_of_unique_M_modules==1) else 5
# Number of dimensions in the action representations going into the NN's action input.
# This can differ from action_dims above. Here we add the RGBD values of the affordance pixel.
# These RGBD values are retrieved on-the-fly from the input state image (1st step) and predictions
# of states preceding the action (2nd and later steps).
action_dims_at_nn_input = action_dims+4 if aff_pixel else action_dims
# Number of auxilliary dimensions.
# These are additional state ouputs from the M module that do not pass through the decoder.
aux_dims = 0
# Number of channels in state images (here: RGBD -> 4 channels)
channels = 4
# Number of output channels in the decoder module.
# When we do differential prediction, we need an additional mask layer for blending the predicted and preceding states.
nn_out_channels = channels+differential_prediction
# Input and output state representation formats.
state_dims_in = (128,128,channels)
state_dims_out = (128,128,channels)
nn_state_dims_out = (128,128,nn_out_channels)

# Number of hidden state variables
hidden_dims = 0

# Number of auxilliary state variables.
state_extra_dims = 0


### Network architecture

# Activation function for hidden layers
activation_function_hidden = jnp.tanh
# Activation function for output layers
activation_function_out = lambda x: jnp.clip(x,0,1)

# Bandwidth of the latent state representation.
M_state_bandwidth = 256
# Bandwidth of the memory trace.
M_memory_bandwidth = 256

# Architecture definitions for all modules.
# Each module should start with a 'type':'input' definition giving input shape.
# Subsequent layers can have types 'dense' or 'conv' for dense and convolutional layers.
# No convolutional layers should appear in the M module definition.
# The 'count' element can be used to construct multiple layers with the same definition.
# When a conv layer follows a dense layer, 'reshape_input' should be defined to indicate how 
# the output of the preceding dense layer should be reshaped into an nD volume.
# Dense layers should define the number of outputs using the 'nO' element.
# Convolutional layers should specify their kernel shape using 'kernel' element.
# Optionally, convolutional layers can define 'strides' or 'upscale' to change the resolution 
# of the activation array by integer values.
# Kernel, strides and upscale elements should match the dimensionality of the activation array.
M_layers = 10
architectures = {
    'E':({'type':'input','shape':state_dims_in},
         {'type':'conv','kernel':(3,3,state_dims_in[-1],8),'strides':(2,2)},
         {'type':'conv','kernel':(3,3,8,8),'strides':(2,2)},
         {'type':'dense','nO':8192},
         {'type':'dense','nO':4096},
         {'type':'dense','nO':M_state_bandwidth},
         ),
    'M':({'type':'input','shape':M_state_bandwidth+state_extra_dims+M_memory_bandwidth+hidden_dims+action_dims_at_nn_input},
         {'type':'dense','nO':M_state_bandwidth+M_memory_bandwidth,'count':M_layers-1},
         {'type':'dense','nO':M_state_bandwidth+M_memory_bandwidth+aux_dims},
         ),
    'D':({'type':'input','shape':M_state_bandwidth},
         {'type':'dense','nO':4096},
         {'type':'dense','nO':8192},
         {'type':'conv','kernel':(3,3,8,8),'upscale':(2,2),'reshape_input':(32,32,8)},
         {'type':'conv','kernel':(3,3,8,nn_out_channels),'upscale':(2,2)},
         ),
    }

# Skip connection lengths in M module.
# Must be in ascending order.
skips = [1,2]

# Whether to apply local response normalisation in modules E & D.
use_local_response_normalisation_ED = False

# Whether to apply local response normalisation to latent state representations.
use_local_response_normalisation_latents = True

# For M module with skip connections.
# If True, all incoming activation branches into a layer
# are scaled by the inverse of the number of incoming branches.
# This could help to keep activation variance in check.
scale_by_incoming_branches = False

# For M module with skip connections.
# If True, ecah layer receives at most one incoming skip connection.
# When multiple incoming skip connections would exist, only activation from the longest is added.
only_one_incoming_skip_per_layer = True


### Training hyperparameters

# Initial signSGD learning rate.
initial_learning_rate = 5e-5
# Total number of training iterations to run.
n_training_iterations = int(1e6)
# Interval for evaluating performance on the validation state (in training iterations).
training_validation_interval = 10000
# Number of sequential no-improvement evaluations on the validation set at which the learning rate should be reduced.
stale_count_limit = 5
# Multiplication factor by which the learning rate is reduced at each reduction. Float in [0,1].
learning_rate_reduction_factor = 0.5


### Batching

# Training batch size
n_batch = 32

# Toggle to disable multiprocessing for debug purposes.
enable_multiprocessing = True

# Number of processes used for batch generation.
# If batch generation is bottlenecking training speed, increasing this may help.
# Must be >=0 (to disable multiprocessing, set enable_multiprocessing to False).
n_batch_generation_processes = 4


### Visualisation

# Number of examples to include in the visualisations generated during training.
# Once every training_test_interval iterations, prediction results for a test batch and a 
# train batch are written out as images to the "training_images" directory inside the run directory.
# This parameter sets the maximum number of examples to include in these images.
n_examples_in_training_image = 16
# Vertical and horizontal separation between panels within an example. Even positive integer.
image_panel_separation = 2
# Vertical separation between examples. Even positive integer.
image_example_separation = 12


### Monitoring & saving

# Interval for printing training loss (in training iterations).
training_print_interval = 100
# Interval for running a test batch, printing test loss, and generating training images (in training iterations).
training_test_interval = 1000
# Interval for saving the training state (in training iterations).
training_save_interval = 2000
# Number of superseded run state files to keep (as backup). Use -1 to keep all.
n_old_run_files_to_keep = 1
# Names for custom scores returned by calculate_scores in the data_manager class.
# Used for display of custom scores during training.
custom_scores = ['state_distance']


