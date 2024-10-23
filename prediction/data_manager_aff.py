import numpy as np
from glob import glob
import imageio
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from functools import partial
import time
import importlib
from data_manager import Data_manager
import config

# Absolute path to utils.py to avoid name clash with yolo utils when we load this module for planning.
utils_path = '/home/.../prediction/utils.py'

# Path(s) to the dataset(s) to train on.
data_paths = ['/PATH/TO/SOURCE/DATASET']

# Path to store preprocessed_data to.
preprocessed_data_file_path = 'preprocessed_data.npy'

# Number of sequences to allocate to the test and validation sets.
# The training set will contain all remaining examples.
n_test = 500
n_validation = 500
# Maximum number of sequences to load.
# Use None for no limit.
max_data = None

# Camera view selection for reading dataset files.
view = 'top'
# Integer factor for downscaling state images from the dataset.
downscale_factor = 2
# Number of pixel to crop off each edge of state images from the dataset.
edge_crop = 128

# Augmentation settings.
place_augmentation = 1
place_augmentation_range = ((3,3),(6,0))
mirror_augmentation = 1
aff_position_noise_augmentation = 1
aff_symmetry_augmentation = 1
aff_pos_noise = (0.025,0.05,0.025)
aff_angle_noise = 0.05

# camera specs for perspective drawing
fov = 50
cp_near = 0.5
cp_far = 1.72
# set up perspective matrix
S = 1/(np.tan(fov/2*np.pi/180))
pm = np.zeros((4,4))
pm[0,0] = S
pm[1,1] = S
pm[2,2] = -cp_far/(cp_far-cp_near)
pm[2,3] = -(cp_far*cp_near)/(cp_far-cp_near)
pm[3,2] = -1

# max number of free params across all affordances/modules.
# note: input for affordances with fewer free params will be zero-padded.
n_free_param_slots = 1

# list of affordance types, used to map affordance names to category integers
aff_type_list = ['grasp', 'place', 'turn']




utils = importlib.machinery.SourceFileLoader('utils',utils_path).load_module()

def msplit(string, param_name):
    temp = string.split(param_name+':(',1)
    pp = [float(v) for v in temp[1].split(')',1)[0].split(', ')] if len(temp)>1 else []
    return pp + [0] * (n_free_param_slots-len(pp))


def process_aff_line(line):
    try:
        aff_type, params = line.split(' ',1)
        aff_type_index = aff_type_list.index(aff_type)
        
        pos = msplit(params,'position')
        rot = msplit(params,'rotation')
        free = msplit(params,'free_params')
        sym = params.split('symmetry:')[1][0]=='T'
    except:
        print('Failed to parse executed affordance data:')
        print(line)
        print('in file:')
        print(aff_file)
        raise ValueError('Invalid line in affordance data.')
    
    action = pos+[rot[1]]+free # only y element of rotation
        
    action[3] = action[3]%360
    action[3] /= 360.0
    if action[3]>0.875:
        action[3] -= 1
    action[-1] /= 90.0
    return aff_type_index, action, sym


def apply_shift_augmentation_to_actions(actions,sx,sz):
    actions = np.array(actions)
    ax = 0.5+2*0.55*actions[...,0]
    az = 0.5+2*0.55*actions[...,2]
    ax = (ax+sx+1)%1
    az = (az-sz+1)%1
    actions[...,0] = (ax-0.5)/(2*0.55)
    actions[...,2] = (az-0.5)/(2*0.55)
    return actions
        
        
def apply_mirror_augmentation_to_actions(actions):
    actions = np.array(actions)
    if actions.ndim==1:
        dummy_dim = True
        actions = actions[None]
    else:
        dummy_dim = False
    actions[:,0] *= -1
    actions[:,4] *= -1
    for s in range(actions.shape[0]):
        angle = actions[s,3]
        mirror_angle = -(angle+0.25)-0.25
        while mirror_angle<-0.125:
            mirror_angle += 1
        while mirror_angle>=0.875:
            mirror_angle -= 1
        actions[s,3] = mirror_angle
    return actions[0] if dummy_dim else actions
        
        
def apply_sym_augmentation_to_actions(actions,sym):
    actions = np.array(actions)
    if sym and np.random.randint(2):
        actions[3] += 0.5
        changed = 1
    if actions[3] > 0.75+0.125:
        actions[3] -= 1
    elif actions[3] < -0.125:
        actions[3] += 1
    return actions


def apply_noise_augmentation_to_actions(actions):
    actions = np.array(actions)
    actions[:,:3] += np.random.uniform(-1,1,actions[:,:3].shape)*np.array(aff_pos_noise)[None]
    actions[:,3] += np.random.uniform(-aff_angle_noise,aff_angle_noise,actions[:,3].shape)
    return actions
    

class Data_manager_aff(Data_manager):

    def __init__(self,training_mode,load_dataset):
        
        if load_dataset:
            
            try:
                self.data = np.load(preprocessed_data_file_path,allow_pickle=True)[()]
                input('Data loaded from preprocessed file. Press Enter to continue or Ctrl+C to reload from source data.')
            except:
                print('Loading from source files')
                state_data = []
                action_data = []
                action_type_data = []
                action_symmetry_data = []
                index_data = []
                feasible_aff_data = []
                
                executed_actions = []
                feasible_actions = []
                for data_path in data_paths:
                    print('reading from data path:', data_path)
                    executed_affs = glob(data_path+'/*_ExecutedAffordance.txt')
                    executed_actions = executed_actions+sorted(executed_affs)
                    print('added', len(executed_affs), 'aff examples. current total:', len(executed_actions))
                    
                    
                executed_actions = executed_actions[:max_data]
                
                current_s_data = None
                example_bad = False
                n_bad_examples = 0
                for i, g in enumerate(executed_actions):
                    utils.print_progress(i,len(executed_actions))
                    path, fname = g.rsplit('/',1)
                    if fname[0] == '_': continue
                    s_data, s_step = fname.split('_')[:2]
                    aff_file = path+'/'+s_data+'_'+s_step+'_ExecutedAffordance.txt'
                    print('reading executed aff file:', aff_file)
                    try:
                        with open(aff_file,'r') as f: line = f.readline()
                    except:
                        continue
                    
                    aff_type_index, action, sym = process_aff_line(line)
                    
                    feasible_aff_list = None
                    
                    
                    i_step = int(s_step)
                    
                    if i_step == 0:
                        rgb_path = path+'/'+s_data+'_'+s_step+'_'+view+'_RGB.png'
                        try:
                            rgb_before = imageio.imread(rgb_path)[...,:3]
                        except:
                            print('failed to read:', rgb_path)
                        d_path = path+'/'+s_data+'_'+s_step+'_'+view+'_D.png'
                        try:
                            d_before = imageio.imread(d_path)[...,:1]
                        except:
                            print('failed to read:', d_path)
                        d_before = 255-d_before
                        d_before = np.clip(15*d_before.astype(int)-15,0,255)
                        rgbd_before = np.concatenate([rgb_before,d_before],axis=-1)
                        rgbd_before = rgbd_before[edge_crop:-edge_crop,edge_crop:-edge_crop]
                        rgbd_before = rgbd_before.reshape(rgbd_before.shape[0]//downscale_factor,downscale_factor,rgbd_before.shape[1]//downscale_factor,downscale_factor,4)
                        rgbd_before = rgbd_before.mean((1,3))
                        rgbd_before = rgbd_before.astype(np.uint8)
                    try:
                        rgb_after = imageio.imread(path+'/'+s_data+'_'+str(i_step+1)+'_'+view+'_RGB.png')[...,:3]
                        d_after = imageio.imread(path+'/'+s_data+'_'+str(i_step+1)+'_'+view+'_D.png')[...,:1]
                        d_after = 255-d_after
                        d_after = np.clip(15*d_after.astype(int)-15,0,255)
                        rgbd_after = np.concatenate([rgb_after,d_after],axis=-1)
                        rgbd_after = rgbd_after[edge_crop:-edge_crop,edge_crop:-edge_crop]
                        rgbd_after = rgbd_after.reshape(rgbd_after.shape[0]//downscale_factor,downscale_factor,rgbd_after.shape[1]//downscale_factor,downscale_factor,4)
                        rgbd_after = rgbd_after.mean((1,3))
                        rgbd_after = rgbd_after.astype(np.uint8)
                    except:
                        continue
                    
                    # continue to next example
                    if s_data != current_s_data:
                        state_data.append([])
                        action_data.append([])
                        action_type_data.append([])
                        action_symmetry_data.append([])
                        index_data.append(int(s_data))
                        feasible_aff_data.append([])
                        current_s_data = s_data
                        example_bad = False
                    elif example_bad:
                        continue
                        
                    if i_step == 0:
                        state_data[-1].append(rgbd_before)
                    
                    state_data[-1].append(rgbd_after)
                    action_data[-1].append(action)
                    action_type_data[-1].append(aff_type_index)
                    action_symmetry_data[-1].append(sym)
                    feasible_aff_data[-1].append(feasible_aff_list)
                    
                    if example_bad:
                        print('removing bad example:', s_data)
                        del state_data[-1]
                        del action_data[-1]
                        del action_type_data[-1]
                        del action_symmetry_data[-1]
                        del feasible_aff_data[-1]
                
                utils.print_progress()
                n_data = len(action_data)
                print('collected', n_data, 'examples (affordance executions)')
                print('bad examples:', n_bad_examples)
                
                # shuffle data using fixed seed
                rng_state = np.random.get_state()
                np.random.seed(1)
                order = np.arange(n_data)
                np.random.shuffle(order)
                np.random.set_state(rng_state)
                
                state_data2 = []
                action_data2 = []
                action_type_data2 = []
                action_symmetry_data2 = []
                index_data2 = []
                feasible_aff_data2 = []
                
                for i in order:
                    state_data2.append(np.array(state_data[i]))
                    action_data2.append(np.array(action_data[i]))
                    action_type_data2.append(np.array(action_type_data[i]))
                    action_symmetry_data2.append(np.array(action_symmetry_data[i]))
                    index_data2.append(index_data[i])
                    feasible_aff_data2.append(feasible_aff_data[i])
                
                state_data = state_data2
                action_data = action_data2
                action_type_data = action_type_data2
                action_symmetry_data = action_symmetry_data2
                index_data = index_data2
                feasible_aff_data = feasible_aff_data2
                
                n_train = n_data-n_test-n_validation
                if n_train <= 0:
                    print('Insufficient examples to realise the requested subset split')
                    print('n_data (', n_data, ') <= n_test (', n_test, ') + n_validation (', n_validation, ')')
                    if max_data is not None:
                        print('Note: max_data is set to:', max_data)
                    raise ValueError('Requested dataset split cannot be realised.')
                    
                print('n_train:', n_train, 'n_test:', n_test, 'n_validation:', n_validation)
                self.data = {}
                self.data['train'] = {}
                self.data['train']['n_data'] = n_train
                self.data['train']['states'] = state_data[:n_train]
                self.data['train']['actions'] = action_data[:n_train]
                self.data['train']['action_types'] = action_type_data[:n_train]
                self.data['train']['action_symmetry'] = action_symmetry_data[:n_train]
                self.data['train']['index'] = index_data[:n_train]
                
                self.data['test'] = {}
                self.data['test']['n_data'] = n_test
                self.data['test']['states'] = state_data[n_train:n_train+n_test]
                self.data['test']['actions'] = action_data[n_train:n_train+n_test]
                self.data['test']['action_types'] = action_type_data[n_train:n_train+n_test]
                self.data['test']['action_symmetry'] = action_symmetry_data[n_train:n_train+n_test]
                self.data['test']['index'] = index_data[n_train:n_train+n_test]
                
                self.data['validation'] = {}
                self.data['validation']['n_data'] = n_validation
                self.data['validation']['states'] = state_data[n_train+n_test:]
                self.data['validation']['actions'] = action_data[n_train+n_test:]
                self.data['validation']['action_types'] = action_type_data[n_train+n_test:]
                self.data['validation']['action_symmetry'] = action_symmetry_data[n_train+n_test:]
                self.data['validation']['index'] = index_data[n_train+n_test:]
                
                np.save(preprocessed_data_file_path,self.data)
            
            # find background rgbd for use as fill value
            self.background_rgbd = self.data['train']['states'][0][0][0,0,:]/255.0
            
            for subset in ('train','test','validation'):
                print('subset:', subset)
                for data_type in ('states','actions','action_types'):
                    print(data_type, ':', len(self.data[subset][data_type]))
            
            
    def make_batch(self,rng_key,subset,n_examples,augmentation):
        
        # randomly seed local numpy randomiser (individual processes receive identical copies from the spawning process)
        rng_key, k = jax.random.split(rng_key)
        seed = jax.random.randint(k,[1],-(2**31),2**31-1)[0] # apparently limited to int32 (even when specifying dtype=int64)
        seed = np.int64(seed)+2**31
        np.random.seed(seed)
        
        n_data = self.data[subset]['n_data']
        
        step_lengths = np.array([a.shape[0] for a in self.data[subset]['actions']])
        n_steps = np.random.randint(1,5);
        
        viable_data_indices = np.where(step_lengths>=n_steps)[0]
        
        example_selection = np.random.choice(viable_data_indices,n_examples).astype(int)
        step_selection = [np.random.randint(s-n_steps+1) for s in step_lengths[example_selection]]
        
        states = np.array([self.data[subset]['states'][i][j:j+n_steps+1]/255.0 for i,j in zip(example_selection,step_selection)])
        actions = np.array([self.data[subset]['actions'][i][j:j+n_steps] for i,j in zip(example_selection,step_selection)])
        action_types = np.array([self.data[subset]['action_types'][i][j:j+n_steps] for i,j in zip(example_selection,step_selection)])
        action_symmetry = np.array([self.data[subset]['action_symmetry'][i][j:j+n_steps] for i,j in zip(example_selection,step_selection)])
        
        for i in range(n_examples):
                    
            if augmentation and place_augmentation:
                
                px = np.random.randint(-place_augmentation_range[0][0],place_augmentation_range[0][1]+1)
                pz = np.random.randint(-place_augmentation_range[1][0],place_augmentation_range[1][1]+1)
                    
                states[i] = np.roll(states[i],(pz,px),(1,2))
                if (px<0):
                    states[i,:,:,px:] = self.background_rgbd[None,None,None]
                if (px>0):
                    states[i,:,:,:px] = self.background_rgbd[None,None,None]
                if (pz<0):
                    states[i,:,pz:] = self.background_rgbd[None,None,None]
                if (pz>0):
                    states[i,:,:pz] = self.background_rgbd[None,None,None]
                
                for s in range(n_steps):
                    x,z = actions[i,s,0], actions[i,s,2]
                    sx, sz = px/config.state_dims_in[0], pz/config.state_dims_in[1]
                    actions[i,s] = apply_shift_augmentation_to_actions(actions[i,s],sx,sz)
                    
            if augmentation and mirror_augmentation:
                if np.random.randint(2):
                    states[i] = states[i,:,:,::-1]
                    actions[i] = apply_mirror_augmentation_to_actions(actions[i])
                        
            # randomly flip some symmetrical affordances 180 deg (0.5)
            if augmentation and aff_symmetry_augmentation:
                for s in range(n_steps):
                    sym = (action_symmetry[i,s] or action_types[i,s]==2)
                    actions[i,s] = apply_sym_augmentation_to_actions(actions[i,s],sym)
            
            if augmentation and aff_position_noise_augmentation:
                actions[i] = apply_noise_augmentation_to_actions(actions[i])
            
        if config.number_of_unique_M_modules==1:
            actions = np.concatenate((actions,action_types[...,None]),axis=-1)
            action_types *= 0
        
        batch = {'states': states,
                 'actions': actions,
                 'M_indices': action_types,
                 }
        
        return batch
    
    
    def apply_action_modification(self,full_state,action):

        if config.aff_pixel:
            x, z, y, angle = action[:4]
                    
            xyzw = jnp.array((x,y,z,1))
            p_xyzw = jnp.matmul(xyzw,pm)
            x, y, z, w = p_xyzw
                    
            x = jnp.clip(jnp.floor((0.5+0.5*x)*128),0,127).astype(int)
            y = jnp.clip(jnp.floor((0.5-0.5*y)*128),0,127).astype(int)
                    
            full_state = jnp.array(full_state)
            pixel = full_state[y,x]
                    
            action = jnp.concatenate((action,pixel),axis=0)
        return action
    
    
    def get_example(self,subset,i_example):
        
        states = np.array(self.data[subset]['states'][i_example])/255.0
        actions = np.array(self.data[subset]['actions'][i_example])
        action_types = np.array(self.data[subset]['action_types'][i_example])
        data_index = self.data[subset]['index'][i_example]
        
        if config.number_of_unique_M_modules==1:
            actions = np.concatenate((actions,action_types[...,None]),axis=-1)
            action_types *= 0
        
        example = {'states': states,
                   'actions': actions,
                   'M_indices': action_types,
                   'data_index': data_index,
                   }
        
        return example
    
    
    def get_example_count(self,subset):
        # return the number of example sequences in the indicated subset.
        return self.data[subset]['n_data']
            
    def get_step_count(self,subset=None,index=None):
        # return the number of steps in the indicated sequence.
        return self.data[subset]['actions'][index].shape[0]
    
    def calculate_scores(self,predicted_state,ground_truth_state,predicted_aux,ground_truth_aux):
        # Calculate score(s) for the given prediction.
        # Should return a list of score values.
        state_score = (((predicted_state[:,:,:4]-ground_truth_state)**2)).mean()
        return [state_score,0]
    
    
    @staticmethod
    @jax.jit
    def state_loss_training(predicted_state,ground_truth_state):
        losses = ((ground_truth_state-predicted_state)**2).sum(-1)
        while losses.ndim>1:
            losses = jnp.mean(losses,axis=1)
        return jnp.mean(losses), losses
