import importlib
import os
import errno
import jax
import numpy as np
from copy import deepcopy
import shutil
import glob
import config

data_manager = None

# makes dir if it does not exist.
# if clear is true and dir exists, its content is cleared.
def makedir(path,clear=False):
    try:
        os.makedirs(path)
        print('created directory:',path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            if clear: clear_dir(path)
        else:
            raise

def clear_dir(path):
    files = glob.glob(path+'/*')
    for f in files:
        try:
            os.remove(f)
        except:
            shutil.rmtree(f)

# Call with i, n to display that i steps out of n steps are completed.
# Call without arguments or with i==n to finalise.
def print_progress(i=1,n=1,w=60):
    r = i/n
    p = np.round(100*r)
    i = int(np.round(w*r))
    print('  ['+('▆'*i)+' '*(w-i)+'] '+str(p)+'%\r',end='')
    if r==1:
        print('')

# Returns the data manager, initialising it if necessary
def get_data_manager(training_mode=None,load_dataset=None):
    global data_manager
    if data_manager is None:
        if training_mode is None or load_dataset is None:
            print('*'*60)
            print('ERROR: First call to get_data_manager() should specify training_mode and load_dataset arguments')
            print('*'*60)
            exit(0)
        dm_module, dm_class = config.data_manager_class.rsplit('.',1)
        dm_module = importlib.import_module(dm_module,package=os.path.abspath(__file__))
        dm_class = getattr(dm_module, dm_class)
        data_manager = dm_class(training_mode,load_dataset)
    return data_manager

def copy_source_files(dest):
    dm_module, dm_class = config.data_manager_class.rsplit('.',1)
    shutil.copy2(dm_module+'.py',dest)
    vis_module, vis_class = config.visualiser_class.rsplit('.',1)
    shutil.copy2(vis_module+'.py',dest)
    for f in ['pEMD.py','data_manager.py','config.py','utils.py']+config.source_files_to_copy:
        shutil.copy2(f,dest)

def get_visualiser(setup=True):
    vis_module, vis_class = config.visualiser_class.rsplit('.',1)
    vis_module = importlib.import_module(vis_module,package=os.path.abspath(__file__))
    vis_class = getattr(vis_module, vis_class)
    return vis_class(setup)

def init_rng():
    # create initial RNG key
    if config.rng_seed is None:
        seed = np.random.randint(np.iinfo(np.int64).min,np.iinfo(np.int64).max)
    else:
        seed = config.rng_seed
    rng_key = jax.random.PRNGKey(seed)
    print('Initialised RNG with seed:', seed)
    print('Initial RNG key:', rng_key)
    return rng_key

def dim_match(shape,tgt):
    for s,t in zip(shape,tgt):
        if t is not None and s!=t:
            return False
    return True

def report_shape(data,name='[?]'):
    try:
        print(name, 'of type', type(data), 'has shape:', data.shape)
    except:
        shape = []
        while True:
            try:
                shape.append(len(data))
                data = data[0]
            except:
                break
        print(name, 'of type', type(data), 'has shape:', shape)

def check_and_complete_batch(batch,subset,n_examples):
    
    n_examples = len(batch['states']) # TODO: clean this up (batch size may vary for small subsets)
    
    # check whether the batch is properly structured and report problems
    batch_bad = False
    batch_keys = batch.keys()
    checks = [('states',(n_examples,None)+config.state_dims_in),('actions',(n_examples,None,config.action_dims))]
    if config.aux_dims>0:
        checks.append(('auxes',(n_examples,None,config.aux_dims)))
    if config.hidden_dims>0:
        checks.append(('hidden',(n_examples,config.hidden_dims)))
    if config.hidden_dims>0:
        checks.append(('state_extras',(n_examples,None,config.state_extra_dims)))
    if config.number_of_unique_M_modules > 1:
        checks.append(('M_indices',(n_examples,None)))
    
    for (batch_key, dims) in checks:
        
        # check existence of required fields
        if batch_key not in batch_keys:
            print('*'*60)
            print("ERROR: Missing field in batch: '", batch_key, "'", sep="")
            print('*'*60)
            batch_bad = True
            continue
        
        # check convertability to numpy arrays
        try:
            batch[batch_key] = np.array(batch[batch_key])
        except:
            print('*'*60)
            print("ERROR: Could not convert batch['", batch_key, "'] to numpy array.", sep="")
            print('*'*60)
            batch_bad = True
            continue
    
        # check array shapes
        #if batch[batch_key].ndim != len(dims) or batch[batch_key].shape[0] != n_examples or batch[batch_key].shape[2:] != dims:
        if batch[batch_key].ndim != len(dims) or not dim_match(batch[batch_key].shape,dims):
            print('*'*60)
            print("ERROR: Array batch['", batch_key, "'] has incorrect shape: ", batch[batch_key].shape, ".", sep="")
            print("Shape should be: ", dims, ".", sep="")
            print('*'*60)
            batch_bad = True
    
    if batch_bad:
        exit(0)
    
    # check consistency between field shapes
    if batch['actions'].shape[1] != batch['states'].shape[1]-1:
        print('*'*60)
        print("ERROR: Step count inconsistency between 'actions' array and 'states' array")
        print("Batch has", batch['actions'].shape[1], "actions/example but", batch['states'].shape[1], "states/example.")
        print("A batch with n actions/example should have n+1 states/example.")
        print('*'*60)
        print(batch['states'].shape,batch['actions'].shape)
        batch_bad = True
    if config.aux_dims > 0:
        if batch['actions'].shape[1] != batch['auxes'].shape[1]:
            print('*'*60)
            print("ERROR: Step count inconsistency between 'actions' array and 'auxes' array")
            print("Batch has", batch['actions'].shape[1], "actions/example but", batch['auxes'].shape[1], "auxes/example.")
            print("A batch with n actions/example should have n auxes/example.")
            print('*'*60)
            batch_bad = True
    if config.number_of_unique_M_modules > 1:
        if batch['M_indices'].shape[1] != batch['actions'].shape[1]:
            print('*'*60)
            print("ERROR: Step count inconsistency between 'actions' array and 'M_indices' array")
            print("Batch has", batch['actions'].shape[1], "actions/example but", batch['M_indices'].shape[1], "M_indices/example.")
            print("A batch with n actions/example should have n M_indices/example.")
            print('*'*60)
            batch_bad = True
    
    # check shape of meta data field (if given)
    if 'meta_data' in batch_keys:
        try:
            n_meta = len(batch['meta_data'])
            if n_meta != n_examples:
                print('*'*60)
                print("ERROR: meta_data count should match the batch size (", n_examples, "), but is ", n_meta, sep="")
                print('*'*60)
                batch_bad = True
        except TypeError:
            print('*'*60)
            print("ERROR: meta_data should be a list or array of length equal to the batch size (", n_examples, ").", sep="")
            print('*'*60)
            batch_bad = True
            
        n_states = batch['states'].shape[1]
        for md in batch['meta_data']:
            try:
                if len(md) not in (1,n_states):
                    print('*'*60)
                    print("ERROR: meta_data for each example should have length equal to the number of states in the example "
                          "(if meta data is associated with states) or 1 (if meta data is associated with sequences), but the "
                          "data module returned meta data with length ", len(md), ".", sep="")
                    print('*'*60)
                    batch_bad = True
                    break
            except TypeError:
                print('*'*60)
                print("ERROR: meta_data for each example should be a list or array of length equal to the number of states in the "
                      "example (if meta data is associated with states) or 1 (if meta data is associated with sequences).", sep="")
                print('*'*60)
                batch_bad = True
                break
    
    if batch_bad:
        exit(0)

    # fill additional fields
    batch['subset'] = subset
    batch['n_examples'] = n_examples
    batch['n_steps'] = batch['actions'].shape[1]
    #if 'meta_data' not in batch_keys:
    #    batch['meta_data'] = [None]
    if 'auxes' not in batch_keys:
        batch['auxes'] = np.zeros(batch['actions'].shape[:2]+(0,))
    if 'hidden' not in batch_keys:
        batch['hidden'] = np.zeros(batch['actions'].shape[:1]+(0,))
    if 'state_extras' not in batch_keys:
        batch['state_extras'] = np.zeros(batch['actions'].shape[:2]+(0,))
    if 'M_indices' not in batch_keys:
        batch['M_indices'] = np.zeros(batch['actions'].shape[:2],dtype=int)
    return batch


def request_example(subset,i_example):
    batch = data_manager.get_example(subset,i_example)
    # TODO: allow midding batch dimension?
    batch['states'] = batch['states'][None]
    if 'state_extras' in batch:
        batch['state_extras'] = batch['state_extras'][None]
    if 'hidden' in batch:
        batch['hidden'] = batch['hidden'][None]
    if 'actions' in batch:
        batch['actions'] = batch['actions'][None]
    if 'M_indices' in batch:
        batch['M_indices'] = batch['M_indices'][None]
    if 'auxes' in batch:
        batch['auxes'] = batch['auxes'][None]
    if 'meta_data' in batch:
        batch['meta_data'] = [batch['meta_data']]
    try:
        batch = check_and_complete_batch(batch,subset,1)
    except:
        print("*"*60)
        print('request_example failed! note: stopped requiring singleton batch dimension for get_example.')
        print("*"*60)
        input('...')
        
    return batch


#def convert_example_to_planning_task(rng_key,example,from_step,n_steps):
    #task = {}
    #for key,item in example.items():
        #last = 1 if key == 'states' else 0
        #if hasattr(item,'ndim'):
            #task[key] = np.repeat(item[:,from_step:from_step+n_steps+last],config.n_strains,0)
        #elif key == 'meta_data':
            #task[key] = [[deepcopy(md) for md in item[0][from_step:from_step+n_steps+1]] for _ in range(config.n_strains)]
        #else:
            #task[key] = example[key]
    #task['n_steps'] = n_steps
    #task = data_manager.generate_equivalents(task)
    #return task


def convert_example_to_planning_task(rng_key,example,from_step,n_steps):
    print('converting example to task. from:', from_step, 'n_steps:', n_steps)
    
    task = {}
    task['states'] = np.repeat(example['states'][:,from_step:from_step+n_steps+1],config.n_strains,0)
    task['state_extras'] = np.repeat(example['state_extras'][:,from_step:from_step+n_steps+1],config.n_strains,0)
    
    if 0:
        task['actions'] = np.repeat(example['actions'][:,from_step:from_step+n_steps],config.n_strains,0)
        print('*'*60)
        print('DEBUG ONLY: keeping original actions in planning task')
        print('*'*60)
        input('...')
    else:
        task['actions'] = None
    
    # planning does not infer which M modules to use so we have to keep this information
    #print('convert/example/M_indices',example['M_indices'])
    task['M_indices'] = np.repeat(example['M_indices'][:,from_step:from_step+n_steps],config.n_strains,0)
    #print('convert/task/M_indices',task['M_indices'])
    
    task['auxes'] = None
    #task['meta_data'] = [[deepcopy(md) for md in example['meta_data'][from_step:from_step+n_steps+1]] for _ in range(config.n_strains)]
    #print("task['meta_data']:")
    #print(report_shape(task['meta_data']))
    if 'meta_data' in example:
        task['meta_data'] = [[deepcopy(example['meta_data'][0][from_step:from_step+n_steps+1])] for _ in range(config.n_strains)]
    #print("task['meta_data']:")
    #print(report_shape(task['meta_data']))
    #input('...')
    task['hidden'] = np.repeat(example['hidden'],config.n_strains,0)
    task['n_steps'] = n_steps
    print("task['n_steps']:", task['n_steps'])
    task = data_manager.generate_equivalents(task)
    
    # Copy miscellaneous keys (if any)
    for key in example:
        if key not in task:
            task[key] = example[key]
    
    return task


# for compatibility with hidden variable inference
# TODO: move to a format where 'states' is eliminated.
# having 'current_state' and 'goal_states' is sufficient for action planning.
# having 'past_states' (including current) is sufficient for hidden var inference.
def convert_example_to_planning_task_v2(rng_key,example,from_step,n_steps):
    print('converting example to task. from:', from_step, 'n_steps:', n_steps)
    
    task = {}
    task['goal_state'] = np.repeat(example['states'][:,from_step+n_steps],config.n_strains,0)
    task['current_state'] = np.repeat(example['states'][:,from_step],config.n_strains,0)
    #task['state_extras'] = np.repeat(example['state_extras'][:,from_step:from_step+n_steps],config.n_strains,0)
    #task['current_state_extras'] = np.repeat(example['state_extras'][:,from_step],config.n_strains,0)
    
    if 0:
        task['actions'] = np.repeat(example['actions'][:,from_step:from_step+n_steps],config.n_strains,0)
        print('*'*60)
        print('DEBUG ONLY: keeping original actions in planning task')
        print('*'*60)
        input('...')
    else:
        task['actions'] = None
    
    # planning does not infer which M modules to use so we have to keep this information
    #print('convert/example/M_indices',example['M_indices'])
    task['M_indices'] = np.repeat(example['M_indices'][:,from_step:from_step+n_steps],config.n_strains,0)
    #print('convert/task/M_indices',task['M_indices'])
    
    task['auxes'] = None
    #task['meta_data'] = [[deepcopy(md) for md in example['meta_data'][from_step:from_step+n_steps+1]] for _ in range(config.n_strains)]
    #print("task['meta_data']:")
    #print(report_shape(task['meta_data']))
    if 'meta_data' in example:
        task['meta_data'] = [[deepcopy(example['meta_data'][0][from_step:from_step+n_steps+1])] for _ in range(config.n_strains)]
    #print("task['meta_data']:")
    #print(report_shape(task['meta_data']))
    #input('...')
    task['hidden'] = np.repeat(example['hidden'],config.n_strains,0)
    task['n_steps'] = n_steps
    task['past_n_steps'] = 0
    
    task = data_manager.generate_equivalents(task)
    
    # initialise blank history
    #task['past_states'] = np.zeros((config.n_strains,0)+config.state_dims_in)
    task['past_states'] = task['current_state'][:,None]
    task['past_state_extras'] = np.zeros((config.n_strains,0)+(config.state_extra_dims,))
    #task['past_state_extras'] = task['current_state_extras'][:,None]
    task['past_M_indices'] = np.zeros((config.n_strains,0),dtype=int)
    task['past_actions'] = np.zeros((config.n_strains,0,config.action_dims))
    
    # Copy miscellaneous keys (if any)
    for key in example:
        if key not in task:
            task[key] = example[key]
    
    return task


def update_task(task,new_state,new_state_extras,new_hidden=None,new_meta_data=None):
    task = data_manager.generate_equivalents(task,undo=True)
    
    #TODO: keep previously generated plan in some strains?
    new_task = {}
    new_task['actions'] = None
    new_task['states'] = task['states']
    # length of sequence for the states element is irrelevant for planning, only first and last element are used.
    # since only these two elements may be given, we should not shorten the state sequence.
    new_task['states'][:,0] = new_state[None]
    if new_state_extras is not None:
        new_task['state_extras'] = task['state_extras'][:,1:]
        new_task['state_extras'][:,0] = new_state_extras[None]
    if new_meta_data is not None:
        new_task['meta_data'] = [[deepcopy(new_meta_data)] for _ in range(config.n_strains)]
    if new_hidden is not None:
        new_task['hidden'][:] = new_hidden[None]
        
    new_task['M_indices'] = task['M_indices'][:,1:]
        
    new_task['n_steps'] = task['n_steps']-1
    
    # Copy unchanged and miscellaneous keys
    for key in task:
        if key not in new_task:
            new_task[key] = task[key]
    
    new_task = data_manager.generate_equivalents(new_task)
    
    return new_task


# for compatibility with hidden variable inference
def update_task_v2(task,performed_action,new_state,performed_state_extras,new_hidden=None,new_meta_data=None,performed_M_index=None):
    task = data_manager.generate_equivalents(task,undo=True)
    
    print('updating hidden inference task with:')
    print('performed_action:', performed_action)
    print('performed_state_extras:', performed_state_extras)
    #input('...')
    
    #TODO: keep previously generated plan in some strains?
    new_task = {}
    new_task['past_actions'] = np.concatenate((task['past_actions'],np.repeat(performed_action[None,None],config.n_strains,0)),axis=1)
    new_task['goal_state'] = task['goal_state']
    new_task['current_state'] = np.repeat(new_state[None],config.n_strains,0)
    print('v2 update / past_states, current_state:', task['past_states'].shape, task['current_state'].shape)
    new_task['past_states'] = np.concatenate((task['past_states'],new_task['current_state'][:,None]),axis=1)
    
    if performed_state_extras is not None:
        print('past_state_extras was:', task['past_state_extras'].shape)
        #new_task['current_state_extras'] = np.repeat(new_state_extras[None],config.n_strains,0)
        new_task['past_state_extras'] = np.concatenate((task['past_state_extras'],np.repeat(performed_state_extras[None,None],config.n_strains,axis=0)),axis=1)
        print('past_state_extras updated to:', new_task['past_state_extras'].shape)
        #input('...')
        #temp = np.repeat(new_state_extras[None,None],config.n_strains,0)
        #print('temp:', temp.shape, "task['state_extras'][:,1:]", task['state_extras'][:,1:].shape)
        #new_task['state_extras'] = np.concatenate((temp,task['state_extras'][:,1:]),axis=1)
    if new_meta_data is not None:
        new_task['meta_data'] = [[deepcopy(new_meta_data)] for _ in range(config.n_strains)]
    if new_hidden is not None:
        new_task['hidden'][:] = new_hidden[None]
        
    if config.number_of_unique_M_modules==1:
        performed_M_index=0
    else:
        if performed_M_index is None:
            print('ERROR: performed_M_index must be specified for task update')
            exit(0)
            
    new_task['M_indices'] = task['M_indices'][:,1:]
    new_task['past_M_indices'] = np.concatenate((task['past_M_indices'],np.full((config.n_strains,1),performed_M_index)),axis=1)
    print('past_M_indices updated to:', new_task['past_M_indices'].shape)
        
    new_task['n_steps'] = task['n_steps']-1
    new_task['past_n_steps'] = task['past_n_steps']+1
    
    # Copy unchanged and miscellaneous keys
    for key in task:
        if key not in new_task:
            new_task[key] = task[key]
    
    new_task = data_manager.generate_equivalents(new_task)
    
    print('task content after v2 update:')
    for key, item in new_task.items():
        if type(item) in (list,np.ndarray):
            report_shape(item,'  '+key)
            print('contains NaN?', np.isnan(item).any())
            #print(' ', key, ':', item.shape)
        else:
            print(' ', key, ':', item)
    
    #input('...')
    return new_task
