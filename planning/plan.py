import argparse
import jax
print('JAX is running on', jax.lib.xla_bridge.get_backend().platform)
from jax import numpy as jnp
import time
import numpy as np
import sys
import os
import shutil
import imageio
import sys
import glob
import errno
import torch
import shutil
import matplotlib.pyplot as plt
import importlib
from functools import partial

recognition_dir = '/home/solvi/models/aff_source_cleanup/recognition/'
prediction_dir = '/home/solvi/models/aff_source_cleanup/prediction/'

# recognition module
sys.path.append(recognition_dir)
import detect
sys.path.remove(recognition_dir)

# prediction module
sys.path.append(prediction_dir)
import config
import pEMD

emd_utils = importlib.machinery.SourceFileLoader('emd_utils',prediction_dir+'utils.py').load_module()
import visualiser_aff as visualiser

### SETTINGS

# Recognition batch size.
n_recognition_parallel = 256
# Prediction batch size.
batch_size = 128
# Whether to zero-pad batches to full size.
# Zero-padding ensures that all batches are the same size.
# This avoids on-the-fly JAX JIT compilation time costs.
# (JAX functions are recompiled for different batch sizes.)
pad_prediction_batches_to_full_batch_size = True

# Resolution multiplier for visualisation outputs.
image_size_multiplier = 6

# Toggles whether to filter affordances by symmetry.
# Symmetrical affordances produce the same outcomes,
# so we can save computational costs by running only
# one side of symmetrical affordance pairs.
symmetry_filtering = 1

# Scaling & cropping of input state images.
downscale_factor = 2
edge_crop = 128

# Additional visualisation options.
# Note that the selection plan is always visualised.
# These toggles enable visualisation of all
# recognition/prediction/matching during planning.
# Enabling these increases planning time cost.
draw_recognition = False
draw_predictions = False
draw_matched_regions = False

# Coordinate correction factor for translating affordance
# coordinates between recognition and prediction networks.
coord_multiplier = 1/1.075

# Margins for duplicate affordance filtering.
duplicate_aff_threshold_xyz = (0.025,0.025,0.1)
duplicate_aff_threshold_angle = 0.05

###

# Makes directory if it does not exist.
def makedir(path):
    try:
        os.makedirs(path)
        print('created directory:',path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def clear_dir(path_to_dir):
    files = glob.glob(path_to_dir+'/*')
    for f in files:
        os.remove(f)


def upscale_image(im):
    im = np.repeat(im,image_size_multiplier,axis=0)
    im = np.repeat(im,image_size_multiplier,axis=1)
    return im


def preprocess_state(rgb_path,out_dir,crop_file=None):
    
    d_path = rgb_path[:-7]+'D.png'
    if not os.path.exists(d_path):
        print('Depth file not found:', d_path)
        raise FileNotFoundError('No state depth image found at path: '+d_path)
    
    im = imageio.imread(rgb_path)[:,:,:3]
    im = im[edge_crop:-edge_crop,edge_crop:-edge_crop]
    im = im.reshape(im.shape[0]//downscale_factor,downscale_factor,im.shape[1]//downscale_factor,downscale_factor,3)
    im = im.mean((1,3))
    rgb = im
    
    im = imageio.imread(d_path)[:,:,:1]
    im = im[edge_crop:-edge_crop,edge_crop:-edge_crop]
    im = im.reshape(im.shape[0]//downscale_factor,downscale_factor,im.shape[1]//downscale_factor,downscale_factor)
    im = im.mean((1,3))
    d = im
    
    if crop_file is not None:
        with(open(crop_file,'r')) as f:
            crop = [int(s) for s in f.readline().split()]
        crop = np.array(crop).reshape((2,2))
        crop -= edge_crop
        crop = crop/downscale_factor
        crop = np.round(crop).astype(int)
        rgb = rgb[crop[0,1]:crop[1,1],crop[0,0]:crop[1,0]]
        d = d[crop[0,1]:crop[1,1],crop[0,0]:crop[1,0]]
    
    if out_dir is not None:
        makedir(out_dir)
        imageio.imwrite(out_dir+'/RGB.png',np.uint8(rgb))
        imageio.imwrite(out_dir+'/Depth.png',np.uint8(d))
    
    return np.concatenate((rgb,d[:,:,None]),axis=-1)


def branch_id_to_name(branch_id):
    return '_'.join((str(i).zfill(3) for i in branch_id))


def init_queue(queue):
    queue['branch_id'] = []
    queue['state'] = []
    queue['action'] = []
    queue['M_module'] = []
    queue['raw_aff_sequence'] = []
    return queue


# Loss and best-matched region for a prediction given a target patch.
@partial(jax.jit,static_argnums=(2,3))
def loss_wrt_target_patch_jax(prediction,patch,prediction_shape,patch_shape):
    
    sh = patch_shape
    n_positions = sh[0]*sh[1]
    
    nx = int(np.floor(prediction_shape[0]/sh[0]))
    ny = int(np.floor(prediction_shape[1]/sh[1]))
    
    patch = jnp.tile(patch,(nx,ny,1))[:,:,:,None]
    
    prediction = jnp.pad(prediction,((0,sh[0]),(0,sh[1]),(0,0)))
    crops = []
    for x in range(sh[0]):
        for y in range(sh[1]):
            crops.append(prediction[x:x+nx*sh[0],y:y+ny*sh[1]])
    crops = jnp.stack(crops,axis=-1)
            
    d = (crops-patch)**2
    dr = d.reshape(d.shape[0]//sh[0],sh[0],d.shape[1]//sh[1],sh[1],sh[2],n_positions)
    dm = dr.mean((1,3,4))
    loss = dm.min()
    ix, iy, ip = jnp.unravel_index(dm.argmin(),dm.shape)
    pp = crops[...,:sh[2],:].reshape(d.shape[0]//sh[0],sh[0],d.shape[1]//sh[1],sh[1],sh[2],n_positions)
    region = pp[ix,:,iy,:,:,ip]
    return loss, region

loss_wrt_target_patch_jax_batched = jax.vmap(loss_wrt_target_patch_jax,in_axes=(0,None,None,None))


def batched_patch_matching(goal,pr):
    pa = goal
    if args.rgb:
        pr = pr[...,:3]
        pa = pa[...,:3]
        print('Ignoring depth')
    if args.depth_only:
        pr = pr[...,3:]
        pa = pa[...,3:]
        print('Ignoring colour')

    t = time.time()
    loss_batch, region_batch = loss_wrt_target_patch_jax_batched(pr,pa,pr[0].shape,pa.shape)
    matching_time_cost = time.time()-t
    print('Matching time cost:', matching_time_cost, '(batch size:', pr.shape[0], ')')
    return pa, loss_batch, region_batch, matching_time_cost


def execute_predictions(branches,queue,rec_queue,goal):
    if not len(queue['state']): return branches, queue, rec_queue, 0, 0
    print('Processing prediction batch')
    states = np.array(queue['state'])
    actions = np.array(queue['action'])
    M_modules = np.array(queue['M_module'])
    raw_aff_sequence = np.array(queue['raw_aff_sequence'])
    
    if pad_prediction_batches_to_full_batch_size:
        pad = batch_size-states.shape[0]
        states = np.pad(states,((0,pad),(0,0),(0,0),(0,0)),mode='constant')
        actions = np.pad(actions,((0,pad),(0,0),(0,0)),mode='constant')
        M_modules = np.pad(M_modules,((0,pad),(0,0)),mode='constant')

    t = time.time()
    predicted_states, predicted_auxes = prediction_module.predict(states,
                                                                  actions,
                                                                  M_index_sequence=M_modules)
    prediction_time_cost = time.time()-t
    print('Prediction time cost:', prediction_time_cost, '(batch size:', states.shape[0], ')')
        
    pa, loss_batch, region_batch, matching_time_cost = batched_patch_matching(goal,predicted_states[:,-1])
            
    time_cost_sum = 0
    for i, (branch_id, aff_params, aff_types, raw_affs, prediction) in enumerate(zip(queue['branch_id'],actions,M_modules,raw_aff_sequence,predicted_states)):
        t2 = time.time()
        loss, region = loss_batch[i], region_batch[i]
        time_cost = time.time()-t2
        time_cost_sum += time_cost
        pred_file_rgb = None
        
        if draw_predictions:
            id_string = branch_id_to_name(branch_id)
            # reorder aff coordinate format for drawing
            aff_draw = aff_params[-1]
            im = visualiser.draw_panel(False,prediction[-1,...,:4],action=aff_draw,M_module=aff_types[-1]).swapaxes(0,1)
            imageio.imwrite(out_dir+'/'+id_string+'__prediction(loss'+str(loss)+').png',np.uint8(np.clip(im,0,255)))
        
        if draw_matched_regions:
            region_im = np.concatenate((region[...,:3],np.repeat(region[...,3:4],3,-1)),1)
            imageio.imwrite(out_dir+'/'+id_string+'__region.png',np.uint8(255*np.clip(region_im,0,1)))
            region_im = np.concatenate((pa[...,:3],np.repeat(pa[...,3:4],3,-1)),1)
            imageio.imwrite(out_dir+'/'+id_string+'__regionTGT.png',np.uint8(255*np.clip(region_im,0,1)))
       
        branches[branch_id] = {'state': prediction[-1], 
                               'path': pred_file_rgb,
                               'aff_param_sequence': aff_params,
                               'aff_type_sequence': aff_types,
                               'raw_aff_sequence': raw_affs,
                               'loss': loss,
                               'region': region,
                               }
        rec_queue['branch_id'].append(branch_id)
        rec_queue['state'].append(prediction[-1])
        
    print('Summed patch evaluation time cost:', time_cost_sum)
    print('Prediction post-processing time cost:', time.time()-t)
    queue = init_queue(queue)
    return branches, queue, rec_queue, prediction_time_cost, matching_time_cost
    

def queue_prediction(branches,queue,rec_queue,patch,branch_id,input_state_rgbd,aff_input,M_module,raw_aff_sequence):
    queue['branch_id'].append(branch_id)
    queue['state'].append(input_state_rgbd)
    queue['action'].append(aff_input)
    queue['M_module'].append(M_module)
    queue['raw_aff_sequence'].append(raw_aff_sequence)
    prediction_time_cost, matching_time_cost = 0, 0
    if len(queue['branch_id']) == batch_size:
        print('Queue reached batch size --> executing prediction')
        branches, queue, rec_queue, prediction_time_cost, matching_time_cost = execute_predictions(branches,queue,rec_queue,patch)
    return branches, queue, rec_queue, prediction_time_cost, matching_time_cost


def parallel_recognition(branches, rec_queue):
    
    print('Recognition queue size:', len(rec_queue['state']))
    aff_chunks = []
    time_cost = 0
    for i in range(0,len(rec_queue['state']),n_recognition_parallel):
        affs, raw_time_cost = recognition_module.run_cases_parallel(rec_queue['state'][i:i+n_recognition_parallel])
        time_cost += raw_time_cost
        aff_chunks.append(affs)
        print('Detected affs', np.array(affs).shape)
    
    if len(aff_chunks):
        aff_lists = np.concatenate(aff_chunks)
        
        for i, branch_id in enumerate(rec_queue['branch_id']):
            branches[branch_id]['aff_list'] = aff_lists[i]
            print(i, 'Branch_id', branch_id, 'got aff list:', branches[branch_id]['aff_list'].shape)
    
    rec_queue = init_queue(rec_queue)
    return branches, rec_queue, time_cost


def force_jax_compiles(depth,goal):
    print('Forcing jax compiles up to depth:', depth)
    for d in range(1,depth+1):
        states = np.zeros((batch_size,)+config.state_dims_in)
        actions = np.zeros((batch_size,d,config.action_dims))
        M_modules = np.zeros((batch_size,d),dtype=int)
        predicted_states, predicted_auxes = prediction_module.predict(states,actions,M_index_sequence=M_modules)
        print('Depth', d, 'passed')
    batched_patch_matching(goal,predicted_states[:,-1])
    print('Patch matching passed')


def process_recognition_result(aff_list):
    if aff_list is None: return []
    
    aff_list[:,:2] = (aff_list[:,:2]/args.img_size-0.5)
            
    confs = aff_list[:,4]
    order = np.argsort(confs)[::-1] # argsort by confidence
    aff_list = aff_list[order] # order affs by confidence
    
    for i, aff in enumerate(aff_list):
        if aff[4] < args.conf_thres:
            aff_list = aff_list[:i]
            break
    
    # add grasp adds for turn affs (grasp affs coinciding with turn affs are not detected separately)
    aff_list = list(aff_list)
    for i, aff in enumerate(aff_list):
        M_module = aff[5].astype(int)
        if M_module == 2:
            aff2 = aff.copy()
            aff2[5] = 0
            aff_list.insert(i+1,aff2)
    aff_list = np.array(aff_list)
    return aff_list


def dig_width(args,rgb_path,goal):
    
    d_file = rgb_path[:-7]+'Depth.png'
    if not os.path.exists(d_file):
        print('Depth file not found:', d_file)
        raise FileNotFoundError('No state depth image found at path: '+d_file)
    
    force_jax_compiles(args.depth,goal)
    
    recognition_time_cost_per_depth = np.zeros(args.depth)
    prediction_time_cost_per_depth = np.zeros(args.depth)
    matching_time_cost_per_depth = np.zeros(args.depth)
    
    n_recognition_passes = 0
    n_prediction_passes = 0
    
    input_state_rgb = imageio.imread(rgb_path)[...,:3]
    input_state_d = imageio.imread(d_file)[...,None]
    input_state_d = (255-input_state_d).astype(float)
    input_state_d = np.clip(15*input_state_d-15,0,255)
    input_state_rgbd = np.concatenate([input_state_rgb,input_state_d],axis=-1)
    input_state_rgbd = input_state_rgbd.astype(np.float32)/255
    pred_file_rgb = out_dir+'/_init__state_RGB.png'
    pred_file_d = out_dir+'/_init__state_Depth.png'
    
    queue = init_queue({})
    rec_queue = init_queue({}) # adds some keys that are not used for recognition
    
    branches = {}
    branches[tuple()] = {'state': input_state_rgbd,
                         'path': rgb_path,
                         'aff_param_sequence': np.zeros((0,5)),
                         'aff_type_sequence': np.zeros(0),
                         'raw_aff_sequence': np.zeros((0,8)),
                         'loss': 0 if args.negative_goal else np.inf,
                         'region': None}
    rec_queue['branch_id'].append(tuple())
    rec_queue['state'].append(input_state_rgbd)
    
    
    branch_ids_todo = [tuple()]
    for d in range(args.depth):
        branch_ids_todo_next = []
        print('PROCESSING DEPTH LEVEL', d)
        print('Branch IDs left to process:', branch_ids_todo)
        
        branches, rec_queue, recognition_time_cost = parallel_recognition(branches,rec_queue)
        recognition_time_cost_per_depth[d] += recognition_time_cost
        
        for branch_id in branch_ids_todo:
            
            if len(branch_id) < d: continue # previously processed branches
            
            branch = branches[branch_id]

            print('Branch:')
            print('  ID:', branch_id)
            print('  Path:', branch['path'])
            
            name = branch_id_to_name(branch_id)            
            aff_list = branch['aff_list']
            n_recognition_passes += 1
            
            if draw_recognition:
                recognition_module.draw_visual(branch['state'],aff_list,out_dir+'/rec_'+name+'.png')
            
            if len(aff_list)==0: continue
            if aff_list is None: continue
            
            aff_list = process_recognition_result(aff_list)
            
            print('Number of affordances:', len(aff_list))
            
            i_aff_parametrised = 0
            for i_aff, aff in enumerate(aff_list):
                
                x, y, angle, prob, conf, cls, z, sym = aff[:8]
                if conf < args.conf_thres:
                    print('confidence below threshold --> break')
                    break
                
                xyz = np.array((x,y,z))
                M_module = cls.astype(int)
                duplicate = False
                for j_aff, aff2 in enumerate(aff_list[:i_aff]):
                    x2, y2, angle2, prob2, conf2, cls2, z2 = aff2[:7]
                    xyz2 = np.array((x2,y2,z2))
                    M_module2 = cls2.astype(int)
                    if M_module2 == M_module:
                        if (np.abs(xyz-xyz2)<duplicate_aff_threshold_xyz).all() and ((symmetry_filtering and sym>=0.5) or np.abs(angle-angle2) < duplicate_aff_threshold_angle):
                            duplicate = True
                            break
                
                if duplicate:
                    continue
                
                turn_parametrisations = [-1,1] if M_module == 2 else [0]
                for turn in turn_parametrisations:
                    aff_input = np.array((coord_multiplier*x,z,coord_multiplier*-y,angle,turn)) # xyz&angle
                    new_branch_id = branch_id+(i_aff_parametrised,)
                    raw_aff_with_free = np.concatenate((aff[:7],[turn]),axis=0)
                    raw_aff_sequence = np.concatenate((branch['raw_aff_sequence'],raw_aff_with_free[None]),axis=0)
                    aff_param_sequence = np.concatenate((branch['aff_param_sequence'],aff_input[None]),axis=0)
                    aff_type_sequence = np.concatenate((branch['aff_type_sequence'],[M_module]),axis=0).astype(int)
                    branches, queue, rec_cueue, prediction_time_cost, matching_time_cost = queue_prediction(branches,queue,rec_queue,goal,new_branch_id,input_state_rgbd,aff_param_sequence,aff_type_sequence,raw_aff_sequence)
                    n_prediction_passes += 1
                    prediction_time_cost_per_depth[d] += prediction_time_cost
                    matching_time_cost_per_depth[d] += matching_time_cost
                    branch_ids_todo_next.append(new_branch_id)
                    i_aff_parametrised += 1
            
        print('Executing remaining predictions')
        branches, queue, rec_queue, prediction_time_cost, matching_time_cost = execute_predictions(branches,queue,rec_queue,goal)
        prediction_time_cost_per_depth[d] += prediction_time_cost
        matching_time_cost_per_depth[d] += matching_time_cost
        branch_ids_todo = branch_ids_todo_next
            
    
    results = []
    for branch_id, branch in branches.items():
        results.append((branch_id,
                        branch['loss'],
                        branch['state'],
                        branch['region'],
                        branch['raw_aff_sequence']))
        
    print('Stats:')
    print('  Recognition passes:', n_recognition_passes)
    print('  Prediction passes:', n_prediction_passes)
    print('Time costs:')
    for d in range(args.depth):
        print('  Depth:', d)
        print('    Recognition:', recognition_time_cost_per_depth[d])
        print('    Prediction: ', prediction_time_cost_per_depth[d])
        print('    Matching:   ', matching_time_cost_per_depth[d])
    print('  Total:')
    print('    Recognition:', recognition_time_cost_per_depth.sum())
    print('    Prediction: ', prediction_time_cost_per_depth.sum())
    print('    Matching:   ', matching_time_cost_per_depth.sum())
    input('Press ENTER to continue...')
    
    with open(solution_dir+'/stats.txt','w') as f:
        f.write('Solution cost stats\n')
        f.write('Passes:\n')
        f.write('  Recognition: '+str(n_recognition_passes)+'\n')
        f.write('  Prediction: '+str(n_prediction_passes)+'\n')
        f.write('Time costs:\n')
        for d in range(args.depth):
            f.write('  Depth: '+str(d)+'\n')
            f.write('    Recognition: '+str(recognition_time_cost_per_depth[d])+'\n')
            f.write('    Prediction:  '+str(prediction_time_cost_per_depth[d])+'\n')
            f.write('    Matching:    '+str(matching_time_cost_per_depth[d])+'\n')
        f.write('  Total:\n')
        f.write('    Recognition: '+str(recognition_time_cost_per_depth.sum())+'\n')
        f.write('    Prediction:  '+str(prediction_time_cost_per_depth.sum())+'\n')
        f.write('    Matching:    '+str(matching_time_cost_per_depth.sum())+'\n')
    
    np.savez(solution_dir+'/stats',
             n_recognition_passes=n_recognition_passes,
             n_prediction_passes=n_prediction_passes,
             recognition_time_cost_per_depth=recognition_time_cost_per_depth,
             prediction_time_cost_per_depth=prediction_time_cost_per_depth,
             matching_time_cost_per_depth=matching_time_cost_per_depth)
        
    return results, branches


def run_case(args):
    global recognition_module, prediction_module
    prediction_module = pEMD.NN(args.run_name,iteration=args.iteration,validation_best=not args.last)
    recognition_module = detect.aff_recognition_module(args)
    print('JAX is running on', jax.lib.xla_bridge.get_backend().platform)
    
    patch = preprocess_state(args.task+'/goal/unity_state_top_RGB.png',
                            None,
                            args.task+'/goal/crop.txt')
    # depth rescaling
    patch[...,3] = 255-patch[...,3]
    patch[...,3] = 15*patch[...,3]-15
    
    makedir(args.task+'/patch')
    imageio.imwrite(args.task+'/patch/RGB.png',np.uint8(patch[...,:3]))
    imageio.imwrite(args.task+'/patch/Depth.png',np.uint8(patch[...,3]))
    patch /= 255.0
    
    rgb_source_path = args.task+'/start'
    rgb_files = glob.glob(rgb_source_path+'/unity_state_*_RGB.png')
    if not len(rgb_files):
        print('No RGB source files found in path:', rgb_source_path)
        raise FileNotFoundError('No initial state image files found in task directory: '+rgb_source_path)
    
    for rgb_path in rgb_files:
        
        print('Processing rgb file:', rgb_path)
        preprocess_state(rgb_path,args.task+'/processed')
        
        t = time.time()
        results, branches = dig_width(args,args.task+'/processed/RGB.png',patch)
        search_time_cost = time.time()-t
            
    if args.negative_goal:
        i_best = np.argmax([r[1] for r in results])
    else:
        i_best = np.argmin([r[1] for r in results])
        
    branch_id, loss, prediction, region, aff_sequence = results[i_best]
    print('Best sequence:', branch_id)
    for aff in aff_sequence:
        print('    ', aff)
    
    if prediction is not None:
        print('Writing visual for best sequence')
        im = upscale_image(prediction[:,:,:3])
        imageio.imwrite(solution_dir+'/best_predicted_outcome.png',np.uint8(np.clip(255*im,0,255)))
    if region is not None:# and region != 0:
        im = upscale_image(region[:,:,:3])
        imageio.imwrite(solution_dir+'/best_predicted_region.png',np.uint8(255*np.clip(im,0,1)))
        im = upscale_image(region[:,:,3])
        imageio.imwrite(solution_dir+'/best_predicted_region_D.png',np.uint8(255*np.clip(im,0,1)))
        im = upscale_image(patch[:,:,:3])
        imageio.imwrite(solution_dir+'/goal_patch.png',np.uint8(255*np.clip(im,0,1)))
        im = upscale_image(patch[:,:,3])
        imageio.imwrite(solution_dir+'/goal_patch_D.png',np.uint8(255*np.clip(im,0,1)))
        
    # write selected affordance sequence
    with open(solution_dir+'/plan.txt','w') as f:
        for aff in aff_sequence:
            x,y,angle,_,_,cls,z,turn = aff
            f.write(str(int(cls))+' '+str(x)+' '+str(y)+' '+str(z)+' '+str(angle)+' '+str(turn)+'\n')
    
    # visualise sequence
    state = branches[tuple()]['state']
    length = len(branch_id)
    ims = []
    rec_ims = []
    for s in range(length):
        partial_id = branch_id[:s+1]
        print('s:', s, 'partial_id:', partial_id)
        branch = branches[partial_id]
        aff_params = branch['aff_param_sequence']
        aff_types = branch['aff_type_sequence']
        id_string = branch_id_to_name(partial_id)
        recognition_module.run_case(state,args.conf_thres,True,solution_dir+'/seq'+str(s)+'a_affordances_'+id_string+'.png')
        # draw prediction image
        im = visualiser.draw_panel(False,state[...,:4],action=aff_params[-1],M_module=aff_types[-1]).swapaxes(0,1)
        # upscale prediction image (for consistency with recognition image and reduced blur in ms word...)
        im = upscale_image(im)
        ims.append(im)
        # save prediction image
        imageio.imwrite(solution_dir+'/seq'+str(s)+'b_action_'+id_string+'.png',np.uint8(np.clip(im,0,255)))
        state = branch['state']
    
    id_string = branch_id_to_name(branch_id)
    im = visualiser.draw_panel(False,state[...,:4]).swapaxes(0,1)
    im = upscale_image(im)
    ims.append(im)
    imageio.imwrite(solution_dir+'/seq'+str(len(branch_id))+'b_predicted_outcome_'+id_string+'.png',np.uint8(np.clip(im,0,255)))
    
    ims_vert = [np.pad(im,((2*image_size_multiplier,2*image_size_multiplier),(0,0),(0,0)),mode='constant',constant_values=255) for im in ims]
    ims_hori = [np.pad(im[:,:im.shape[1]//2],((0,0),(2*image_size_multiplier,2*image_size_multiplier),(0,0)),mode='constant',constant_values=255) for im in ims]
    im_vert = np.concatenate(ims_vert,axis=0)
    im_hori = np.concatenate(ims_hori,axis=1)
    im_vert = im_vert[2*image_size_multiplier:-2*image_size_multiplier,:]
    im_hori = im_hori[:,2*image_size_multiplier:-2*image_size_multiplier]
    
    imageio.imwrite(solution_dir+'/plan_RGBD_vertical.png',np.uint8(np.clip(im_vert,0,255)))
    im_vert = im_vert[:,:im.shape[1]//2]
    imageio.imwrite(solution_dir+'/plan_RGB_vertical.png',np.uint8(np.clip(im_vert,0,255)))
    imageio.imwrite(solution_dir+'/plan_RGB_horizontal.png',np.uint8(np.clip(im_hori,0,255)))
    
    for s in range(length):
        partial_id = branch_id[:s]
        if draw_recognition:
            id_string = branch_id_to_name(partial_id)
            rec_im_path = out_dir+'/rec_'+id_string+'.png'
            rec_im = imageio.imread(rec_im_path)
        else:
            branch = branches[partial_id]
            rec_im = recognition_module.draw_visual(branch['state'],branch['aff_list'])
        rec_ims.append(rec_im)
    
    rec_ims_vert = [np.pad(rec_im,((2*image_size_multiplier,2*image_size_multiplier),(0,0),(0,0)),mode='constant',constant_values=255) for rec_im in rec_ims]
    rec_ims_hori = [np.pad(rec_im,((0,0),(2*image_size_multiplier,2*image_size_multiplier),(0,0)),mode='constant',constant_values=255) for rec_im in rec_ims]
    
    rec_im_vert = np.concatenate(rec_ims_vert,axis=0)
    rec_im_hori = np.concatenate(rec_ims_hori,axis=1)
    
    rec_im_vert = rec_im_vert[2*image_size_multiplier:-2*image_size_multiplier,:]
    rec_im_hori = rec_im_hori[:,2*image_size_multiplier:-2*image_size_multiplier]
    
    imageio.imwrite(solution_dir+'/recognitions_vertical.png',np.uint8(np.clip(rec_im_vert,0,255)))
    imageio.imwrite(solution_dir+'/recognitions_horizontal.png',np.uint8(np.clip(rec_im_hori,0,255)))
    print('Total search time:', search_time_cost)
    

if __name__ == '__main__':

    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name',type=str,
                        help='Prediction net (EMD) run name.')
    parser.add_argument('--task', type=str, default='', 
                        help='Path to the task directory containing the task to plan for.')
    parser.add_argument('-d','--depth',type=int,default=4,
                        help='Depth to expanding search tree to (defaults to 4).')
    parser.add_argument('-n, --negative', dest='negative_goal', default=False, action='store_true',
                        help='Negative goal mode: tries to avoid/eliminate the goal patch (defaults to False).')
    parser.add_argument('-c, --rgb', dest='rgb', default=False, action='store_true',
                        help='Use RGB (no D) for goal matching.')
    parser.add_argument('-z, --depth_only', dest='depth_only', default=False, action='store_true',
                        help='Use D (no RGB) for goal matching.')
    parser.add_argument('-l, --last', dest='last', default=False, action='store_true',
                        help='Forces loading of the last iteration instead of the validation-best net (defaults to False).')
    parser.add_argument('-i','--iteration',type=int,default=None,
                        help='Training iteration to load (defaults to highest iteration found).')
    #parser.add_argument('-s','--set',type=str,default='test',
    #                    help='Data subset to evaluate. should be one of train/validation/test. (defaults to test).')
    #parser.add_argument('-m','--marker',type=str,default='',
    #                    help='Marker to identify result directory.'+
    #                         'When running multiple planning evaluations in parallel, set different markers for each to prevent them from interfering.')
    #parser.add_argument('-q', '--quantitative_evaluation',dest='quantitative_evaluation_mode', default=False, action='store_true',
    #                    help='Quantitative evaluation of ability to predict future affordances.')
    #parser.add_argument('-b', '--sampling_budget',type=int,default=-1,
    #                    help='Sampling budget for quantitative evaluation of DAF mode (default: verify GT).')
    
    # args from yolo
    parser.add_argument('--weights', nargs='+', type=str, default='yolov4-p5.pt', help='Recognition net (ScaledYOLOv4) model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=128, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.75, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    
    args = parser.parse_args()
    
    goal_dir = args.task+'/goal'
    out_dir = args.task+'/out'
    solution_dir = args.task+'/solution'
    args.output = args.task+'/recognition'
    
    makedir(out_dir)
    clear_dir(out_dir)
    
    makedir(solution_dir)
    clear_dir(solution_dir)
    
    makedir(args.output)
    clear_dir(args.output)
    
    print('get visualiser')
    visualiser = emd_utils.get_visualiser(True)
    
    if args.rgb:
        print('using only colour (no depth) for goal matching')
    if args.negative_goal:
        print('using negative goal')
        
    run_case(args)
