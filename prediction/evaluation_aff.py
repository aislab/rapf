import argparse
import jax
import time
import numpy as np
import sys
import os
import shutil
import imageio
import config
import pEMD
import utils
from data_manager_aff import data_paths
import visualiser_aff as visualiser

goal_patch_path = ('goal_patch_RGB.png', 'goal_patch_D.png')

data_manager = None

visualisation = True
draw_from_all_steps = False

max_sequence_length = 4

max_tasks = 500

def variations(im_rgbd,ims_rgbd,ims_rgb):
    im_rgb = im_rgbd[:,:im_rgb.shape[1]//2]
    im_rgbd = np.pad(im_rgbd,((1,1),(1,1),(0,0)),mode='constant',constant_values=255)
    im_rgb = np.pad(im_rgb,((1,1),(1,1),(0,0)),mode='constant',constant_values=255)
    ims_rgb.append(im_rgb)
    ims_rgbs.append(im_rgbd)
    
def mod(im,includeD):
    if not includeD:
        im = im[:,:im.shape[1]//2]
    im = np.pad(im,((1,1),(1,1),(0,0)),mode='constant',constant_values=255)
    return im

def evaluate_prediction(args):
    out_dir = args.run_name+'/prediction_evalution'
    utils.makedir(out_dir)
    nn = pEMD.NN(args.run_name,iteration=args.iteration,validation_best=not args.last)
    scores_per_step = [[] for _ in range(max_sequence_length)]
    scores_per_step_masked = [[] for _ in range(max_sequence_length)]
    
    n_tasks = data_manager.get_example_count(args.set)
    if max_tasks > 0: n_tasks = min(n_tasks,max_tasks)
    
    for i_data in range(n_tasks):
        print('i_data:', i_data)
        example = utils.request_example(args.set,i_data)
        
        for from_step in range(example['n_steps'] if (visualisation and draw_from_all_steps) else 1):
        
            predicted_states, predicted_auxes = nn.predict(example['states'][0][from_step],
                                                        example['actions'][0][from_step:],
                                                        state_extra_sequence=example['state_extras'][0][from_step:],
                                                        hidden=example['hidden'][0],
                                                        M_index_sequence=example['M_indices'][0][from_step:])
            
            if from_step==0:
                for s, pr in enumerate(predicted_states):
                    #pr = predicted_states[s,...,:4]
                    gt = example['states'][0][s+1,...,:4]
                    d = np.abs(gt-pr).mean()
                    print(s,d)
                    scores_per_step[s].append(d)
                    mask = np.where(gt==example['states'][0][s,...,:4],0,1)
                    d = (mask*np.abs(gt-pr)).sum()/mask.sum()
                    scores_per_step_masked[s].append(d)
            
            if visualisation:
                ims_pr = []
                ims_pr_state_only = []
                im = visualiser.draw_panel(False,example['states'][0][from_step][:,:,:4],
                                                 action=example['actions'][0][from_step],
                                                 M_module=example['M_indices'][0][from_step]).swapaxes(0,1)
                ims_pr.append(im)
                im = visualiser.draw_panel(False,example['states'][0][from_step][:,:,:4]).swapaxes(0,1)
                ims_pr_state_only.append(im)
                for s, pr in enumerate(predicted_states): 
                    if s == len(predicted_states)-1:
                        a, m = None, None
                    else:
                        a = example['actions'][0][from_step+s+1]
                        m = example['M_indices'][0][from_step+s+1]
                    im = visualiser.draw_panel(False,pr[...,:4],action=a,M_module=m).swapaxes(0,1)
                    ims_pr.append(im)
                    im = visualiser.draw_panel(False,pr[...,:4]).swapaxes(0,1)
                    ims_pr_state_only.append(im)
                
                ims_gt = []
                ims_gt_state_only = []
                for s, gt in enumerate(example['states'][0][from_step:]):
                    if s == example['states'][0][from_step:].shape[0]-1:
                        a, m = None, None
                    else:
                        a = example['actions'][0][from_step+s]
                        m = example['M_indices'][0][from_step+s]
                    im = visualiser.draw_panel(False,gt[...,:4],action=a,M_module=m).swapaxes(0,1)
                    ims_gt.append(im)
                    im = visualiser.draw_panel(False,gt[...,:4]).swapaxes(0,1)
                    ims_gt_state_only.append(im)
                
                im_gt_rgbd_v = np.concatenate([mod(im,1) for im in ims_gt],axis=0)
                im_gt_rgbd_h = np.concatenate([mod(im,1) for im in ims_gt],axis=1)
                im_gt_rgbd_v_so = np.concatenate([mod(im,1) for im in ims_gt_state_only],axis=0)
                im_gt_rgbd_h_so = np.concatenate([mod(im,1) for im in ims_gt_state_only],axis=1)
                
                im_gt_rgb_v = np.concatenate([mod(im,0) for im in ims_gt],axis=0)
                im_gt_rgb_h = np.concatenate([mod(im,0) for im in ims_gt],axis=1)
                im_gt_rgb_v_so = np.concatenate([mod(im,0) for im in ims_gt_state_only],axis=0)
                im_gt_rgb_h_so = np.concatenate([mod(im,0) for im in ims_gt_state_only],axis=1)
                
                im_pr_rgbd_v = np.concatenate([mod(im,1) for im in ims_pr],axis=0)
                im_pr_rgbd_h = np.concatenate([mod(im,1) for im in ims_pr],axis=1)
                im_pr_rgbd_v_so = np.concatenate([mod(im,1) for im in ims_pr_state_only],axis=0)
                im_pr_rgbd_h_so = np.concatenate([mod(im,1) for im in ims_pr_state_only],axis=1)
                
                im_pr_rgb_v = np.concatenate([mod(im,0) for im in ims_pr],axis=0)
                im_pr_rgb_h = np.concatenate([mod(im,0) for im in ims_pr],axis=1)
                im_pr_rgb_v_so = np.concatenate([mod(im,0) for im in ims_pr_state_only],axis=0)
                im_pr_rgb_h_so = np.concatenate([mod(im,0) for im in ims_pr_state_only],axis=1)
                
                im = np.concatenate((im_gt_rgbd_v,
                                     im_pr_rgbd_v,
                                     im_gt_rgbd_v_so,
                                     im_pr_rgbd_v_so),axis=1)
                imageio.imwrite(out_dir+'/'+args.set+str(i_data)+'_step'+str(from_step)+'_rgbd_v.png',np.uint8(np.clip(im,0,255)))
                
                im = np.concatenate((im_gt_rgbd_h,
                                     im_pr_rgbd_h,
                                     im_gt_rgbd_h_so,
                                     im_pr_rgbd_h_so),axis=0)
                imageio.imwrite(out_dir+'/'+args.set+str(i_data)+'_step'+str(from_step)+'_rgbd_h.png',np.uint8(np.clip(im,0,255)))
                
                im = np.concatenate((im_gt_rgb_v,
                                     im_pr_rgb_v_so),axis=1)
                imageio.imwrite(out_dir+'/'+args.set+str(i_data)+'_step'+str(from_step)+'_compact_v.png',np.uint8(np.clip(im,0,255)))
                
                im = np.concatenate((im_gt_rgb_h,
                                     im_pr_rgb_h_so),axis=0)
                imageio.imwrite(out_dir+'/'+args.set+str(i_data)+'_step'+str(from_step)+'_compact_h.png',np.uint8(np.clip(im,0,255)))
                
        
    for s in range(max_sequence_length):
        print('step:', s)
        print('  n:', len(scores_per_step[s]))
        score_mean = np.mean(scores_per_step[s])
        score_std = np.std(scores_per_step[s])
        masked_score_mean = np.mean(scores_per_step_masked[s])
        masked_score_std = np.std(scores_per_step_masked[s])
        print('  score:', score_mean, '(', score_std, ')')
        print('  masked:', masked_score_mean, '(', masked_score_std, ')')
    print('all steps:')
    score_mean = np.mean(np.concatenate(scores_per_step))
    score_std = np.std(np.concatenate(scores_per_step))
    masked_score_mean = np.mean(np.concatenate(scores_per_step_masked))
    masked_score_std = np.std(np.concatenate(scores_per_step_masked))
    print('  score:', score_mean, '(', score_std, ')')
    print('  masked:', masked_score_mean, '(', masked_score_std, ')')
    

if __name__ == '__main__':

    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('run_name',type=str)
    parser.add_argument('-l, --last', dest='last', default=False, action='store_true',
                        help='Forces loading of the last iteration instead of the validation-best net (defaults to False).')
    parser.add_argument('-i','--iteration',type=int,default=None,
                        help='Training iteration to load (defaults to highest iteration found).')
    parser.add_argument('-s','--set',type=str,default='test',
                        help='Data subset to evaluate. should be one of train/validation/test. (defaults to test).')
    parser.add_argument('-m','--marker',type=str,default='',
                        help='Marker to identify result directory.'+
                             'When running multiple planning evaluations in parallel, set different markers for each to prevent them from interfering.')
    args = parser.parse_args()
    
    print('Run name:', args.run_name)

    if args.set not in ('test','train','validation'):
        print('*'*60)
        print('ERROR: Invalid value for "set" flag.')
        print('Should be one of train/validation/test.')
        print('*'*60)
        exit(0)
    
    data_manager = utils.get_data_manager(False,True)
    visualiser = utils.get_visualiser(True)
    
    evaluate_prediction(args)
        
    data_manager.shutdown()
