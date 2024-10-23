"""
This script performs evaluation of a trained affordance recognition network.
"""

import argparse
import numpy as np
import os
import imageio
import glob
import matplotlib.pyplot as plt
import detect
import errno
plt.ioff()


dataset_dir = './mydata/DATASET_NAME/'

n_affordance_types = 3
xy_threshold = 0.025
z_threshold = 0.05
angle_threshold = 5

duplicate_aff_threshold_xy = 0.025
duplicate_aff_threshold_z = 0.025
duplicate_aff_threshold_angle = 0.05


# makes dir if it does not exist.
# if clear is true and dir exists, its content is cleared.
def makedir(path,clear=False):
    try:
        os.makedirs(path)
        print('created directory:',path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
            

def find_best_threshold(confidences,spuriousness):
    aff_classification_record = confidences
    aff_gt_record = 1-spuriousness
    order = np.argsort(aff_classification_record)
    aff_classification_record = aff_classification_record[order]
    aff_gt_record = aff_gt_record[order]
    best_threshold_set = []
    best_correct_count = 0
    for i in range(len(aff_classification_record)-1):
        threshold = (aff_classification_record[i]+aff_classification_record[i+1])/2
        n_correct = (aff_gt_record[:i+1]==0).sum()+(aff_gt_record[i+1:]==1).sum()
        print('Correct @ threshold', threshold, ':', n_correct)
        if n_correct == best_correct_count:    
            best_threshold_set.append(threshold)
        if n_correct > best_correct_count:
            best_correct_count = n_correct
            best_threshold_set = [threshold]
    print('Best threshold set:', best_threshold_set)
    best_threshold = np.mean(best_threshold_set)
    print('Best correct count:', best_correct_count, '/', len(aff_gt_record), '=', best_correct_count/len(aff_gt_record))
    print('at threshold:', best_threshold)
    print('NOTE: threshold is NOT automatically set to this value')
    
    fig, ax = plt.subplots()
    cc = ['green' if g else 'red' for g in aff_gt_record]
    ax.scatter(np.arange(len(aff_classification_record)),aff_classification_record,c=cc)
    plt.savefig('threshold_'+args.set+'.png')
    

def evaluate(args):
    rgb_paths = sorted(glob.glob(os.path.join(dataset_dir+'/'+args.set, '[0-9]*_RGB.png')))
    if args.n_data is not None:
        rgb_paths = rgb_paths[:int(args.n_data)]
        
    makedir('recognition_evaluation',True)
    recognition_module = detect.aff_recognition_module(args)

    n_aff_matched = np.zeros(n_affordance_types)
    n_aff_total = np.zeros(n_affordance_types)
    n_detections_per_class = np.zeros(n_affordance_types)
    n_spurious = 0
    plot_aff_index = 0

    plot_fig, plot_ax = plt.subplots(1)

    confidences = []
    spuriousness = []

    for i_example, rgb_path in enumerate(rgb_paths):
        
        d_file = rgb_path[:-7]+'Depth.png'
        input_state_rgb = imageio.imread(rgb_path)[...,:3]
        input_state_d = imageio.imread(d_file)[...,None]
        input_state_rgbd = np.concatenate([input_state_rgb,input_state_d],axis=-1)
        input_state_rgbd = input_state_rgbd.astype(np.float32)/255
        gt_im = input_state_rgbd.copy()
        
        aff_list = recognition_module.run_case(input_state_rgbd,args.conf_thres,True,'recognition_evaluation/'+args.set+str(i_example).zfill(4)+'.png')
        
        if aff_list is None:
            aff_list = np.zeros((0,8))
            
        print('Initial yield:', aff_list.shape)
        if len(aff_list):
            print('Confidence range:', aff_list[:,4].min(), aff_list[:,4].max())
        aff_list = aff_list[aff_list[:,4]>=args.conf_thres]
        print('Left after confidence thresholding:', aff_list.shape)
        
        confs = aff_list[:,4]
        order = np.argsort(confs)[::-1] # argsort by confidence
        aff_list = aff_list[order] # order affs by confidence
        
        aff_list[:,:2] /= args.img_size
        
        rec_cls = aff_list[:,5]
        for cls in rec_cls:
            n_detections_per_class[int(cls)] += 1
        
        # filter out duplicates
        keep = []
        for (i_aff, aff) in enumerate(aff_list):
            duplicate = False
            x, y, angle, prob, conf, cls, z = aff[:7]
            xy = np.array((x,y))
            M_module = cls.astype(int)
            for j_aff, aff2 in enumerate(aff_list[:i_aff]):
                x2, y2, angle2, prob2, conf2, cls2, z2 = aff2[:7]
                xy2 = np.array((x2,y2))
                M_module2 = cls2.astype(int)
                if M_module2 == M_module:
                    if np.linalg.norm((xy-xy2)) < duplicate_aff_threshold_xy and \
                       np.abs(z-z2) < duplicate_aff_threshold_z and \
                       np.abs(angle-angle2) < duplicate_aff_threshold_angle:
                        duplicate = True
                        break
            if not duplicate:
                keep.append(i_aff)
                
        aff_list = aff_list[keep]
        print('Left after duplicate supression:', aff_list.shape)
        
        # normalise
        rec_xy = aff_list[:,:2]
        rec_z = aff_list[:,6:7]
        rec_sym = aff_list[:,7]
        rec_xyz = np.concatenate((rec_xy,rec_z),axis=1)
        rec_angle = aff_list[:,2]*360
        rec_cls = aff_list[:,5]
        rec_conf = aff_list[:,4]
        
        aff_path = rgb_path[:-3]+'txt'
        with(open(aff_path)) as f:
            lines = f.readlines()
        gt = np.array([[float(v) for v in line.split()] for line in lines])
        gt_output_file_name = 'recognition_evaluation/'+args.set+str(i_example).zfill(4)+'gt.png'
        recognition_module.draw_gt(gt_im,gt,gt_output_file_name)
        gt_cls = gt[:,0]
        gt_xy = gt[:,1:3]
        gt_z = gt[:,4:5]
        gt_sym = gt[:,5]
        gt_xyz = np.concatenate((gt_xy,gt_z),axis=1)
        gt_angle = gt[:,3]*360
        
        n_ok = 0
        spurious = np.ones(aff_list.shape[0])
        for i_gt in range(len(gt)):
            cls = int(gt_cls[i_gt])
            print('Matching to ground truth:',cls,gt_xyz[i_gt],gt_angle[i_gt])
            n_aff_total[cls] += 1
            same_cls = np.where(cls==rec_cls)[0]
            d_xy = np.linalg.norm(gt_xy[i_gt:i_gt+1]-rec_xy[same_cls],axis=-1)
            d_z = np.linalg.norm(gt_z[i_gt:i_gt+1]-rec_z[same_cls],axis=-1)
            ii_rec = same_cls[np.logical_and(d_xy<xy_threshold,d_z<z_threshold)]
            ii_rec2 = []
            min_a_diff = np.inf
            for i_rec in ii_rec:
                rec_a = rec_angle[i_rec]
                gt_a = gt_angle[i_gt]
                a_diff = np.abs(gt_a-rec_a)
                if (a_diff<angle_threshold):
                    if np.sign(rec_sym[i_rec]-0.5) == np.sign(gt_sym[i_gt]-0.5):
                        ii_rec2.append(i_rec)
                        spurious[i_rec] = False
                        if a_diff<min_a_diff:
                            min_a_diff = a_diff
                            best_i_rec = i_rec
            
            if not len(ii_rec2):
                print('  no valid match')
                continue
                
            dd = np.linalg.norm(gt_xyz[i_gt:i_gt+1]-rec_xyz[ii_rec2],axis=-1)
            
            d = dd.min()
            print('Ground truth:', cls,gt_xyz[i_gt], gt_angle[i_gt])
            print('Matched to:  ', rec_xyz[best_i_rec], rec_angle[best_i_rec])
            n_ok += 1
            n_aff_matched[cls] += 1
            
        n_spurious += spurious.sum()
        
        confidences.append(rec_conf)
        spuriousness.append(spurious)
        
        for i in range(len(aff_list)):
            plot_ax.scatter(plot_aff_index,rec_conf[i],s=2,color='red' if spurious[i] else 'black')
            plot_aff_index += 1
            
        print('Example', i_example, 'score:', n_ok/len(gt), '(', n_ok, '/', len(gt), ')')
        
    confidences = np.concatenate(confidences)
    spuriousness = np.concatenate(spuriousness)
    find_best_threshold(confidences,spuriousness)
        
    print('Total score per affordance type:', n_aff_matched/n_aff_total, '(all:', n_aff_matched.sum()/n_aff_total.sum(), ')')
    print('Detections:', n_detections_per_class, '(all:', np.sum(n_detections_per_class).astype(int),')')
    print('Matched:   ', n_aff_matched, '(all:', np.sum(n_aff_matched).astype(int),')')
    print('Actual:    ', n_aff_total, '(all:', np.sum(n_aff_total).astype(int),')')
    print('Spurious recs:', n_spurious, '  mean spurious recs per image:', n_spurious/len(rgb_paths))
        
    print('Writing plot...')
    plot_file_name = 'recognition_evaluation/'+args.set+'_spurious.png'
    plot_fig.savefig(plot_file_name,dpi=300)
    print('Done')


if __name__ == '__main__':

    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('-s','--set',type=str,default='test',
                        help='Data subset to evaluate. should be one of train/validation/test. (defaults to test).')
    parser.add_argument('-n','--n_data',type=str,default=None,
                        help='Maximum number of examples to evaluate (defaults to all examples in subset).')
    
    # Arguments from yolo
    parser.add_argument('--weights', nargs='+', type=str, default='yolov4-p5.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=128, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.8261324763298035, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    
    args = parser.parse_args()
    
    if args.set not in ('test','train','valid'):
        print('*'*60)
        print('ERROR: Invalid value for "set" flag.')
        print('Should be one of train/valid/test.')
        print('*'*60)
        raise ValueError('Invalid value for "set" flag')
    
    evaluate(args)
        
