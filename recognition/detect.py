
import argparse
import os
import platform
import shutil
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, plot_one_box, apply_camera_perspective, strip_optimizer)
from utils.torch_utils import select_device, load_classifier, time_synchronized

import imageio
import numpy as np
from cfg import NUM_CH

aff_colours = [[255,127,127],
               [127,255,127],
               [127,127,255]]

size_multiplier = 6

class aff_recognition_module():
    def __init__(self,opt,save_img=False):
        self.opt = opt
        out, weights, view_img, save_txt, self.imgsz = \
            opt.output, opt.weights, opt.view_img, opt.save_txt, opt.img_size

        # Initialize
        self.device = select_device(opt.device)
        if os.path.exists(out):
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        self.model = attempt_load(weights, map_location=self.device)  # load FP32 model
        self.imgsz = check_img_size(self.imgsz, s=self.model.stride.max())  # check img_size
        if self.half:
            self.model.half()  # to FP16

        # Second-stage classifier
        self.classify = False
        if self.classify:
            modelc = load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=self.device)['model'])  # load weights
            modelc.to(self.device).eval()

        # Get names
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

        # Run inference
        img = torch.zeros((1, NUM_CH, self.imgsz, self.imgsz), device=self.device)  # init img
        _ = self.model(img.half() if self.half else img) if self.device.type != 'cpu' else None  # run once


    def draw_gt(self,im,gt,save_path):
        im = np.repeat(im,size_multiplier,axis=0)
        im = np.repeat(im,size_multiplier,axis=1)
        im *= 255
        sh = im.shape
        for aff in gt:
            gt_cls = aff[0]
            gt_xy = aff[1:3]*sh[:2]
            gt_z = aff[4]
            gt_sym = aff[5]
            gt_angle = aff[3]*360
            i_cls = int(gt_cls)
            plot_one_box(gt_xy, im, label=i_cls, color=aff_colours[int(gt_cls)], z=gt_z, sym=gt_sym, angle=gt_angle, line_thickness=1)
        imageio.imwrite(save_path, im[:,:,:3])
        
        
    def apply_perspective(self,x,y,z):
        return apply_camera_perspective(x,y,z)
        
        
    def run_case(self,source,conf_threshold,save_image=True,output_filepath=None):
        
        t0 = time.time()
        
        out, weights, view_img, save_txt = \
            self.opt.output, self.opt.weights, self.opt.view_img, self.opt.save_txt
        
        if type(source) is str:
            webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')
        
            # Set Dataloader
            vid_path, vid_writer = None, None
            if webcam:
                view_img = True
                cudnn.benchmark = True  # set True to speed up constant image size inference
                dataset = LoadStreams(source, img_size=self.imgsz)
            else:
                save_img = True
                dataset = LoadImages(source, img_size=self.imgsz)
            dataset_mode = dataset.mode
            
        else:
            webcam = False
            save_img = save_image
            view_img = False
            im = np.array(source)
            im = np.transpose(im,(2,0,1))
            dataset = [[output_filepath,im*255.0,(255*source).astype(np.uint8),None]]
            dataset_mode = 'images'
        
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            with torch.no_grad():
                pred = self.model(img, augment=self.opt.augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes, agnostic=self.opt.agnostic_nms)
            t2 = time_synchronized()

            # Apply Classifier
            if self.classify:
                pred = apply_classifier(pred, modelc, img, im0s)

            # upscale image
            im0s = np.repeat(im0s,size_multiplier,axis=0)
            im0s = np.repeat(im0s,size_multiplier,axis=1)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if det is None: continue
                if webcam:  # batch_size >= 1
                    p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
                else:
                    p, s, im0 = path, '', im0s

                if output_filepath is None:
                    save_path = str(Path(out) / Path(p).name)
                else:
                    save_path = output_filepath
                    
                txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset_mode == 'video' else '')
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                original_xy = det[:, :2]
                
                if isinstance(det, torch.Tensor):
                    det = np.array(det.cpu())
                
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det_scaled_xy = scale_coords(img.shape[2:], det[:, :2].copy(), im0.shape).round()
                    
                    confs = det[:,4]
                    
                    # Print results
                    for c in np.unique(det[:, -1]):
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += '%g %ss, ' % (n, self.names[int(c)])  # add to string
                    
                    # Write results
                    for i_aff, vals in enumerate(det):
                        if isinstance(vals, torch.Tensor):
                            vals = np.array(vals.cpu())
                        xya = vals[:3]
                        prob = vals[3]
                        conf = vals[4]
                        cls = vals[5]
                        free = vals[6:]
                        if save_txt:  # Write to file
                            if conf > conf_threshold:
                                with open(txt_path + '.txt', 'a') as f:
                                    f.write(('%g ' * 7 + '\n') % (cls, conf, *xya, prob, *free))  # label format
                            
                        if save_img or view_img:  # Add bbox to image
                            if conf > conf_threshold:
                                print(i_aff)
                                print('  cls:', cls)
                                print('  xya:', xya)
                                print('  prob:', prob)
                                print('  conf:', conf)
                                print('  free:', free)
                                i_cls = int(cls)
                                label = '%s' % (self.names[i_cls])
                                xy = det_scaled_xy[i_aff]
                                plot_one_box(xy, im0, label=i_cls, color=aff_colours[int(cls)], z=free[0], sym=free[1], angle=360*xya[2], line_thickness=1, conf=conf)
                    
                print('Done. (%.3fs)' % (t2 - t1))
                    
                # Stream results
                if view_img:
                    cv2.imshow(p, im0)
                    if cv2.waitKey(1) == ord('q'):  # q to quit
                        raise StopIteration

                # Save results (image with detections)
                if save_img:
                    if dataset_mode == 'images':
                        imageio.imwrite(save_path, np.uint8(im0[:,:,:3]))
                        print('saved to:', save_path)
                    else:
                        if vid_path != save_path:  # new video
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()  # release previous video writer

                            fourcc = 'mp4v'  # output video codec
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                        vid_writer.write(im0)
            
        if save_txt or save_img:
            print('Results saved to %s' % Path(out))
            if platform == 'darwin' and not self.opt.update:  # MacOS
                os.system('open ' + save_path)
        
        print('Done. (%.3fs)' % (time.time() - t0))
        return det
    
    
    def run_cases_parallel(self,ims):
        ims = np.array(ims)
        print('Running detection on image batch of shape', ims.shape, 'with range', ims.min(), ims.max())
        ims = np.transpose(ims,(0,3,1,2))
        ims = torch.from_numpy(ims).to(self.device)
        ims = ims.half() if self.half else ims.float()  # uint8 to fp16/32

        # Inference
        t = time_synchronized()
        with torch.no_grad():
            affs = self.model(ims, augment=self.opt.augment)[0]
            affs = non_max_suppression(affs, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes, agnostic=self.opt.agnostic_nms)
        raw_time_cost = time.time() - t
        
        print('Done. (%.3fs)' % raw_time_cost)
        affs = [(np.empty((0,8)) if aff is None else np.array(aff.cpu())) for aff in affs]
        print('Detected affs', len(affs))
        
        return affs, raw_time_cost
    
    
    def draw_visual(self,image,det,output_filepath=None):

        out, weights, view_img, save_txt = \
            self.opt.output, self.opt.weights, self.opt.view_img, self.opt.save_txt
        
        im = 255*image.copy()
        
        # upscale image
        im = np.repeat(im,size_multiplier,axis=0)
        im = np.repeat(im,size_multiplier,axis=1)
        im0 = im
        
        if output_filepath is not None:
            p, s = output_filepath, ''
            save_path = output_filepath
        
            txt_path = str(Path(out) / Path(p).stem)
            s += '%gx%g ' % image.shape[:2]  # print string
        
        gn = torch.tensor(im.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        
        if isinstance(det, torch.Tensor):
            det = np.array(det.cpu())
        
        if det is not None and len(det):
            # Rescale boxes from img_size to im size
            det_scaled_xy = scale_coords(image.shape[:2], det[:, :2].copy(), im.shape).round()
            
            confs = det[:,4]
            
            # Write results
            for i_aff, vals in enumerate(det):
                if isinstance(vals, torch.Tensor):
                    vals = np.array(vals.cpu())
                xya = vals[:3]
                prob = vals[3]
                conf = vals[4]
                cls = vals[5]
                free = vals[6:]
                if save_txt:  # Write to file
                    if conf > conf_threshold:#self.opt.conf_thres:
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 7 + '\n') % (cls, conf, *xya, prob, *free))  # label format
                    
                print(i_aff)
                print('  cls:', cls)
                print('  xya:', xya)
                print('  prob:', prob)
                print('  conf:', conf)
                print('  free:', free)
                i_cls = int(cls)
                label = '%s' % (self.names[i_cls])
                xy = det_scaled_xy[i_aff]
                plot_one_box(xy, im, label=i_cls, color=aff_colours[int(cls)], z=free[0], sym=free[1], angle=360*xya[2], line_thickness=1, conf=conf)
        
        # Stream results
        if view_img:
            cv2.imshow(p, im)
            if cv2.waitKey(1) == ord('q'):  # q to quit
                raise StopIteration

        # Save results (image with detections)
        if output_filepath is not None:
            imageio.imwrite(save_path,np.uint8(im[:,:,:3]))
            print('Image saved to:', save_path)
            print('Results saved to %s' % Path(out))
            if platform == 'darwin' and not self.opt.update:  # MacOS
                os.system('open ' + save_path)
        
        return im[:,:,:3]
    

def setup():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov4-p5.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.9, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['']:
                recognition_module = aff_recognition_module(opt)
                strip_optimizer(opt.weights)
        else:
            recognition_module = aff_recognition_module(opt)
    return recognition_module, opt


if __name__ == '__main__':
    recognition_module, opt = setup()
    recognition_module.run_case(opt.source)
    
