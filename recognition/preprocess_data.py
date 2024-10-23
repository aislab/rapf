
"""
Copyright 2024 Autonomous Intelligence and Systems (AIS) Lab, Shinshu University & EPSON AVASYS Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation 
files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, 
modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software 
is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES 
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE 
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR 
IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""
"""
This script processes a dataset generated using the simulation environment into a suitable format for training/evaluation of 
the affordance recognition network.
"""

import os
import glob
import shutil
import numpy as np
import errno
import imageio

# The directory containing the dataset to process
src_dirs = ['PATH/TO/SOURCE/DATASET']
# The directory to save the processed dataset to
dst_dir = './mydata/DATASET_NAME/'

# Maximum manipulation sequence length in the dataset
max_sequence_length = 4

# Whether to shuffle the dataset.
# Shuffling should be off if the data split needs to match the prediction side.
shuffle_data = False

aff_file_name_base = 'affordances'

view = 'top'

# Number of examples to allocate to the test and validation sets.
# The training set will contain all remaining examples.
n_test = 500
n_validation = 500

# Integer factor for downscaling state images from the dataset.
downscale_factor = 2
# Number of pixel to crop off each edge of state images from the dataset.
crop = 128


# Whether to add symmetrical copies of symmetrical affordances.
add_symmetricals = 1

file_limit = None

# List of affordance types, used to map affordance names to category integers
aff_type_list = ['grasp', 'place', 'turn']


def pad(path):
    path, name = path.rsplit('/',1)
    return path+'/'+name.zfill(10)

src_files = []
for src_dir in src_dirs:
    src_files += sorted(glob.glob(os.path.join(src_dir, '[0-9]*_0_affordances.txt')))

print('found', len(src_files), 'txt files')

# Shuffle data using fixed seed
if shuffle_data:
    rng_state = np.random.get_state()
    np.random.seed(1)
    order = np.arange(len(src_files))
    np.random.shuffle(order)
    np.random.set_state(rng_state)
    src_files2 = [src_files[i] for i in order]
    src_files = src_files2

if (file_limit is not None) and (file_limit>0) and (len(src_files)>file_limit):
    print('limiting data to', file_limit)
    src_files = src_files[:file_limit]
len_src_files = len(src_files)

train_num = int(len_src_files) - n_test - n_validation
val_num = n_validation
test_num = n_test

train_files = src_files[:train_num]
val_files = src_files[train_num:(train_num + val_num)]
test_files = src_files[(train_num + val_num):(train_num + val_num + test_num)]

print('Example count:', len_src_files)
print('Splitting into:')
print('  Test:      ', len(test_files))
print('  Validation:', len(val_files))
print('  Train:     ', len(train_files))

try: shutil.rmtree(dst_dir)
except: pass

dst_train_dir = os.path.join(dst_dir, 'train')
dst_val_dir = os.path.join(dst_dir, 'valid')
dst_test_dir = os.path.join(dst_dir, 'test')

dst_dir_list = [
    dst_test_dir,
    dst_val_dir,
    dst_train_dir,
    ]


# Call with i, n to display that i out of n steps are completed.
# Call without arguments or with i==n to finalise.
def print_progress(i=1,n=1,w=60):
    r = i/n
    p = np.round(100*r)
    i = int(np.round(w*r))
    print('  ['+('▆'*i)+' '*(w-i)+'] '+str(p)+'%\r',end='')
    if r==1:
        print('')


# Makes dir if it does not exist
def makedir(path):
    try:
        os.makedirs(path)
        print('created directory:',path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def msplit(string, param_name):
    temp = string.split(param_name+':(',1)
    if len(temp)>1:
        return [float(v) for v in temp[1].split(')',1)[0].split(', ')]
    return []


for d in dst_dir_list:
    makedir(d)
    

def copy_src_to_dst(src_file_list, dst_dir):
    out_no = 0
    
    for i_data, src_file in enumerate(src_file_list):
        print_progress(i_data,len(src_file_list))
        
        src_dir, src_file_bn = os.path.split(src_file)
        parts = src_file_bn.split('_')
        data_no = parts[0]
        data_no_int = int(data_no)
        step_no = parts[1]
        
        for step_no in range(max_sequence_length):
            out_no_str = str(out_no).zfill(6)
            # txt
            dst_txt = os.path.join(dst_dir, f'{out_no_str}_RGB.txt')
            with open(dst_txt, mode='w') as f_out:
                src_txt = os.path.join(src_dir, f'{data_no_int}_{step_no}_'+aff_file_name_base+'.txt')
                if not os.path.exists(src_txt):
                    continue
                
                with open(src_txt) as f_in:
                    lines = f_in.readlines()
                    prev_angle_list = []
                    prev_aff_type_index = -1
                    prev_pos = None
                    temp_aff_list = []
                    for line in lines:
                        aff_type, params = line.split(' ',1)
                        
                        pos = msplit(params,'position')
                        rot = msplit(params,'rotation')
                        sym = params.split('symmetry:')[1][0]=='T'
                        
                        temp_aff_list.append((aff_type,pos,rot,sym))
                        
                    i_delete = []
                    for (aff_type, pos, rot, sym) in temp_aff_list:
                        if aff_type == 'turn':
                            try:
                                i = temp_aff_list.index(('grasp',pos,rot,sym))
                                i_delete.append(i)
                            except:
                                pass
                    for i in i_delete[::-1]: # delete in reverse order to avoid invalidating indices during iteration
                        del temp_aff_list[i]
                    
                    if add_symmetricals:
                        additions = []
                        for (aff_type, pos, rot, sym) in temp_aff_list:
                            if sym:
                                rot2 = [rot[0],rot[1]+180,rot[2]]
                                additions.append((aff_type,pos,rot2,sym))
                        temp_aff_list += additions
                    
                    for (aff_type, pos, rot, sym) in temp_aff_list:
                        
                        aff_type_index = aff_type_list.index(aff_type)
                        x, y, z = pos
                        vx = np.clip(0.5+x*1.075,0,1)
                        vz = np.clip(0.5-z*1.075,0,1)
                        
                        angle = (rot[1]%360)
                        # normalising to [-45,135] simplifies loss calculation
                        if angle>315: angle-=360
                        angle /= 360
                        
                        symstr = '1' if sym else '0'
                        line = str(aff_type_index)+' '+str(vx)+' '+str(vz)+' '+str(angle)+' '+str(y)+' '+symstr+'\n'
                        f_out.write(line)
            
            # RGB
            src_rgb_filename = f'{data_no_int}_{step_no}_'+view+'_RGB.png'
            dst_rgb_filename = f'{out_no_str}_RGB.png'
            src_rgb = os.path.join(src_dir, src_rgb_filename)
            dst_rgb = os.path.join(dst_dir, dst_rgb_filename)

            # Read colour image
            im = imageio.imread(src_rgb)[:,:,:3]
            im = im[crop:-crop,crop:-crop]
            im = im.reshape(im.shape[0]//downscale_factor,downscale_factor,im.shape[1]//downscale_factor,downscale_factor,3)
            im = im.mean((1,3))
            imageio.imwrite(dst_rgb,np.uint8(im))

            # Depth
            src_depth_filename = f'{data_no_int}_{step_no}_'+view+'_D.png'
            dst_depth_filename = f'{out_no_str}_Depth.png'
            src_depth = os.path.join(src_dir, src_depth_filename)
            dst_depth = os.path.join(dst_dir, dst_depth_filename)

            # Read depth image
            im = imageio.imread(src_depth)[:,:,:1]
            # boost depth
            im = (255-im).astype(float)
            im = np.clip(15*im-15,0,255)
            im = im[crop:-crop,crop:-crop]
            im = im.reshape(im.shape[0]//downscale_factor,downscale_factor,im.shape[1]//downscale_factor,downscale_factor)
            im = im.mean((1,3))
            imageio.imwrite(dst_depth,np.uint8(im))
            
            out_no += 1
    
    print_progress()

print('Building test set...')
copy_src_to_dst(test_files, dst_test_dir)
print('Building validation set...')
copy_src_to_dst(val_files, dst_val_dir)
print('Building training set...')
copy_src_to_dst(train_files, dst_train_dir)

short_names = [name[0] for name in aff_type_list]

with open(dst_dir+'data.yaml','w') as f:
    f.write('train: '+dst_train_dir+'\n')
    f.write('test: '+dst_test_dir+'\n')
    f.write('val: '+dst_val_dir+'\n')
    f.write('\n')
    f.write('nc: '+str(len(aff_type_list))+'\n')
    f.write("names: "+str(short_names)+'\n')
