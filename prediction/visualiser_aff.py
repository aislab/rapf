import numpy as np
import config
import data_manager_aff as data_manager

aff_colours = [[255,127,127],
               [127,255,127],
               [127,127,255]]


if data_manager.view=='top':
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
    

class Visualiser_aff():
    
    def __init__(self,setup=False):
        print('visualiser setup')
        
    def close(self):
        print('visualiser closing')
        
    def draw_panel(self,is_prediction,state,state_extras=None,hidden=None,action=None,M_module=None,aux=None,meta_data=None,scores=None,state_index=0,return_image=True):

        rgbd = state[...,:4]
        rgbd = np.floor(255*rgbd).astype(int)
        d = np.tile(rgbd[...,3:4],(1,1,3))
        
        if state.shape[-1] == 4:
            im = np.concatenate((rgbd[...,:3],d),axis=1)
        else:
            sigma = state[...,4:]
            print('draw_panel/sigma range:', sigma.min(),sigma.max())
            #sigma = sigma/sigma.max()
            sigma = np.floor(255*sigma).astype(int)
            d_sigma = np.tile(sigma[...,3:4],(1,1,3))
            im = np.concatenate((rgbd[...,:3],d,sigma[...,:3],d_sigma),axis=1)
        
        if action is not None:
            
            if M_module is None:
                aff_colour = 255
            else:
                aff_colour = aff_colours[M_module]
            
            if data_manager.view=='top':
                x, z, y, angle = action[:4]
                
                xyzw = np.array((x,y,z,1))
                #print('pred/xyzw:', xyzw)
                p_xyzw = np.matmul(xyzw,pm)
                #print('pred/p_xyzw:', p_xyzw)
                #input('...')
                x, y, z, w = p_xyzw
                
                x = np.clip(np.floor((0.5+0.5*x)*128).astype(int),0,127)
                y = np.clip(np.floor((0.5-0.5*y)*128).astype(int),0,127)
                #x = np.clip(np.floor(128*(0.5+2*s*0.5*x)).astype(int),0,127)
                #z = np.clip(np.floor(128*(0.5-2*s*0.5*z)).astype(int),0,127)
                
                try:
                    s = 2 # marker size
                    im[y-s:y+s+1,x-s:x+s+1] = 255
                    #im[y-5:y+6,x-5] = aff_colour
                    #im[y-5:y+6,x+5] = aff_colour
                    #im[y-5,x-5:x+6] = aff_colour
                    #im[y+5,x-5:x+6] = aff_colour
                    #im[y-5,x-5] = 0
                    #im[y-5,x+5] = 0
                    #im[y+5,x-5] = 0
                    #im[y+5,x+5] = 0
                    if action[4]!=0:
                        sgn = np.sign(action[4])
                        for r in (-3*s,3*s):
                            for rr in range(0,90,5):
                                mx = np.clip(np.round(x+r*np.cos(np.deg2rad(360*angle+sgn*rr))),0,127).astype(int)
                                my = np.clip(np.round(y+r*np.sin(np.deg2rad(360*angle+sgn*rr))),0,127).astype(int)
                                im[my,mx,:] = 0
                    
                    #angle += 0.05
                    #print('angle', angle)
                    #mirror_angle = -(angle+0.25)-0.25
                    #print('mirrors to', mirror_angle)
                    #while mirror_angle<-0.125:
                        #mirror_angle += 1
                    #while mirror_angle>=0.875:
                        #mirror_angle -= 1
                    #print('normalises to', mirror_angle)
                    #if not (-0.125 < mirror_angle < 0.875):
                        #print('mirror angle out of range:', mirror_angle)
                        #input('...')
                    
                    for r in range(-5*s,5*s+1):
                        if np.abs(r)<s:continue
                        
                        mx = np.clip(np.round(x+r*np.cos(np.deg2rad(360*angle))),0,127).astype(int)
                        my = np.clip(np.round(y+r*np.sin(np.deg2rad(360*angle))),0,127).astype(int)
                        im[my,mx,:] = [255,0,0] if r<0 else ([0,255,0] if r>0 else 0)
                        
                        #mx = np.clip(np.round(x+r*np.cos(np.deg2rad(360*mirror_angle))),0,127).astype(int)
                        #my = np.clip(np.round(y+r*np.sin(np.deg2rad(360*mirror_angle))),0,127).astype(int)
                        #im[my,mx,:] = [255,127,127] if r<0 else ([127,255,127] if r>0 else 0)
                        
                    im[y-s+1:y+s,x-s+1:x+s] = aff_colour
                            
                    # debug - direct visual of raw angle value
                    if 0:
                        d = int(12*angle)
                        im[y-2,x:x+d] = 0
                        im[y-2,x+12] = 0
                        
                    #im[mz,mx,1] = 255
                    #im[y,x] = 0
                except:
                    print('draw panel: exception while drawing action')
            
            if data_manager.view=='front':
                x, y, z, angle = action
                y -= 0.97
                print(x,y)
                x = np.clip(np.floor(128*(0.5+2*0.45*x)).astype(int),0,127)
                y = np.clip(np.floor(128*(0.5-2*0.45*y)).astype(int),0,127)
                print(x,y)
                try:
                    im[y-8:y+9,x-8] = 0
                    im[y-8:y+9,x+8] = 0
                    im[y-8,x-8:x+9] = 0
                    im[y+8,x-8:x+9] = 0
                    im[y-8,x-8] = aff_colour
                    im[y-8,x+8] = aff_colour
                    im[y+8,x-8] = aff_colour
                    im[y+8,x+8] = aff_colour
                    #for r in range(2,10):
                        #mx = np.clip(np.floor(x+r*np.sin(np.deg2rad(360*angle))),0,127).astype(int)
                        #mz = np.clip(np.floor(z+r*np.cos(np.deg2rad(360*angle))),0,127).astype(int)
                        #im[mz,mx,:] = 0
                        #im[mz,mx,1] = 255
                    im[y,x] = 0
                except:
                    print('draw panel: exception while drawing action')
                    input('...')
                    
        return im.swapaxes(0,1)

    def draw_panel_daf(self,
                       is_prediction,
                       state,state_extras=None,
                       hidden=None,
                       action=None,
                       M_module=None,
                       aff_classifications=None,
                       fA_data=None,
                       GT_mode=False,
                       state_index=0,
                       return_image=True):
        im = self.draw_panel(is_prediction,state,state_extras,hidden,action,M_module,aux=None,meta_data=None,state_index=state_index,return_image=return_image)
        im = im.swapaxes(0,1)
        
        #print('fA_data:', len(fA_data))
        #print(fA_data)
        
        for i, (action, cl) in enumerate(zip(fA_data,aff_classifications)):
            #print('action:', action)
            #print('gt:', gt, 'cl:', cl)
            im[:3,3*i:3*i+3] = int(255*cl[0])
            im[4:7,3*i:3*i+3] = int(255*np.round(cl[0]))
            #if GT_mode and cl==0: continue
            
            M_module = int(action[-1])
            if M_module is None:
                aff_colour = 255
            else:
                aff_colour = aff_colours[M_module]
            
            if data_manager.view=='top':
                x, z, y, angle = action[:4]
                
                xyzw = np.array((x,y,z,1))
                p_xyzw = np.matmul(xyzw,pm)

                x, y, z, w = p_xyzw
                
                x = np.clip(np.floor((0.5+0.5*x)*128).astype(int),0,127)
                y = np.clip(np.floor((0.5-0.5*y)*128).astype(int),0,127)
                
                try:
                    s = 2 # marker size
                    im[y-s:y+s+1,x-s:x+s+1] = int(255*cl[0])
                    if action[4]!=0:
                        sgn = np.sign(action[4])
                        for r in (-3*s,3*s):
                            for rr in range(0,90,5):
                                mx = np.clip(np.round(x+r*np.cos(np.deg2rad(360*angle+sgn*rr))),0,127).astype(int)
                                my = np.clip(np.round(y+r*np.sin(np.deg2rad(360*angle+sgn*rr))),0,127).astype(int)
                                im[my,mx,:] = 0
                    
                    for r in range(-5*s,5*s+1):
                        if np.abs(r)<s:continue
                        
                        mx = np.clip(np.round(x+r*np.cos(np.deg2rad(360*angle))),0,127).astype(int)
                        my = np.clip(np.round(y+r*np.sin(np.deg2rad(360*angle))),0,127).astype(int)
                        im[my,mx,:] = [255,0,0] if r<0 else ([0,255,0] if r>0 else 0)
                        
                        #mx = np.clip(np.round(x+r*np.cos(np.deg2rad(360*mirror_angle))),0,127).astype(int)
                        #my = np.clip(np.round(y+r*np.sin(np.deg2rad(360*mirror_angle))),0,127).astype(int)
                        #im[my,mx,:] = [255,127,127] if r<0 else ([127,255,127] if r>0 else 0)
                        
                    im[y-s+1:y+s,x-s+1:x+s] = aff_colour
                    im[y,x] = int(255*cl[0])
                    #im[y-s:y,x-s:x-s+2] = int(255*gt[0])
                    #im[y-s+1,x-s:x-s+2] = int(255*cl[0])
                            
                        #im[mz,mx,1] = 255
                    #im[y,x] = 0
                except:
                    print('draw panel: exception while drawing action')
        
        return im.swapaxes(0,1)
