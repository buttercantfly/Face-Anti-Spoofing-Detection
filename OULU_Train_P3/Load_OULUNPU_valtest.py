'''
Code of 'Searching Central Difference Convolutional Networks for Face Anti-Spoofing' 
By Zitong Yu & Zhuo Su, 2019

If you use the code, please cite:
@inproceedings{yu2020searching,
    title={Searching Central Difference Convolutional Networks for Face Anti-Spoofing},
    author={Yu, Zitong and Zhao, Chenxu and Wang, Zezheng and Qin, Yunxiao and Su, Zhuo and Li, Xiaobai and Zhou, Feng and Zhao, Guoying},
    booktitle= {CVPR},
    year = {2020}
}

Only for research purpose, and commercial use is not allowed.

MIT License
Copyright (c) 2020 
'''

from __future__ import print_function, division
import os
import torch
import pandas as pd
#from skimage import io, transform
import cv2
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pdb
import math
import os 


frames_total = 8    # each video 8 uniform samples
 
face_scale = 1.3  #default for test and val 
#face_scale = 1.1  #default for test and val

def crop_face_from_scene(image,face_name_full, scale):
    f=open(face_name_full,'r')
    lines=f.readlines()
    y1,x1,w,h=[float(ele) for ele in lines[:4]]
    f.close()
    y2=y1+w
    x2=x1+h

    y_mid=(y1+y2)/2.0
    x_mid=(x1+x2)/2.0
    h_img, w_img = image.shape[0], image.shape[1]
    #w_img,h_img=image.size
    w_scale=scale*w
    h_scale=scale*h
    y1=y_mid-w_scale/2.0
    x1=x_mid-h_scale/2.0
    y2=y_mid+w_scale/2.0
    x2=x_mid+h_scale/2.0
    y1=max(math.floor(y1),0)
    x1=max(math.floor(x1),0)
    y2=min(math.floor(y2),w_img)
    x2=min(math.floor(x2),h_img)

    #region=image[y1:y2,x1:x2]
    region=image[x1:x2,y1:y2]
    return region




class Normaliztion_valtest(object):
    """
        same as mxnet, normalize into [-1, 1]
        image = (image - 127.5)/128
    """
    def __call__(self, sample):
        image_x, val_map_x, spoofing_label = sample['image_x'],sample['val_map_x'] ,sample['spoofing_label']
        new_image_x = (image_x - 127.5)/128     # [-1,1]
        return {'image_x': new_image_x, 'val_map_x': val_map_x , 'spoofing_label': spoofing_label}


class ToTensor_valtest(object):
    """
        Convert ndarrays in sample to Tensors.
        process only one batch every time
    """

    def __call__(self, sample):
        image_x, val_map_x, spoofing_label = sample['image_x'],sample['val_map_x'] ,sample['spoofing_label']
        
        # swap color axis because    BGR2RGB
        # numpy image: (batch_size) x T x H x W x C
        # torch image: (batch_size) x T x C X H X W
        image_x = image_x[:,:,:,::-1].transpose((0, 3, 1, 2))
        image_x = np.array(image_x)
        # print(image_x)
        
        val_map_x = np.array(val_map_x)
        # print(val_map_x)
                        
        spoofing_label_np = np.array([0],dtype=np.long)
        spoofing_label_np[0] = spoofing_label
        
        return {'image_x': torch.from_numpy(image_x.astype(np.float64)).float(), 'val_map_x': torch.from_numpy(val_map_x.astype(np.float64)).float(),'spoofing_label': torch.from_numpy(spoofing_label_np.astype(np.long)).long()}


# /home/ztyu/FAS_dataset/OULU/Train_images/          6_3_20_5_121_scene.jpg        6_3_20_5_121_scene.dat
# /home/ztyu/FAS_dataset/OULU/IJCB_re/OULUtrain_images/        6_3_20_5_121_depth1D.jpg
class Spoofing_valtest(Dataset):

    def __init__(self, info_list, root_dir, val_map_dir,  transform=None):

        self.landmarks_frame = pd.read_csv(info_list, delimiter=',', header=None)
        self.root_dir = root_dir
        self.val_map_dir = val_map_dir
        self.transform = transform
        self.count = 0

    def __len__(self):
        return len(self.landmarks_frame[0])

    
    def __getitem__(self, idx):
        #print(self.landmarks_frame.iloc[idx, 0])
        videoname = str(self.landmarks_frame.iloc[idx, 0])
        image_path = os.path.join(self.root_dir, videoname)
        val_map_path = os.path.join(self.val_map_dir, videoname)
        # print(self.count,'\n')
        self.count+=1
        # print(videoname)
        # print(image_path)
        files_total = 0
        for image_name in os.listdir(image_path):
            if os.path.isfile(os.path.join(image_path, image_name)):
                files_total += 1
        # print(files_total, '\n')

		    
        spoofing_label = self.landmarks_frame.iloc[idx, 1]
        if spoofing_label == 1:
            spoofing_label = 1            # real
            image_x, val_map_x = self.get_single_image_x(image_path, val_map_path, videoname)
        else:
            # print('line 141 : spoof photo')
            spoofing_label = 0

            image_x = np.zeros((frames_total, 256, 256, 3))
            val_map_x = np.ones((frames_total, 32, 32))

            count = 0
            while (count < 8):
                image_id = np.random.randint(1, files_total - 1)
                # print(count)

                for temp in range(100):
                    s = "_%03d_scene" % image_id
                    image_name = videoname + s + '.jpg'
                    image_path2 = os.path.join(image_path, image_name)

                    if os.path.exists(image_path2):  # some scene.dat are missing
                        image_x_temp = cv2.imread(image_path2)
                        if image_x_temp is not None:
                            # print(image_name, ' exist\n')
                            break
                        else:
                            # print(map_name, ' exist but can`t read\n')
                            image_id = np.random.randint(1, files_total - 1)
                    else:
                        image_id = np.random.randint(1, files_total - 1)

                image_x[count, :, :, :] = image_x_temp
                count += 1
            # print('out of while loop !!!!!!!!!!!!\n')



        sample = {'image_x': image_x, 'val_map_x':val_map_x , 'spoofing_label': spoofing_label}


        # print('get image get map//////////////')
        # print(image_x.shape)
        # print(val_map_x.shape)
        if self.transform:
            sample = self.transform(sample)
        # print('transform the image and map\n')
        return sample

    def get_single_image_x(self, image_path, val_map_path, videoname):

        files_total = 0
        for image_name in os.listdir(image_path):
            if os.path.isfile(os.path.join(image_path, image_name)):
                files_total += 1
        # print(videoname)
        # print(files_total,'\n')
        
        image_x = np.zeros((frames_total, 256, 256, 3))
        val_map_x = np.ones((frames_total, 32, 32))
        
        # random choose 1 frame
        count = 0
        while(count < 8):
            image_id = np.random.randint(1,files_total-1)
            # print(count)
            
            for temp in range(100):
                # s = "-%04d" % image_id
                # s1 = "-%04d_depth1D" % image_id
                s = "_%03d_scene" % image_id
                s1 = "_%03d_depth1D" % image_id
                image_name = videoname + s + '.jpg'
                map_name = videoname + s1 + '.jpg'

                # print(map_name)
                image_path2 = os.path.join(image_path, image_name)
                val_map_path2 = os.path.join(val_map_path, map_name)
                
                if  os.path.exists(image_path2) and os.path.exists(val_map_path2)  :    # some scene.dat are missing
                    image_x_temp = cv2.imread(image_path2)
                    val_map_x_temp2 = cv2.imread(val_map_path2, 0)
                
                    if val_map_x_temp2 is not None and image_x_temp is not None:
                        # print(map_name, ' exist')
                        # print(image_name, ' exist\n')
                        break
                    else:
                        # print(map_name, ' exist but can`t read\n')
                        image_id = np.random.randint(1, files_total - 1)
                else:
                    # print(image_name, ' not exist\n')
                    image_id = np.random.randint(1, files_total - 1)
                    
            # RGB
            # print('valtest line 225')
            image_path2 = os.path.join(image_path, image_name)
            image_x_temp = cv2.imread(image_path2)
            # print(map_name, ' jump out of for loop exist\n')

            # gray-map
            # print('valtest line 231')
            val_map_x_temp = cv2.imread(val_map_path2, 0)
            # print(image_x_temp.shape)
            if image_x_temp.shape != (256,256,3):
                print('???????????????????\n')

            image_x[count,:,:,:] = image_x_temp
            # transform to binary mask --> threshold = 0 
            temp = cv2.resize(val_map_x_temp, (32, 32))
            val_map_x[count,:,:] = np.where(temp < 1, temp, 1)
            count += 1
            # print(count,'\n')

        # print('out of while loop\n')
            
			
        return image_x, val_map_x



            
 


