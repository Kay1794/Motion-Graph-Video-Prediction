import sys
import os
import numpy as np
import json
import torch
from torch.utils.data import Dataset, DataLoader
import time
import cv2
from tqdm import tqdm
from matplotlib import pyplot
import warnings
from multiprocessing import Manager
from matplotlib import pyplot as plt
from data.data_utils import *
import configs
import math
from copy import deepcopy as cp
import random
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import shutil
import gzip
import torchvision


class UCFPredDataset(Dataset):
    def __init__(self,config,split,is_train=False):
        self.config = config
        self.val_data = []
        self.data_root = config['dataroot']
        self.is_train = is_train
        self.split = split
        
        if split=='train':
            self.data_list = open(os.path.join(self.data_root,'train.txt'),'r').readlines()
        else:
            if config['val_subset']=='all':
                self.data_list = open(os.path.join(self.data_root,'test.txt'),'r').readlines()
            else:
                self.data_list = open(os.path.join(self.data_root,config['val_subset']+'_test.txt'),'r').readlines()
        
 
    def __len__(self):

        return len(self.data_list)
        
    def getimg(self, line,idx=0,depth=False):
        file_names = line.split(' ')
        if file_names[-1][-1] == '\n':
            file_names[-1] = file_names[-1][:-1]

        imgs = []
        for i in range(len(file_names)):
            cur_filename = file_names[i]        
            im = cv2.imread(os.path.join(self.data_root,cur_filename))
            h, w, c = im.shape
            side_len = min(w, h)
            crop_img = self.CenterCrop(im, side_len, side_len)
            final_img = cv2.resize(im, self.config['in_res']).copy()
            if depth:
                final_img = final_img / 255. * self.depth_scale[cur_filename]
                if max_depth < self.depth_scale[cur_filename]:
                    max_depth = self.depth_scale[cur_filename]

            imgs.append(final_img)
   
        return  imgs
    
    def CenterCrop(self,img,crop_w,crop_h):
        center = np.array(img.shape[:2]) / 2
        x = center[1] - crop_w / 2
        y = center[0] - crop_h / 2

        crop_img = img[int(y):int(y + crop_h), int(x):int(x + crop_w)]
        return crop_img
    
    def __getitem__(self, index):

        imgs = self.getimg(self.data_list[index])
        seq = (torch.from_numpy(np.asarray(imgs))/255.-0.5).to(self.config['device']).float()
        seq = torch.flip(seq,dims=(-1,))
        if self.split == 'train' and self.config['flip_aug']:
            flag = random.uniform(0,1)
            if flag < 0.5:
                seq = torch.flip(seq,dims=[1])
            flag = random.uniform(0,1)
            if flag < 0.5:
                seq = torch.flip(seq,dims=[2])
            flag = random.uniform(0,1)
        if self.split == 'train' and self.config['rot_aug']:
            flag = random.uniform(0,1)
            if flag < 0.5:
                k = random.randint(1, 3)
                seq = torch.rot90(seq,dims=(1,2),k=k)


        return seq[:self.config['total_len']], seq[self.config['prev_len']:self.config['total_len'],...,:self.config['n_channel']]#n, c, h , w



class STRPM_UCFPredDataset(Dataset):



    def __init__(self, config, split, is_train=False):
        self.split = split
        self.config = config
        self.h5_path = config['dataroot']+'/ucf_prediction_official'
        if self.split == 'train':
            print('Loading train dataset')
            self.dataset = h5py.File(self.h5_path+'_'+split+'.h5', "r")
            print('Loading train dataset finished, with size:', len(self.dataset.keys()))
        else:
            print('Loading test dataset')
            self.dataset = h5py.File(self.h5_path + '_test.h5', "r")
            print('Loading test dataset finished, with size:', len(self.dataset.keys()))

    def preprocess_img(self, img_list):
        T,H,W,C = img_list.shape
        img_list = img_list.reshape(H*T,W,C)
        img_list = cv2.cvtColor(img_list,cv2.COLOR_BGR2RGB).reshape(T,H,W,C)
        if self.config['range'] == 1.:
            norm_img_list = img_list / 255.
            norm_img_list_tensor = norm_img_list- 0.5
        else:
            norm_img_list_tensor = img_list

        return norm_img_list_tensor

    def __len__(self):
        return len(self.dataset.keys())


    def __getitem__(self, idx):

        data_slice = np.asarray(self.dataset.get(str(idx)))

        sample = self.preprocess_img(data_slice)
        prev_frames_tensor = sample.copy()[:self.config['prev_len']]
        fut_frames_tensor = sample.copy()[self.config['prev_len']:(self.config['prev_len']+self.config['fut_len'])]

        fut_frames_tensor = torch.from_numpy(fut_frames_tensor.copy()).float().to(self.config['device'])
        prev_frames_tensor = torch.from_numpy(prev_frames_tensor.copy()).float().to(self.config['device'])


        if self.split == 'train' and self.config['flip_aug']:
            flag = random.uniform(0,1)
            if flag < 0.5:
                
                prev_frames_tensor = torch.flip(prev_frames_tensor,dims=[1])
                fut_frames_tensor = torch.flip(fut_frames_tensor,dims=[1])
            flag = random.uniform(0,1)
            if flag < 0.5:
                prev_frames_tensor = torch.flip(prev_frames_tensor,dims=[2])
                fut_frames_tensor = torch.flip(fut_frames_tensor,dims=[2])

        if self.split == 'train' and self.config['rot_aug']:
            flag = random.uniform(0,1)
            if flag < 0.5:
                k = random.randint(1, 3)
                prev_frames_tensor = torch.rot90(prev_frames_tensor,dims=(1,2),k=k)
                fut_frames_tensor = torch.rot90(fut_frames_tensor,dims=(1,2),k=k)
        prev_frames_tensor = torch.cat([prev_frames_tensor,fut_frames_tensor],dim=0)

        return prev_frames_tensor,fut_frames_tensor



# ------------ Dataloader code from CVPR2023-DMVFN ---------------

class CityTrainDataset(Dataset):
    def __init__(self,config,split,is_train=False):
        self.config = config
        self.is_train = is_train
        self.path = config['dataroot'] + '/' + split + '/'
        
        self.train_data = sorted(os.listdir(self.path))


    def __len__(self):
        return len(self.train_data) * 10

    def getimg(self, index):
        
        data_name = self.train_data[index//10]
        frame_idx = index % 10
        data_path = os.path.join(self.path, data_name)
        frame_list = sorted(os.listdir(data_path))
        imgs = []
        for i in range(frame_idx * 3, frame_idx * 3 + 3):
            im = cv2.imread(os.path.join(data_path,frame_list[i]))
            im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
            imgs.append(im)
        return  imgs
            
    def __getitem__(self, index):
        imgs = self.getimg(index)
        # imgs = self.aug_seq(imgs, 256, 256)
        length = len(imgs)
        if self.is_train:
            if random.randint(0, 1):
                for i in range(length):
                    imgs[i] = cv2.rotate(imgs[i], cv2.ROTATE_180)
            if random.uniform(0, 1) < 0.5:
                for i in range(length):
                    imgs[i] = imgs[i][:, :, ::-1]
            if random.uniform(0, 1) < 0.5:
                for i in range(length):
                    imgs[i] = imgs[i][::-1]
            if random.uniform(0, 1) < 0.5:
                for i in range(length):
                    imgs[i] = imgs[i][:, ::-1]
        for i in range(length):
            imgs[i] = torch.from_numpy(imgs[i].copy())
        input_seq =  torch.stack(imgs, 0)/255.-0.5#n, c, h , w

        return input_seq,input_seq[-1:]



# ------------ Dataloader code from CVPR2023-DMVFN ---------------

class CityTrainDataset(Dataset):
    def __init__(self,config,split,is_train=False):
        self.config = config
        self.is_train = is_train
        self.path = config['dataroot'] + '/' + split + '/'
        
        self.train_data = sorted(os.listdir(self.path))


    def __len__(self):
        return len(self.train_data) * 10

    def getimg(self, index):
        
        data_name = self.train_data[index//10]
        frame_idx = index % 10
        data_path = os.path.join(self.path, data_name)
        frame_list = sorted(os.listdir(data_path))
        imgs = []
        for i in range(frame_idx * 3, frame_idx * 3 + 3):
            im = cv2.imread(os.path.join(data_path,frame_list[i]))
            imgs.append(im)
        return  imgs
            
    def __getitem__(self, index):
        imgs = self.getimg(index)
        # imgs = self.aug_seq(imgs, 256, 256)
        length = len(imgs)
        if self.is_train:
            if random.randint(0, 1):
                for i in range(length):
                    imgs[i] = cv2.rotate(imgs[i], cv2.ROTATE_180)
            if random.uniform(0, 1) < 0.5:
                for i in range(length):
                    imgs[i] = imgs[i][:, :, ::-1]
            if random.uniform(0, 1) < 0.5:
                for i in range(length):
                    imgs[i] = imgs[i][::-1]
            if random.uniform(0, 1) < 0.5:
                for i in range(length):
                    imgs[i] = imgs[i][:, ::-1]
        for i in range(length):
            imgs[i] = torch.from_numpy(imgs[i].copy())
        input_seq =  torch.stack(imgs, 0)/255.-0.5#n, c, h , w

        return input_seq,input_seq[-1:]

        

class KittiTrainDataset(Dataset): 
##File version
    def __init__(self,config,split,is_train=False):
        self.config = config
        self.is_train = is_train
        if split == 'train':
            self.path = config['dataroot'] +  '/train/'
            self.train_data = sorted(os.listdir(self.path))
        else:
            self.path = config['dataroot'] +  '/test/'
            self.train_data = sorted(os.listdir(self.path))
            

    def __len__(self):
        return len(self.train_data*3)


    def aug_seq(self, imgs):
        ih, iw, _ = imgs[0].shape
        for i in range(len(imgs)):
            imgs[i] = imgs[i]
        return imgs

    def getimg(self, index):
        # all sequence
        data_name = self.train_data[index//3]
        frame_id = index % 3
        data_path = os.path.join(self.path, data_name)
        frame_list = sorted(os.listdir(data_path))
        imgs = []
        for i in range(frame_id*3,frame_id*3+3):
            im = cv2.imread(os.path.join(data_path, frame_list[i]))
            im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
            imgs.append(im)
        return  imgs
            
    def __getitem__(self, index):
        imgs = self.getimg(index)
        imgs = self.aug_seq(imgs)
        length = len(imgs)
        if self.is_train:
            if random.uniform(0, 1) < 0.5:
                for i in range(length):
                    imgs[i] = imgs[i][:, :, ::-1]
            if random.uniform(0, 1) < 0.5:
                for i in range(length):
                    imgs[i] = imgs[i][::-1]
            if random.uniform(0, 1) < 0.5:
                for i in range(length):
                    imgs[i] = imgs[i][:, ::-1]
        input_seq = torch.from_numpy(np.asarray(imgs)) /255.-0.5


        return input_seq,input_seq[-1:]#n, c, h , w



class KittiValDataset(Dataset):
    def __init__(self,config,split,is_train=False):
        self.config = config
        self.val_data = []
        self.video_path = config['dataroot'] + '/test/'
        self.video_data = sorted(os.listdir(self.video_path))
        for i in self.video_data:
            self.val_data.append(os.path.join(self.video_path, i))
        self.val_data = sorted(self.val_data)
 
    def __len__(self):
        return len(self.val_data)
        
    def getimg(self, index):
        data = self.val_data[index]
        img_list = sorted(os.listdir(data))
        imgs = []
        for i in range(2,9,1):
            im = cv2.imread(os.path.join(data, img_list[i]))
            imgs.append(im)
        return  imgs
    
    def __getitem__(self, index):
        imgs = self.getimg(index)
        name = self.video_data[index]
        seq = (torch.from_numpy(np.asarray(imgs))/255.-0.5).to(self.config['device'])
        seq = torch.flip(seq,dims=(-1,))
        return seq[:self.config['total_len']], seq[self.config['prev_len']:self.config['total_len']]#n, c, h , w