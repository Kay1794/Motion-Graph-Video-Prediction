import torch
import numpy as np
from torch.optim import lr_scheduler
from torch.autograd import Variable
from operator import add
#from model import *
#from simvp_model import *
import time
import math
from utils import *
from copy import deepcopy as cp
from layers import *
import torch.optim as optim

class Module(object):
    def __init__(self, model, config, optimizer, criterion):
        
        self.model = model
        self.time_record = []
        self.optimizer = optimizer
        self.epoch = -1
        if config['cos_restart']:
            print('Using CosineAnnealingLR_Restart Scheduler!')
            self.scheduler = CosineAnnealingLR_Restart(optimizer,T_period=list(config['t_period']),restarts=list(np.cumsum(config['t_period'])[:-1]),last_epoch =self.epoch,ratio=config['restart_ratio'])
        else:
            self.scheduler = None
        self.criterion = criterion
        self.config = config
        
        self.mode = 'train'
        self.loss_list = config['loss_list']
        self.time_weight = None
        
    def cal_loss(self, pred_data, gt_data, type='recon',device='cuda',epoch=-1):
        

        if type in ['recon']: 
            
            pred_data = pred_data.reshape(gt_data.shape)
            loss = self.criterion['recon'](pred_data,gt_data)    
            return loss
        
        elif type.find('percep')>-1:

            if len(pred_data.shape) == 5:
                B,T,h,w,c = pred_data.shape
            elif len(pred_data.shape) == 4:
                T = self.config['fut_len']
                B = pred_data.shape[0]//T
                _,h,w,c = pred_data.shape
            pred_data = pred_data.reshape(B*T,h,w,c).permute(0,3,1,2)
            gt_data = gt_data.reshape(B*T,h,w,c).permute(0,3,1,2)

            loss = torch.mean(self.criterion[type](pred_data.clone(),\
            gt_data.clone().to(device))).mean() * 0.5 * 0.25

            return loss



    def step(self, data,epoch=-1):

        '''
        :param data: dictionary, key = {'input_img'}
        :return: recon_img, loss_dict
        '''
        gt_data = data['gt_img'].clone()
        output_list = self.forward_data(data)
        recon_img = output_list['recon_img']

        # cal_loss
        loss = torch.tensor(0.).to(self.config['device'])
        loss_dict = {}
        for key in self.loss_list:
            if key == 'recon':
                loss_dict[key] = self.cal_loss(recon_img, gt_data.clone(), key,loss.device)
            elif key == 'percep':
                loss_dict[key] = self.cal_loss(recon_img, gt_data, key,loss.device) 

        for key in loss_dict.keys():
            loss += loss_dict[key]
            loss_dict[key] = loss_dict[key].item()
                
        loss_dict['total'] = loss.item()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if 'ldl_d' in self.config['loss_list']:
            loss_dict['ldl_d'] = self.cal_loss(recon_img.clone(), gt_data.clone(), 'ldl_d',loss.device)
            loss_dict['ldl_d'] = loss_dict['ldl_d'].item()

        if 'pred_sim_matrix' in output_list:
            for i in range(len(output_list['pred_sim_matrix'])):
                output_list['pred_sim_matrix'][i] = output_list['pred_sim_matrix'][i].cpu().detach().numpy()
                output_list['gt_sim_matrix'][i] = output_list['vis_sim_matrix'][i].cpu().detach().numpy()
                output_list['vis_sim_matrix'][i] = None
        output_list['recon_img'] = output_list['recon_img'].cpu().detach()

        return output_list, loss_dict

    def val(self, data,epoch):
        '''
        :param data: dictionary, key = {'input_img'}
        :return: recon_img, loss
        '''
        gt_data = data['gt_img'].clone()
        output_list = self.forward_data(data,update_sim_matrix=True,inference=True)
        recon_img = output_list['recon_img']
        gt_data = gt_data.reshape(recon_img.shape)


        # cal_lossmid
        loss = torch.tensor(0.).to(self.config['device'])
        loss_dict = {}
        for key in self.loss_list:
            if key in ['recon']:
                loss_dict[key] = self.cal_loss(recon_img, gt_data.clone(), 'recon',loss.device)
            elif key == 'percep':
                loss_dict[key] = self.cal_loss(recon_img, gt_data, key,loss.device)

        for key in loss_dict.keys():
            loss += loss_dict[key]
            loss_dict[key] = loss_dict[key].item()
        loss_dict['total'] = loss.item()
        output_list['recon_img'] = output_list['recon_img']

        return output_list, loss_dict
    

    def forward_data(self,data,update_sim_matrix=True,inference=False):
        '''
        :param data: dictionary, key = {'input_img'}
        :return: recon_img_stack
        '''
        img_stack = data['input_img'] # N, H, W, C
        # return None
        if self.config['long_term']:
            output_list = self.model.module.long_term_forward(img_stack)
        else:
            output_list = self.model(img_stack,inference=inference)


        return output_list






