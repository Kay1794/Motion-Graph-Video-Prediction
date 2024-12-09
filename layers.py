import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import torchvision
import time
import torchist
from utils import *


class ConvBlock(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, config,in_channels, out_channels, kernel=3,mid_channels=None,bn=True,motion=False,dilation=1):
        super().__init__()
        self.config = config
        if not mid_channels:
            mid_channels = out_channels

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )  if ((config['use_bn']) or (config['motion_use_bn'] and motion)) else  nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, bias=False,dilation=dilation),
            nn.ReLU(inplace=True)
        ) 

    def forward(self, x):
        return self.conv(x)

class ResBlock(nn.Module):
    def __init__(self, config,in_channels, out_channels, downsample=False,upsample=False,skip=False,factor=2,motion=False):
        super().__init__()
        self.upsample = upsample
        self.config = config
        self.maxpool= None
        
        if downsample:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2)
            if factor == 4:
                self.maxpool = nn.MaxPool2d(2)
            
        elif upsample:
            self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=factor, stride=factor)
            self.shortcut = nn.Sequential(nn.Upsample(scale_factor=factor, mode='nearest'),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels)) \
            if (motion) else nn.Sequential(nn.Upsample(scale_factor=factor, mode='nearest'),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1))

        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.shortcut = None

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)


    def forward(self, input):
        if self.shortcut is None:
            shortcut = input.clone()
        else:
            shortcut = self.shortcut(input)
        input = nn.ReLU()(self.conv1(input))
        input = nn.ReLU()(self.conv2(input))
        #if not self.upsample:
        input = input + shortcut
        if not self.maxpool is None:
            input = self.maxpool(input)
        return nn.LeakyReLU()(input)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self,config, in_channels, out_channels, skip=True,scale=2,bn=True,motion=False):
        super().__init__()
        factor = scale

        if skip:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=factor, stride=factor)
            self.conv = ConvBlock(config,out_channels, out_channels,bn=bn,motion=motion)

        else:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=factor, stride=factor)
            self.conv = ConvBlock(config,out_channels*2, out_channels,bn=bn,motion=motion)

    def forward(self, x1, x2=None):

        x1 = self.up(x1)
        if x2 is None:
            return self.conv(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        

    def forward(self, x):
        return self.conv(x)


class ImageEncoder(nn.Module):
    def __init__(self, config):
        super(ImageEncoder, self).__init__()


        base_channel = config['base_channel']
        scale = config['downsample_scale']
        self.scale_num = len(scale)
        n_channel = config['n_channel'] * (config['shuffle_scale']**2)
        self.config = config
        self.inconv = nn.Conv2d(n_channel, base_channel, 3, 1, 1)
        layers = []
        self.block_rrdb = nn.Sequential(*layers)

        pre_downsample_block_list = []
        
        for i in range(self.scale_num-2):
            pre_downsample_block_list.append(ResBlock(config,base_channel * (2**(i)),base_channel* (2**((i+1))),downsample=True,factor=scale[i]))
        self.pre_downsample_block = nn.ModuleList(pre_downsample_block_list)
        
        if self.scale_num >= 2: 
            self.downsample_high = ResBlock(config,base_channel*(2**((self.scale_num-2))),base_channel*(2**((self.scale_num-1))),downsample=True,factor=scale[-2])
        self.downsample_low = ResBlock(config,base_channel*(2**((self.scale_num-1))),base_channel*(2**((self.scale_num))),downsample=True,factor=scale[-1])

    def forward(self, x, save_all=False):
        in_feat = []
        x = self.inconv(x)
        x1 = self.block_rrdb(x)
        x2 = x1
        in_feat.append(x2.clone())
        for i in range(self.scale_num-2):
            x2 = self.pre_downsample_block[i](x2) 
            in_feat.append(x2.clone())
        if self.scale_num >= 2:
            x_high = self.downsample_high(x2)
            in_feat.append(x_high.clone())
        else:
            x_high = x2
        x_low = self.downsample_low(x_high)
        in_feat.append(x_low.clone())


        for i in range(len(in_feat)-self.config['scale_in_use']):
            in_feat[i] = None


        return in_feat


class MotionDecoder(nn.Module):
    def __init__(self, config):
        super(MotionDecoder, self).__init__()
        self.config = config
        base_channel = config['tendency_len'] + config['pos_len']
        scale = config['downsample_scale']
        self.scale_num = len(scale)
        out_channel = config['out_edge_num'] * 3

        factor = 1
        
        upsample_block_list = []
        for i in range(0,self.scale_num,1):
            upsample_block_list.append(nn.Sequential(
                ResBlock(config,base_channel,base_channel,upsample=False,motion=True),
                ResBlock(config,base_channel,base_channel,upsample=False,motion=True),
                ResBlock(config,base_channel,base_channel,upsample=True,motion=True)
                ))
        self.upsample_block =  nn.ModuleList(upsample_block_list)

        self.outc_list = nn.ModuleList()
        for i in range(0,self.scale_num,1):
            self.outc_list.append(OutConv(base_channel, out_channel))

    def upscale_flow(self,flow,shape,scale):

        motion_up = F.interpolate(flow,shape)
        b,c,h,w = motion_up.shape
        motion_up = motion_up.reshape(b,c//3,3,h,w)
        motion_up[:,:,:2] *= scale
        motion_up =motion_up.reshape(b,c,h,w)

        return motion_up


    def forward(self, in_feat):
        x = in_feat
        logits_list = []
        resize_logits_list = []
        
        for i in range(self.scale_num-1,-1,-1):

            if i<self.scale_num-1:
                motion_up = self.upscale_flow(logits_list[-1],(x.shape[-2:]),self.config['downsample_scale'][i-1])
                logits_list.append(motion_up + self.outc_list[i](x.clone()))  
            else:
                
                logits_list.append(self.outc_list[i](x.clone()))

            x = self.upsample_block[i](x)
        
        motion_up = self.upscale_flow(logits_list[-1],(x.shape[-2:]),self.config['downsample_scale'][i-1])
        logits_list.append(motion_up + self.outc_list[-1](x))
        h,w = logits_list[-1].shape[-2:]
        for i in range(len(logits_list)-1):
            cur_h = logits_list[i].shape[-2]
            resize_logits_list.append(self.upscale_flow(logits_list[i],(h,w),h//cur_h).clone())
        resize_logits_list.append(logits_list[-1].clone())


        return resize_logits_list


#---------------------------------#

class SpatialAtt(nn.Module):
    def __init__(self, config,img_feat=False,edge_type='spatial'):
        super(SpatialAtt, self).__init__()
        self.config = config
        cur_feat_len = config['tendency_len'] + config['pos_len']

        self.net = nn.Sequential(nn.Conv2d(cur_feat_len,cur_feat_len,3,1,1),
        nn.BatchNorm2d(cur_feat_len),
        nn.LeakyReLU())

    def forward(self,graph_feat,edge,weight=None,debug=False):

            
        B,T,HW,C = graph_feat.shape
        graph_feat_spatial = graph_feat.reshape(B*T,self.config['mat_size'][0],self.config['mat_size'][1],C).permute(0,3,1,2)
        graph_feat = self.net(graph_feat_spatial)
        graph_feat = graph_feat.reshape(B,T,C,HW).permute(0,1,3,2)

        return graph_feat


class GraphAtt(nn.Module):
    def __init__(self, config,img_feat=False,edge_type='forward'):
        super(GraphAtt, self).__init__()
        self.config = config
        self.edge_type = edge_type
        self.img_feat = img_feat
        
        head_num = 1
        if img_feat:
            graph_feat_len = img_feat
        else:
            graph_feat_len = config['tendency_len'] + config['pos_len']


        if not img_feat:

            self.att_layer = nn.Linear(graph_feat_len * head_num ,graph_feat_len * head_num)
            self.fuse = nn.Linear(graph_feat_len * (head_num+1),graph_feat_len)
            self.fuse_norm = nn.GroupNorm(1, graph_feat_len)
            self.activate = nn.LeakyReLU()
            self.norm = nn.GroupNorm(1, graph_feat_len * head_num)
            
            self.dist = nn.Sequential(
            nn.Linear(2 , graph_feat_len),
            nn.LeakyReLU(inplace=True),
            nn.Linear(graph_feat_len,graph_feat_len),
            nn.GroupNorm(1, graph_feat_len)
            )

    def forward(self,graph_feat,edge,weight=None,debug=False,position=None):

        B,T,HW,C = graph_feat.shape
        graph_feat = graph_feat.reshape(-1,C)

        position = position.clone().repeat(B,1,1,1).reshape(B*T*HW,-1)
        copy_graph_feat = torch.zeros_like(graph_feat)

        if self.edge_type == 'forward':
            node_id_pre = torch.stack([edge[:,0],edge[:,1],edge[:,2]],dim=1)
            node_id_suc = torch.stack([edge[:,0],edge[:,1]+1,edge[:,3]],dim=1)
        elif self.edge_type == 'backward':
            node_id_pre = torch.stack([edge[:,0],edge[:,1]+1,edge[:,2]],dim=1)
            node_id_suc = torch.stack([edge[:,0],edge[:,1],edge[:,3]],dim=1)
        elif self.edge_type == 'spatial':
            node_id_pre = torch.stack([edge[:,0],edge[:,1],edge[:,2]],dim=1)
            node_id_suc = torch.stack([edge[:,0],edge[:,1],edge[:,3]],dim=1)
        
        flat_id_pre = torchist.ravel_multi_index(node_id_pre,(B,T,HW))
        flat_id_suc = torchist.ravel_multi_index(node_id_suc,(B,T,HW))

        
        
        if self.img_feat:

            copy_graph_feat.index_add_(0,flat_id_suc,graph_feat[flat_id_pre].clone()* weight.unsqueeze(-1))
            graph_feat = copy_graph_feat.reshape(B*T,H,W,C).permute(0,3,1,2)
        else:
            copy_graph_feat = copy_graph_feat.unsqueeze(-2).reshape(B*T*HW,-1)
            feat_to_add = graph_feat[flat_id_pre].clone()
            value  = feat_to_add

            tendency_dist = graph_feat[flat_id_suc][...,-self.config['tendency_len']:] -feat_to_add[...,-self.config['tendency_len']:]
            pos_dist = position[flat_id_suc]-position[flat_id_pre]

            dist_emb = self.dist(pos_dist)
            att_result = self.att_layer(value + dist_emb)
            copy_graph_feat.index_add_(0,flat_id_suc,att_result * weight.unsqueeze(-1))
            copy_graph_feat = self.norm(copy_graph_feat)
            
            graph_feat[torch.unique(flat_id_suc)] = self.activate(self.fuse_norm(self.fuse(torch.cat([graph_feat[torch.unique(flat_id_suc)],copy_graph_feat[torch.unique(flat_id_suc)]],dim=-1))))

        if self.img_feat:
            graph_feat = graph_feat.reshape(B,T,C,H,W)
        else:
            graph_feat = graph_feat.reshape(B,T,HW,-1)

        return graph_feat


