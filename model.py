import torch
from torchvision.models.resnet import *
import torch.nn.functional as F
import torch.nn as nn
from matplotlib import pyplot as plt
from layers import *
import math
import time
from utils import *

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        self.prev_len = config['prev_len']
        self.fut_len = config['fut_len']
        self.available_pred_len = 1 if config['long_term'] else config['fut_len'] # currently only support pred 1 future frame at a time
        self.coord_cuda = torch.zeros((config['batch'],config['total_len'],config['mat_size'][0]*config['mat_size'][1],self.config['edge_num'],2), dtype=int).to('cuda')
        self.n_channel = config['n_channel']
        if self.config['shuffle_scale'] > 1:
            self.unshuffle = nn.PixelUnshuffle(self.config['shuffle_scale'])
            self.shuffle = nn.PixelShuffle(self.config['shuffle_scale'])

        self.encoder = ImageEncoder(config)


        # feature space2depth

        self.feature_shuffle = []
        self.feature_unshuffle = []
        self.feature_scale = []
        for i in range(len(config['downsample_scale'])):
            feat_shuffle_scale = 1
            for s in range(len(config['downsample_scale'])-1,i-1,-1):
                feat_shuffle_scale *= config['downsample_scale'][s]
            
            self.feature_scale.append(feat_shuffle_scale)
            self.feature_shuffle.append(nn.PixelShuffle(feat_shuffle_scale))
            self.feature_unshuffle.append(nn.PixelUnshuffle(feat_shuffle_scale))

        self.feature_scale.append(1)
        self.feature_shuffle = nn.ModuleList(self.feature_shuffle)
        self.feature_unshuffle = nn.ModuleList(self.feature_unshuffle)
            
        
        #Motion indices
        self.motion_indices = motion_node_indices(self.config).to(self.config['device']) #1,T,HW,3 (x,y,t), range 0~1
        self.motion_indices[...,2] = 0

            

        # node encoder 

        self.tdc_len = config['tendency_len']
        if (self.config['tendency_len'] > 0):
            
            self.tdc_encoder = nn.Sequential(nn.Linear(3 ,self.tdc_len),
            nn.GroupNorm(1,self.tdc_len),
            nn.LeakyReLU(),
            nn.Linear(self.tdc_len,self.tdc_len),
            nn.GroupNorm(1,self.tdc_len),
            nn.LeakyReLU(),
            nn.Linear(self.tdc_len,self.tdc_len),
            nn.GroupNorm(1,self.tdc_len),
            nn.LeakyReLU(),
            )
        self.pos_len = self.config['pos_len']
        if (self.config['pos_len'] > 0):
            
            self.pos_encoder = nn.Sequential(nn.Linear(2 ,self.pos_len),
            nn.GroupNorm(1,self.pos_len),
            nn.LeakyReLU(),
            nn.Linear(self.pos_len,self.pos_len),
            nn.GroupNorm(1,self.pos_len),
            nn.LeakyReLU()
            )
        

        # graph attention for motion prediction
        spatial_att_list = []
        temporal_forward_att_list = []
        temporal_backward_att_list = []
        for i in range(self.config['scale_in_use']):
            spatial_att = []
            temporal_forward_att = []
            temporal_backward_att = []
            for j in range(config['pred_att_iter_num']):

                spatial_att.append(SpatialAtt(config, edge_type = 'spatial'))
                if 'forward' in self.config['edge_list']:
                    temporal_forward_att.append(GraphAtt(config, edge_type = 'forward'))
                if 'backward' in self.config['edge_list']:
                    spatial_att.append(SpatialAtt(config, edge_type = 'spatial'))
                    temporal_backward_att.append(GraphAtt(config, edge_type = 'backward'))

            decoder_len = self.tdc_len + self.pos_len
            spatial_att.append(nn.Sequential(
                nn.Conv3d(decoder_len,decoder_len, kernel_size=(3,3,3), stride=(1,1,1), padding='same'),
                nn.BatchNorm3d(decoder_len),
                nn.LeakyReLU(),
                nn.Conv3d(decoder_len,decoder_len, kernel_size=(3,3,3), stride=(1,1,1), padding='same'),
                nn.BatchNorm3d(decoder_len),
                nn.LeakyReLU()
            ))

            spatial_att_list.append(nn.ModuleList(spatial_att))
            temporal_forward_att_list.append(nn.ModuleList(temporal_forward_att))
            temporal_backward_att_list.append(nn.ModuleList(temporal_backward_att))

        self.spatial_att_list = nn.ModuleList(spatial_att_list)
        self.temporal_forward_att_list = nn.ModuleList(temporal_forward_att_list)
        self.temporal_backward_att_list = nn.ModuleList(temporal_backward_att_list)

        
        # motion decoder
        decoder_len = self.tdc_len + self.pos_len

        self.motion_fuse = nn.Sequential(
            nn.Conv3d(decoder_len,decoder_len, kernel_size=(self.config['scale_in_use'],3,3), stride=(1,1,1), padding=(0,1,1)),
            nn.BatchNorm3d(decoder_len),
            nn.LeakyReLU()
        )
        self.motion_indices_upsample = motion_node_indices_upsample(self.config) #1,T,HW,3 (x,y,t), range 0~1
        n_channel = config['n_channel'] * (config['shuffle_scale']**2)
        feat_len = config['base_channel']
        self.motion_upsampler = MotionDecoder(config)


    def graph_construct(self,sim_feat,B,T):
        N = sim_feat.shape[0]   
        c = sim_feat.shape[1]
        h = sim_feat.shape[2]
        w = sim_feat.shape[3]
        sim_feat = sim_feat.reshape(B, T, -1, h, w)
        
        mat_hw = self.config['mat_size'][0] *  self.config['mat_size'][1]
        gt_motion = None
        weight_map =None

        '''
        Calculate
        '''
        mat_list = {}
        for mat_name in self.config['edge_list']:
            mat_list[mat_name] = build_similarity_matrix(sim_feat.clone(),self.config,mat_type=mat_name)
            
        '''
        Build motion graph
        '''
        edge_list = {}
        weight_list = {}
        for mat_name in self.config['edge_list']:
            if mat_name == 'spatial':
                edge_list[mat_name] = None
                weight_list[mat_name] = None
                continue

            edge,weight,X,Y = build_graph_edge(mat_list[mat_name],self.config,gt=(mat_name == 'gt_graph'),coord_cuda=self.coord_cuda)

            if self.config['edge_normalize']:
                weight = edge_normalize(edge,weight,B,mat_hw)

            if mat_name == 'forward':
                node_init = X.clone() #normalize coords

            edge_list[mat_name] = edge.clone()
            weight_list[mat_name] = weight.clone()

        return edge_list,weight_list,gt_motion,node_init,weight_map

    
    def multiflow_compose(self,graph_feat,flow_list,img=False):
        N = graph_feat[-1].shape[0]   
        c = graph_feat[-1].shape[1]
        h = graph_feat[-1].shape[2]
        w = graph_feat[-1].shape[3]
        pred_feat_list = [None for i in range(len(graph_feat))]
        scale_num = -1
        for i in range(len(graph_feat)):

            if graph_feat[i] is None:
                continue
            else:
                cur_feat = graph_feat[i].clone()
                cur_flow = flow_list[i].clone()


            if img:
                N = graph_feat[i].shape[0]   
                c = graph_feat[i].shape[1]
                h = graph_feat[i].shape[2]
                w = graph_feat[i].shape[3]
            
            #------Unshuffle------#


            scale_num += 1
            
            B,T,hw,K,_ = cur_flow.shape
            BT,ori_c,ori_h,ori_w = cur_feat.shape

            
            if not (ori_h == h and ori_w == w):
                
                # test if eligible for shuffling

                
                if (ori_h != h*self.feature_scale[i]) or (ori_w != w*self.feature_scale[i]):
                    cur_feat = F.interpolate(cur_feat,(h*self.feature_scale[i],w*self.feature_scale[i]))
                if not img:
                    cur_feat = self.compose_unshuffle[i](cur_feat.clone())

                    
                    
            BT,cur_c,_,_ = cur_feat.shape
            cur_feat = cur_feat.reshape(B,-1,cur_c,h,w)[:,:self.prev_len] 
            
            #------- Compose--------#
            pred_feat = multi_warp(cur_feat,cur_flow)



            #---------------------#

            pred_feat_list[i] = pred_feat.clone()


            if not (ori_h == h and ori_w == w):
                if not img:
                    pred_feat_list[i] = self.feature_shuffle[i](pred_feat_list[i].clone())
                if (ori_h != h*self.feature_scale[i]) or (ori_w != w*self.feature_scale[i]): 
                    pred_feat_list[i] = F.interpolate(pred_feat_list[i],(ori_h,ori_w)).clone()

        return pred_feat_list

    def long_term_forward(self, input_image):
        output_list = {}
        pred_img_list = []
        B, T, H, W, C = input_image.shape
        cur_input_seq = input_image.clone()[:,:self.prev_len]
        
        for i in range(self.fut_len):

            cur_output = self.forward(cur_input_seq,inference=True)
            pred_img = cur_output['recon_img'].reshape(B,-1,H,W,C)
            pred_img_list.append(pred_img.clone())
            cur_input_seq = torch.cat((cur_input_seq[:,1:],pred_img),dim=1)

        '''
        prepare for output
        '''
        output_list['recon_img'] = torch.cat(pred_img_list,dim=1)

        return output_list
            

    def forward(self, input_image,inference=False,visualization=False):

        ori_input = input_image.clone()
        output_list = {}
        


        '''
        spatial feature extraction
        '''
        t = time.time()
        start_time = t
        # input_image = input_image.unsqueeze(0)
        B, T, H, W, C = input_image.shape

        # T, H, W, C = input_image.shape
        # B = 1
        self.cur_B = B
        input_image = input_image.reshape(-1, H, W, C)  # B*T,H,W,C
        input_image = input_image.permute(0, 3, 1, 2)
        
        if self.config['shuffle_scale'] > 1:
            input_image = self.unshuffle(input_image)
        input_image_raw = input_image.clone()
        raw_img_wh = input_image_raw.shape[-2:]
        emb_feat_list = self.encoder(input_image) # N, C, H, W
        #-----------#
        
        '''
        sim matrix calculation
        '''
        output_list['gt_motion'] = []
        output_list['pred_motion'] = []
        edge_list = []
        weight_list = []
        node_init_list = []
        non  = 0
        for i in range(len(self.config['downsample_scale'])+1):
            if emb_feat_list[i] is None:
                output_list['gt_motion'].append(None)
                edge_list.append(None)
                weight_list.append(None)
                node_init_list.append(None)
                non += 1
                continue
            cur_feat = emb_feat_list[i].clone()


            sim_feat = cur_feat
            
            if i != len(self.config['downsample_scale']):
                sim_feat = self.feature_unshuffle[i](sim_feat.clone())
        
            edge,weight,gt_motion,node_init,weight_map = self.graph_construct(sim_feat,B,T)

            node_init[:,:,:,:,:2] -= (self.motion_indices.clone().repeat([B,1,1,1,1]))[:,:node_init.shape[1],:,:,:2] #record the offset
            node_init_list.append(torch.cat([node_init,torch.zeros_like(node_init[:,-1:])],dim=1)) #B,T,HW,K,3; Add maskd last frame info

            edge_list.append(edge)
            weight_list.append(weight)

        '''
        Motion prediction steps:
        1. Init node with indices & Node encoding
        2. Spatial - Temporal Interaction
        3. Motion Decodeing
        4. Weight Prediction
        '''

        start = 0 #record the id of the first scale in use
        pred_edge_list = []
        pred_weight_list = []
        pred_flow_list = []
        tendency_feat_list = []
        for j in range(len(emb_feat_list)):
            if not(emb_feat_list[j] is None):
                b_,t_,hw_,k_,c_ = node_init_list[j].shape
                if (self.config['tendency_len'] > 0):
                    tendency_feat = torch.max(self.tdc_encoder(node_init_list[j].reshape(-1,3)).reshape(b_,t_,hw_,k_,-1),dim=-2)[0]
                    
                    tendency_feat_list.append(tendency_feat.clone())

                    init_node_feat = tendency_feat

                if self.config['pos_len'] > 0:
                    normalize_pos = self.motion_indices.clone().repeat([B,1,1,1,1])[...,0,:2].reshape(-1,2)
                    normalize_pos[:,0] /= (self.config['mat_size'][0]-1.)
                    normalize_pos[:,1] /= (self.config['mat_size'][1]-1.)
                    pos_id = self.pos_encoder(normalize_pos).reshape(b_,t_,hw_,-1)

                    init_node_feat = torch.cat([pos_id,init_node_feat],dim=-1)
                cur_node = init_node_feat.clone()
                for i in range(self.config['pred_att_iter_num']):
                    if 'spatial' in self.config['edge_list']:
                        idx = i if ( ('backward' not in self.config['edge_list']))else i*2

                        cur_node = self.spatial_att_list[j-start][idx](cur_node,edge_list[j]['spatial'],weight_list[j]['spatial'])
                    if 'forward' in self.config['edge_list']:
                        cur_node = self.temporal_forward_att_list[j-start][i](cur_node,edge_list[j]['forward'],weight_list[j]['forward'],position=self.motion_indices[...,0,:2].clone())
                    
                    if 'backward' in self.config['edge_list']:
                        if 'spatial' in self.config['edge_list']:
                            cur_node = self.spatial_att_list[j-start][i*2+1](cur_node,edge_list[j]['spatial'],weight_list[j]['spatial'])
                        cur_node = self.temporal_backward_att_list[j-start][i](cur_node,edge_list[j]['backward'],weight_list[j]['backward'],position=self.motion_indices[...,0,:2].clone())

                w,h = self.config['mat_size']
                cur_node = cur_node.reshape(b_,t_,h,w,-1).permute(0,4,1,2,3)
                cur_node = self.spatial_att_list[j-start][-1](cur_node)
                cur_node = cur_node.permute(0,2,3,4,1).reshape(b_,t_,hw_,-1)
                    
                pred_flow_list.append(cur_node.clone())           

            else:
                output_list['pred_motion'].append(None)
                pred_edge_list.append(None)
                pred_weight_list.append(None)
                pred_flow_list.append(None)
                start += 1



        valid_flow = []
        for i in range(len(pred_flow_list)):
            if pred_flow_list[i] is None:
                continue
            else:
                valid_flow.append(pred_flow_list[i].clone())
        
        multi_scale_motion = torch.stack(valid_flow,dim=2)
        b_,t_,s_,hw_,c_ = multi_scale_motion.shape
        h,w = self.config['mat_size']

        multi_scale_motion = multi_scale_motion.reshape(b_*t_,s_,h,w,c_).permute(0,4,1,2,3)
        fused_motion = self.motion_fuse(multi_scale_motion).squeeze(2)
                
        output_list['pred_motion'] = []
        pred_flow_list = []
        pred_node_flow_list = self.motion_upsampler(fused_motion)
        length  = len(pred_node_flow_list)
        for f_id in range(len(pred_node_flow_list)):
            pred_node_flow = pred_node_flow_list[length-f_id-1] # scale in pred_node_flow_list is from small to large
            pred_node_flow = pred_node_flow.reshape(b_,t_,self.config['out_edge_num'],3,-1).permute(0,1,4,2,3)
            pred_node_flow[...,:,-1] = pred_node_flow[...,:,-1].exp() / (1. + torch.sum(pred_node_flow[...,:,-1].exp(),dim=-1).unsqueeze(-1))
            pred_node_motion = pred_node_flow.clone()


            pred_node_motion[...,:2] = pred_node_motion[...,:2]+ (self.motion_indices_upsample[0].clone().repeat([B,1,1,1,1]))[...,:pred_node_flow.shape[-2],:2]                        
            pred_flow_list.append(pred_node_flow.clone())
            output_list['pred_motion'].append(pred_node_motion.clone())



        shuffle_image_list = []
        shuffle_image_list.append(input_image_raw.clone())
        warped_image_list = self.multiflow_compose(shuffle_image_list,[pred_flow_list[0]],img=True)
        output_list['warped_img'] = []
        output_list['warped_img'].append(warped_image_list[0].clone())
        if self.config['shuffle_scale'] > 1:
            output_list['warped_img'][0] = self.shuffle(output_list['warped_img'][0])
        output_list['warped_img'][0] = output_list['warped_img'][0].permute(0,2,3,1)

        output_list['recon_img'] = output_list['warped_img'][0]
        return output_list








