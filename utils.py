import numpy as np
import os
from matplotlib import pyplot as plt
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.color import rgb2yuv
import torch.nn.functional as F
import math
import torchvision
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
import cv2
import lpips
import random
from torch_geometric.utils import softmax as group_softmax
from torch_geometric.utils import scatter as geo_scatter
import torchist
import softsplat
from pytorch_msssim import ms_ssim
from torch.nn.utils import spectral_norm
import torchvision.models as models

device = torch.device("cuda")
grid = None

lpips = lpips.LPIPS(net='alex').cuda()
metrics_to_save = []

def update_config(cur_config,args):
    cur_config['nepoch'] = args.nepoch
    cur_config['dataset'] = args.dataset
    cur_config['loss_list'] = args.loss_list
    cur_config['eval_list'] = args.eval_list
    cur_config['exp_name'] = args.exp_name
    cur_config['range'] = args.img_range
    cur_config['base_channel'] = args.base_channel
    cur_config['top_k'] = args.top_k
    cur_config['batch'] = args.batch
    cur_config['shuffle_scale'] = args.shuffle_scale
    cur_config['flip_aug'] = args.flip_aug
    cur_config['rot_aug'] = args.rot_aug
    cur_config['cos_restart'] = args.cos_restart
    cur_config['t_period'] = np.asarray(args.t_period)
    cur_config['val_subset'] = args.val_subset
    cur_config['long_term'] = args.long_term
    cur_config['lr'] = args.lr
    cur_config['restart_ratio'] = args.restart_ratio
    cur_config['scale_in_use'] = args.scale_in_use
    cur_config['edge_list'] = args.edge_list
    cur_config['window_length'] = args.window_length
    cur_config['pos_len'] = args.pos_len
    if len(args.downsample_scale) > 0:
        cur_config['downsample_scale'] = args.downsample_scale
    if args.prev_len != -1:
        cur_config['prev_len'] = args.prev_len
    if args.fut_len != -1:
        cur_config['fut_len'] = args.fut_len
        cur_config['total_len'] = cur_config['fut_len'] + cur_config['prev_len']
    if args.long_term and args.fut_len == 1:
        print('Conflict! Long term mode while fut_len = 1')
        exit()

    # set mat size
    lowres_scale = np.prod(cur_config['downsample_scale'])
    lowres_scale *= args.shuffle_scale

    cur_config['mat_size'] = [cur_config['in_res'][0]//lowres_scale,cur_config['in_res'][1]//lowres_scale]

    '''
    Graph related
    '''
    cur_config['pred_att_iter_num'] = args.pred_att_iter_num
    cur_config['edge_num'] = int(cur_config['mat_size'][0] * cur_config['mat_size'][1] * args.top_k)
    cur_config['out_edge_num'] = cur_config['edge_num'] if args.out_edge_num == -1 else args.out_edge_num
    cur_config['edge_softmax'] = args.edge_softmax
    cur_config['edge_normalize'] = args.edge_normalize
    cur_config['tendency_len'] = args.tendency_len

    return cur_config
    


class MeanShift(nn.Conv2d):
    def __init__(self, data_mean, data_std, data_range=1, norm=True):
        c = len(data_mean)
        super(MeanShift, self).__init__(c, c, kernel_size=1)
        std = torch.Tensor(data_std)
        self.weight.data = torch.eye(c).view(c, c, 1, 1)
        if norm:
            self.weight.data.div_(std.view(c, 1, 1, 1))
            self.bias.data = -1 * data_range * torch.Tensor(data_mean)
            self.bias.data.div_(std)
        else:
            self.weight.data.mul_(std.view(c, 1, 1, 1))
            self.bias.data = data_range * torch.Tensor(data_mean)
        self.requires_grad = False

def unravel_index(indices,shape,coord_cuda):
    r"""Converts flat indices into unraveled coordinates in a target shape.
    from: https://github.com/pytorch/pytorch/issues/35674#issuecomment-739492875

    This is a `torch` implementation of `numpy.unravel_index`.

    Args:
        indices: A tensor of indices, (*, N).
        shape: The targeted shape, (D,).

    Returns:
        unravel coordinates, (*, N, D).
    """

    shape = torch.tensor(shape).to('cuda')
    indices = indices.to('cuda') % torch.prod(shape)  # prevent out-of-bounds indices
    coord_size = indices.size() + shape.size()
    # return None
    coord = coord_cuda[:coord_size[0],:coord_size[1],:coord_size[2],:coord_size[3],:coord_size[4]].clone()
    for i, dim in enumerate(reversed(shape)):
        coord[..., i] = indices % dim
        # indices = torchindices / dim
        indices = torch.div(indices,dim,rounding_mode ='trunc')

    return coord.flip(-1)

def ravel_index(indices,shape):
    r"""Converts flat indices into unraveled coordinates in a target shape.
    from: https://github.com/pytorch/pytorch/issues/35674#issuecomment-739492875

    This is a `torch` implementation of `numpy.unravel_index`.

    Args:
        indices: A tensor of indices, (*, N,2).
        shape: The targeted shape, (D,).

    Returns:
        ravel coordinates, (*, N).
    """

    coord = torch.zeros(indices.size()[:-1]).to(indices.device)
    coord = indices[...,0] * shape[1] + indices[...,1]

    return (coord+0.5).long()
    


def augmentation_pred(config,prev_frames_tensor,fut_frames_tensor):
    if config['flip_aug']:
        flag = random.uniform(0,1)
        if flag < 0.5:
            
            prev_frames_tensor = torch.flip(prev_frames_tensor,dims=[2])
            fut_frames_tensor = torch.flip(fut_frames_tensor,dims=[2])
        if config['name'].find('ucf') > -1:
            flag = random.uniform(0,1)
            if flag < 0.5 :
                prev_frames_tensor = torch.flip(prev_frames_tensor,dims=[3])
                fut_frames_tensor = torch.flip(fut_frames_tensor,dims=[3])

    if config['rot_aug']:
        flag = random.uniform(0,1)
        if flag < 0.5:
            if config['name'].find('ucf') > -1:
                k = random.randint(1, 3)
            else:
                k = 2
            prev_frames_tensor = torch.rot90(prev_frames_tensor,dims=(2,3),k=k)
            fut_frames_tensor = torch.rot90(fut_frames_tensor,dims=(2,3),k=k)

    return prev_frames_tensor,fut_frames_tensor

def augmentation_recon(config,frame_tensor):
    if config['flip_aug']:
        flag = random.uniform(0,1)
        if flag < 0.5:
            
            frame_tensor = torch.flip(frame_tensor,dims=[1])
        if config['name'].find('ucf') > -1:
            flag = random.uniform(0,1)
            if flag < 0.5 :
                prev_frames_tensor = torch.flip(prev_frames_tensor,dims=[2])
                fut_frames_tensor = torch.flip(fut_frames_tensor,dims=[2])

    if config['rot_aug']:
        flag = random.uniform(0,1)
        if flag < 0.5:
            if config['name'].find('ucf') > -1:
                k = random.randint(1, 3)
            else:
                k = 2
            prev_frames_tensor = torch.rot90(prev_frames_tensor,dims=(1,2),k=k)
            fut_frames_tensor = torch.rot90(fut_frames_tensor,dims=(1,2),k=k)

    return prev_frames_tensor,fut_frames_tensor


def feat_compose(source_feat,sim_matrix):
    '''

    :param source_feat: previous feats at time t, (B,c,h,w)
    :param sim_matrix: composition guide of time t to future t' (B,h,w,h,w)
    :return: fut_feat: composed feats for time t' (B,c,h,w)
    '''

    B,c,h,w = source_feat.shape
    source_feat = source_feat.reshape(B,c,h*w)
    sim_matrix = sim_matrix.reshape(B,h*w,h*w).permute(0,2,1)
    fut_feat = torch.bmm(source_feat,sim_matrix)
    fut_feat = fut_feat.reshape(B,c,h,w)

    return fut_feat

def get_feature_patch(feat,kernel):
    #input: B,T,C,H,W
    B,T,C,H,W = feat.shape
    h_w,w_w = kernel
    
    feat = feat.reshape(B*T,C,H,W)
    pad_h = h_w//2
    pad_w = w_w//2
    pad_feat = F.pad(feat, (pad_w, pad_w, pad_h, pad_h ), mode='constant')

    patches = F.unfold(pad_feat, kernel_size=kernel).view(B,T, C, h_w,w_w,H,W).permute(0,1,2,5,6,3,4).contiguous()

    return patches

def fold_feature_patch(feat,kernel):
    #input: B,h,w,ws,ws,c
    
    B,h,w,h_w,w_w,c = feat.shape
    pad_h = h_w//2
    pad_w = w_w//2
    feat = feat.permute(0,5,3,4,1,2).reshape(B,c*h_w*w_w,h*w)
    weight  = torch.ones_like(feat)
    feature_map = F.fold(feat, kernel_size=kernel,output_size=(h+pad_h*2,w+pad_w*2))
    weight = F.fold(weight, kernel_size=kernel,output_size=(h+pad_h//2*2,w+pad_w//2*2))
    feature_map = feature_map[:,:,pad_h:-pad_h,pad_w:-pad_w]
    weight = weight[:,:,pad_h:-pad_h,pad_w:-pad_w]
    feature_map /= weight

    return feature_map

def motion_node_indices(config,B=1,T=None):
    if T is None:
        T = config['prev_len']
    res_x = config['mat_size'][0]
    res_y = config['mat_size'][1]
    ts = torch.linspace(0,T-1,steps=T)
    ys = torch.linspace(0, res_y-1, steps=res_y)
    xs = torch.linspace(0, res_x-1, steps=res_x)
    t, x, y = torch.meshgrid(ts, xs, ys)
    single_graph = torch.stack([x.float(),y.float(),t],dim=-1).to(config['device']) #normalized index

    node_indices = single_graph.unsqueeze(0).repeat([B,1,1,1,1]).reshape(B,T,-1,1,3).repeat([1,1,1,config['edge_num'],1])

    
    return node_indices

def motion_node_indices_upsample(config,B=1,T=None):
    
    if T is None:
        T = config['prev_len']
    ts = torch.linspace(0,T-1,steps=T)
    node_indices_list = []
    res_x = config['in_res'][0] // 2 if config['shuffle_scale'] == 2 else config['in_res'][0]
    res_y = config['in_res'][1] // 2 if config['shuffle_scale'] == 2 else config['in_res'][1]

    res_x_list = [res_x // (2**i) for i in range(len(config['downsample_scale'])+1)]
    res_y_list = [res_y // (2**i) for i in range(len(config['downsample_scale'])+1)]
    for i in range(len(res_x_list)):
        res_x = res_x_list[i]
        res_y = res_y_list[i]
        ys = torch.linspace(0, res_y-1, steps=res_y)
        xs = torch.linspace(0, res_x-1, steps=res_x)
        t, x, y = torch.meshgrid(ts, xs, ys)
        single_graph = torch.stack([x.float(),y.float(),t],dim=-1).to(config['device']) #normalized index
        node_indices = single_graph.unsqueeze(0).repeat([B,1,1,1,1]).reshape(B,T,-1,1,3).repeat([1,1,1,config['out_edge_num'],1])
        node_indices_list.append(node_indices.clone().to(config['device']))
    
    return node_indices_list

def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros'):
    """Warp an image or feature map with optical flow
    Args:
        x (Tensor): size (B, C, H, W)
        flow (Tensor): size (N, H, W, 2), normal value
        interp_mode (str): 'nearest' or 'bilinear'
        padding_mode (str): 'zeros' or 'border' or 'reflection'

    Returns:
        Tensor: warped image or feature map
    """
    assert x.size()[-2:] == flow.size()[1:3]
    B, C, H, W = x.size()
    # mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False
    grid = grid.type_as(x)
    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(W - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(H - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode)
    return output 




def build_similarity_matrix(emb_feats,config,mat_type='forward',window_mask=False):
    '''

    :param emb_feats: a sequence of embeddings for every frame (N,T,c,h,w)
    :return: similarity matrix (N, T-1, h*w, h*w) current frame --> next frame
    '''
    B,T,c,h,w = emb_feats.shape
    emb_feats = emb_feats.permute(0,1,3,4,2) #  (B,T,h,w,c)
    normalize_feats = emb_feats.clone() / (torch.norm(emb_feats.clone(),dim=-1,keepdim=True)+1e-6) #  (B,T,h,w,c)

    if mat_type == 'spatial':
        cur_frame = normalize_feats[:,:config['prev_len']].clone().reshape(-1,h*w,c) # (B*(T-1),h*w,c)
        similarity_matrix = torch.bmm(cur_frame,cur_frame.clone().permute(0,2,1)).reshape(B,config['prev_len'],h*w,h*w)
    elif mat_type == 'forward':
        prev_frame = normalize_feats[:,:(config['prev_len']-1)].reshape(-1,h*w,c) # (B*(T-1),h*w,c)
        next_frame = normalize_feats[:,1:config['prev_len']].reshape(-1,h*w,c) # (B*(T-1),h*w,c)                                                 
        similarity_matrix = torch.einsum('bij,bjk->bik', (prev_frame,next_frame.permute(0,2,1))).reshape(B,-1,h*w,h*w)
    elif mat_type == 'backward':
        prev_frame = normalize_feats[:,:(config['prev_len']-1)].reshape(-1,h*w,c) # (B*(T-1),h*w,c)
        next_frame = normalize_feats[:,1:config['prev_len']].reshape(-1,h*w,c) # (B*(T-1),h*w,c)                                                      
        similarity_matrix = torch.einsum('bij,bjk->bik', (next_frame,prev_frame.permute(0,2,1))).reshape(B,config['prev_len']-1,h*w,h*w)
    elif mat_type == 'gt_graph':
        exist_frame = normalize_feats[:,:config['prev_len']].reshape(B,config['prev_len'],1,h*w,c).repeat((1,1,config['fut_len'],1,1)).reshape(-1,h*w,c)
        gt_frame = normalize_feats[:,config['prev_len']:].reshape(B,1,config['fut_len'],h*w,c).repeat((1,config['prev_len'],1,1,1)).reshape(-1,h*w,c)                                                      
        similarity_matrix = torch.einsum('bij,bjk->bik', (exist_frame,gt_frame.permute(0,2,1))).reshape(B,config['prev_len'] * config['fut_len'],h*w,h*w)

    similarity_matrix[similarity_matrix<0] = 0.
    return similarity_matrix



def retrieve_diag(similar_matrix,config):
    B,T,hw,hw = similar_matrix.shape
    h = config['mat_size'][0]
    w = config['mat_size'][1]
    diagonal_mask = torch.zeros(hw,hw).to(similar_matrix.device).bool() #(h*w,h*w)
    diagonal_mask.fill_diagonal_(True)
    diagonal_mask = diagonal_mask.reshape(1,1,hw,hw).repeat(B,T,1,1).bool()
    diag_matrix = similar_matrix[diagonal_mask].reshape(B,T,h,w)

    return diag_matrix

def sim_matrix_softmax(similar_matrix):
    B,T,hw,hw = similar_matrix.shape
    similar_matrix = similar_matrix.reshape(similar_matrix.shape[0],similar_matrix.shape[1],-1)
    similar_matrix = F.softmax(similar_matrix,dim=-1)
    
    return similar_matrix.reshape(B,T,hw,hw)


def transform_graph_edge(config,pred_motion,pred_weight,gt=False):
    #shape: (B,T,HW,K,1),(B,T,HW,K,2)
    B = pred_motion.shape[0]
    T = pred_motion.shape[1]
    hw = pred_motion.shape[2]
    k_num = pred_motion.shape[3]

    pred_motion[...,0] *= (config['mat_size'][0]-1)
    pred_motion[...,1] *= (config['mat_size'][1]-1)
    

    pred_motion_ravel = ravel_index((pred_motion+0.5).long(),config['mat_size'])

    Bs = torch.linspace(0,B-1,steps=B)
    Ts = torch.linspace(0, T-1, steps=T)
    HWs = torch.linspace(0, hw-1, steps=hw)
    ks = torch.linspace(0, k_num-1, steps=k_num)

    b,t,hw,k_num = torch.meshgrid(Bs, Ts, HWs, ks, indexing='ij')
    structure = (torch.stack([b.reshape(-1),t.reshape(-1),hw.reshape(-1),k_num.reshape(-1)],dim=-1)+0.5).long().to(config['device'])
    edge = structure.clone()
    edge[:,-1] = pred_motion_ravel.reshape(-1)
    weight = pred_weight.reshape(-1)

    return edge,weight

def edge_softmax(edge,weight,batch,hw):

    raveled_index = torchist.ravel_multi_index(torch.stack([edge[:,0],edge[:,1],edge[:,3]],dim=-1),shape=(batch,int(torch.max(edge[:,1]+1)),hw))
    # print(torch.unique(raveled_index).shape)
    # exit()
    weight = group_softmax(weight,(raveled_index+0.5).long())

    return weight

def edge_normalize(edge,weight,batch,hw):

    raveled_index = (torchist.ravel_multi_index(torch.stack([edge[:,0],edge[:,1],edge[:,3]],dim=-1),shape=(batch,int(torch.max(edge[:,1]+1)),hw))+0.5).long()
    weight_sum = geo_scatter(weight,raveled_index,reduce='sum')
    
    div_weight = weight_sum[raveled_index]
    
    weight /= (div_weight + 1e-6)

    return weight


def build_graph_edge(similarity_matrix,config,gt=False,coord_cuda=None):

    B = similarity_matrix.shape[0]
    T_prime = similarity_matrix.shape[1]
    hw = similarity_matrix.shape[2]
    new_similarity_matrix = similarity_matrix.clone()
    select_num = max(config['edge_num'],1)

    top_k,indices = torch.topk(new_similarity_matrix,select_num,dim=-1) #shape: B,T,HW,K

    if gt:
        diag_element = retrieve_diag(similarity_matrix,config)
        weight_map = torch.min(diag_element,dim=1)[0] * (-1.) + 1.

    else:
        weight_map = None

    B = top_k.shape[0]
    T = top_k.shape[1]
    hw = top_k.shape[2]
    k_num = top_k.shape[3]

    Bs = torch.linspace(0,B-1,steps=B).to(config['device'])
    Ts = torch.linspace(0, T-1, steps=T).to(config['device'])
    HWs = torch.linspace(0, hw-1, steps=hw).to(config['device'])
    ks = torch.linspace(0, k_num-1, steps=k_num).to(config['device'])


    b,t,hw,k_num = torch.meshgrid(Bs, Ts, HWs, ks, indexing='ij')

    structure = (torch.stack([b.reshape(-1).to(config['device']),t.reshape(-1).to(config['device']),hw.reshape(-1).to(config['device']),k_num.reshape(-1).to(config['device'])],dim=-1)+0.5).long().to(config['device'])
    edge = structure.clone()
    

    unraveled_indices = unravel_index(indices.clone(),config['mat_size'],coord_cuda)

    motion_gt = torch.cat([unraveled_indices.float().to(config['device']),top_k.clone().unsqueeze(-1)],dim=-1) #B,T,HW,K,3 (x,y,weight)
    
    edge[:,-1] = indices.reshape(-1)
    weight = top_k.reshape(-1)
    
    
    return edge,weight,motion_gt,weight_map



def multi_warp(img, flow,last_only=False):
    '''
    img: B,T,C,H,W
    flow: B,T,HW,K,3
    '''
    B,T,C,H,W = img.shape
    K = flow.shape[-2]
    img = img.unsqueeze(2).repeat(1,1,K,1,1,1) #B,T,K,C,H,W
    flow = flow.reshape(B,T,H,W,K,3).permute(0,1,4,5,2,3)
    
    if last_only:
        img = img[:,-1:]
        flow = flow[:,-1:]
        T = 1

    
    flow = flow.reshape(B*T*K,3,H,W)
    weight = flow[:,-1:] # BTK,1,H,W
    img = img.reshape(B*T*K,C,H,W)
    
    flow = flow[:,:2]
    flow = torch.cat([flow[:,1:2],flow[:,:1]],dim=1)

    # Customized softmax Splatting
    
    '''
    splat_source = torch.cat([img,torch.ones_like(img[:,-1:])],dim=1) * weight.exp()
    BTK = B*T*K
    output = softsplat.softsplat(tenIn=splat_source, tenFlow=flow[:,:2], tenMetric=None, strMode='sum')
    weight_map = output[:,-1:]
    output = output[:,:-1]

    output = output.reshape(-1,T*K,C,H,W)
    weight_map = weight_map.reshape(-1,T*K,1,H,W)
    weight_map = torch.sum(weight_map,dim=1)
    output = torch.sum(output,dim=1)
    output /= (weight_map + 1e-6)
    '''
    
    
    # Customized Normalization Splatting
    splat_source = torch.cat([img,torch.ones_like(img[:,-1:])],dim=1) * weight
    BTK = B*T*K
    output = softsplat.softsplat(tenIn=splat_source, tenFlow=flow[:,:2], tenMetric=None, strMode='sum')
    weight_map = output[:,-1:]
    output = output[:,:-1]

    output = output.reshape(-1,T*K,C,H,W)
    weight_map = weight_map.reshape(-1,T*K,1,H,W)
    weight_map = torch.sum(weight_map,dim=1)
    output = torch.sum(output,dim=1)
    output /= (weight_map + 1e-6)

    return output.reshape(-1,C,H,W)

def check_folder(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    return folder_path




class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f',is_val=False):
        self.name = name
        self.fmt = fmt
        self.is_val = is_val
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        #fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        fmtstr = '{name} {avg' + self.fmt + '}'
        if self.is_val:
            fmtstr = 'val {name} {avg' + self.fmt + '}'

        return fmtstr.format(**self.__dict__)


def metric_print(metrics,epoch_num,step_num,t,last_iter=False):
    '''

    :param metrics: metric list
    :param epoch_num: epoch number
    :param step_num: step number
    :param t: time duration
    :return: string
    '''
    if last_iter:
        base = 'Epoch {} \t'.format(epoch_num)
    else:
        base = 'Epoch {} iter {}\t'.format(epoch_num, step_num)
    for key in metrics.keys():
        base = base + str(metrics[key]) + '\t'
    final = base + 'takes {} s'.format(t)
    return final


def update_metrics(metrics,loss_dict):
    for key in loss_dict.keys():
        metrics[key].update(loss_dict[key])

    return metrics


def img_valid(img):
    img = img + 0.5
    img = img
    img[img < 0] = 0.
    img[img > 1.] = 1.

    return img

def img_clamp(img):

    img[img < 0] = 0.
    img[img > 255] = 255
    if torch.is_tensor(img):
        img = img.cpu().numpy()
    img = img.astype(np.uint8)

    return img

def torch_img_clamp_normalize(img):

    img[img < 0] = 0.
    img[img > 255] = 255
    img /= 255.

    return img

def MAE(true,pred):
    return (np.abs(pred-true)).sum()

def MSE(true,pred):
    return ((pred-true)**2).sum()


def visualization_check_video(save_path,epoch,image_list,valid=False,is_train=False,matrix=False,config=None,long_term=False):
    plt.clf()
    if is_train:
        save_file = save_path + '/visual_check_train_'
    else:
        save_file = save_path + '/visual_check_'

    if matrix:
        save_file = save_file + 'matrix_'
    save_file = save_file + str(epoch) + '.png'
    sample_num = len(image_list)
    vis_image = []
    dtf_image = []
    diff_prev_image = [] # diff_prev
    diff_gt_image = [] # diff_gt
    for i in range(sample_num):
        gt_seq,recon_seq = image_list[i]
        if gt_seq.shape[0] > 20 or long_term:

            gt = np.hstack(gt_seq)
            recon = np.hstack(recon_seq)
            if valid:
                gt = img_clamp(gt)
                recon = img_clamp(recon)
            else:
                gt = img_valid(gt)
                recon = img_valid(recon)
            recon_range = np.zeros_like(gt)
            recon_range[:,-recon.shape[1]:,:] = recon
            recon = recon_range
            vis_image.append(np.vstack([gt,recon]))
            
        else:
            gt = np.hstack(gt_seq[-2:])
        
            recon = np.hstack(recon_seq)
            if valid:
                gt = img_clamp(gt)
                recon = img_clamp(recon)
            else:
                gt = img_valid(gt)
                recon = img_valid(recon)

        
            overlay = recon.copy()
            gt_frame = img_clamp(gt_seq[-1]) if valid else img_valid(gt_seq[-1])
            last_frame = img_clamp(gt_seq[-2]) if valid else img_valid(gt_seq[-2])
            
            overlay_last = last_frame*.5 + overlay *.5
            overlay_gt = gt_frame*.5 + overlay *.5
            real_overlay = gt_frame *.5 + last_frame *.5 
            if valid:
                overlay_last = overlay_last.astype(np.uint8)
                overlay_gt = overlay_gt.astype(np.uint8)
                real_overlay = real_overlay.astype(np.uint8)
            vis_image.append(np.hstack([gt,recon,overlay_last,overlay_gt,real_overlay]))

    whole_image = np.vstack(vis_image)

    if whole_image.shape[-1] == 1:
        plt.imshow(whole_image,interpolation="nearest",cmap='gray')
    else:
        plt.imshow(whole_image,interpolation="nearest")
    plt.axis('off')
    plt.savefig(save_file, dpi=400, bbox_inches ="tight", pad_inches = 0)


def image_evaluation(image_list,gt_image_list,eval_metrics,valid=False):

    
    size = image_list.shape
    if len(size) > 4:
        image_list = image_list.reshape(size[0]*size[1],size[2],size[3],size[4])
    size = gt_image_list.shape
    if len(size) > 4:
        gt_image_list = gt_image_list.reshape(size[0]*size[1],size[2],size[3],size[4])
    for i in range(image_list.shape[0]):
        if valid:
            image = img_clamp(image_list[i]) /255.
            gt_image = img_clamp(gt_image_list[i]) / 255.
        else:
            image = img_valid(image_list[i])
            gt_image = img_valid(gt_image_list[i])

        gt_image_gpu = gt_image.clone()
        images_gpu = image.clone()

        gt_image = gt_image.cpu().numpy()
        image = image.cpu().numpy()

        for key in eval_metrics:
            
            if key == 'psnr':
                eval_metrics[key].update(psnr(gt_image.copy(),image.copy()))
            if key == 'psnr_y':
                eval_metrics[key].update(psnr(rgb2yuv(gt_image.copy())[:,:,:1],rgb2yuv(image.copy()))[:,:,:1])
            elif key == 'ssim':
                # eval_metrics[key].update(ssim(gt_image.copy(),image.copy(),channel_axis=2,data_range=1))
                eval_metrics[key].update(ssim(gt_image.copy(),image.copy(),multichannel=True))
            elif key == 'mae':
                eval_metrics[key].update(MAE(gt_image.copy(),image.copy()))
            elif key == 'mse':
                eval_metrics[key].update(MSE(gt_image.copy(),image.copy()))
            elif key == 'lpips':
                eval_metrics[key].update(lpips(images_gpu.permute(2,0,1).unsqueeze(0).cuda(),gt_image_gpu.permute(2,0,1).unsqueeze(0).cuda()).item())
            elif key == 'ms_ssim':
                eval_metrics[key].update(ms_ssim(X=images_gpu.permute(2,0,1).unsqueeze(0), Y=gt_image_gpu.permute(2,0,1).unsqueeze(0),data_range=1.0, size_average=True))
    return eval_metrics



class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, rank=0):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        pretrained = True
        self.vgg_pretrained_features = models.vgg19(pretrained=pretrained).features
        self.normalize = MeanShift([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], norm=True).cuda()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, X, Y, indices=None):
        X = self.normalize(X)
        Y = self.normalize(Y)
        indices = [2, 7, 12, 21, 30]
        weights = [1.0/2.6, 1.0/4.8, 1.0/3.7, 1.0/5.6, 10/1.5]
        k = 0
        loss = 0
        for i in range(indices[-1]):
            X = self.vgg_pretrained_features[i](X)
            Y = self.vgg_pretrained_features[i](Y)
            if (i+1) in indices:
                loss += weights[k] * (X - Y.detach()).abs().mean() * 0.1
                k += 1
        return loss


class CosineAnnealingLR_Restart(_LRScheduler):
    def __init__(self, optimizer, T_period, restarts=None, weights=None, eta_min=1e-5, last_epoch=-1,ratio=0.5):
        self.T_period = list(T_period)
        self.T_max = self.T_period[0]  # current T period
        self.eta_min = eta_min
        self.restarts = list(restarts)
        self.restart_weights = [ratio ** (i+1) for i in range(len(restarts))]
        self.last_restart = 0
        print('restart ratio: ',ratio,' T_period: ',T_period,' minimum lr: ',eta_min)
        assert len(self.restarts) == len(
            self.restart_weights), 'restarts and their weights do not match.'
        super(CosineAnnealingLR_Restart, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch == 0:
            return self.base_lrs
        elif self.last_epoch in self.restarts:
            self.last_restart = self.last_epoch
            self.T_max = self.T_period[list(self.restarts).index(self.last_epoch) + 1]
            weight = self.restart_weights[list(self.restarts).index(self.last_epoch)]
            return [group['initial_lr'] * weight for group in self.optimizer.param_groups]
        elif (self.last_epoch - self.last_restart - 1 - self.T_max) % (2 * self.T_max) == 0:
            return [
                group['lr'] + (base_lr - self.eta_min) * (1 - math.cos(math.pi / self.T_max)) / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        return [(1 + math.cos(math.pi * (self.last_epoch - self.last_restart) / self.T_max)) /
                (1 + math.cos(math.pi * ((self.last_epoch - self.last_restart) - 1) / self.T_max)) *
                (group['lr'] - self.eta_min) + self.eta_min
                for group in self.optimizer.param_groups]


class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. ' f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * F.l1_loss(pred, target, weight, reduction=self.reduction)

