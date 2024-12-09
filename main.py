import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
import sys
import argparse
import os
from shutil import copytree, copy
from utils import *
from tqdm import tqdm
from model import *
from module import *
import configs
from copy import deepcopy as cp
import pytorch_ssim
import lpips
import gc
import torch.backends.cudnn as cudnn

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def main(config,args):
    num_epochs = args.nepoch
    need_log = args.log
    num_workers = args.nworker

    start_epoch = 0
    best_psnr = 0.

    # config log info for training mode
    torch.autograd.set_detect_anomaly(True)
    if args.mode == 'train' and need_log:
        logger_root = args.logpath if args.logpath != '' else 'results'

        if args.resume == '':
            time_stamp = time.strftime("%m-%d_%H-%M")

            model_save_path = check_folder(os.path.join(logger_root, args.dataset))
            model_save_path = check_folder(os.path.join(model_save_path, args.method))
            model_save_path = check_folder(os.path.join(model_save_path, args.exp_name
                                                        + '_'+time_stamp))
            log_file_name = os.path.join(model_save_path, 'log.txt')
            saver = open(log_file_name, "w")
            saver.write("GPU number: {}\n".format(torch.cuda.device_count()))
            saver.write('Running Config: '+str(config)+'\n')
            saver.flush()

            # Logging the details for this experiment
            saver.write("command line: {}\n".format(" ".join(sys.argv[0:])))
            saver.write("Save to: "+model_save_path+'\n')
            saver.write(args.__repr__() + "\n\n")
            saver.flush()

            # Copy the code files as logs
            copytree('data', os.path.join(model_save_path, 'data'))
            python_files = [f for f in os.listdir('.') if f.endswith('.py')]
            for f in python_files:
                copy(f, model_save_path)
        else:
#
            model_save_path = args.resume[:args.resume.rfind('/')]
            
            log_file_name = os.path.join(model_save_path, 'log.txt')
            saver = open(log_file_name, "a")
            saver.write("GPU number: {}\n".format(torch.cuda.device_count()))
            
            saver.flush()
            # Logging the details for this experiment
            saver.write("command line: {}\n".format(" ".join(sys.argv[1:])))
            saver.write(args.__repr__() + "\n\n")
            saver.write('Running log: '+str(config)+'\n')
        
            saver.flush()
    else:
        model_save_path = None
        saver = open('tmp.txt', "a")

    # Specify gpu device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config['device'] = device
    device_num = torch.cuda.device_count()
    print("device number", device_num)

    # Load data

    if args.mode == 'train':
        trainset = PredDataset(config=config, split='train', is_train=True)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch, shuffle=True,num_workers=num_workers)
        config['train_disp_freq'] = max(1,len(trainset) // args.batch  // args.display_freq)
        config['train_steps'] = len(trainloader)

        valset = PredDataset(config=config, split='test', is_train=False)
        valloader = torch.utils.data.DataLoader(valset, args.batch, shuffle=False, num_workers=num_workers)
        config['val_disp_freq'] = max(1,len(valset) // args.batch // args.display_freq)
        print("Training dataset size:", len(trainset))
        print("Validation dataset size:", len(valset))
        if args.log:
            config['model_save_path'] = model_save_path

    elif args.mode in ['val']:
        
        valset = PredDataset(config=config, split='test', is_train=False)
        valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch, shuffle=False, num_workers=num_workers)
        config['val_disp_freq'] = len(valset) // args.batch // args.display_freq
        model_save_path = args.resume[:args.resume.rfind('/')]
        log_file_name = os.path.join(model_save_path, 'validation_log.txt')
        saver = open(log_file_name, "a")
        saver.write("Validation on : {}\n".format(str(args.resume)))
        saver.flush()
        print("Validation dataset size:", len(valset))


    # ----------------------------#
    # build model
    # ----------------------------#

    model = Model(config)
    model = model.to(device)
    model = nn.DataParallel(model)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Model Parameter num: ',pytorch_total_params)
    if args.log:
        saver.write('Model Parameter num: '+str(pytorch_total_params)+'\n')
        saver.flush()
    
    # ----------------------------#
    # specify optimizer

    if args.optimizer == 'adamw':
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    if args.log:
        saver.write('Trainable Parameter num: '+str(pytorch_total_params)+'\n')
        saver.flush()

    # specify creterion
    reduction = 'mean'
    motion_reduction = 'sum'
    criterion = {'recon': torch.nn.MSELoss(reduction=reduction),'percep': VGGPerceptualLoss().cuda()}

    if 'percep' in config['loss_list']:
        criterion['recon'] =  torch.nn.L1Loss(reduction=reduction)

    module = Module(model, config, optimizer, criterion)
    module.loss_list = args.loss_list
    # ------------------------------#
    # load model
    if args.resume != '' or args.mode in ['val']:
        checkpoint = torch.load(args.resume)
        message = module.model.load_state_dict(checkpoint['model_state_dict'],strict=False)
        print(message)
        start_epoch = checkpoint['epoch'] + 1
        if 'best_psnr' in checkpoint:
            best_psnr = checkpoint['best_psnr']
            if np.isinf(best_psnr):
                best_psnr = -1
        print("Load model from {}, at epoch {}".format(args.resume, start_epoch - 1))
        module.epoch = start_epoch -1

    # --------------- TRAINING CODE ---------------#
    best_model_name = 'best_psnr_0to1.pth'
    if args.mode == 'train':

        for epoch in range(start_epoch, num_epochs + 1):
            selected_display = []
            start_time = time.time()
            lr = module.optimizer.param_groups[0]['lr']
            print("Epoch {}, learning rate {}".format(epoch, lr))

            if need_log:
                saver.write("epoch: {}, lr: {}\t".format(epoch, lr))
                saver.flush()

            metrics = {}
            metrics['total'] = AverageMeter('Total loss', ':.6f')  # for motion prediction error
            for key in args.loss_list:
                metrics[key] = AverageMeter(key + ' loss', ':.6f')

            module.model.train()
            it = time.time()

            for i, sample in enumerate(trainloader, 0):
                img,gt = sample

                data = {}
                data['input_img'] = img.to(config['device']).type(dtype=torch.float32)
                data['gt_img'] = gt.to(config['device']).type(dtype=torch.float32)

                output_list, loss_dict = module.step(data,epoch)
                recon_img = output_list['recon_img']    
                metrics = update_metrics(metrics,loss_dict)
                recon_img = recon_img.reshape(img.shape[0], config['fut_len'], config['in_res'][0],config['in_res'][1], -1)
                if i % config['train_disp_freq'] == 0:
                    selected_display.append(
                        cp((data['input_img'][0].cpu().detach().numpy().copy(), recon_img[0].cpu().detach().numpy().copy())))
                    message = metric_print(metrics,epoch,i,str(time.time() - it))
                    it = time.time()
                    print(message)
            if not module.scheduler is None:
                module.scheduler.step()
            if 'ldl_g' in config['loss_list']:
                module.scheduler_d.step()
            message = metric_print(metrics, epoch, -1, str(time.time() - start_time),True)
            
            print(message)

            if need_log:
                saver.write(message+'\n')
                saver.flush()
                
                save_dict = {'epoch': epoch,
                        'model_state_dict': module.model.state_dict(),
                        'optimizer_state_dict': module.optimizer.state_dict(),
                        'loss': metrics['total'].avg,
                        'best_psnr': best_psnr}
                if not module.scheduler is None:
                    save_dict['scheduler_state_dict'] = module.scheduler.state_dict()
                torch.save(save_dict, os.path.join(model_save_path, 'latest.pth'))
                print('---------------------------')
                if (args.energy_save_mode and ((epoch % int(config['t_period'][0])) < (0.8 * (config['t_period'][0]))) and (epoch % 5 !=0)):
                    continue
                
                print('Validation on Epoch ',epoch)
                print('---------------------------')
                
                    
                val_metrics,module = validate(valloader, module, config, model_save_path, saver, epoch)
                
                
                save_dict = {'epoch': epoch,
                                'model_state_dict': module.model.state_dict(),
                                'optimizer_state_dict': module.optimizer.state_dict(),
                                'loss': metrics['total'].avg,
                                'best_psnr': best_psnr}

                if not module.scheduler is None:
                    save_dict['scheduler_state_dict'] = module.scheduler.state_dict()
                
                if val_metrics['psnr'].avg > best_psnr:
                    best_psnr = val_metrics['psnr'].avg
                    save_dict['best_psnr'] = best_psnr
                    torch.save(save_dict, os.path.join(model_save_path, best_model_name))
                if (epoch > 0) and (epoch %10 == 0):
                    best_model_name = 'best_psnr_'+str(epoch // 10)+'to'+str(epoch // 10+1)+'.pth'
                    best_psnr = 0. #reinitialize
                torch.save(save_dict, os.path.join(model_save_path, 'latest.pth'))
                visualization_check_video(model_save_path, epoch, selected_display, valid=(config['range'] == 255),is_train=True,config=config,long_term = config['fut_len'] > 1)

            else:
                validate(valloader, module, config, None, None, None)

            del selected_display
            gc.collect()
            
            print('---------------------------')
 
    elif args.mode == 'val':
        print('Validate on epoch ', module.epoch)
        validate(valloader, module, config, model_save_path, saver, -1)


def validate(valloader,module,config,model_save_path,saver,epoch):
    
    
    module.model.eval()
    val_metrics = {}
    eval_metrics = {}
    for key in config['loss_list']:
        val_metrics[key] = AverageMeter(key + ' loss', ':.6f',is_val=True)
    for key in config['eval_list']:
        eval_metrics[key] = AverageMeter(key, ':.6f',is_val=True)
    val_metrics['total'] = AverageMeter('Total loss', ':.6f',is_val=True)  # for motion prediction error
    it = time.time()
    start_time = time.time()
    selected_display = []
    metrics_to_save = []
    with torch.no_grad():

        for i, sample in enumerate(valloader, 0):

            img,gt = sample
            data = {}
            data['input_img'] = img.to(config['device']).type(dtype=torch.float32)
            data['gt_img'] = gt.to(config['device']).type(dtype=torch.float32)

            output_list, loss_dict = module.val(data,epoch)
            recon_img = output_list['recon_img']
            data['gt_img'] = data['gt_img']

            val_metrics = update_metrics(val_metrics,loss_dict)
            recon_img = recon_img.reshape(img.shape[0],config['fut_len'],config['in_res'][0],config['in_res'][1],-1)
            eval_metrics = image_evaluation(recon_img.detach(),data['gt_img'].detach(),eval_metrics,valid=(config['range']==255))
            
            if i % config['val_disp_freq'] == 0:
                selected_display.append(cp((data['input_img'][0].cpu().detach().numpy(), recon_img[0].cpu().detach().numpy())))

                message = metric_print(val_metrics, epoch, i, str(time.time() - it))
                print(message)
                it = time.time()
            del output_list

    val_metrics = {**val_metrics,**eval_metrics}
    message = metric_print(val_metrics, epoch, -1, str(time.time() - start_time), True)
    print(message)


    visualization_check_video(model_save_path, epoch, selected_display, valid=(config['range'] == 255),config=config,long_term = config['fut_len'] > 1)


    if not saver is None:
        saver.write('Validation epoch ' + str(epoch) + '_' + message+'\n')
    del selected_display

    return val_metrics,module


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #model related 
    parser.add_argument('--method', default='ours_graph', type=str, help='Which method to be test, ours/SimVP',choices= ['pretrain','recon','ours_graph','simvp'])
    parser.add_argument('--base_channel', default=32, type=int, help='input image range')
    parser.add_argument('--shuffle_scale', default=1, type=int, help='Pixel shuffle')
    parser.add_argument('--top_k', default=0.005, type=float, help='select top k')
    parser.add_argument('--out_edge_num', default=-1, type=int, help='out edge num')
    parser.add_argument('--long_term', action='store_true', help='do long term forward')
    parser.add_argument('--fut_len', default=-1, type=int, help='If -1, follow the initial config parameter, otherwise, update config')
    parser.add_argument('--prev_len', default=-1, type=int, help='If -1, follow the initial config parameter, otherwise, update config')
    parser.add_argument('--downsample_scale', nargs="*", type=int,default=[], help='downsample ratio, the length of the list indicating the downsample times')
    parser.add_argument('--window_length', default=-1, type=int, help='window ratio for temporal similarity matrix')
    parser.add_argument('--scale_in_use', default=4, type=int, help='how many scales of features used for composition')
    parser.add_argument('--pred_att_iter_num', default=1, type=int, help='iteration number for graph attention')
    parser.add_argument('--edge_list', nargs="*", type=str, default=['forward','backward','spatial'], help='edge type list currently includes: i) forward, ii)backward, iii)spatial')
    parser.add_argument('--edge_softmax', action='store_true', help='Apply softmax on edge ')
    parser.add_argument('--edge_normalize', action='store_true', help='Apply normalization on edge ')
    parser.add_argument('--spatial_conv', action='store_true', help='Using conv2D as the spatial attention module ')
    parser.add_argument('--tendency_len', default=0, type=int,help='Add tendency embedding to graph attention ')
    parser.add_argument('--motion_fuse', action='store_true', help='Fuse Multiscale Motion')
    parser.add_argument('--motion_upsample', action='store_true', help='Upsample Motion to the largest scale')
    parser.add_argument('--tdc_pool', default='max', help='pooling method for pointnet',choices= ['max','avg'])
    parser.add_argument('--pos_len', default=0, type=int, help='Add position id to node feature')
    
    
    #data related
    parser.add_argument('--dataset', default='ucf_4to1', help='choose dataset',choices= ['ucf_4to1','strpm','kitti','city'])
    parser.add_argument('--img_range', default=1, type=int, help='input image range')
    parser.add_argument('--flip_aug', action='store_true', help='flip augmentation')
    parser.add_argument('--rot_aug', action='store_true', help='rotation augmentation')
    parser.add_argument('--val_subset', default='all', type=str, help='Which subset to use',choices= ['hard','intermediate','easy','all'])
    
    
    # training related
    parser.add_argument('--mode', default=None, help='Train/Val mode',choices=['train','val'])
    parser.add_argument('--resume', default='', type=str, help='The path to the saved model that is loaded to resume training')
    parser.add_argument('--batch', default=32, type=int, help='Batch size')
    parser.add_argument('--nepoch', default=10, type=int, help='Number of epochs')
    parser.add_argument('--display_freq', default=6, type=int, help='display frequency')
    parser.add_argument('--nworker', default=0, type=int, help='Number of workers')
    parser.add_argument('--lr', default=0.001, type=float, help='Initial learning rate')
    parser.add_argument('--cos_restart', action='store_true', help='use cosine restart scheduler')
    parser.add_argument('--restart_ratio', default=0.5, type=float, help='lr drop ratio in cosine restart lr scheduler')
    parser.add_argument('--t_period', nargs="*", type=int,default=[], help='consine restart scheduler period')
    parser.add_argument('--optimizer', default='adamw', help='Optimizer choice',choices=['adamw'])
    

    #loss related
    parser.add_argument('--loss_list', nargs="*", type=str, default=['recon'], help='loss type list')
    parser.add_argument('--eval_list', nargs="*", type=str, default=['psnr','ssim'], help='loss type list')
    

    #display/save related
    parser.add_argument('--energy_save_mode', action='store_true', help='Only validate when needed')
    parser.add_argument('--log', action='store_true', help='Whether to log')
    parser.add_argument('--logpath', default='/mnt/team/t-yiqizhong/Summer2023/video_prediction/results/', help='The path to the output log file')    
    parser.add_argument('--exp_name', default='', help='The name of the experiment')
    

    
    args = parser.parse_args()
    print(args)
    torch.manual_seed(1024)
    cur_config = None

    # Set dataset
    if args.dataset.find('ucf')>-1:
        cur_config = configs.ucf_config
        from data.Dataloader import UCFPredDataset as PredDataset
    elif args.dataset.find('strpm')>-1:
        cur_config = configs.strpm_ucf_config
        from data.Dataloader import STRPM_UCFPredDataset as PredDataset
    elif args.dataset.find('kitti')>-1:
        cur_config = configs.kitti_config
        from data.Dataloader import KittiTrainDataset as PredDataset
        if args.mode == 'val':
           from data.Dataloader import KittiValDataset as PredDataset
    elif args.dataset.find('city')>-1:
        cur_config = configs.city_config
        from data.Dataloader import CityTrainDataset as PredDataset

    set_seed(1) # from SimVP
    cur_config = update_config(cur_config,args)
    
    print('\n+++++++++++++++++++++++++++++')
    print('Matrix size: ',cur_config['mat_size'])
    print('Select top ', cur_config['edge_num'], " egdes")
    print('+++++++++++++++++++++++++++++\n')
    
    main(cur_config,args)



