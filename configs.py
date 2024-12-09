import numpy as np

'''
Dataset root
'''
lambda_14 = False
lambda_11 = False

ucf_config = {}
ucf_config['name'] = 'ucf'
ucf_config['dataroot'] = '/data/ours_ucf/'
ucf_config['in_res'] = (512,512)
ucf_config['prev_len'] = 4
ucf_config['fut_len'] = 1
ucf_config['total_len'] = ucf_config['prev_len'] + ucf_config['fut_len']
ucf_config['eval_list'] = ['psnr','ssim']
ucf_config['shuffle_setting'] = True
ucf_config['downsample_scale'] = np.asarray([2,4,2])
ucf_config['num_probe'] = 9
ucf_config['n_channel'] = 3

#----------------------------------#

strpm_ucf_config = {}
for key in ucf_config.keys():
    strpm_ucf_config[key] = ucf_config[key]
strpm_ucf_config['dataroot'] = '/data/strpm_ucf/'

#----------------------------------#

kitti_config = {}
kitti_config['name'] = 'kitti'
kitti_config['dataroot'] = '/data/kitti_2to1' # '/home/yiqizhong/project/video_prediction/dataset/KITTI_2to1/'

kitti_config['in_res'] = (256,832)
kitti_config['prev_len'] = 2
kitti_config['fut_len'] = 1
kitti_config['total_len'] = 3
kitti_config['eval_list'] = ['psnr','ssim','psnr_y','lpips']
kitti_config['downsample_scale'] = np.asarray([2,2,2])
kitti_config['n_channel'] = 3

#----------------------------------#

city_config = {}
city_config['name'] = 'city'
city_config['dataroot'] = '/data/cityscapes/'
city_config['in_res'] = (512,1024)
city_config['prev_len'] = 2
city_config['fut_len'] = 1
city_config['total_len'] = 3
city_config['eval_list'] = ['psnr','ssim','psnr_y','lpips']
city_config['downsample_scale'] = np.asarray([2,2,2])
city_config['n_channel'] = 3