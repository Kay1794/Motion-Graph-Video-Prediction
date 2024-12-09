## This is the official repo for NeurIPS 2024 paper - Motion Graph Unleashed: A Novel Approach to Video Prediction [paper](https://arxiv.org/pdf/2410.22288)

# Prepare
```bash
git clone https://github.com/Kay1794/Motion-Graph-Video-Prediction.git
cd Motion-Graph-Video-Prediction
conda env create -f environment.yml
```
# Dataset
UCF Sports Dataset 
+ [Data download](https://www.crcv.ucf.edu/data/ucf_sports_actions.zip)
+ [MMVP split](./dataset/ucf_mmvp_split)
  - Unzip downloaded file
  - Modify ucf_config\['dataroot'\] in config.py to the unziped folder
  - Copy split txt files to ucf_config\['dataroot'\]
+ [STRPM split](./dataset/ucf_strpm_split)
  - ⬜️Dataloader
  - ⬜️Implement steps

KITTI & Cityscapse
+ We follow the data preparation steps of [CVPR2023 DMVFN](https://github.com/hzwer/CVPR2023-DMVFN)

# Run
Training on UCF Sports MMVP Split
```bash
python main.py --mode train --scale_in_use 4 --base_channel 16 --downsample_scale 2 2 2 --exp baseline_old_env --cos_restart --rot_aug --flip_aug --loss_list recon --edge_normalize --pred_att_iter_num 3 --tendency_len 16 --edge_list backward forward spatial --t_period 300 --nepoch 300 --eval_list psnr ssim lpips --logpath ./results/ --shuffle_scale 2 --pos_len 4 --loss_list recon --top_k 0.01 --batch 16 --dataset ucf_4to1 --energy_save_mode --log
```
Validation on UCF Sports MMVP Split
```bash
python main.py --mode train  --scale_in_use 4 --base_channel 16  --downsample_scale 2 2 2 --exp baseline_fix --cos_restart --rot_aug --flip_aug --loss_list recon --edge_normalize --pred_att_iter_num 3 --tendency_len 16 --edge_list backward forward spatial  --t_period 300 --nepoch 300 --eval_list psnr ssim lpips --logpath /mnt/team/t-yiqizhong/projects/video_prediction/results/ --shuffle_scale 2 --pos_len 4 --loss_list recon --top_k 0.01 --batch 16 --dataset ucf_4to1 --resume ./pretrained_model/ucf_mmvp_split.pth --mode val
```
