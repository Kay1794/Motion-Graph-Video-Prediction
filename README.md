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



