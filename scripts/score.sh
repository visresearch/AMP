#!/bin/bash  
#SBATCH -o out/slurm/%j.scdg5e-4.log 
#SBATCH -J scdg5e-4

#SBATCH -p A6000-ni

#SBATCH --nodes=1 

##SBATCH -w gpu15

#SBATCH --gres=gpu:8
#SBATCH --ntasks=64

# source ~/.bashrc
# conda activate py39torch231


NGPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

# FRAME=open_clip
# CONFIG=./configs/open_clip_BigG_coco_entropy.yaml
# CONFIG=./configs/open_clip_g_coco_entropy.yaml
# CONFIG=configs/prune_dist/open_clip_BigG_entropy_dist.yaml

# FRAME=eva_clip
# CONFIG=./configs/eva_clip_bigE_plus_coco.yaml
# CONFIG=./configs/eva_clip_bigE_plus_coco_entropy.yaml
# CONFIG=./configs/eva_clip8b_coco_entropy.yaml
# CONFIG=configs/prune_dist/eva_clip_bigE_plus_entropy_dist.yaml

FRAME=dinov2
CONFIG=./configs/dinov2_coco.yaml
# CONFIG=./configs/prune_dist/dinov2_coco.yaml


XFORMERS_DISABLED=1 torchrun --nnodes=1 --nproc_per_node=$NGPU --master-port=29502 main_score.py --config_file $CONFIG --frame $FRAME
