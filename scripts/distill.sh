#!/bin/bash  
#SBATCH -o out/slurm/disG1e-6.%j.out ##作业的输出信息文件  
#SBATCH -J disG1e-6 ##作业名  
#SBATCH -p A6000-ni

##SBATCH -w gpu17

#SBATCH --nodes=1 ##申请1个节点  
#SBATCH --gres=gpu:8
#SBATCH --ntasks=64

##SBATCH --gres=gpu:4
##SBATCH --ntasks=32

##SBATCH --gres=gpu:1
##SBATCH --ntasks=8

source ~/.bashrc
conda activate py39torch231

NGPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

FRAME=open_clip
# CONFIG=configs/distill_open_clip_g.yaml
# CONFIG=configs/distill_open_clip_BigG.yaml
CONFIG=configs/distill_dist/distill_open_clip_BigG.yaml

# FRAME=eva_clip
# CONFIG=configs/distill_clip_BigE.yaml
# CONFIG=configs/distill_clip8B.yaml

# FRAME=dinov2
# CONFIG=configs/distill_dinov2.yaml


torchrun --nnodes=1 --nproc_per_node=$NGPU --master-port=29501 distill.py --config_file $CONFIG  --frame $FRAME

