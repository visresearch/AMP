#!/bin/bash  
#SBATCH -o out/slurm/prune.%j.txt 
#SBATCH -J prune 
##SBATCH -p 3080ti-shen
#SBATCH -p A6000-ni
##SBATCH --nodes=1 
##SBATCH --ntasks=16

#SBATCH --nodes=1 
#SBATCH --ntasks=16

# source ~/.bashrc
# conda activate py39torch231

# CONFIG=./configs/open_clip_coco.yaml
CONFIG=./configs/open_clip_BigG_coco.yaml

python main_prune.py --config_file $CONFIG

