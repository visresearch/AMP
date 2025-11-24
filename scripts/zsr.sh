#!/bin/bash  
#SBATCH -o out/slurm/zsr8Bdis.%j.log 
#SBATCH -J zsr8Bdis 
#SBATCH -p A6000-ni
##SBATCH -w gpu17
#SBATCH --nodes=1 

##SBATCH --gres=gpu:8
##SBATCH --ntasks=64

##SBATCH --gres=gpu:4
##SBATCH --ntasks=32

##SBATCH --gres=gpu:2
##SBATCH --ntasks=16

#SBATCH --gres=gpu:1
#SBATCH --ntasks=8

source ~/.bashrc
conda activate py39torch231


# ------------------ Open Clip ------------------
# TYPE=open_clip
# MODEL=ViT-g-14
# BATCHSIZE=1536
# BATCHSIZE=512
# non_visual_pretrained=/public/scccse/model_weight/CLIP-ViT-g-14-laion2B-s34B-b88K/open_clip_pytorch_model_non_visual.pth

# TYPE=open_clip
# MODEL=ViT-bigG-14
# BATCHSIZE=128
# non_visual_pretrained=/public/scccse/model_weight/CLIP-ViT-bigG-14-laion2B-39B-b160k/open_clip_pytorch_model_non_visual.pth

# ------------------ Eva Clip ------------------
# TYPE=eva_clip
# MODEL=EVA02-CLIP-bigE-14-plus
# BATCHSIZE=64
# non_visual_pretrained=/public/scccse/model_weight/EVA-CLIP/EVA02_CLIP_E_psz14_plus_s9B_non_visual.pth

TYPE=eva_clip
MODEL=EVA-CLIP-8B
BATCHSIZE=128
non_visual_pretrained=/public/scccse/model_weight/EVA-CLIP/EVA_CLIP_8B_psz14_s9B_non_visual.pth

# NGPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
NGPU=1

# ------------------- Dataset -------------------
DATA=(
    flickr30k
    mscoco_captions
)

ROOT=(
    /public/scccse/dataset/Flick30k
    /public/scccse/dataset/COCO2014
)

CKPT_PATHS=(
    out/distill/EVA-CLIP-8B/eva_clip_2025-04-18_21-03-54/ckpt/student_EVA-CLIP-8B
)

EPOCHS=(10)
# EPOCHS=(10)
# EPOCHS=(18 19 20)
CKPTS=()

for i in "${!CKPT_PATHS[@]}";
do
    echo ${CKPT_PATHS[$i]}
    for j in "${!EPOCHS[@]}";
    do
        CKPTS+=("${CKPT_PATHS[$i]}_${EPOCHS[$j]}.pth")
        echo ${CKPTS[-1]}
    done
done

# CKPTS=(out/eva_clip_bigE_coco/model_layer_wise_thresh_0.02_entropy/model_layer_0.pth)

for i in "${!CKPTS[@]}";
do
    echo ${CKPTS[$i]}
    for j in "${!DATA[@]}";
    do
        echo ${DATA[$j]}
        torchrun --nnodes=1 --nproc_per_node=$NGPU --master-port=29501 \
            eval/eval_zsc.py \
            --model $MODEL \
            --model_type $TYPE \
            --use_prune \
            --pretrained ${CKPTS[$i]} \
            --non_visual_pretrained $non_visual_pretrained \
            --task "zeroshot_retrieval" \
            --batch_size $BATCHSIZE \
            --dataset ${DATA[$j]} \
            --dataset_root ${ROOT[$j]} \
            --only_vision
    done
done


