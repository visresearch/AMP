#!/bin/bash  
#SBATCH -o out/slurm/zscG.%j.log 
#SBATCH -J zscG 
#SBATCH -p A6000-ni
#SBATCH -w gpu17
#SBATCH --nodes=1 

#SBATCH --gres=gpu:8
#SBATCH --ntasks=64

##SBATCH --gres=gpu:4
##SBATCH --ntasks=32

##SBATCH --gres=gpu:2
##SBATCH --ntasks=16

##SBATCH --gres=gpu:1
##SBATCH --ntasks=8

source ~/.bashrc
conda activate py39torch231


# ------------------ Open Clip ------------------
# TYPE=open_clip
# MODEL=ViT-g-14
# BATCHSIZE=1536
# CLF_PATH=out/clfs/open_clip_g_text_clf/text_classifier_weight+viT-g-14

TYPE=open_clip
MODEL=ViT-bigG-14
BATCHSIZE=1024
CLF_PATH=out/clfs/open_clip_big_g_text_clf/text_classifier


# ------------------ Eva Clip ------------------
# TYPE=eva_clip
# MODEL=EVA02-CLIP-bigE-14-plus
# BATCHSIZE=256
# CLF_PATH=out/clfs/eva_clip_text_clf/text_classifier_weight+clipe

# TYPE=eva_clip
# MODEL=EVA-CLIP-8B
# BATCHSIZE=128
# CLF_PATH=out/clfs/eva_clip8b_text_clf/text_classifier_weight+clip8b

NGPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

# ------------------- Dataset -------------------
DATA=(
    imagenet1k
    imagenet-a
    imagenet-r
    imagenetv2
    imagenet_sketch
    objectnet
)

ROOT=(
    /public/scccse/dataset/ILSVRC2012
    /public/datasets/ImageNet-A/imagenet-a
    /public/datasets/ImageNet-r
    /public/datasets/ImageNetV2
    /public/datasets/ImageNet-sketch/sketch
    /public/datasets/objectnet/OpenDataLab___ObjectNet/raw/objectnet-1.0
)

CLFS=()

for i in "${!DATA[@]}";
do
    echo ${DATA[$i]}

    CLFS+=("${CLF_PATH}_${DATA[$i]}.pth")

    echo ${CLFS[$i]}
done

CKPT_PATHS=(
    # out/distill/ViT-bigG-14/open_clip_2025-05-13_15-23-20/ckpt/student_ViT-bigG-14
    out/distill/ViT-bigG-14/open_clip_2025-05-13_16-35-55/ckpt/student_ViT-bigG-14
)

EPOCHS=(10)
# EPOCHS=(20)
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

        # ---------- exsting text classifier ----------
        torchrun --nnodes=1 --nproc_per_node=$NGPU --master-port=29503 \
            eval/eval_zsc.py \
            --model $MODEL \
            --model_type $TYPE \
            --use_prune \
            --pretrained ${CKPTS[$i]} \
            --batch_size $BATCHSIZE \
            --load_clfs ${CLFS[$j]} \
            --dataset ${DATA[$j]} \
            --dataset_root ${ROOT[$j]} \
            --only_vision

        # ---------- without text classifier ----------
        # torchrun --nnodes=1 --nproc_per_node=$NGPU --master-port=29503 \
        #     eval/eval_zsc.py \
        #     --model $MODEL \
        #     --model_type $TYPE \
        #     --use_prune \
        #     --pretrained ${CKPTS[$i]} \
        #     --batch_size $BATCHSIZE \
        #     --save_clf ${CLFS[$j]} \
        #     --dataset ${DATA[$j]} \
        #     --dataset_root ${ROOT[$j]}
    done
done


