#!/bin/bash  
#SBATCH -o out/slurm/knn.%j.out ##作业的输出信息文件  
#SBATCH -J knn ##作业名
  
##SBATCH --nodes=1 ##申请1个节点  

#SBATCH -p A6000-ni
#SBATCH --gres=gpu:8 ##每个作业占用的GPU数量 *
#SBATCH --ntasks=64

##SBATCH -p 3080ti-shen
##SBATCH --gres=gpu:2 ##每个作业占用的GPU数量 *
##SBATCH --ntasks=52

source ~/.bashrc
conda activate py39torch231

NGPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)


# --------------------- score ckpt ---------------------
# CKPT_PATHS=()
# CKPT_PATHS=()

# THRESHS=(0.005)
# THRESHS=(0.005)

# for i in "${!THRESHS[@]}";
# do
#     CKPT_PATHS+=(
#         # out/vit_giant2_prune/dinov2_2025-02-21_15-07-57/ckpt/model_layer_wise_thresh_${THRESHS[$i]}_t20
#         out/vit_giant2_prune/the_same_one
#     )
# done
# for i in "${!THRESHS[@]}";
# do
#     CKPT_PATHS+=(
#         # out/vit_giant2_prune/dinov2_2025-02-21_15-07-57/ckpt/model_layer_wise_thresh_${THRESHS[$i]}_t20
#         out/vit_giant2_prune/the_same_one
#     )
# done

# BATCH=1024
# # BATCH=512
# # BATCH=128
# # BATCH=64
# # BATCH=32
# # BATCH=16
# BATCH=1024
# # BATCH=512
# # BATCH=128
# # BATCH=64
# # BATCH=32
# # BATCH=16

# CKPT=()
# CKPT=()

# IDS=(0)
# IDS=(0)

# l=${!IDS[@]}
# l=${!IDS[@]}

# for i in "${!CKPT_PATHS[@]}";
# do
#     for j in "${!IDS[@]}";
#     do
#         CKPT+=("${CKPT_PATHS[$i]}/model_layer_${IDS[$j]}.pth")
# for i in "${!CKPT_PATHS[@]}";
# do
#     for j in "${!IDS[@]}";
#     do
#         CKPT+=("${CKPT_PATHS[$i]}/model_layer_${IDS[$j]}.pth")

#         echo ${CKPT[-1]}
#     done
# done
#         echo ${CKPT[-1]}
#     done
# done

# idx=0
# idx=0

# for i in "${!CKPT_PATHS[@]}";
# do
#     for j in "${!IDS[@]}";
#     do
#         echo ${CKPT[${idx}]}
#         torchrun --nnodes=1 --nproc_per_node=$NGPU --master-port=29502 eval/eval_knn.py --frame dinov2 --config_file configs/dinov2_coco.yaml --output_dir ${CKPT_PATHS[$i]}/${IDS[$j]} --batch_size $BATCH --pretrained ${CKPT[${idx}]}
#         let idx++
#     done
# done
# for i in "${!CKPT_PATHS[@]}";
# do
#     for j in "${!IDS[@]}";
#     do
#         echo ${CKPT[${idx}]}
#         torchrun --nnodes=1 --nproc_per_node=$NGPU --master-port=29502 eval/eval_knn.py --frame dinov2 --config_file configs/dinov2_coco.yaml --output_dir ${CKPT_PATHS[$i]}/${IDS[$j]} --batch_size $BATCH --pretrained ${CKPT[${idx}]}
#         let idx++
#     done
# done

# --------------------- distill ckpt ---------------------
# BATCH=1024
# CKPT_PATH=out/distill/vit_giant2/dinov2_2025-04-12_22-25-13/ckpt
# EPOCHS=(24 25 26 27 28 29 30)

# for i in "${!EPOCHS[@]}";
# do
#     CKPT=$CKPT_PATH/student_vit_giant2_${EPOCHS[$i]}.pth

#     torchrun --nnodes=1 --nproc_per_node=$NGPU --master-port=29502 eval/eval_knn.py --frame dinov2 --config_file configs/dinov2_coco.yaml --output_dir $CKPT_PATH/${EPOCHS[$i]} --batch_size $BATCH --pretrained $CKPT
# done

# --------------------- single ckpt ---------------------

BATCH=1024
# # BATCH=128
# # BATCH=96
# # BATCH=64
# # BATCH=32


CKPT=out/vit_giant2_prune/dinov2_2025-05-13_21-37-01/ckpt/model_thresh_0.0003_t20/model_layer.pth
OUT=out/vit_giant2_prune/dinov2_2025-05-13_21-37-01/ckpt/model_thresh_0.0003_t20/
CONFIG=configs/prune_dist/dinov2_coco.yaml

torchrun --nnodes=1 --nproc_per_node=$NGPU --master-port=29502 eval/eval_knn.py --frame dinov2 --config_file $CONFIG --output_dir $OUT --batch_size $BATCH --pretrained $CKPT