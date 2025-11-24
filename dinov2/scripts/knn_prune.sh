#!/bin/bash  
#SBATCH -o gpu.%j.out ##作业的输出信息文件  
#SBATCH -J knn ##作业名  
#SBATCH -p A6000-ni
#SBATCH -w gpu17 
#SBATCH --nodes=1 ##申请1个节点  
#SBATCH --gres=gpu:1 ##每个作业占用的GPU数量 *


# ---------------- vit-base ----------------
python dinov2/run/eval/knn.py \
    --config-file dinov2/configs/eval/vitb14_pretrain_prune.yaml \
    --pretrained-weights /public/scccse/model_weight/dinov2/prune/dinov2_vitb14_pretrain_prune.pth \
    --output-dir /public/scccse/model_weight/dinov2/knn_prune \
    --train-dataset ImageNet:split=TRAIN:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --val-dataset ImageNet:split=VAL:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --batch-size 2048

# {"('full', 10) Top 1": 82.04799890518188}
# {"('full', 10) Top 5": 92.93400049209595}
# {"('full', 20) Top 1": 81.9379985332489}
# {"('full', 20) Top 5": 94.10399794578552}
# {"('full', 100) Top 1": 80.5679976940155}
# {"('full', 100) Top 5": 95.23000121116638}
# {"('full', 200) Top 1": 79.63399887084961}
# {"('full', 200) Top 5": 95.35199999809265}

# ---------------- vit-large ----------------
python dinov2/run/eval/knn.py \
    --config-file dinov2/configs/eval/vitl14_pretrain_prune.yaml \
    --pretrained-weights ./out/dinov2_vitl14_pretrain_prune.pth \
    --output-dir /public/scccse/model_weight/dinov2/knn_prune \
    --train-dataset ImageNet:split=TRAIN:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --val-dataset ImageNet:split=VAL:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --batch-size 2048


# {"('full', 10) Top 1": 83.81400108337402}
# {"('full', 10) Top 5": 93.55599880218506}
# ---------------- {"('full', 20) Top 1": 83.68800282478333}
# {"('full', 20) Top 5": 94.70400214195251}
# {"('full', 100) Top 1": 82.68600106239319}
# {"('full', 100) Top 5": 95.96999883651733}
# {"('full', 200) Top 1": 82.10800290107727}
# {"('full', 200) Top 5": 96.10199928283691}

# ---------------- vit-giant2 ----------------
python dinov2/run/eval/knn.py \
    --config-file dinov2/configs/eval/vitg14_pretrain_prune.yaml \
    --pretrained-weights ./out/dinov2_vitg14_pretrain_prune.pth \
    --output-dir /public/scccse/model_weight/dinov2/knn_prune \
    --train-dataset ImageNet:split=TRAIN:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --val-dataset ImageNet:split=VAL:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --batch-size 1024

# ---------------- {"('full', 10) Top 1": 83.70800018310547}
# {"('full', 10) Top 5": 93.09599995613098}
# {"('full', 20) Top 1": 83.65399837493896}
# {"('full', 20) Top 5": 94.19599771499634}
# {"('full', 100) Top 1": 82.70800113677979}
# {"('full', 100) Top 5": 95.3980028629303}
# {"('full', 200) Top 1": 82.1340024471283}
# {"('full', 200) Top 5": 95.67000269889832}