#!/bin/bash  
#SBATCH -o gpu.%j.out ##作业的输出信息文件  
#SBATCH -J knn ##作业名  
#SBATCH -p A6000-ni
#SBATCH -w gpu17 
#SBATCH --nodes=1 ##申请1个节点  
#SBATCH --gres=gpu:1 ##每个作业占用的GPU数量 *


# ---------------- vit-small ----------------
python dinov2/run/eval/knn.py \
    --config-file dinov2/configs/eval/vits14_pretrain.yaml \
    --pretrained-weights /public/scccse/model_weight/dinov2/dinov2_vits14_pretrain.pth \
    --output-dir /public/scccse/model_weight/dinov2/dinov2_vitb14_pretrain \
    --train-dataset ImageNet:split=TRAIN:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --val-dataset ImageNet:split=VAL:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --batch-size 2048

# {"('full', 10) Top 1": 79.00599837303162}
# {"('full', 10) Top 5": 91.62799715995789}
# ---------------- {"('full', 20) Top 1": 79.1379988193512}
# {"('full', 20) Top 5": 92.96000003814697}
# {"('full', 100) Top 1": 77.89400219917297}
# {"('full', 100) Top 5": 94.30400133132935}
# {"('full', 200) Top 1": 77.1619975566864}
# {"('full', 200) Top 5": 94.39799785614014}

# ---------------- vit-base ----------------
python dinov2/run/eval/knn.py \
    --config-file dinov2/configs/eval/vitb14_pretrain.yaml \
    --pretrained-weights /public/scccse/model_weight/dinov2/dinov2_vitb14_pretrain.pth \
    --output-dir /public/scccse/model_weight/dinov2/dinov2_vitb14_pretrain \
    --train-dataset ImageNet:split=TRAIN:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --val-dataset ImageNet:split=VAL:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --batch-size 2048

# {"('full', 10) Top 1": 82.03999996185303}
# {"('full', 10) Top 5": 92.74600148200989}
# ---------------- {"('full', 20) Top 1": 82.10999965667725}
# {"('full', 20) Top 5": 93.92399787902832}
# {"('full', 100) Top 1": 81.42200112342834}
# {"('full', 100) Top 5": 95.23000121116638}
# {"('full', 200) Top 1": 80.87999820709229}
# {"('full', 200) Top 5": 95.47200202941895}

# ---------------- vit-large ----------------
python dinov2/run/eval/knn.py \
    --config-file dinov2/configs/eval/vitl14_pretrain.yaml \
    --pretrained-weights /public/scccse/model_weight/dinov2/dinov2_vitl14_pretrain.pth \
    --output-dir /public/scccse/model_weight/dinov2/dinov2_vitb14_pretrain \
    --train-dataset ImageNet:split=TRAIN:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --val-dataset ImageNet:split=VAL:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --batch-size 2048

# {"('full', 10) Top 1": 83.51600170135498}
# {"('full', 10) Top 5": 93.21600198745728}
# ---------------- {"('full', 20) Top 1": 83.50399732589722}
# {"('full', 20) Top 5": 94.3560004234314}
# {"('full', 100) Top 1": 82.8819990158081}
# {"('full', 100) Top 5": 95.64599990844727}
# {"('full', 200) Top 1": 82.5380027294159}
# {"('full', 200) Top 5": 95.96400260925293}

# ---------------- vit-giant ----------------
python dinov2/run/eval/knn.py \
    --config-file dinov2/configs/eval/vitg14_pretrain.yaml \
    --pretrained-weights /public/scccse/model_weight/dinov2/dinov2_vitg14_pretrain.pth \
    --output-dir /public/scccse/model_weight/dinov2/dinov2_vitb14_pretrain \
    --train-dataset ImageNet:split=TRAIN:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --val-dataset ImageNet:split=VAL:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --batch-size 1024

# {"('full', 10) Top 1": 83.50800275802612}
# {"('full', 10) Top 5": 92.89000034332275}
# ---------------- {"('full', 20) Top 1": 83.52599740028381}
# {"('full', 20) Top 5": 93.99799704551697}
# {"('full', 100) Top 1": 82.49599933624268}
# {"('full', 100) Top 5": 95.21600008010864}
# {"('full', 200) Top 1": 81.88999891281128}
# {"('full', 200) Top 5": 95.49000263214111}



