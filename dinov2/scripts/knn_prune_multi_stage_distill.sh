

# ------------------------------------------------ 8 blocks + 8 blocks + 8 blocks + 8 blocks + 8 blocks ------------------------------------------------

# ---------------- vit-giant2 (32-39_1, 24-31_1024, 16-23_1536, 8-15_1536, 0-7_1536) ----------------
python dinov2/run/eval/knn.py \
    --config-file dinov2/configs/eval/vitg14_pretrain_prune.yaml \
    --pretrained-weights ./out/dinov2_multi_stage/32-39_1+24-31_1024+16-23_1536+8-15_1536+0-7_1536/dinov2_vitg14_pretrain_prune.pth \
    --output-dir /public/scccse/model_weight/dinov2/knn_prune/multi_stage/giant2/32-39_1+24-31_1024+16-23_1536+8-15_1536+0-7_1536 \
    --train-dataset ImageNet:split=TRAIN:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --val-dataset ImageNet:split=VAL:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --batch-size 1792

# {"('full', 10) Top 1": 70.7099974155426}
# {"('full', 10) Top 5": 85.98399758338928}
# {"('full', 20) Top 1": 70.63599824905396}
# {"('full', 20) Top 5": 87.71399855613708}
# {"('full', 100) Top 1": 68.46200227737427}
# {"('full', 100) Top 5": 88.91400098800659}
# {"('full', 200) Top 1": 66.93199872970581}
# {"('full', 200) Top 5": 88.45199942588806}


# ---------------- distilled vit-giant2 (32-39_1, 24-31_1024, 16-23_1536, 8-15_1536, 0-7_1536) ----------------
python dinov2/run/eval/knn.py \
    --config-file dinov2/configs/eval/vitg14_pretrain_prune.yaml \
    --pretrained-weights /public/scccse/model_weight/dinov2/distill/eval/manual_12500/distill_32-39_1+24-31_1024+16-23_1536+8-15_1536+0-7_1536.pth \
    --output-dir /public/scccse/model_weight/dinov2/knn_prune/multi_stage/giant2/32-39_1+24-31_1024+16-23_1536+8-15_1536+0-7_1536/distill \
    --train-dataset ImageNet:split=TRAIN:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --val-dataset ImageNet:split=VAL:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --batch-size 2048

# {"('full', 10) Top 1": 82.014000415802}
# {"('full', 10) Top 5": 92.85600185394287}
# {"('full', 20) Top 1": 81.80000185966492}
# {"('full', 20) Top 5": 94.10399794578552}
# {"('full', 100) Top 1": 80.33400177955627}
# {"('full', 100) Top 5": 95.35999894142151}
# {"('full', 200) Top 1": 79.43000197410583}
# {"('full', 200) Top 5": 95.41599750518799}

python dinov2/run/eval/knn.py \
    --config-file dinov2/configs/eval/vitg14_pretrain_prune.yaml \
    --pretrained-weights /public/scccse/model_weight/dinov2/distill2/eval/training_62499/distill_32-39_1+24-31_1024+16-23_1536+8-15_1536+0-7_1536.pth \
    --output-dir /public/scccse/model_weight/dinov2/knn_prune/multi_stage/giant2/32-39_1+24-31_1024+16-23_1536+8-15_1536+0-7_1536/distill2 \
    --train-dataset ImageNet:split=TRAIN:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --val-dataset ImageNet:split=VAL:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --batch-size 2048

# {"('full', 10) Top 1": 83.95199775695801}
# {"('full', 10) Top 5": 93.69400143623352}
# {"('full', 20) Top 1": 83.88000130653381}
# {"('full', 20) Top 5": 94.7700023651123}
# {"('full', 100) Top 1": 82.71999955177307}
# {"('full', 100) Top 5": 95.95199823379517}
# {"('full', 200) Top 1": 81.9819986820221}
# {"('full', 200) Top 5": 96.11200094223022}


python dinov2/run/eval/knn.py \
    --config-file dinov2/configs/eval/vitg14_pretrain_prune.yaml \
    --pretrained-weights /public/scccse/model_weight/dinov2/distill3/eval/training_62499/distill_32-39_1+24-31_1024+16-23_1536+8-15_1536+0-7_1536.pth \
    --output-dir /public/scccse/model_weight/dinov2/knn_prune/multi_stage/giant2/32-39_1+24-31_1024+16-23_1536+8-15_1536+0-7_1536/distill3 \
    --train-dataset ImageNet:split=TRAIN:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --val-dataset ImageNet:split=VAL:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --batch-size 2048

# {"('full', 10) Top 1": 83.70800018310547}
# {"('full', 10) Top 5": 93.59400272369385}
# {"('full', 20) Top 1": 83.5319995880127}
# {"('full', 20) Top 5": 94.69000101089478}
# {"('full', 100) Top 1": 82.56800174713135}
# {"('full', 100) Top 5": 95.85599899291992}
# {"('full', 200) Top 1": 81.80999755859375}
# {"('full', 200) Top 5": 96.05799913406372}

python dinov2/run/eval/knn.py \
    --config-file dinov2/configs/eval/vitg14_pretrain_prune.yaml \
    --pretrained-weights /public/scccse/model_weight/dinov2/distill4/eval/training_62499/distill_32-39_1+24-31_1024+16-23_1536+8-15_1536+0-7_1536.pth \
    --output-dir /public/scccse/model_weight/dinov2/knn_prune/multi_stage/giant2/32-39_1+24-31_1024+16-23_1536+8-15_1536+0-7_1536/distill4 \
    --train-dataset ImageNet:split=TRAIN:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --val-dataset ImageNet:split=VAL:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --batch-size 2048

# {"('full', 10) Top 1": 83.92800092697144}
# {"('full', 10) Top 5": 93.6959981918335}
# {"('full', 20) Top 1": 83.8699996471405}
# {"('full', 20) Top 5": 94.80000138282776}
# {"('full', 100) Top 1": 82.78800249099731}
# {"('full', 100) Top 5": 95.93799710273743}
# {"('full', 200) Top 1": 81.96600079536438}
# {"('full', 200) Top 5": 96.13000154495239}


python dinov2/run/eval/knn.py \
    --config-file dinov2/configs/eval/vitg14_pretrain_prune.yaml \
    --pretrained-weights ./out/dinov2/dinov2_2024-08-01_22-25-42/ckpt/student_dinov2_vit_giant2_5.pth \
    --output-dir ./out/dinov2/dinov2_2024-08-01_22-25-42/ckpt/knn \
    --train-dataset ImageNet:split=TRAIN:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --val-dataset ImageNet:split=VAL:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --batch-size 2048

# {"('full', 10) Top 1": 83.80200266838074}
# {"('full', 10) Top 5": 93.63800287246704}
# {"('full', 20) Top 1": 83.77000093460083}
# {"('full', 20) Top 5": 94.76600289344788}
# {"('full', 100) Top 1": 82.69400000572205}
# {"('full', 100) Top 5": 95.95999717712402}
# {"('full', 200) Top 1": 81.9920003414154}
# {"('full', 200) Top 5": 96.13199830055237}

python dinov2/run/eval/knn.py \
    --config-file dinov2/configs/eval/vitg14_pretrain_prune.yaml \
    --pretrained-weights ./out/dinov2/dinov2_2024-08-01_17-06-56/ckpt/student_dinov2_vit_giant2_5.pth \
    --output-dir ./out/dinov2/dinov2_2024-08-01_17-06-56/ckpt/knn \
    --train-dataset ImageNet:split=TRAIN:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --val-dataset ImageNet:split=VAL:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --batch-size 2048

# {"('full', 10) Top 1": 83.24199914932251}
# {"('full', 10) Top 5": 93.62000226974487}
# {"('full', 20) Top 1": 83.15799832344055}
# {"('full', 20) Top 5": 94.74800229072571}
# {"('full', 100) Top 1": 81.9599986076355}
# {"('full', 100) Top 5": 95.86399793624878}
# {"('full', 200) Top 1": 81.07600212097168}
# {"('full', 200) Top 5": 96.06000185012817}
