
# ------------------------------------------------ 8 blocks + 8 blocks + 8 blocks + 8 blocks + 8 blocks ------------------------------------------------##SBATCH -w gpu16 

# ---------------- vit-giant2 (32-39_1, 24-31_1024, 16-23_1536, 8-15_1536, 0-7_1536) ----------------
python dinov2/run/eval/knn.py \
    --config-file dinov2/configs/eval/vitg14_pretrain_prune.yaml \
    --pretrained-weights ./out/dinov2_multi_stage_abs_sort/32-39_1+24-31_1024+16-23_1536+8-15_1536+0-7_1536/dinov2_vitg14_pretrain_prune.pth \
    --output-dir /public/scccse/model_weight/dinov2/knn_prune/multi_stage_abs_sort/giant2/32-39_1+24-31_1024+16-23_1536+8-15_1536+0-7_1536 \
    --train-dataset ImageNet:split=TRAIN:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --val-dataset ImageNet:split=VAL:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --batch-size 1792



# ---------------- vit-giant2 (32-39_1, 24-31_1024) ----------------
python dinov2/run/eval/knn.py \
    --config-file dinov2/configs/eval/vitg14_pretrain_prune.yaml \
    --pretrained-weights ./out/dinov2_multi_stage_abs_sort/32-39_1+24-31_1024/dinov2_vitg14_pretrain_prune.pth \
    --output-dir /public/scccse/model_weight/dinov2/knn_prune/multi_stage_abs_sort/giant2/32-39_1+24-31_1024 \
    --train-dataset ImageNet:split=TRAIN:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --val-dataset ImageNet:split=VAL:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --batch-size 1792

# {"('full', 10) Top 1": 81.20200037956238}
# {"('full', 10) Top 5": 92.89000034332275}
# {"('full', 20) Top 1": 81.03399872779846}
# {"('full', 20) Top 5": 94.10799741744995}
# {"('full', 100) Top 1": 79.25999760627747}
# {"('full', 100) Top 5": 95.14600038528442}
# {"('full', 200) Top 1": 78.18400263786316}
# {"('full', 200) Top 5": 95.07399797439575}