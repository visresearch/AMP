

# ------------------------------------------------ 8 blocks + 8 blocks + 8 blocks + 8 blocks + 8 blocks ------------------------------------------------

# ---------------- vit-giant2 (32-39_1, 24-31_1024, 16-23_1536, 8-15_1536, 0-7_1536) ----------------
python dinov2/run/eval/knn.py \
    --config-file dinov2/configs/eval/vitg14_pretrain_prune.yaml \
    --pretrained-weights ./out/dinov2_multi_stage_abs/32-39_1+24-31_1024+16-23_1536+8-15_1536+0-7_1536/dinov2_vitg14_pretrain_prune.pth \
    --output-dir /public/scccse/model_weight/dinov2/knn_prune/multi_stage_abs/giant2/32-39_1+24-31_1024+16-23_1536+8-15_1536+0-7_1536 \
    --train-dataset ImageNet:split=TRAIN:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --val-dataset ImageNet:split=VAL:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --batch-size 1792

# {"('full', 10) Top 1": 70.906001329422}
# {"('full', 10) Top 5": 86.080002784729}
# {"('full', 20) Top 1": 70.97399830818176}
# {"('full', 20) Top 5": 87.7839982509613}
# {"('full', 100) Top 1": 68.48400235176086}
# {"('full', 100) Top 5": 88.9680027961731}
# {"('full', 200) Top 1": 66.99000000953674}
# {"('full', 200) Top 5": 88.5699987411499}


# ------------------------------------------------ 8 block + 8 blocks ------------------------------------------------
# ---------------- vit-giant2 (32-39_1, 24-31_1024) ----------------
python dinov2/run/eval/knn.py \
    --config-file dinov2/configs/eval/vitg14_pretrain_prune.yaml \
    --pretrained-weights ./out/dinov2_multi_stage_abs/32-39_1+24-31_1024/dinov2_vitg14_pretrain_prune.pth \
    --output-dir /public/scccse/model_weight/dinov2/knn_prune/multi_stage_abs/giant2/32-39_1+24-31_1024 \
    --train-dataset ImageNet:split=TRAIN:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --val-dataset ImageNet:split=VAL:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --batch-size 1792

# {"('full', 10) Top 1": 81.42200112342834}
# {"('full', 10) Top 5": 92.96200275421143}
# {"('full', 20) Top 1": 81.28600120544434}
# {"('full', 20) Top 5": 94.19999718666077}
# {"('full', 100) Top 1": 79.69599962234497}
# {"('full', 100) Top 5": 95.22600173950195}
# {"('full', 200) Top 1": 78.66799831390381}
# {"('full', 200) Top 5": 95.18200159072876}

# ------------------------------------------------ 8 blocks ------------------------------------------------
# ---------------- vit-giant2 (32-39_768) ----------------
python dinov2/run/eval/knn.py \
    --config-file dinov2/configs/eval/vitg14_pretrain_prune.yaml \
    --pretrained-weights ./out/dinov2_multi_stage_abs/32-39_768/dinov2_vitg14_pretrain_prune.pth \
    --output-dir /public/scccse/model_weight/dinov2/knn_prune/multi_stage_abs/giant2/32-39_768 \
    --train-dataset ImageNet:split=TRAIN:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --val-dataset ImageNet:split=VAL:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --batch-size 1792

# {"('full', 10) Top 1": 84.14199948310852}
# {"('full', 10) Top 5": 93.8539981842041}
# {"('full', 20) Top 1": 84.20600295066833}
# {"('full', 20) Top 5": 94.88400220870972}
# {"('full', 100) Top 1": 83.3400011062622}
# {"('full', 100) Top 5": 96.15399837493896}
# {"('full', 200) Top 1": 82.67800211906433}
# {"('full', 200) Top 5": 96.33399844169617}

# ---------------- vit-giant2 (32-39_1024) ----------------
python dinov2/run/eval/knn.py \
    --config-file dinov2/configs/eval/vitg14_pretrain_prune.yaml \
    --pretrained-weights ./out/dinov2_multi_stage_abs/32-39_1024/dinov2_vitg14_pretrain_prune.pth \
    --output-dir /public/scccse/model_weight/dinov2/knn_prune/multi_stage_abs/giant2/32-39_1024 \
    --train-dataset ImageNet:split=TRAIN:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --val-dataset ImageNet:split=VAL:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --batch-size 1792

# {"('full', 10) Top 1": 84.17199850082397}
# {"('full', 10) Top 5": 93.71799826622009}
# {"('full', 20) Top 1": 84.08200144767761}
# {"('full', 20) Top 5": 94.80800032615662}
# {"('full', 100) Top 1": 83.15200209617615}
# {"('full', 100) Top 5": 96.04799747467041}
# {"('full', 200) Top 1": 82.60999917984009}
# {"('full', 200) Top 5": 96.27799987792969}

# ---------------- vit-giant2 (32-39_512) ----------------
python dinov2/run/eval/knn.py \
    --config-file dinov2/configs/eval/vitg14_pretrain_prune.yaml \
    --pretrained-weights ./out/dinov2_multi_stage_abs/32-39_512/dinov2_vitg14_pretrain_prune.pth \
    --output-dir /public/scccse/model_weight/dinov2/knn_prune/multi_stage_abs/giant2/32-39_512 \
    --train-dataset ImageNet:split=TRAIN:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --val-dataset ImageNet:split=VAL:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --batch-size 1792

# {"('full', 10) Top 1": 84.17199850082397}
# {"('full', 10) Top 5": 93.99600028991699}
# {"('full', 20) Top 1": 84.12200212478638}
# {"('full', 20) Top 5": 95.00600099563599}
# {"('full', 100) Top 1": 83.37399959564209}
# {"('full', 100) Top 5": 96.28199934959412}
# {"('full', 200) Top 1": 82.65799880027771}
# {"('full', 200) Top 5": 96.4460015296936}

# ---------------- vit-giant2 (32-39_256) ----------------
python dinov2/run/eval/knn.py \
    --config-file dinov2/configs/eval/vitg14_pretrain_prune.yaml \
    --pretrained-weights ./out/dinov2_multi_stage_abs/32-39_256/dinov2_vitg14_pretrain_prune.pth \
    --output-dir /public/scccse/model_weight/dinov2/knn_prune/multi_stage_abs/giant2/32-39_256 \
    --train-dataset ImageNet:split=TRAIN:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --val-dataset ImageNet:split=VAL:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --batch-size 1792

# {"('full', 10) Top 1": 84.16399955749512}
# {"('full', 10) Top 5": 94.07200217247009}
# {"('full', 20) Top 1": 84.13199782371521}
# {"('full', 20) Top 5": 95.10999917984009}
# {"('full', 100) Top 1": 83.29399824142456}
# {"('full', 100) Top 5": 96.2719976902008}
# {"('full', 200) Top 1": 82.56999850273132}
# {"('full', 200) Top 5": 96.48600220680237}

# ---------------- vit-giant2 (32-39_128) ----------------
python dinov2/run/eval/knn.py \
    --config-file dinov2/configs/eval/vitg14_pretrain_prune.yaml \
    --pretrained-weights ./out/dinov2_multi_stage_abs/32-39_128/dinov2_vitg14_pretrain_prune.pth \
    --output-dir /public/scccse/model_weight/dinov2/knn_prune/multi_stage_abs/giant2/32-39_128 \
    --train-dataset ImageNet:split=TRAIN:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --val-dataset ImageNet:split=VAL:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --batch-size 1792

# {"('full', 10) Top 1": 84.06400084495544}
# {"('full', 10) Top 5": 94.01599764823914}
# {"('full', 20) Top 1": 84.04600024223328}
# {"('full', 20) Top 5": 95.11399865150452}
# {"('full', 100) Top 1": 83.12199711799622}
# {"('full', 100) Top 5": 96.28599882125854}
# {"('full', 200) Top 1": 82.41999745368958}
# {"('full', 200) Top 5": 96.47600054740906}

