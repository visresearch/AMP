#!/bin/bash  
#SBATCH -o gpu.%j.out ##作业的输出信息文件  
#SBATCH -J knn ##作业名  
#SBATCH -p A6000-ni
#SBATCH -w gpu17 
#SBATCH --nodes=1 ##申请1个节点  
#SBATCH --gres=gpu:1 ##每个作业占用的GPU数量 *


# ------------------------------------------------ 8 block + 8 blocks ------------------------------------------------
# ---------------- vit-giant2 (32-39_1, 24-31_4) ----------------
python dinov2/run/eval/knn.py \
    --config-file dinov2/configs/eval/vitg14_pretrain_prune.yaml \
    --pretrained-weights ./out/dinov2_multi_stage/32-39_1+24-31_4/dinov2_vitg14_pretrain_prune.pth \
    --output-dir /public/scccse/model_weight/dinov2/knn_prune/multi_stage/giant2/32-39_1+24-31_4 \
    --train-dataset ImageNet:split=TRAIN:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --val-dataset ImageNet:split=VAL:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --batch-size 1792

# {"('full', 10) Top 1": 76.31400227546692}
# {"('full', 10) Top 5": 89.9940013885498}
# {"('full', 20) Top 1": 76.02999806404114}
# {"('full', 20) Top 5": 91.40999913215637}
# {"('full', 100) Top 1": 73.41799736022949}
# {"('full', 100) Top 5": 92.03400015830994}
# {"('full', 200) Top 1": 71.81599736213684}
# {"('full', 200) Top 5": 91.51600003242493}

# ---------------- vit-giant2 (32-39_1, 24-31_512) ----------------
python dinov2/run/eval/knn.py \
    --config-file dinov2/configs/eval/vitg14_pretrain_prune.yaml \
    --pretrained-weights ./out/dinov2_multi_stage/32-39_1+24-31_512/dinov2_vitg14_pretrain_prune.pth \
    --output-dir /public/scccse/model_weight/dinov2/knn_prune/multi_stage/giant2/32-39_1+24-31_512 \
    --train-dataset ImageNet:split=TRAIN:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --val-dataset ImageNet:split=VAL:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --batch-size 1792

# {"('full', 10) Top 1": 80.2299976348877}
# {"('full', 10) Top 5": 92.3479974269867}
# {"('full', 20) Top 1": 79.93999719619751}
# {"('full', 20) Top 5": 93.62800121307373}
# {"('full', 100) Top 1": 78.04399728775024}
# {"('full', 100) Top 5": 94.52800154685974}
# {"('full', 200) Top 1": 76.78200006484985}
# {"('full', 200) Top 5": 94.39600110054016}

# ---------------- vit-giant2 (32-39_1, 24-31_1024) ----------------
python dinov2/run/eval/knn.py \
    --config-file dinov2/configs/eval/vitg14_pretrain_prune.yaml \
    --pretrained-weights ./out/dinov2_multi_stage/32-39_1+24-31_1024/dinov2_vitg14_pretrain_prune.pth \
    --output-dir /public/scccse/model_weight/dinov2/knn_prune/multi_stage/giant2/32-39_1+24-31_1024 \
    --train-dataset ImageNet:split=TRAIN:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --val-dataset ImageNet:split=VAL:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --batch-size 1792

# {"('full', 10) Top 1": 81.26800060272217}
# {"('full', 10) Top 5": 92.89399981498718}
# {"('full', 20) Top 1": 81.19000196456909}
# {"('full', 20) Top 5": 94.15000081062317}
# {"('full', 100) Top 1": 79.54400181770325}
# {"('full', 100) Top 5": 95.1799988746643}
# {"('full', 200) Top 1": 78.53800058364868}
# {"('full', 200) Top 5": 95.14399766921997}

# ------------------------------------------------ 8 block + 8 blocks + 8 blocks ------------------------------------------------

# ---------------- vit-giant2 (32-39_1, 24-31_1024, 16-23_1536) ----------------
python dinov2/run/eval/knn.py \
    --config-file dinov2/configs/eval/vitg14_pretrain_prune.yaml \
    --pretrained-weights ./out/dinov2_multi_stage/32-39_1+24-31_1024+16-23_1536/dinov2_vitg14_pretrain_prune.pth \
    --output-dir /public/scccse/model_weight/dinov2/knn_prune/multi_stage/giant2/32-39_1+24-31_1024+16-23_1536 \
    --train-dataset ImageNet:split=TRAIN:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --val-dataset ImageNet:split=VAL:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --batch-size 1792

# {"('full', 10) Top 1": 78.28400135040283}
# {"('full', 10) Top 5": 91.430002450943}
# {"('full', 20) Top 1": 78.03199887275696}
# {"('full', 20) Top 5": 92.7839994430542}
# {"('full', 100) Top 1": 76.13400220870972}
# {"('full', 100) Top 5": 93.73800158500671}
# {"('full', 200) Top 1": 74.8520016670227}
# {"('full', 200) Top 5": 93.51800084114075}

# ------------------------------------------------ 8 block + 8 blocks + 8 blocks + 8 blocks ------------------------------------------------

# ---------------- vit-giant2 (32-39_1, 24-31_1024, 16-23_1536, 8-15_1536) ----------------
python dinov2/run/eval/knn.py \
    --config-file dinov2/configs/eval/vitg14_pretrain_prune.yaml \
    --pretrained-weights ./out/dinov2_multi_stage/32-39_1+24-31_1024+16-23_1536+8-15_1536/dinov2_vitg14_pretrain_prune.pth \
    --output-dir /public/scccse/model_weight/dinov2/knn_prune/multi_stage/giant2/32-39_1+24-31_1024+16-23_1536+8-15_1536 \
    --train-dataset ImageNet:split=TRAIN:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --val-dataset ImageNet:split=VAL:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --batch-size 1792

# {"('full', 10) Top 1": 75.75799822807312}
# {"('full', 10) Top 5": 89.60999846458435}
# {"('full', 20) Top 1": 75.49999952316284}
# {"('full', 20) Top 5": 91.18000268936157}
# {"('full', 100) Top 1": 73.46799969673157}
# {"('full', 100) Top 5": 92.13799834251404}
# {"('full', 200) Top 1": 72.13199734687805}
# {"('full', 200) Top 5": 91.76200032234192}

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

# ------------------------------------------------ 8 block + 4 blocks ------------------------------------------------
# ---------------- vit-giant2 (32-39_1, 28-31_512) ----------------
python dinov2/run/eval/knn.py \
    --config-file dinov2/configs/eval/vitg14_pretrain_prune.yaml \
    --pretrained-weights ./out/dinov2_multi_stage/32-39_1+28-31_512/dinov2_vitg14_pretrain_prune.pth \
    --output-dir /public/scccse/model_weight/dinov2/knn_prune/multi_stage/giant2/32-39_1+28-31_512 \
    --train-dataset ImageNet:split=TRAIN:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --val-dataset ImageNet:split=VAL:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --batch-size 1792

# {"('full', 10) Top 1": 82.79200196266174}
# {"('full', 10) Top 5": 93.50799918174744}
# {"('full', 20) Top 1": 82.69400000572205}
# {"('full', 20) Top 5": 94.69199776649475}
# {"('full', 100) Top 1": 81.27599954605103}
# {"('full', 100) Top 5": 95.75799703598022}
# {"('full', 200) Top 1": 80.3820013999939}
# {"('full', 200) Top 5": 95.75600028038025}

# ---------------- vit-giant2 (32-39_1, 28-31_1024) ----------------
python dinov2/run/eval/knn.py \
    --config-file dinov2/configs/eval/vitg14_pretrain_prune.yaml \
    --pretrained-weights ./out/dinov2_multi_stage/32-39_1+28-31_1024/dinov2_vitg14_pretrain_prune.pth \
    --output-dir /public/scccse/model_weight/dinov2/knn_prune/multi_stage/giant2/32-39_1+28-31_1024 \
    --train-dataset ImageNet:split=TRAIN:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --val-dataset ImageNet:split=VAL:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --batch-size 1792

# {"('full', 10) Top 1": 83.04399847984314}
# {"('full', 10) Top 5": 93.60799789428711}
# {"('full', 20) Top 1": 82.93200135231018}
# {"('full', 20) Top 5": 94.74800229072571}
# {"('full', 100) Top 1": 81.63800239562988}
# {"('full', 100) Top 5": 95.87799906730652}
# {"('full', 200) Top 1": 80.74399828910828}
# {"('full', 200) Top 5": 95.89800238609314}

# ---------------- vit-giant2 (32-39_1, 28-31_256) ----------------
python dinov2/run/eval/knn.py \
    --config-file dinov2/configs/eval/vitg14_pretrain_prune.yaml \
    --pretrained-weights ./out/dinov2_multi_stage/32-39_1+28-31_256/dinov2_vitg14_pretrain_prune.pth \
    --output-dir /public/scccse/model_weight/dinov2/knn_prune/multi_stage/giant2/32-39_1+28-31_256 \
    --train-dataset ImageNet:split=TRAIN:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --val-dataset ImageNet:split=VAL:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --batch-size 1792

# {"('full', 10) Top 1": 82.58799910545349}
# {"('full', 10) Top 5": 93.26199889183044}
# {"('full', 20) Top 1": 82.35999941825867}
# {"('full', 20) Top 5": 94.48599815368652}
# {"('full', 100) Top 1": 80.85200190544128}
# {"('full', 100) Top 5": 95.524001121521}
# {"('full', 200) Top 1": 80.01800179481506}
# {"('full', 200) Top 5": 95.61399817466736}

# ---------------- vit-giant2 (32-39_1, 28-31_1536) ----------------
python dinov2/run/eval/knn.py \
    --config-file dinov2/configs/eval/vitg14_pretrain_prune.yaml \
    --pretrained-weights ./out/dinov2_multi_stage/32-39_1+28-31_1536/dinov2_vitg14_pretrain_prune.pth \
    --output-dir /public/scccse/model_weight/dinov2/knn_prune/multi_stage/giant2/32-39_1+28-31_1536 \
    --train-dataset ImageNet:split=TRAIN:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --val-dataset ImageNet:split=VAL:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --batch-size 1792

# {"('full', 10) Top 1": 83.30399990081787}
# {"('full', 10) Top 5": 93.71200203895569}
# {"('full', 20) Top 1": 83.25999975204468}
# {"('full', 20) Top 5": 94.84999775886536}
# {"('full', 100) Top 1": 81.98800086975098}
# {"('full', 100) Top 5": 95.96199989318848}
# {"('full', 200) Top 1": 81.08800053596497}
# {"('full', 200) Top 5": 96.03400230407715}

# ------------------------------------------------ 8 block + 2 blocks ------------------------------------------------
# ---------------- vit-giant2 (32-39_1, 30-31_512) ----------------
python dinov2/run/eval/knn.py \
    --config-file dinov2/configs/eval/vitg14_pretrain_prune.yaml \
    --pretrained-weights ./out/dinov2_multi_stage/32-39_1+30-31_512/dinov2_vitg14_pretrain_prune.pth \
    --output-dir /public/scccse/model_weight/dinov2/knn_prune/multi_stage/giant2/32-39_1+30-31_512 \
    --train-dataset ImageNet:split=TRAIN:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --val-dataset ImageNet:split=VAL:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --batch-size 1792

# {"('full', 10) Top 1": 83.48600268363953}
# {"('full', 10) Top 5": 93.77400279045105}
# {"('full', 20) Top 1": 83.38800072669983}
# {"('full', 20) Top 5": 94.89200115203857}
# {"('full', 100) Top 1": 82.16800093650818}
# {"('full', 100) Top 5": 96.0640013217926}
# {"('full', 200) Top 1": 81.4079999923706}
# {"('full', 200) Top 5": 96.13999724388123}

# ---------------- vit-giant2 (32-39_1, 30-31_1536) ----------------
python dinov2/run/eval/knn.py \
    --config-file dinov2/configs/eval/vitg14_pretrain_prune.yaml \
    --pretrained-weights ./out/dinov2_multi_stage/32-39_1+30-31_1536/dinov2_vitg14_pretrain_prune.pth \
    --output-dir /public/scccse/model_weight/dinov2/knn_prune/multi_stage/giant2/32-39_1+30-31_1536 \
    --train-dataset ImageNet:split=TRAIN:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --val-dataset ImageNet:split=VAL:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --batch-size 1792

# {"('full', 10) Top 1": 83.65600109100342}
# {"('full', 10) Top 5": 93.83000135421753}
# {"('full', 20) Top 1": 83.58799815177917}
# {"('full', 20) Top 5": 94.96999979019165}
# {"('full', 100) Top 1": 82.4400007724762}
# {"('full', 100) Top 5": 96.14800214767456}
# {"('full', 200) Top 1": 81.61200284957886}
# {"('full', 200) Top 5": 96.22399806976318}