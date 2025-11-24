#!/bin/bash  
#SBATCH -o gpu.%j.out ##作业的输出信息文件  
#SBATCH -J knn ##作业名  
#SBATCH -p A6000-ni
#SBATCH -w gpu17 
#SBATCH --nodes=1 ##申请1个节点  
#SBATCH --gres=gpu:1 ##每个作业占用的GPU数量 *



# ---------------- vit-giant2 (32-39) ----------------
python dinov2/run/eval/knn.py \
    --config-file dinov2/configs/eval/vitg14_pretrain_prune.yaml \
    --pretrained-weights ./out/dinov2_part/32-39/dinov2_vitg14_pretrain_prune.pth \
    --output-dir /public/scccse/model_weight/dinov2/knn_prune/part/giant2/32-40 \
    --train-dataset ImageNet:split=TRAIN:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --val-dataset ImageNet:split=VAL:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --batch-size 1792

#★{"('full', 10) Top 1": 84.02799963951111}
# {"('full', 10) Top 5": 93.51400136947632}
# {"('full', 20) Top 1": 83.9460015296936}
# {"('full', 20) Top 5": 94.56999897956848}
# {"('full', 100) Top 1": 82.95400142669678}
# {"('full', 100) Top 5": 95.80199718475342}
# {"('full', 200) Top 1": 82.3199987411499}
# {"('full', 200) Top 5": 96.00600004196167}

# ---------------- vit-giant2 (24-39) ----------------
python dinov2/run/eval/knn.py \
    --config-file dinov2/configs/eval/vitg14_pretrain_prune.yaml \
    --pretrained-weights ./out/dinov2_part/24-39/dinov2_vitg14_pretrain_prune.pth \
    --output-dir /public/scccse/model_weight/dinov2/knn_prune/part/giant2/24-39 \
    --train-dataset ImageNet:split=TRAIN:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --val-dataset ImageNet:split=VAL:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --batch-size 1792

#★{"('full', 10) Top 1": 83.2539975643158}
# {"('full', 10) Top 5": 93.21200251579285}
# {"('full', 20) Top 1": 83.13599824905396}
# {"('full', 20) Top 5": 94.29600238800049}
# {"('full', 100) Top 1": 82.01000094413757}
# {"('full', 100) Top 5": 95.54200172424316}
# {"('full', 200) Top 1": 81.35200142860413}
# {"('full', 200) Top 5": 95.76600193977356}


# ---------------- vit-giant2 (32-39) (pruned_hidden_size: 1024) ----------------
python dinov2/run/eval/knn.py \
    --config-file dinov2/configs/eval/vitg14_pretrain_prune.yaml \
    --pretrained-weights ./out/dinov2_part/32-39_1024/dinov2_vitg14_pretrain_prune.pth \
    --output-dir /public/scccse/model_weight/dinov2/knn_prune/part/giant2/32-39_1024 \
    --train-dataset ImageNet:split=TRAIN:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --val-dataset ImageNet:split=VAL:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --batch-size 1792

#★{"('full', 10) Top 1": 84.20199751853943}
# {"('full', 10) Top 5": 93.76599788665771}
# {"('full', 20) Top 1": 84.14000272750854}
# {"('full', 20) Top 5": 94.8199987411499}
# {"('full', 100) Top 1": 83.21200013160706}
# {"('full', 100) Top 5": 95.95999717712402}
# {"('full', 200) Top 1": 82.52400159835815}
# {"('full', 200) Top 5": 96.22200131416321}

# ---------------- vit-giant2 (32-39) (pruned_hidden_size: 1280) ----------------
python dinov2/run/eval/knn.py \
    --config-file dinov2/configs/eval/vitg14_pretrain_prune.yaml \
    --pretrained-weights ./out/dinov2_part/32-39_1280/dinov2_vitg14_pretrain_prune.pth \
    --output-dir /public/scccse/model_weight/dinov2/knn_prune/part/giant2/32-39_1280 \
    --train-dataset ImageNet:split=TRAIN:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --val-dataset ImageNet:split=VAL:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --batch-size 1792

# {"('full', 10) Top 1": 84.10400152206421}
# {"('full', 10) Top 5": 93.62800121307373}
# {"('full', 20) Top 1": 84.06599760055542}
# {"('full', 20) Top 5": 94.68799829483032}
# {"('full', 100) Top 1": 83.14599990844727}
# {"('full', 100) Top 5": 95.91799974441528}
# {"('full', 200) Top 1": 82.48800039291382}
# {"('full', 200) Top 5": 96.15200161933899}

# ---------------- vit-giant2 (32-39) (pruned_hidden_size: 768) ----------------
python dinov2/run/eval/knn.py \
    --config-file dinov2/configs/eval/vitg14_pretrain_prune.yaml \
    --pretrained-weights ./out/dinov2_part/32-39_768/dinov2_vitg14_pretrain_prune.pth \
    --output-dir /public/scccse/model_weight/dinov2/knn_prune/part/giant2/32-39_768 \
    --train-dataset ImageNet:split=TRAIN:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --val-dataset ImageNet:split=VAL:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --batch-size 1792

#★{"('full', 10) Top 1": 84.27000045776367}
# {"('full', 10) Top 5": 93.93600225448608}
# {"('full', 20) Top 1": 84.197998046875}
# {"('full', 20) Top 5": 94.96999979019165}
# {"('full', 100) Top 1": 83.2319974899292}
# {"('full', 100) Top 5": 96.14800214767456}
# {"('full', 200) Top 1": 82.56800174713135}
# {"('full', 200) Top 5": 96.32800221443176}

# ---------------- vit-giant2 (32-39) (pruned_hidden_size: 512) ----------------
python dinov2/run/eval/knn.py \
    --config-file dinov2/configs/eval/vitg14_pretrain_prune.yaml \
    --pretrained-weights ./out/dinov2_part/32-39_512/dinov2_vitg14_pretrain_prune.pth \
    --output-dir /public/scccse/model_weight/dinov2/knn_prune/part/giant2/32-39_512 \
    --train-dataset ImageNet:split=TRAIN:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --val-dataset ImageNet:split=VAL:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --batch-size 1792

#★{"('full', 10) Top 1": 84.22600030899048}
# {"('full', 10) Top 5": 93.99799704551697}
# {"('full', 20) Top 1": 84.22399759292603}
# {"('full', 20) Top 5": 95.0380027294159}
# {"('full', 100) Top 1": 83.27800035476685}
# {"('full', 100) Top 5": 96.21000289916992}
# {"('full', 200) Top 1": 82.57399797439575}
# {"('full', 200) Top 5": 96.41199707984924}

# ---------------- vit-giant2 (32-39) (pruned_hidden_size: 384) ----------------
python dinov2/run/eval/knn.py \
    --config-file dinov2/configs/eval/vitg14_pretrain_prune.yaml \
    --pretrained-weights ./out/dinov2_part/32-39_384/dinov2_vitg14_pretrain_prune.pth \
    --output-dir /public/scccse/model_weight/dinov2/knn_prune/part/giant2/32-39_384 \
    --train-dataset ImageNet:split=TRAIN:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --val-dataset ImageNet:split=VAL:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --batch-size 1792

# {"('full', 10) Top 1": 84.13000106811523}
# {"('full', 10) Top 5": 93.98000240325928}
# {"('full', 20) Top 1": 84.1759979724884}
# {"('full', 20) Top 5": 95.05800008773804}
# {"('full', 100) Top 1": 83.21599960327148}
# {"('full', 100) Top 5": 96.25599980354309}
# {"('full', 200) Top 1": 82.53600001335144}
# {"('full', 200) Top 5": 96.44399881362915}

# ---------------- vit-giant2 (32-39) (pruned_hidden_size: 192) ----------------
python dinov2/run/eval/knn.py \
    --config-file dinov2/configs/eval/vitg14_pretrain_prune.yaml \
    --pretrained-weights ./out/dinov2_part/32-39_192/dinov2_vitg14_pretrain_prune.pth \
    --output-dir /public/scccse/model_weight/dinov2/knn_prune/part/giant2/32-39_192 \
    --train-dataset ImageNet:split=TRAIN:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --val-dataset ImageNet:split=VAL:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --batch-size 1792

# {"('full', 10) Top 1": 83.9900016784668}
# {"('full', 10) Top 5": 93.99999976158142}
# {"('full', 20) Top 1": 84.06400084495544}
# {"('full', 20) Top 5": 95.08200287818909}
# {"('full', 100) Top 1": 83.11399817466736}
# {"('full', 100) Top 5": 96.25399708747864}
# {"('full', 200) Top 1": 82.41999745368958}
# {"('full', 200) Top 5": 96.47600054740906}

# ---------------- vit-giant2 (32-39) (pruned_hidden_size: 128) ----------------
python dinov2/run/eval/knn.py \
    --config-file dinov2/configs/eval/vitg14_pretrain_prune.yaml \
    --pretrained-weights ./out/dinov2_part/32-39_128/dinov2_vitg14_pretrain_prune.pth \
    --output-dir /public/scccse/model_weight/dinov2/knn_prune/part/giant2/32-39_128 \
    --train-dataset ImageNet:split=TRAIN:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --val-dataset ImageNet:split=VAL:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --batch-size 1792

# {"('full', 10) Top 1": 83.99400115013123}
# {"('full', 10) Top 5": 94.00200247764587}
# {"('full', 20) Top 1": 84.01600122451782}
# {"('full', 20) Top 5": 95.10800242424011}
# {"('full', 100) Top 1": 83.0839991569519}
# {"('full', 100) Top 5": 96.23000025749207}
# {"('full', 200) Top 1": 82.38599896430969}
# {"('full', 200) Top 5": 96.45799994468689}

# ---------------- vit-giant2 (32-39) (pruned_hidden_size: 64) ----------------
python dinov2/run/eval/knn.py \
    --config-file dinov2/configs/eval/vitg14_pretrain_prune.yaml \
    --pretrained-weights ./out/dinov2_part/32-39_64/dinov2_vitg14_pretrain_prune.pth \
    --output-dir /public/scccse/model_weight/dinov2/knn_prune/part/giant2/32-39_64 \
    --train-dataset ImageNet:split=TRAIN:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --val-dataset ImageNet:split=VAL:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --batch-size 1792

# {"('full', 10) Top 1": 83.98399949073792}
# {"('full', 10) Top 5": 94.00799870491028}
# {"('full', 20) Top 1": 83.97200107574463}
# {"('full', 20) Top 5": 95.11399865150452}
# {"('full', 100) Top 1": 83.04399847984314}
# {"('full', 100) Top 5": 96.27400040626526}
# {"('full', 200) Top 1": 82.32399821281433}
# {"('full', 200) Top 5": 96.44399881362915}

# ---------------- vit-giant2 (32-39) (pruned_hidden_size: 32) ----------------
python dinov2/run/eval/knn.py \
    --config-file dinov2/configs/eval/vitg14_pretrain_prune.yaml \
    --pretrained-weights ./out/dinov2_part/32-39_32/dinov2_vitg14_pretrain_prune.pth \
    --output-dir /public/scccse/model_weight/dinov2/knn_prune/part/giant2/32-39_32 \
    --train-dataset ImageNet:split=TRAIN:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --val-dataset ImageNet:split=VAL:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --batch-size 1792

# {"('full', 10) Top 1": 83.9460015296936}
# {"('full', 10) Top 5": 94.00799870491028}
# {"('full', 20) Top 1": 83.94799828529358}
# {"('full', 20) Top 5": 95.13599872589111}
# {"('full', 100) Top 1": 83.06599855422974}
# {"('full', 100) Top 5": 96.28599882125854}
# {"('full', 200) Top 1": 82.29600191116333}
# {"('full', 200) Top 5": 96.4460015296936}

# ---------------- vit-giant2 (32-39) (pruned_hidden_size: 16) ----------------
python dinov2/run/eval/knn.py \
    --config-file dinov2/configs/eval/vitg14_pretrain_prune.yaml \
    --pretrained-weights ./out/dinov2_part/32-39_16/dinov2_vitg14_pretrain_prune.pth \
    --output-dir /public/scccse/model_weight/dinov2/knn_prune/part/giant2/32-39_16 \
    --train-dataset ImageNet:split=TRAIN:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --val-dataset ImageNet:split=VAL:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --batch-size 1792

# {"('full', 10) Top 1": 83.94200205802917}
# {"('full', 10) Top 5": 94.00399923324585}
# {"('full', 20) Top 1": 83.92000198364258}
# {"('full', 20) Top 5": 95.1259970664978}
# {"('full', 100) Top 1": 82.99400210380554}
# {"('full', 100) Top 5": 96.27400040626526}
# {"('full', 200) Top 1": 82.22399950027466}
# {"('full', 200) Top 5": 96.42000198364258}

# ---------------- vit-giant2 (32-39) (pruned_hidden_size: 8) ----------------
python dinov2/run/eval/knn.py \
    --config-file dinov2/configs/eval/vitg14_pretrain_prune.yaml \
    --pretrained-weights ./out/dinov2_part/32-39_8/dinov2_vitg14_pretrain_prune.pth \
    --output-dir /public/scccse/model_weight/dinov2/knn_prune/part/giant2/32-39_8 \
    --train-dataset ImageNet:split=TRAIN:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --val-dataset ImageNet:split=VAL:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --batch-size 1792

# {"('full', 10) Top 1": 83.92999768257141}
# {"('full', 10) Top 5": 93.98599863052368}
# {"('full', 20) Top 1": 83.89599919319153}
# {"('full', 20) Top 5": 95.11200189590454}
# {"('full', 100) Top 1": 82.95599818229675}
# {"('full', 100) Top 5": 96.25800251960754}
# {"('full', 200) Top 1": 82.18799829483032}
# {"('full', 200) Top 5": 96.42599821090698}

# ---------------- vit-giant2 (32-39) (pruned_hidden_size: 4) ----------------
python dinov2/run/eval/knn.py \
    --config-file dinov2/configs/eval/vitg14_pretrain_prune.yaml \
    --pretrained-weights ./out/dinov2_part/32-39_4/dinov2_vitg14_pretrain_prune.pth \
    --output-dir /public/scccse/model_weight/dinov2/knn_prune/part/giant2/32-39_4 \
    --train-dataset ImageNet:split=TRAIN:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --val-dataset ImageNet:split=VAL:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --batch-size 1792

# {"('full', 10) Top 1": 83.93800258636475}
# {"('full', 10) Top 5": 93.98599863052368}
# {"('full', 20) Top 1": 83.8699996471405}
# {"('full', 20) Top 5": 95.08000016212463}
# {"('full', 100) Top 1": 82.94600248336792}
# {"('full', 100) Top 5": 96.26799821853638}
# {"('full', 200) Top 1": 82.14200139045715}
# {"('full', 200) Top 5": 96.41799926757812}

# ---------------- vit-giant2 (32-39) (pruned_hidden_size: 2) ----------------
python dinov2/run/eval/knn.py \
    --config-file dinov2/configs/eval/vitg14_pretrain_prune.yaml \
    --pretrained-weights ./out/dinov2_part/32-39_2/dinov2_vitg14_pretrain_prune.pth \
    --output-dir /public/scccse/model_weight/dinov2/knn_prune/part/giant2/32-39_2 \
    --train-dataset ImageNet:split=TRAIN:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --val-dataset ImageNet:split=VAL:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --batch-size 1792

# {"('full', 10) Top 1": 83.95400047302246}
# {"('full', 10) Top 5": 93.94599795341492}
# {"('full', 20) Top 1": 83.85199904441833}
# {"('full', 20) Top 5": 95.09000182151794}
# {"('full', 100) Top 1": 82.92800188064575}
# {"('full', 100) Top 5": 96.26200199127197}
# {"('full', 200) Top 1": 82.15000033378601}
# {"('full', 200) Top 5": 96.43200039863586}

# ---------------- vit-giant2 (32-39) (pruned_hidden_size: 1) ----------------
python dinov2/run/eval/knn.py \
    --config-file dinov2/configs/eval/vitg14_pretrain_prune.yaml \
    --pretrained-weights ./out/dinov2_part/32-39_1/dinov2_vitg14_pretrain_prune.pth \
    --output-dir /public/scccse/model_weight/dinov2/knn_prune/part/giant2/32-39_1 \
    --train-dataset ImageNet:split=TRAIN:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --val-dataset ImageNet:split=VAL:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra \
    --batch-size 1792

#★{"('full', 10) Top 1": 83.94200205802917}
# {"('full', 10) Top 5": 93.94800066947937}
# {"('full', 20) Top 1": 83.8479995727539}
# {"('full', 20) Top 5": 95.08600234985352}
# {"('full', 100) Top 1": 82.91599750518799}
# {"('full', 100) Top 5": 96.24599814414978}
# {"('full', 200) Top 1": 82.14200139045715}
# {"('full', 200) Top 5": 96.41799926757812}

# ---------------- vit-giant2 (prune 32-39 blocks) (pruned_hidden_size: 0) ----------------
# {"('full', 10) Top 1": 79.61000204086304}
# {"('full', 10) Top 5": 91.94599986076355}
# {"('full', 20) Top 1": 79.50599789619446}
# {"('full', 20) Top 5": 93.1339979171753}
# {"('full', 100) Top 1": 77.674001455307}
# {"('full', 100) Top 5": 94.04000043869019}
# {"('full', 200) Top 1": 76.50399804115295}
# {"('full', 200) Top 5": 93.87999773025513}