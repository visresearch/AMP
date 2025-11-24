


python dinov2/run/train/distill.py \
    --config-file dinov2/configs/train/vitg14_distill.yaml \
    --output-dir /public/scccse/model_weight/dinov2/distill2 \
    --ngpus 8 \
    train.dataset_path=ImageNet:split=TRAIN:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra

python dinov2/run/train/distill.py \
    --config-file dinov2/configs/train/vitg14_distill.yaml \
    --output-dir /public/scccse/model_weight/dinov2/distill2 \
    --ngpus 4 \
    --eval-only \
    train.dataset_path=ImageNet:split=TRAIN:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra

python dinov2/run/train/distill.py \
    --config-file dinov2/configs/train/vitg14_distill.yaml \
    --output-dir /public/scccse/model_weight/dinov2/distill3 \
    --nodelist gpu17 \
    train.dataset_path=ImageNet:split=TRAIN:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra

python dinov2/run/train/distill.py \
    --config-file dinov2/configs/train/vitg14_distill.yaml \
    --output-dir /public/scccse/model_weight/dinov2/distill4 \
    --nodelist gpu17 \
    train.dataset_path=ImageNet:split=TRAIN:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra
