import argparse

import sys
from pathlib import Path
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import torch
from torch.distributed import init_process_group, destroy_process_group

import os
from dinov2.eval.knn import eval_knn_with_model
from dinov2.eval.metrics import AccuracyAveraging

from module.module import create_dinov2_by_weight, create_model_by_weight


def get_args():
    parser = argparse.ArgumentParser("Zero-Shot Classifition")
    parser.add_argument("--frame", type=str)
    parser.add_argument("--config_file", type=str)
    parser.add_argument("--train_dataset_str", type=str)
    parser.add_argument("--val_dataset_str", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--pretrained", type=str)

    parser.set_defaults(
        frame = 'eva_clip',
        config_file = './configs/eva/eval.yaml',
        train_dataset_str = 'ImageNet:split=TRAIN:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra',
        val_dataset_str = 'ImageNet:split=VAL:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra',
        output_dir = './out/knn',
        # batch_size = 256,
        batch_size = 512,
    )

    args = parser.parse_args()

    return args


def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def knn_eval(args):
    from omegaconf import OmegaConf

    ddp_setup()

    cfg = OmegaConf.load(args.config_file)

    if args.frame == 'eva_clip':
        model, _, transform = create_model_by_weight(cfg.model.name, args.pretrained, non_visual_pretrained=None, only_vision=True, frame=args.frame)
        model = model.half()
        
        model.norm = torch.nn.Identity()
        model.head = torch.nn.Identity()
    elif args.frame == 'open_clip':
        model, _, transform = create_model_by_weight(cfg.model.name, args.pretrained, non_visual_pretrained=None, only_vision=True, frame=args.frame)
        model = model.half()
        
        model.ln_post = torch.nn.Identity()
        model.proj = None
    elif args.frame == 'dinov2':
        model = create_dinov2_by_weight(cfg.model, args.pretrained, cfg.model.input_size)
    else:
        NotImplementedError(f'Not recognize {args.frame}')

    model.half().cuda()
    
    split = args.pretrained.split('/')
    out_dir = '/'.join(split[:-1])
    os.makedirs(out_dir, exist_ok=True)
    # os.makedirs(args.output_dir, exist_ok=True)

    eval_knn_with_model(
        model=model,
        # output_dir=args.output_dir,
        output_dir=out_dir,
        train_dataset_str=args.train_dataset_str,
        val_dataset_str=args.val_dataset_str,
        nb_knn=[10, 20, 100, 200],
        temperature=0.07,
        autocast_dtype=torch.half,
        accuracy_averaging=AccuracyAveraging.MEAN_ACCURACY,
        transform=None,
        gather_on_cpu=False,
        batch_size=args.batch_size,
        num_workers=8,
        n_per_class_list=[-1],
        n_tries=1,
    )

    destroy_process_group()

    return



if __name__ == '__main__':
    args = get_args()
    knn_eval(args)
