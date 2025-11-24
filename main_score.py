import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"

import torch
from torch.distributed import destroy_process_group, get_world_size, init_process_group
from torch.utils.data import DataLoader, DistributedSampler

import torchvision.datasets as datasets

from clip.open_clip import get_tokenizer as openclip_tokenizer, create_model_and_transforms as create_openclip_model_and_transforms
from clip.eva_clip import get_tokenizer as evaclip_tokenizer, create_model_and_transforms as create_evaclip_model_and_transforms

from module.module import build_model_dinov2
from module.transforms import get_transforms_dinov2
from module.module_ckpt import set_model_ckpt

from utils.logger import console_logger, Logger
from utils.misc import copy_files

from parse import get_args_score
from importance import ImportanceEstimatorCrossDataset
from layer_wise.layer_score import LayerScoreEstimator
from score import ScoreEstimator
from data.coco_caption import COCOCaptionMerge


def main(args, cfg):
    # -------------- Distribute --------------
    init_process_group(backend="nccl")
    rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(rank)
    world_size = get_world_size()
    # --------------------- Log ---------------------
    if args.frame == 'open_clip' or args.frame == 'eva_clip':
        args.log_dir = f'./out/{cfg.model.name}_prune'
    elif args.frame == 'dinov2':
        args.log_dir = f'./out/{cfg.model.arch}_prune'
    else:
        raise NotImplementedError
    
    if rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)

        logger_tb = Logger(args.log_dir, name=args.frame)
        logger_console = console_logger(logger_tb.log_dir, 'console')
        dst_dir = os.path.join(logger_tb.log_dir, 'code/')
        copy_files('./', dst_dir, args.exclude_file_list)

        logger_console.info(
            "{}".format(args).replace(', ', ',\n')
        )
    else:
        logger_tb, logger_console = None, None

    # -------------- Model --------------

    dtype = torch.bfloat16
    device = torch.device(f'cuda:{rank}')

    if 'use_ckpt' in cfg.model and cfg.model.use_ckpt:
        set_model_ckpt(args.frame, cfg.model.nblock_interval)

    if args.frame == 'open_clip':
        model, transform_train, transform_val = \
            create_openclip_model_and_transforms(cfg.model.name, pretrained=cfg.model.path, precision='bf16', device=device)
        
        tokenizer = openclip_tokenizer(cfg.model.name)
    elif args.frame == 'eva_clip':
        model, transform_train, transform_val = \
            create_evaclip_model_and_transforms(
                cfg.model.name, pretrained=cfg.model.path, 
                precision='bf16', device=device, 
                force_custom_clip=True
            )
        model.text = None
    
        tokenizer = evaclip_tokenizer(cfg.model.name)
    elif args.frame == 'dinov2':
        model = build_model_dinov2(
            cfg.model, 
            pretrained=cfg.model.path, 
            img_size=cfg.model.input_size
        ).bfloat16().cuda()

        transform_val = get_transforms_dinov2(
            cfg.data.input_size, 
            cfg.data.min_crop
        )
        
        tokenizer = None
    else:
        raise NotImplementedError

    # -------------- Dataset --------------
    if cfg.data.name == 'coco':
        dataset = COCOCaptionMerge(
            cfg.data.path, transform_val, tokenizer, 
            split='train', one_caption_per_image=False, 
            # only_image=(args.frame == 'dinov2')
            only_image=True
        )

        dataset_val = COCOCaptionMerge(
            cfg.data.path, transform_val, tokenizer, 
            split='val', one_caption_per_image=False,
            # only_image=(args.frame == 'dinov2')
            only_image=True
        )
    elif cfg.data.name == 'imagenet1k':
        dataset = datasets.ImageFolder(
            os.path.join(cfg.data.path, 'train'), transform=transform_val
        )

        dataset_val = datasets.ImageFolder(
            os.path.join(cfg.data.path, 'val'), transform=transform_val
        )
    else:
        raise NotImplementedError

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        # sampler=DistributedSampler(dataset, shuffle=False),
        sampler=DistributedSampler(dataset, shuffle=True),
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory, 
        prefetch_factor=cfg.data.prefetch_factor,
        drop_last=True,
        persistent_workers=True,
    )

    dataloader_val = DataLoader(
        dataset_val,
        # batch_size=cfg.data.batch_size_val,
        batch_size=cfg.data.batch_size,
        # sampler=DistributedSampler(dataset_val, shuffle=False),
        sampler=DistributedSampler(dataset_val, shuffle=True),
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory, 
        prefetch_factor=cfg.data.prefetch_factor,
        drop_last=True,
        persistent_workers=True,
    )

    if rank == 0:
        logger_console.info(f'length of dataloader: {len(dataloader)}')
        logger_console.info(f'length of dataloader_val: {len(dataloader_val)}')
    
    # -------------- Importance --------------
    score_path, model_path = None, None
    if args.layer_wise_prune:
        if rank == 0:
            score_path = os.path.join(logger_tb.log_dir, f'ckpt/score_layer_wise_thresh_{cfg.prune.loss_inc_thresh}_t{cfg.prune.temp_inv}')

            model_path = os.path.join(logger_tb.log_dir, f'ckpt/model_layer_wise_thresh_{cfg.prune.loss_inc_thresh}_t{cfg.prune.temp_inv}')

        estimator = LayerScoreEstimator(
            cfg, 
            model, 
            dataloader, dataloader_val,
            score_path, model_path,
            loss_inc_thresh=cfg.prune.loss_inc_thresh,
            use_abs=cfg.prune.use_abs,
            loggers=(logger_tb, logger_console)
        )
        
        if cfg.resume.use_resume:
            estimator.resume(
                fname_score=cfg.resume.fname_score,
                fname_model=cfg.resume.fname_model
            )

        estimator.run()
    else:
        if rank == 0:
            score_path = os.path.join(logger_tb.log_dir, f'ckpt/score_thresh_{cfg.prune.loss_inc_thresh}_t{cfg.prune.temp_inv}')

            model_path = os.path.join(logger_tb.log_dir, f'ckpt/model_thresh_{cfg.prune.loss_inc_thresh}_t{cfg.prune.temp_inv}')

        estimator = ScoreEstimator(
            cfg, 
            model, 
            dataloader, dataloader_val,
            score_path, model_path,
            loss_inc_thresh=cfg.prune.loss_inc_thresh,
            use_abs=cfg.prune.use_abs,
            loggers=(logger_tb, logger_console)
        )
        
        if cfg.resume.use_resume:
            estimator.resume(
                fname_score=cfg.resume.fname_score,
                fname_model=cfg.resume.fname_model
            )

        estimator.run()

        # score_path = f'./out/{cfg.model.name}_{cfg.data.name}'
        # estimator = ImportanceEstimatorCrossDataset(model, \
        #     dtype=dtype, dataloader=dataloader)

        # estimator.run()
        # estimator.save_scores(score_path)

    destroy_process_group()

    return


if __name__ == '__main__':
    from omegaconf import OmegaConf

    args = get_args_score()
    cfg = OmegaConf.load(args.config_file)

    main(args, cfg)