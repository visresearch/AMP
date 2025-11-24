import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import datetime
import time
from contextlib import nullcontext

import torch
from torch.utils.data import DataLoader, DistributedSampler, ConcatDataset
from torch.distributed import init_process_group, destroy_process_group, get_world_size, get_rank
from torch.nn.functional import mse_loss, normalize, kl_div, softmax, log_softmax
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, MixedPrecision, StateDictType, FullStateDictConfig


from functools import partial
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

import torchvision.datasets as datasets
from parse import get_args_distill
from module.anyprecision_optimizer import AnyPrecisionAdamW

from clip.eva_clip import eva_vit_model

from utils.logger import console_logger, Logger
from utils.misc import copy_files, AverageMeter, remove_key_prefix
from utils.dist import gather_features

from module.model_wrapper import ModelWrapper
from module.scheduler import CosineScheduler, apply_lr_scheduler, apply_wd_scheduler
from module.module import create_model_by_weight, create_dinov2_by_weight, build_model_dinov2
from module.transforms import get_transforms_clip, get_transforms_dinov2
from module.loss import entropy
from module.module_ckpt import set_model_ckpt_train

from clip.open_clip import create_model_and_transforms as create_open_clip_model_and_transforms
from clip.eva_clip import create_model_and_transforms as create_eva_clip_model_and_transforms

from module.meopt import MEOptimizer

def train_one_epoch(args, cfg, student, teacher, \
        train_loader, optimizer, scaler, \
            schedulers, epoch, loggers):

    student.train()
    teacher.eval()

    rank = get_rank()
    world_size = get_world_size()

    lr_scheduler, wd_scheduler = schedulers
    logger_tb, logger_console = loggers

    niter_per_epoch = len(train_loader)
    train_loader_iter = iter(train_loader)

    autocast = nullcontext \
        if cfg.optim.pure_bf16 else torch.cuda.amp.autocast

    times = AverageMeter('Time')
    losses = AverageMeter('Loss')

    niter_global = epoch * niter_per_epoch

    data, _ = next(train_loader_iter)

    data = data.bfloat16().cuda(non_blocking=True)

    end = time.time()
    end_ = end

    loss_feat, loss_cls = None, None
    loss_entropy = None

    for i in range(niter_per_epoch):
        loss = 0
        if i < niter_per_epoch - 1:
            _data, _ = next(train_loader_iter)
            _data = _data.bfloat16().cuda(non_blocking=True)
        
        if cfg.optim.use_meopt:
            optimizer.set_lr_wd_for_optimizers(
                lr_scheduler[niter_global],
                wd_scheduler[niter_global]
            )
        else:
            apply_lr_scheduler(optimizer, lr_scheduler[niter_global])
            apply_wd_scheduler(optimizer, wd_scheduler[niter_global])
        
        with torch.no_grad():
            target, target_feat = teacher(data)

        with autocast():
            pred, feat = student(data)

            if cfg.optim.loss == 'mse':
                loss_cls = mse_loss(pred, target)
                loss_feat = mse_loss(feat, target_feat)
            elif cfg.optim.loss == 'entropy':
                pred = normalize(pred, dim=-1)
                pred_all = gather_features(
                    pred, 
                    gather_with_grad=True,
                    rank=rank,
                    world_size=world_size
                )
                logit_pred = cfg.optim.temp_inv * pred @ pred_all.T
                logp_pred = log_softmax(logit_pred, dim=-1)

                target = normalize(target, dim=-1)
                target_all = gather_features(
                    target, 
                    gather_with_grad=False,
                    rank=rank,
                    world_size=world_size
                )
                logit_target = cfg.optim.temp_inv * target @ target_all.T
                logp_target = log_softmax(logit_target, dim=-1)

                loss_entropy = 0.5 * (
                    kl_div(logp_pred, logp_target, 
                        reduction="batchmean", log_target=True) +
                    kl_div(logp_target, logp_pred, 
                        reduction="batchmean", log_target=True)
                )
            else:
                raise NotImplementedError

            if loss_cls is not None:
                loss += loss_cls
            if loss_feat is not None:
                loss += loss_feat
            if loss_entropy is not None:
                loss += loss_entropy

        if cfg.optim.use_meopt:
            loss.backward()
        else:
            optimizer.zero_grad(set_to_none=True)

            if cfg.optim.pure_bf16:
                loss.backward()
                optimizer.step()
            else:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        torch.cuda.synchronize()

        if i < niter_per_epoch - 1:
            data = _data

        loss_value = loss.item()
        losses.update(loss_value)
        times.update(time.time() - end)
        end = time.time()
        
        if rank == 0 and ((i + 1) % args.print_freq == 0):

            if logger_console is not None and logger_tb is not None:

                logger_tb.add_scalar('loss_iter', loss_value, niter_global)

                lr = lr_scheduler[niter_global]
                logger_tb.add_scalar('lr_iter', lr, niter_global)

                info = \
                    f'Epoch [{(epoch + 1):04d}][{(i + 1):04d}/{niter_per_epoch:04d}] - ' \
                    + f'batch_time: {times.val:.2f}s, ' \
                    + f'rest_time: {datetime.timedelta(seconds=int(times.avg * (niter_per_epoch * cfg.optim.nepochs - niter_global)))}, ' \
                    + f'lr: {lr:.2e}, '
                
                if loss_cls is not None:
                    loss_cls_val = loss_cls.item()
                    info += f'loss_cls: {loss_cls_val:.2e}, '
                    logger_tb.add_scalar('loss_cls', loss_cls_val, niter_global)
                
                if loss_feat is not None:
                    loss_feat_val = loss_feat.item()
                    info += f'loss_feat: {loss_feat_val:.2e}, '
                    logger_tb.add_scalar('loss_feat', loss_feat_val, niter_global)
                
                if loss_entropy is not None:
                    loss_entropy_val = loss_entropy.item()
                    info += f'loss_entropy: {loss_entropy_val:.2e}, '
                    logger_tb.add_scalar('loss_entropy', loss_entropy_val, niter_global)
                
                info += f'loss: {losses.val:.2e}({losses.avg:.2e})'

                logger_console.info(info)

        niter_global += 1
    
    if logger_console is not None:
        logger_console.info(f'Training Time for 1 epoch: {datetime.timedelta(seconds=time.time() - end_)}')

    return losses.avg


def main(args, cfg):
    # --------------------- Env ---------------------
    init_process_group(backend="nccl")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)

    world_size = get_world_size()
    # --------------------- Log ---------------------
    if args.frame == 'open_clip' or args.frame == 'eva_clip':
        args.log_dir = f'./out/distill/{cfg.model.name}'
    elif args.frame == 'dinov2':
        args.log_dir = f'./out/distill/{cfg.model.arch}'
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

    # --------------------- Model ---------------------
    
    if args.frame == 'open_clip':
        teacher, _, _ = \
            create_open_clip_model_and_transforms(
            cfg.model.name, cfg.model.weight_t,
            precision='bf16' if cfg.optim.pure_bf16 else 'fp32',
            # device=torch.device('cuda', rank)
        )
        teacher = teacher.visual.cuda(rank)

        student, _, _ = create_model_by_weight(
            cfg.model.name, cfg.model.weight_s,
            precision='bf16' if cfg.optim.pure_bf16 else 'fp32',
            # device=torch.device('cuda', rank),
            only_vision=True,
            use_zero_module=cfg.model.use_zero_module,
            frame=args.frame
        )
        # student = student.visual.cuda(rank)
        student = student.cuda(rank)

        transform_train = get_transforms_clip(cfg.data.input_size, cfg.data.min_crop)
    elif args.frame == 'eva_clip':
        teacher, _, _ = \
            create_eva_clip_model_and_transforms(
            cfg.model.name, cfg.model.weight_t,
            precision='bf16' if cfg.optim.pure_bf16 else 'fp32',
            # device=torch.device('cuda', rank)
            force_custom_clip=True
        )
        teacher = teacher.visual.cuda(rank)

        student, _, _ = create_model_by_weight(
            cfg.model.name, cfg.model.weight_s,
            precision='bf16' if cfg.optim.pure_bf16 else 'fp32',
            # device=torch.device('cuda', rank)
            only_vision=True,
            use_zero_module=cfg.model.use_zero_module,
            frame=args.frame
        )
        if 'weight_resume' in cfg.model:
            msg = student.load_state_dict(
                remove_key_prefix(
                    torch.load(cfg.model.weight_resume, weights_only=True, map_location='cpu'), 
                    del_key='model.'
                ),
                strict=False
            )
            if rank == 0:
                print(msg)

        # student = student.visual.cuda(rank)
        student = student.cuda(rank)

        transform_train = get_transforms_clip(cfg.data.input_size, cfg.data.min_crop)
    elif args.frame == 'dinov2':
        teacher = build_model_dinov2(
            cfg.model, 
            pretrained=cfg.model.weight_t, 
            img_size=cfg.model.input_size
        ).bfloat16().cuda()

        student = create_dinov2_by_weight(
            cfg.model,
            pretrained=cfg.model.weight_s, 
            img_size=cfg.model.input_size
        ).bfloat16().cuda()

        transform_train = get_transforms_dinov2(
            cfg.data.input_size, 
            cfg.data.min_crop
        )
    else:
        raise NotImplementedError
    
    if cfg.optim.use_module_ckpt:
        set_model_ckpt_train(student, cfg.optim.max_gpu_memory)

    if args.frame == 'open_clip':
        state_dict_norm = teacher.ln_post.state_dict()
        proj_param = teacher.proj

        teacher.ln_post = torch.nn.Identity()
        student.ln_post = torch.nn.Identity()

        teacher.proj = student.proj = None
    elif args.frame == 'eva_clip':
        state_dict_norm = teacher.norm.state_dict()
        state_dict_head = teacher.head.state_dict()

        teacher.norm = torch.nn.Identity()
        student.norm = torch.nn.Identity()

        teacher.head = torch.nn.Identity()
        student.head = torch.nn.Identity()
    elif args.frame == 'dinov2':
        state_dict_norm = teacher.norm.state_dict()

        teacher.norm = torch.nn.Identity()
        student.norm = torch.nn.Identity()

        teacher.mask_token = None
        student.mask_token = None

    teacher = ModelWrapper(teacher)
    student = ModelWrapper(student)

    if 'distribute_strategy' in cfg.optim \
        and cfg.optim.distribute_strategy == 'fsdp':
        wrapping_policy = partial(transformer_auto_wrap_policy, 
                                    transformer_layer_cls={
                                        eva_vit_model.Block
                                    }
                                )

        precision = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
            cast_forward_inputs=True,
        )

        student = FSDP(student, 
            auto_wrap_policy=wrapping_policy, 
            mixed_precision=precision, 
            sharding_strategy=ShardingStrategy.SHARD_GRAD_OP, 
            device_id=rank, 
            limit_all_gathers=True
        )
    else:
        student = torch.nn.parallel.DistributedDataParallel(
            student, device_ids=[rank], 
            find_unused_parameters=False,
        )

    # --------------------- Dataset ---------------------
    train_dataset = datasets.ImageFolder(
        os.path.join(cfg.data.root, 'train'), transform=transform_train
    )

    if rank == 0:
        logger_console.info(f'Training dataset size: {len(train_dataset)}')

    train_sampler = DistributedSampler(train_dataset)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.data.batch_size_per_gpu,
        shuffle=(train_sampler is None),
        num_workers=cfg.data.num_workers,
        sampler=train_sampler,
        pin_memory=cfg.data.pin_memory,
        drop_last=True,
        prefetch_factor=cfg.data.prefetch_factor,
        persistent_workers=True,
    )

    # --------------------- Optimizer ---------------------
    batch_size = cfg.data.batch_size_per_gpu * world_size
    lr = cfg.optim.base_lr * batch_size / 256

    params = student.parameters()

    if cfg.optim.pure_bf16:
        if cfg.optim.use_meopt:
            optimizer = MEOptimizer(student, 
                opt_cls=AnyPrecisionAdamW,
                lr=lr, betas=(0.9, 0.95),
                momentum_dtype=torch.bfloat16,
                variance_dtype=torch.bfloat16,
                use_kahan_summation=True,
            )
        else:
            optimizer = AnyPrecisionAdamW(params, 
                lr=lr, betas=(0.9, 0.95),
                momentum_dtype=torch.bfloat16,
                variance_dtype=torch.bfloat16,
                use_kahan_summation=True,
            )
        scaler = None
    else:
        if cfg.optim.use_meopt:
            optimizer = MEOptimizer(student, 
                opt_cls=torch.optim.AdamW,
                lr=lr, betas=(0.9, 0.95)
            )
        else:
            optimizer = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95))
            scaler = torch.cuda.amp.GradScaler()

    niter_epoch = len(train_loader)

    lr_scheduler = CosineScheduler(
        base_value=lr,
        final_value=cfg.optim.min_lr,
        warmup_iters=int(cfg.optim.warmup_epochs * niter_epoch),
        total_iters=int(cfg.optim.nepochs * niter_epoch),
    )

    wd_scheduler = CosineScheduler(
        base_value=cfg.optim.weight_decay,
        final_value=cfg.optim.weight_decay_end,
        total_iters=int(cfg.optim.nepochs * niter_epoch),
    )

    if rank == 0:
        save_dir = os.path.join(logger_tb.log_dir, 'ckpt')
        os.makedirs(save_dir, exist_ok=True)
        start_time = time.time()

    epochs_start = cfg.model.epoch_resume if 'epoch_resume' in cfg.model else 0

    for epoch in range(epochs_start, cfg.optim.nepochs):

        train_loader.sampler.set_epoch(epoch)

        loss_epoch = train_one_epoch(args, cfg, student, teacher, train_loader, \
            optimizer, scaler, (lr_scheduler, wd_scheduler), epoch, \
                (logger_tb, logger_console))

        if isinstance(student, FSDP):
            with FSDP.state_dict_type(
                student, StateDictType.FULL_STATE_DICT, 
                FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            ):
                # state_dict = student.model.state_dict()
                state_dict = student.state_dict()
        else:
            if isinstance(student.module, ModelWrapper):
                state_dict = student.module.model.state_dict()
            else:
                state_dict = student.module.state_dict()

        if ((epoch + 1) % args.save_freq == 0 or epoch + 1 == cfg.optim.nepochs) and rank == 0:
            if args.frame == 'dinov2':
                fname = f'student_{cfg.model.arch}_{epoch + 1}.pth'
            else:
                fname = f'student_{cfg.model.name}_{epoch + 1}.pth'
            
            if args.frame == 'open_clip':
                state_dict['proj'] = proj_param
                state_dict['ln_post.weight'] = state_dict_norm['weight']
                if 'bias' in state_dict_norm:
                    state_dict['ln_post.bias'] = state_dict_norm['bias']
            elif args.frame == 'eva_clip':
                state_dict['head.weight'] = state_dict_head['weight']
                if 'bias' in state_dict_head:
                    state_dict['head.bias'] = state_dict_head['bias']
                state_dict['norm.weight'] = state_dict_norm['weight']
                if 'bias' in state_dict_norm:
                    state_dict['norm.bias'] = state_dict_norm['bias']
            elif args.frame == 'dinov2':
                state_dict['norm.weight'] = state_dict_norm['weight']
                if 'bias' in state_dict_norm:
                    state_dict['norm.bias'] = state_dict_norm['bias']
            else:
                raise NotImplementedError

            state_dict = remove_key_prefix(state_dict, del_key='model.')
            torch.save(state_dict, os.path.join(save_dir, fname))
        
        if rank == 0:
            if cfg.optim.use_meopt:
                lr_ = optimizer.lr
            else:
                lr_ = optimizer.param_groups[0]['lr']

            if logger_tb is not None:
                logger_tb.add_scalar('train_loss_epoch', loss_epoch, epoch+1)
                logger_tb.add_scalar('lr_epoch', lr_, epoch+1)
                logger_tb.flush()
            
            if logger_console is not None:
                logger_console.info(f"Epoch: {epoch + 1}, Lr: {lr_}, Loss: {loss_epoch}")


    if rank == 0:
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger_console.info(f'Training time: {total_time_str}')

    torch.distributed.barrier()
    destroy_process_group()

    return


if __name__ == '__main__':
    from omegaconf import OmegaConf
    args = get_args_distill()
    cfg = OmegaConf.load(args.config_file)
    main(args, cfg)

