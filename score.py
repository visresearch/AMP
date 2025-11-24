import os
import time
import datetime
import math
import copy

from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributed import get_rank, get_world_size, all_reduce, ReduceOp

from clip.open_clip.model import CLIP as OpenCLIP
from clip.eva_clip.model import CustomCLIP as EVACLIP
from clip.open_clip.transformer import VisionTransformer
from clip.eva_clip.eva_vit_model import EVAVisionTransformer

from dinov2.models.vision_transformer import DinoVisionTransformer

from module.loss import entropy
from module.module_ckpt import set_model_ckpt

from utils.misc import AverageMeter, binary_search
from utils.dist import gather_features
from prune import prune_mlp_by_idxs, prune_mlp_by_weight, prune_dinov2_by_weight, prune_eva_clip_by_weight


class ScoreEstimator(object):
    def __init__(self, cfg, model:nn.Module, 
            dataloader, dataloader_val, 
            score_path, model_path, 
            use_adaptive_prune=True, 
            loss_inc_thresh=math.log(1.02),
            num_selected_neurons=None, 
            use_abs=True, 
            loggers=None,
        ):

        self.cfg = cfg

        self.logger_tb, self.logger_console = loggers

        self.add_hook_on_input_neuron = True
        self.activations = []
        self.is_ckpt_forward = False

        self.batch_size = dataloader.batch_size 
        self.nbatch_per_epoch = len(dataloader)
        self.dataloader = dataloader

        self.dataloader_val = dataloader_val
        self.batch_size_val = dataloader_val.batch_size 
        
        self.use_abs = use_abs

        self.score_path = score_path
        self.model_path = model_path

        self.num_selected_neurons = num_selected_neurons
        self.use_adaptive_prune = use_adaptive_prune
        self.loss_inc_thresh = loss_inc_thresh
        self.max_num_prune = 6

        self.max_gpu_memory = self.cfg.model.max_gpu_memory

        self.loss_prev_block = None

        self.score = self.score_abs = 0.

        self.rank = get_rank()
        self.world_size = get_world_size()

        self.device = torch.device('cuda', self.rank)

        self.nbatch_accum = torch.zeros([], dtype=torch.float, device=self.device)

        self.losses_val = []

        self.proj = None

        self.frame = ""

        if isinstance(model, OpenCLIP):
            self.frame = "open_clip"
            self.blocks = model.visual.transformer.resblocks

            self.mlp_hidden_size = self.blocks[0].mlp[0].out_features

            self.nlayers = len(self.blocks)
            self.logit_scale = model.logit_scale.exp().item()

            self.scores = torch.zeros((self.nlayers, self.mlp_hidden_size), \
                    dtype=torch.float, device=self.device)
            self.scores_abs = torch.zeros((self.nlayers, self.mlp_hidden_size), \
                dtype=torch.float, device=self.device)
            self.labels = self.labels_val = None

            for p in model.parameters():
                p.requires_grad = False
            
            if self.cfg.prune.only_image:
                self.proj = model.visual.proj
                model.visual.proj = None
                model = model.visual
        elif isinstance(model, EVACLIP):
            self.frame = "eva_clip"
            self.blocks = model.visual.blocks

            self.mlp_hidden_size = self.blocks[0].mlp.fc1.out_features

            self.nlayers = len(self.blocks)
            self.logit_scale = model.logit_scale.exp().item()

            self.scores = torch.zeros((self.nlayers, self.mlp_hidden_size), \
                    dtype=torch.float, device=self.device)
            self.scores_abs = torch.zeros((self.nlayers, self.mlp_hidden_size), \
                dtype=torch.float, device=self.device)

            self.labels = self.labels_val = None
            
            for p in model.parameters():
                p.requires_grad = False

            if self.cfg.prune.only_image:
                self.proj = model.visual.head
                model.visual.head = nn.Identity()
                model = model.visual
        elif isinstance(model, DinoVisionTransformer):
            self.frame = "dinov2"
            self.blocks = model.blocks
            self.mlp_hidden_size = self.blocks[0].mlp.w3.in_features

            self.nlayers = len(self.blocks)

            self.scores = torch.zeros((self.nlayers, self.mlp_hidden_size), \
                    dtype=torch.float, device=self.device)
            self.scores_abs = torch.zeros((self.nlayers, self.mlp_hidden_size), \
                dtype=torch.float, device=self.device)

            for p in model.parameters():
                p.requires_grad = False
        else:
            raise NotImplementedError
    
        self.resume_score = False

        self.model = model

        self.model.eval()

        self.start_layer_id = self.nlayers - 1

        self.use_ckpt_during_prune = self.cfg.model.use_ckpt_during_prune

        if self.use_ckpt_during_prune:
            # gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024 / 1024 # GB

            if isinstance(self.model, OpenCLIP) or isinstance(self.model, EVACLIP):
                set_model_ckpt(
                    self.frame, 
                    block_id_no_ckpt=None,
                    model=self.model.visual,
                    set_x_grad=True,
                    max_gpu_memory=self.max_gpu_memory
                )
            else:
                set_model_ckpt(
                    self.frame, 
                    block_id_no_ckpt=None, 
                    model=self.model,
                    set_x_grad=True,
                    max_gpu_memory=self.max_gpu_memory
                )

    def run(self):
        # nblock_interval = 0
        if not self.resume_score:
            handles_fwd, handles_bwd = [], []
            for layer_id in range(self.start_layer_id, -1, -1):
                if isinstance(self.model, OpenCLIP) or isinstance(self.model, VisionTransformer):
                    handles_fwd.append(self.blocks[layer_id].mlp[-1]\
                        .register_forward_pre_hook(
                        self.record_input_activations
                    ))

                    handles_bwd.append(self.blocks[layer_id].mlp[-1]\
                        .register_full_backward_hook(
                        self.taylor_neuron_importance_1st_order
                    ))
                elif isinstance(self.model, EVACLIP) or isinstance(self.model, EVAVisionTransformer):
                    handles_fwd.append(self.blocks[layer_id].mlp.fc2\
                        .register_forward_pre_hook(
                        self.record_input_activations
                    ))

                    handles_bwd.append(self.blocks[layer_id].mlp.fc2\
                        .register_full_backward_hook(
                        self.taylor_neuron_importance_1st_order
                    ))
                elif isinstance(self.model, DinoVisionTransformer):
                    handles_fwd.append(self.blocks[layer_id].mlp.w3\
                        .register_forward_pre_hook(
                        self.record_input_activations
                    ))

                    handles_bwd.append(self.blocks[layer_id].mlp.w3\
                        .register_full_backward_hook(
                        self.taylor_neuron_importance_1st_order
                    ))
                else:
                    raise NotImplementedError
                
            # --------- Fwd & Bwd ---------
            loss = self.dataset_loop()

            for fwd, bwd in zip(handles_fwd, handles_bwd):
                fwd.remove()
                bwd.remove()
            
            all_reduce(self.scores, op=ReduceOp.SUM)
            all_reduce(self.scores_abs, op=ReduceOp.SUM)
            all_reduce(self.nbatch_accum, op=ReduceOp.SUM)

            self.scores /= self.nbatch_accum
            self.scores_abs /= self.nbatch_accum

        # --------- Pruning ---------
        for layer_id in range(self.start_layer_id, -1, -1):
            if self.use_adaptive_prune:
                self.adaptive_prune(layer_id)
            else:
                self.prune(layer_id)

            if isinstance(self.model, OpenCLIP) or isinstance(self.model, VisionTransformer):
                prune_size = self.blocks[layer_id].mlp[0].out_features
            elif isinstance(self.model, EVACLIP) or isinstance(self.model, EVAVisionTransformer):
                prune_size = self.blocks[layer_id].mlp.fc1.out_features
            elif isinstance(self.model, DinoVisionTransformer):
                prune_size = self.blocks[layer_id].mlp.w3.in_features
            else:
                raise NotImplementedError

            if self.rank == 0:
                self.logger_console.info(f'the size of pruned block {layer_id}: {prune_size}')
        
        self.save()

    def adaptive_prune(self, layer_id):
        if self.loss_prev_block is None:
            self.loss_prev_block = self.dataset_loop_val()

        scores_sorted, neuron_idxs = torch.sort(
            self.scores_abs[layer_id] if self.use_abs else self.scores[layer_id],
            dim=-1, 
            descending=True
        )

        if isinstance(self.model, OpenCLIP) or isinstance(self.model, VisionTransformer):
            fc_in = copy.deepcopy(self.blocks[layer_id].mlp[0])
            fc_out = copy.deepcopy(self.blocks[layer_id].mlp[-1])
        elif isinstance(self.model, EVACLIP) or isinstance(self.model, EVAVisionTransformer):
            fc_in = copy.deepcopy(self.blocks[layer_id].mlp.fc1)
            fc_out = copy.deepcopy(self.blocks[layer_id].mlp.fc2)
        elif isinstance(self.model, DinoVisionTransformer):
            fc_in = copy.deepcopy(self.blocks[layer_id].mlp.w12)
            fc_out = copy.deepcopy(self.blocks[layer_id].mlp.w3)
        else:
            raise NotImplementedError

        loss = None

        losses = [(layer_id, self.mlp_hidden_size, self.loss_prev_block)]

        if self.rank == 0:
            self.logger_console.info(f'the loss of layer {layer_id} with mlp size {self.mlp_hidden_size}: {self.loss_prev_block}')

        mlp_hidden_size_result = self.mlp_hidden_size

        loss_result = None

        left, right = 0, self.mlp_hidden_size

        for i in range(self.max_num_prune):
            mid = (left + right) // 2

            if isinstance(self.model, OpenCLIP) or isinstance(self.model, VisionTransformer):
                self.blocks[layer_id].mlp[0], \
                    self.blocks[layer_id].mlp[-1] = \
                        prune_mlp_by_idxs(
                            fc_in, fc_out,
                            neuron_idxs[:mid]
                )
            elif isinstance(self.model, EVACLIP) or isinstance(self.model, EVAVisionTransformer):
                self.blocks[layer_id].mlp.fc1, \
                    self.blocks[layer_id].mlp.fc2 = \
                        prune_mlp_by_idxs(
                            fc_in, fc_out,
                            neuron_idxs[:mid]
                )
            elif isinstance(self.model, DinoVisionTransformer):
                self.blocks[layer_id].mlp.w12, \
                    self.blocks[layer_id].mlp.w3 = \
                        prune_mlp_by_idxs(
                            fc_in, fc_out,
                            neuron_idxs[:mid],
                            has_gate=True
                )
            else:
                raise NotImplementedError
            
            loss = self.dataset_loop_val()
            # all_reduce(loss, op=ReduceOp.AVG)

            losses.append(
                (layer_id, mid, loss)
            )

            if self.rank == 0:
                self.logger_console.info(f'the loss of layer {layer_id} with mlp size {mid}: {loss}')

            loss_inc = loss - self.loss_prev_block
            # loss_inc = math.fabs(loss - self.loss_prev_block)
            # loss_inc = (loss - self.loss_prev_block) / self.loss_prev_block

            # if loss_inc <= 0 or math.fabs(loss_inc) <= self.loss_inc_thresh:
            if loss_inc <= self.loss_inc_thresh:
                right = mid
                mlp_hidden_size_result = mid
                loss_result = loss
            else:
                left = mid

        if isinstance(self.model, OpenCLIP) or isinstance(self.model, VisionTransformer):
            self.blocks[layer_id].mlp[0], \
                self.blocks[layer_id].mlp[-1] = \
                    prune_mlp_by_idxs(
                        fc_in, fc_out,
                        neuron_idxs[:mlp_hidden_size_result]
            )
        elif isinstance(self.model, EVACLIP) or isinstance(self.model, EVAVisionTransformer):
            self.blocks[layer_id].mlp.fc1, \
                self.blocks[layer_id].mlp.fc2 = \
                    prune_mlp_by_idxs(
                        fc_in, fc_out,
                        neuron_idxs[:mlp_hidden_size_result]
            )
        elif isinstance(self.model, DinoVisionTransformer):
                self.blocks[layer_id].mlp.w12, \
                    self.blocks[layer_id].mlp.w3 = \
                        prune_mlp_by_idxs(
                            fc_in, fc_out,
                            neuron_idxs[:mlp_hidden_size_result],
                            has_gate=True
                )
        else:
            raise NotImplementedError

        if self.rank == 0:
            self.logger_tb.add_scalar('mlp_hsize', mlp_hidden_size_result, self.nlayers - layer_id)

            self.logger_tb.add_scalar('loss', 
                loss_result if loss_result is not None else self.loss_prev_block, 
                self.nlayers - layer_id
            )

        self.loss_prev_block = loss_result
        self.losses_val.append(losses)

        # return mlp_hidden_size_result, loss_result

    def prune(self, layer_id):
        if self.num_selected_neurons is not None:
            _, neuron_idxs = torch.topk(
                self.scores_abs[layer_id] if self.use_abs else self.scores[layer_id],
                k=self.num_selected_neurons,
                largest=True
            )
        else:
            raise NotImplementedError

        if isinstance(self.model, OpenCLIP) or isinstance(self.model, VisionTransformer):
            self.blocks[layer_id].mlp[0], \
                self.blocks[layer_id].mlp[-1] = \
                    prune_mlp_by_idxs(
                        self.blocks[layer_id].mlp[0],
                        self.blocks[layer_id].mlp[-1],
                        neuron_idxs
                )
        elif isinstance(self.model, EVACLIP) or isinstance(self.model, EVAVisionTransformer):
            self.blocks[layer_id].mlp.fc1, \
                self.blocks[layer_id].mlp.fc2 = \
                    prune_mlp_by_idxs(
                        self.blocks[layer_id].mlp.fc1,
                        self.blocks[layer_id].mlp.fc2,
                        neuron_idxs
                )
        elif isinstance(self.model, DinoVisionTransformer):
                self.blocks[layer_id].mlp.w12, \
                    self.blocks[layer_id].mlp.w3 = \
                        prune_mlp_by_idxs(
                            self.blocks[layer_id].mlp.w12, 
                            self.blocks[layer_id].mlp.w3,
                            neuron_idxs,
                            has_gate=True
                )
        else:
            raise NotImplementedError

    def dataset_loop(self):
        timer = AverageMeter('Time')
        losses = AverageMeter('Loss')

        dataloader = iter(self.dataloader)

        if self.cfg.prune.only_image:
            image = next(dataloader)
            if isinstance(image, list) or isinstance(image, tuple):
                image = image[0]
        else:
            image, text = next(dataloader)
            text = text.cuda(non_blocking=True)
        
        image = image.to(torch.bfloat16).cuda(non_blocking=True)

        t0 = time.perf_counter()
        _t0 = t0

        loss_value = 0.

        for i, _data in enumerate(dataloader):
            if self.cfg.prune.only_image:
                if isinstance(_data, list) or isinstance(_data, tuple):
                    _image = _data[0]
                else:
                    _image = _data
            else:
                _image, _text = _data
                _text = _text.cuda(non_blocking=True)

            _image = _image.to(torch.bfloat16).cuda(non_blocking=True) 
            
            if self.cfg.prune.only_image:
                loss = self.step_entropy(image, use_grad=True)
            else:
                loss = self.step(image, text)

            if self.cfg.prune.only_image:
                image = _image
            else:
                image, text = _image, _text

            loss_val = loss.item()
            losses.update(loss_val)
            loss_value += loss_val

            t1 = time.perf_counter()
            timer.update(t1 - t0)
            t0 = t1
            self.nbatch_accum += 1
            niter = i + 1
            if self.rank == 0 and niter % 20 == 0:
                progress = niter / self.nbatch_per_epoch
                info = f'niter: {niter:06d} / {self.nbatch_per_epoch:06d}, ' \
                     + f'progress: {progress:.2f}, ' \
                     + f'iter_time: {timer.val:.2f}({timer.avg:.2f})s, ' \
                     + f'run_time: {datetime.timedelta(seconds=int(timer.sum))}, ' \
                     + f'rest_time: {datetime.timedelta(seconds=int(timer.sum / progress - timer.sum))}, '\
                     + f'loss: {losses.val:.2f}({losses.avg:.2f})'

                self.logger_console.info(info)
            # break
        
        # last batch
        if self.cfg.prune.only_image:
            loss = self.step_entropy(image)
        else:
            loss = self.step(image, text)

        loss_value += loss.item()

        loss_value /= (i + 2)

        if self.rank == 0:
            self.logger_console.info(f'Total evaluation time: {datetime.timedelta(seconds=int(time.perf_counter() - _t0))}')

        return loss_value

    def step(self, image, text):
        ncaption = 1
        if text.dim() == 3:
            batch_size, ncaption, ntoken = text.size()
            text = text.view(-1, ntoken)
        else:
            batch_size, ntoken = text.size()

        with torch.no_grad():
            text_feat = self.model.encode_text(text, normalize=True)
            all_text_feat = gather_features(text_feat, gather_with_grad=False, \
                rank=self.rank, world_size=self.world_size)

        img_feat = self.model.encode_image(image, normalize=True)
        all_img_feat = gather_features(img_feat, gather_with_grad=True, \
            rank=self.rank, world_size=self.world_size)

        if not self.cfg.prune.use_entropy:
            if ncaption > 1:
                nchannel = all_text_feat.size(-1)

                all_text_feat = all_text_feat.view(-1, ncaption, nchannel)\
                    .permute(2, 1, 0).contiguous().view(nchannel, -1)

                logits_per_image = self.logit_scale * img_feat @ all_text_feat

                logits_per_image = logits_per_image.view(batch_size*ncaption, -1)

                logits_per_text = self.logit_scale * text_feat.view(-1, nchannel) @ \
                    all_img_feat.T
                
                if self.labels is None:
                    self.labels = self.rank * self.batch_size + \
                        torch.arange(self.batch_size, device=self.device, \
                            dtype=torch.long)
                    self.labels = self.labels.view(-1, 1).repeat(1, ncaption).view(-1)
            else:
                logits_per_image = self.logit_scale * img_feat @ all_text_feat.T
                logits_per_text = self.logit_scale * text_feat @ all_img_feat.T

                if self.labels is None:
                    self.labels = self.rank * self.batch_size + \
                        torch.arange(self.batch_size, device=self.device, \
                            dtype=torch.long)
        else:
            if ncaption > 1:
                nchannel = all_text_feat.size(-1)
                all_text_feat = all_text_feat.view(-1, nchannel)

        if self.cfg.prune.use_entropy:
            loss = 0.5 * (entropy(img_feat, all_text_feat, self.cfg.prune.temp_inv, use_norm=False) + \
                entropy(text_feat, all_img_feat, self.cfg.prune.temp_inv, use_norm=False)
            )
        else:
            loss = 0.5 * (
                F.cross_entropy(logits_per_image, self.labels) + \
                F.cross_entropy(logits_per_text, self.labels)
            )

        loss.backward()

        all_reduce(loss, op=ReduceOp.AVG)

        torch.cuda.synchronize()

        return loss
    
    def step_entropy(self, image, use_grad=True):
        img_feat = self.model(image)

        img_feat = F.normalize(img_feat, dim=1)

        all_img_feat = gather_features(img_feat, gather_with_grad=use_grad, \
            rank=self.rank, world_size=self.world_size)

        # convert to float32 for softmax and cosine similarity
        logits = self.cfg.prune.temp_inv * img_feat.float() @ all_img_feat.float().T

        p = F.softmax(logits, dim=-1)

        loss = - torch.sum(p * torch.log(p), dim=1).mean()
        
        if use_grad:
            loss.requires_grad_(True)
            loss.backward()

        all_reduce(loss, op=ReduceOp.AVG)

        torch.cuda.synchronize()

        return loss

    def dataset_loop_val(self):

        dataloader = iter(self.dataloader_val)

        if self.cfg.prune.only_image:
            image = next(dataloader)
            if isinstance(image, list) or isinstance(image, tuple):
                image = image[0]
        else:
            image, text = next(dataloader)
            text = text.cuda(non_blocking=True)
        
        image = image.to(torch.bfloat16).cuda(non_blocking=True)

        t0 = time.perf_counter()

        loss_value = 0.

        for i, _data in enumerate(dataloader):
            if self.cfg.prune.only_image:
                if isinstance(_data, list) or isinstance(_data, tuple):
                    _image = _data[0]
                else:
                    _image = _data
            else:
                _image, _text = _data
                _text = _text.cuda(non_blocking=True)

            _image = _image.to(torch.bfloat16).cuda(non_blocking=True) 
            
            if self.cfg.prune.only_image:
                with torch.no_grad():
                    loss = self.step_entropy(image, use_grad=False)
            else:
                loss = self.step_val(image, text)

            loss_value += loss.item()

            if self.cfg.prune.only_image:
                image = _image
            else:
                image, text = _image, _text
            # break
        
        # last batch
        if self.cfg.prune.only_image:
            with torch.no_grad():
                loss = self.step_entropy(image, use_grad=False)
        else:
            loss = self.step_val(image, text)

        loss_value += loss.item()

        # loss_value /= len(self.dataloader_val)
        loss_value /= (i + 2)

        if self.rank == 0:
            self.logger_console.info(f'Total validation time: {datetime.timedelta(seconds=int(time.perf_counter() - t0))}')
        
        return loss_value

    def step_val(self, image, text):
        with torch.no_grad():
            ncaption = 1
            if text.dim() == 3:
                batch_size, ncaption, ntoken = text.size()
                text = text.view(-1, ntoken)
            else:
                batch_size, ntoken = text.size()

            text_feat = self.model.encode_text(text, normalize=True)
            all_text_feat = gather_features(text_feat, gather_with_grad=False, \
                rank=self.rank, world_size=self.world_size)

            img_feat = self.model.encode_image(image, normalize=True)
            all_img_feat = gather_features(img_feat, gather_with_grad=False, \
                rank=self.rank, world_size=self.world_size)

            if not self.cfg.prune.use_entropy:
                if ncaption > 1:
                    nchannel = all_text_feat.size(-1)

                    all_text_feat = all_text_feat.view(-1, ncaption, nchannel)\
                        .permute(2, 1, 0).contiguous().view(nchannel, -1)

                    logits_per_image = self.logit_scale * img_feat @ all_text_feat

                    logits_per_image = logits_per_image.view(batch_size*ncaption, -1)

                    logits_per_text = self.logit_scale * text_feat.view(-1, nchannel) @ \
                        all_img_feat.T
                    
                    if self.labels_val is None:
                        self.labels_val = self.rank * self.batch_size + \
                            torch.arange(self.batch_size, device=self.device, \
                                dtype=torch.long)
                        self.labels_val = self.labels_val.view(-1, 1).repeat(1, ncaption).view(-1)
                else:
                    logits_per_image = self.logit_scale * img_feat @ all_text_feat.T
                    logits_per_text = self.logit_scale * text_feat @ all_img_feat.T

                    if self.labels_val is None:
                        self.labels_val = self.rank * self.batch_size + \
                            torch.arange(self.batch_size, device=self.device, \
                                dtype=torch.long)
            else:
                if ncaption > 1:
                    nchannel = all_text_feat.size(-1)
                    all_text_feat = all_text_feat.view(-1, nchannel)

            if self.cfg.prune.use_entropy:
                loss = 0.5 * (
                    entropy(img_feat, all_text_feat, self.cfg.prune.temp_inv, use_norm=False) + \
                    entropy(text_feat, all_img_feat, self.cfg.prune.temp_inv, use_norm=False)
                )
            else:
                loss = 0.5 * (
                    F.cross_entropy(logits_per_image, self.labels_val) + \
                    F.cross_entropy(logits_per_text, self.labels_val)
                )

            all_reduce(loss, op=ReduceOp.AVG)

        torch.cuda.synchronize()

        return loss

    def save(self):

        if self.rank == 0:
            os.makedirs(self.score_path, exist_ok=True)
            os.makedirs(self.model_path, exist_ok=True)

            scores = {
                'scores': self.scores,
                'scores_abs': self.scores_abs,
                'losses_val': self.losses_val,
            }

            torch.save(
                scores, 
                f'{self.score_path}/scores_layer.pth'
            )

            if self.cfg.prune.only_image and self.proj is not None:
                if isinstance(self.model, VisionTransformer):
                    self.model.proj = self.proj
                elif isinstance(self.model, DinoVisionTransformer) or isinstance(self.model, EVAVisionTransformer):
                    self.model.head = self.proj
                else:
                    raise NotImplementedError

            torch.save(
                self.model.state_dict(), 
                f'{self.model_path}/model_layer.pth'
            )

            if self.cfg.prune.only_image:
                if isinstance(self.model, VisionTransformer):
                    self.model.proj = None
                elif isinstance(self.model, DinoVisionTransformer) or isinstance(self.model, EVAVisionTransformer):
                    self.model.head = nn.Identity()
                else:
                    raise NotImplementedError

    def resume(self, fname_score, fname_model):
        state_dict = torch.load(fname_score, \
            map_location=torch.device('cuda', self.rank))

        self.scores = state_dict['scores']
        self.scores_abs = state_dict['scores_abs']

        self.resume_score = True

        # ------ pruning ------
        if fname_model is not None:
            if isinstance(self.model, OpenCLIP) or isinstance(self.model, VisionTransformer):
                self.model = prune_mlp_by_weight(self.model, fname_model).bfloat16()
            elif isinstance(self.model, EVACLIP) or isinstance(self.model, EVAVisionTransformer):
                self.model = prune_eva_clip_by_weight(self.model, fname_model).bfloat16()
            elif isinstance(self.model, DinoVisionTransformer):
                self.model = prune_dinov2_by_weight(self.model, fname_model).bfloat16()
            else:
                raise NotImplementedError

            for p in self.model.parameters():
                p.requires_grad = False

    # ----------------- Hook -----------------
    def release_parameter_grad_hook(self, param):
        if param.grad is not None:
            param.grad.detach_()
            param.grad = None

    def record_input_activations(self, module, args):
        # print(module, len(self.activations))
        if self.is_ckpt_forward: # 避免ckpt多次调用hook函数
            return

        nlayer_record = len(self.activations)
        if nlayer_record == 0:
            args[0].requires_grad = True # !important

        self.activations.append(args[0].data)

        if nlayer_record >= self.nlayers - 1:
            self.is_ckpt_forward = True


    def taylor_neuron_importance_1st_order(self, module, grad_input, grad_output):
        cur_layer_id = len(self.activations) - 1
        
        if self.add_hook_on_input_neuron:
            self.activations[cur_layer_id] *= grad_input[0].data
        else:
            self.activations[cur_layer_id] *= grad_output[0].data

        self.scores[cur_layer_id] += \
            self.activations[cur_layer_id].mean(dim=0) \
                .sum(dim=0).to(torch.float)

        self.scores_abs[cur_layer_id] += \
            torch.abs(self.activations[cur_layer_id].sum(dim=1)).mean(dim=0) \
                .to(torch.float)
        
        self.activations.pop()

        if cur_layer_id == 0:
            self.is_ckpt_forward = False


