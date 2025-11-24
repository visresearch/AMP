import os
import time
import datetime

from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributed import get_rank, get_world_size

from clip.open_clip.model import CLIP

from utils.misc import AverageMeter
from utils.dist import gather_features

class ImportanceEstimator(object):
    def __init__(self, model:nn.Module, \
        criterion='taylor_neuron_importance_1st_order'):

        # self.model = model
        self.criterion = criterion

        self.handles_backward = []
        self.handles_forward = []
        self.handles_param = []

        self.cur_layer = 0

        self.nlayers = None

        self.scores = None
        self.scores_abs = None

        self.add_hook_on_input_neuron = True

        self.rank = get_rank()
        self.device = torch.device('cuda', self.rank)

        if criterion == 'taylor_neuron_importance_1st_order':
            self.activations = []

            if isinstance(model, CLIP):
                self.nlayers = model.visual.transformer.layers
                mlp_hidden_size = model.visual.transformer.resblocks[0].mlp[0].out_features

                self.scores = torch.zeros((self.nlayers, mlp_hidden_size), \
                    dtype=torch.float, device=self.device)
                self.scores_abs = torch.zeros((self.nlayers, mlp_hidden_size), \
                    dtype=torch.float, device=self.device)

                for i, block in enumerate(model.visual.transformer.resblocks):
                    print(f'add hook for block {i}')
                    self.add_forward_pre_hook(block.mlp[-1]) # the input of last fc
                    self.add_backward_hook(block.mlp[-1]) # the input of last fc

        # release parameter grads of model to save memory
        for p in model.visual.parameters():
            self.handles_param.append(
                p.register_post_accumulate_grad_hook(self.release_parameter_grad_hook)
            )

    def release_parameter_grad_hook(self, param):
        param.grad.detach_()
        param.grad = None

    def add_forward_hook(self, module:nn.Module):
        self.handles_forward.append(
            module.register_forward_hook(self.record_activations)
        )
    
    def add_backward_hook(self, module:nn.Module):
        self.handles_backward.append(
            module.register_full_backward_hook(
                self.taylor_neuron_importance_1st_order
            )
        )
    
    def add_forward_pre_hook(self, module:nn.Module):
        self.handles_forward.append(
            module.register_forward_pre_hook(self.record_input_activations)
        )
    
    def record_activations(self, module, args, output):
        self.activations.append(output.data)
        self.cur_layer += 1
    
    def record_input_activations(self, module, args):
        self.activations.append(args[0].data)
        self.cur_layer += 1

    def taylor_neuron_importance_1st_order(self, module, grad_input, grad_output):
        self.cur_layer -= 1

        if self.add_hook_on_input_neuron:
            self.activations[-1] *= grad_input[0].data
        else:
            self.activations[-1] *= grad_output[0].data

        token_dim = self.activations[-1].size(-1)
        
        self.scores[self.cur_layer] = \
            self.activations[-1].view(-1, token_dim).mean(dim=0)
        
        self.scores_abs[self.cur_layer] = \
            torch.abs(self.activations[-1]).view(-1, token_dim).mean(dim=0)

        self.activations.pop()

    def get_scores(self):
        return self.scores, self.scores_abs

    def reset(self):
        self.activations = []
        self.cur_layer = 0
        self.scores.zero_()
        self.scores_abs.zero_()
    
    def taylor_weight_importance(self):
        NotImplementedError
    
    def remove_handles(self):
        for handle in self.handles_forward:
            handle.remove()

        for handle in self.handles_backward:
            handle.remove()

        for handle in self.handles_param:
            handle.remove()


class ImportanceEstimatorCrossDataset(object):
    def __init__(self, model:nn.Module, dataloader, \
        dtype, \
        criterion='taylor_neuron_importance_1st_order'):

        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            self.model = model.module
        else:
            self.model = model

        # self.dataloader = dataloader
        self.batch_size = dataloader.batch_size 
        self.nbatch_per_epoch = len(dataloader)
        self.dataloader = iter(dataloader)

        self.estimator = ImportanceEstimator(self.model, \
            dtype, criterion)

        self.score_accu = 0.
        self.score_abs_accu = 0.

        self.timer = AverageMeter('Time')
        self.losses = AverageMeter('Loss')

        self.rank = get_rank()
        self.world_size = get_world_size()

        self.device = torch.device('cuda', self.rank)

        self.logit_scale = self.model.logit_scale.exp().item()
        self.labels = self.rank * self.batch_size + \
            torch.arange(self.batch_size, device=self.device, dtype=torch.long)

    def step(self, image, text):

        with torch.no_grad():
            text_feat = self.model.encode_text(text, normalize=True)
            all_text_feat = gather_features(text_feat, gather_with_grad=False, \
                rank=self.rank, world_size=self.world_size)

        img_feat = self.model.encode_image(image, normalize=True)
        all_img_feat = gather_features(img_feat, gather_with_grad=True, \
            rank=self.rank, world_size=self.world_size)

        logits_per_image = self.logit_scale * img_feat @ all_text_feat.T
        logits_per_text = self.logit_scale * text_feat @ all_img_feat.T

        loss = 0.5 * (
            F.cross_entropy(logits_per_image, self.labels) + \
            F.cross_entropy(logits_per_text, self.labels)
        )

        # loss.backward(retain_graph=True)
        loss.backward()
        torch.cuda.synchronize()
        
        scores, scores_abs = self.estimator.get_scores()

        self.score_accu += scores.float()
        self.score_abs_accu += scores_abs.float()

        self.estimator.reset()

        return loss

    def run(self):

        image, text = next(self.dataloader)
        image = image.to(torch.bfloat16).cuda(non_blocking=True)
        text = text.cuda(non_blocking=True)

        t0 = time.perf_counter()
        _t0 = t0

        for i, (_image, _text) in enumerate(self.dataloader):
            _image = _image.to(torch.bfloat16).cuda(non_blocking=True) 
            _text = _text.cuda(non_blocking=True)
            
            loss = self.step(image, text)

            image, text = _image, _text

            self.losses.update(loss.item())

            t1 = time.perf_counter()
            self.timer.update(t1 - t0)
            t0 = t1

            niter = i + 1
            if self.rank == 0 and niter % 100 == 0:
                progress = niter / self.nbatch_per_epoch
                info = f'niter: {niter:06d}, progress: {progress:.2f}, ' \
                     + f'iter_time: {self.timer.val:.2f}({self.timer.avg:.2f})s, ' \
                     + f'run_time: {datetime.timedelta(seconds=int(self.timer.sum))}s, ' \
                     + f'rest_time: {datetime.timedelta(seconds=int(self.timer.sum / progress - self.timer.sum))}s, '\
                     + f'loss: {self.losses.val:.2f}({self.losses.avg:.2f})s, '

                print(info)
        
        # last batch
        self.step(image, text)

        if self.rank == 0:
            print(f'Total evaluation time: {datetime.timedelta(seconds=int(time.perf_counter() - _t0))}')
    
    def save_scores(self, path):
        os.makedirs(path, exist_ok=True)

        fname = f'{path}/score_accu_{self.rank}_{self.world_size}.pth'

        pack = {
            'score_accu': self.score_accu,
            'score_abs_accu': self.score_abs_accu,
        }

        torch.save(pack, fname)
