import os

import torch

from clip.open_clip.model import CLIP
from clip.open_clip.factory import create_model

from parse import get_args_score
from prune import prune_mlp_by_idxs

def merge_importance_score(cfg):

    score_accu, score_abs_accu = [], []

    for i in range(cfg.ckpt.nfile_score):
        fname = f'{cfg.ckpt.score_path}/score_accu_{i}_{cfg.ckpt.nfile_score}.pth'
        scores_and_ntokens = torch.load(fname, map_location='cpu')
        score_accu.append(scores_and_ntokens['score_accu'])
        score_abs_accu.append(scores_and_ntokens['score_abs_accu'])
    
    score_merge = sum(score_accu)
    score_abs_accu_merge = sum(score_abs_accu)

    # print(score_merge)
    # print(score_abs_accu_merge)

    score = {
        'score_accu': score_merge,
        'score_abs_accu': score_abs_accu_merge,
    }

    torch.save(score, f'{cfg.ckpt.score_path}/score_merge.pth')

def preserve_neurons(cfg):
    path = f'{cfg.ckpt.score_path}/score_merge.pth'
    score = torch.load(path, map_location='cpu')

    if cfg.prune.use_abs:
        scores, neuron_idxs = torch.topk(score['score_abs_accu'], k=cfg.prune.num_selected_neurons, largest=True)
    else:
        scores, neuron_idxs = torch.topk(score['score_accu'], k=cfg.prune.num_selected_neurons, largest=True)
    # sorted_scores, neuron_idxs = torch.sort(score, descending=True)
    # print(sorted_scores)

    return neuron_idxs


def prune_model(cfg):
    if not os.path.exists(f'{cfg.ckpt.score_path}/score_merge.pth'):
        merge_importance_score(cfg)

    neuron_idxs = preserve_neurons(cfg)
    # -------------- Model --------------
    model = create_model(cfg.model.name, pretrained=cfg.model.path)

    if isinstance(model, CLIP):
        for idx, layer in enumerate(model.visual.transformer.resblocks):
            layer.mlp[0], layer.mlp[-1] = \
                prune_mlp_by_idxs(
                    layer.mlp[0], layer.mlp[-1], neuron_idxs[idx]
                )
    else:
        raise NotImplementedError

    path = cfg.ckpt.model_path+f"-hidden{cfg.prune.num_selected_neurons}"

    if cfg.prune.use_abs:
        path += '_abs'

    os.makedirs(path, exist_ok=True)
    
    torch.save(model.state_dict(), path+'/ckpt.pth')


if __name__ == '__main__':
    from omegaconf import OmegaConf

    args = get_args_score()
    cfg = OmegaConf.load(args.config_file)

    # merge_importance_score(cfg)
    # preserve_neurons(cfg)
    prune_model(cfg)
