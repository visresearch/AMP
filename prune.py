import torch
from torch import nn
from clip.open_clip.model import CLIP as OpenCLIP

def prune_mlp_by_idxs(in_fc:nn.Linear, out_fc:nn.Linear, idxs:torch.Tensor, has_gate=False):
    # --------------- input fc ---------------
    in_weight = in_fc.weight.data
    in_bias = in_fc.bias.data \
        if in_fc.bias is not None else None

    if has_gate:
        gate_idxs = torch.cat(
            (idxs, idxs + in_weight.shape[0] // 2),
            dim=0
        )
        in_weight_prune = in_weight[gate_idxs]
        in_bias_prune = in_bias[gate_idxs] \
            if in_bias is not None else None
    else:
        in_weight_prune = in_weight[idxs]
        in_bias_prune = in_bias[idxs] \
            if in_bias is not None else None

    in_fc_prune = nn.Linear(
        in_features=in_weight_prune.shape[1], 
        out_features=in_weight_prune.shape[0],
        bias=in_bias_prune is not None
    )

    in_fc_prune.weight.data = in_weight_prune
    if in_bias is not None:
        in_fc_prune.bias.data = in_bias_prune
    
    # --------------- out fc -----------------
    out_weight = out_fc.weight.data
    out_bias = out_fc.bias.data \
        if out_fc.bias is not None else None

    out_weight_prune = out_weight[:, idxs]

    out_fc_prune = nn.Linear(
        in_features=out_weight_prune.shape[1], 
        out_features=out_weight_prune.shape[0],
        bias=out_bias is not None
    )
    out_fc_prune.weight.data = out_weight_prune
    if out_bias is not None:
        out_fc_prune.bias.data = out_bias

    return in_fc_prune, out_fc_prune


def prune_mlp_by_hidden_size(in_fc:nn.Linear, out_fc:nn.Linear, size:int, has_gate=False):
    # --------------- input fc ---------------
    in_weight = in_fc.weight.data
    in_bias = in_fc.bias.data \
        if in_fc.bias is not None else None

    if has_gate:
        hidden_size = in_weight.shape[0] // 2
        in_weight_prune = torch.cat((in_weight[:size], in_weight[hidden_size:(hidden_size + size)]), dim=0)
        in_bias_prune = torch.cat((in_bias[:size], in_bias[hidden_size:(hidden_size + size)]), dim=0) \
            if in_bias is not None else None
    else:
        in_weight_prune = in_weight[:size]
        in_bias_prune = in_bias[:size] \
            if in_bias is not None else None

    in_fc_prune = nn.Linear(
        in_features=in_weight_prune.shape[1], 
        out_features=in_weight_prune.shape[0],
        bias=in_bias_prune is not None
    )

    in_fc_prune.weight.data = in_weight_prune
    if in_bias is not None:
        in_fc_prune.bias.data = in_bias_prune
    
    # --------------- out fc -----------------
    out_weight = out_fc.weight.data
    out_bias = out_fc.bias.data \
        if out_fc.bias is not None else None

    out_weight_prune = out_weight[:, :size]

    out_fc_prune = nn.Linear(
        in_features=out_weight_prune.shape[1], 
        out_features=out_weight_prune.shape[0],
        bias=out_bias is not None
    )
    out_fc_prune.weight.data = out_weight_prune
    if out_bias is not None:
        out_fc_prune.bias.data = out_bias

    return in_fc_prune, out_fc_prune


def prune_mlp_by_weight(model, weight):
    state_dict = torch.load(
        weight, 
        map_location='cpu', 
        weights_only=True
    )

    target_key = 'mlp.c_fc.weight'

    mlp_hidden_sizes = []

    is_clip = isinstance(model, OpenCLIP)

    for k, v in state_dict.items():
        if ('visual.' in k or not is_clip) and target_key in k:
            mlp_hidden_sizes.append(v.size(0))
    
    if is_clip:
        model = model.visual

    nblock = len(model.transformer.resblocks)

    for i in range(nblock):
        model.transformer.resblocks[i].mlp[0], \
            model.transformer.resblocks[i].mlp[-1] = \
            prune_mlp_by_hidden_size(
                model.transformer.resblocks[i].mlp[0], 
                model.transformer.resblocks[i].mlp[-1],
                mlp_hidden_sizes[i]
            )
    
    if model.proj is None:
        if 'proj' in state_dict:
            del state_dict['proj']
        elif 'visual.proj' in state_dict:
            del state_dict['visual.proj']

    model.load_state_dict(state_dict, strict=True)

    return model

def prune_dinov2_by_weight(model, pretrained=None):
    state_dict = torch.load(
        pretrained, 
        map_location='cpu', 
        weights_only=True
    )

    mlp_hidden_sizes = []

    target_key = 'mlp.w3.weight'

    for k, v in state_dict.items():
        if target_key in k:
            mlp_hidden_sizes.append(v.size(1))

    nblock = len(model.blocks)

    for i in range(nblock):
        model.blocks[i].mlp.w12, model.blocks[i].mlp.w3 = \
            prune_mlp_by_hidden_size(
                model.blocks[i].mlp.w12, 
                model.blocks[i].mlp.w3,
                mlp_hidden_sizes[i],
                has_gate=True
            )
    
    model.load_state_dict(state_dict, strict=True)
    
    return model

def prune_eva_clip_by_weight(model, weight):
    state_dict = torch.load(
        weight, 
        map_location='cpu', 
        weights_only=True
    )

    target_key = 'mlp.fc1.weight'

    mlp_hidden_sizes = []

    is_clip = isinstance(model, OpenCLIP)

    for k, v in state_dict.items():
        if ('visual.' in k or not is_clip) and target_key in k:
            mlp_hidden_sizes.append(v.size(0))
    
    if is_clip:
        model = model.visual

    nblock = len(model.blocks)

    for i in range(nblock):
        model.blocks[i].mlp.fc1, \
            model.blocks[i].mlp.fc2 = \
            prune_mlp_by_hidden_size(
                model.blocks[i].mlp.fc1, 
                model.blocks[i].mlp.fc2,
                mlp_hidden_sizes[i]
            )
    
    if not isinstance(model.head, nn.Linear):
        if 'head.weight' in state_dict:
            del state_dict['head.weight']
        if 'head.bias' in state_dict:
            del state_dict['head.bias']

    model.load_state_dict(state_dict, strict=True)

    return model