import os
import sys
sys.path.append(os.getcwd())

import torch

class ZeroModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):

        # return torch.zeros_like(x)
        return 0


from typing import Union
# from clip.open_clip import get_model_config, CLIP, get_cast_dtype, convert_weights_to_lp
from clip.open_clip import create_model_and_transforms as create_open_clip_model_and_transforms
from clip.eva_clip import create_model_and_transforms as create_eva_clip_model_and_transforms

from prune import prune_mlp_by_hidden_size

from dinov2.models import vision_transformer as dino_vit


def create_model_by_weight(
        model_name, pretrained,
        non_visual_pretrained=None,
        precision: str = 'fp32',
        device: Union[str, torch.device] = 'cpu',
        only_vision: bool = False,
        use_zero_module=False,
        frame='open_clip'
    ):

    if frame == 'open_clip':
        model, preprocess_train, preprocess_val = \
            create_open_clip_model_and_transforms(
                model_name,
                pretrained=None,
                precision=precision, 
                device=device
            )
        
        origin_mlp_hidden_size = model.visual.transformer.resblocks[0].mlp[0].out_features
        
        state_dict = torch.load(
            pretrained, 
            map_location=device, 
            weights_only=True
        )

        if non_visual_pretrained is None:
            del_key = 'model.'
            keys = list(state_dict.keys())
            for k in keys:
                if del_key in k:
                    state_dict[k[len(del_key):]] = state_dict[k]
                    del state_dict[k]

        target_key = 'mlp.c_fc.weight'

        mlp_hidden_sizes = []

        for k, v in state_dict.items():
            if only_vision and target_key in k:
                mlp_hidden_sizes.append(v.size(0))
            elif 'visual.' in k and target_key in k:
                mlp_hidden_sizes.append(v.size(0))

        # print(mlp_hidden_sizes)
        assert(len(mlp_hidden_sizes) > 0)

        # print('mlp_hidden_sizes:', mlp_hidden_sizes)
        # print('average hidden size:', sum(mlp_hidden_sizes) / len(mlp_hidden_sizes))

        nblock = len(model.visual.transformer.resblocks)

        # idx = 0

        for i in range(nblock):
            model.visual.transformer.resblocks[i].mlp[0], \
                model.visual.transformer.resblocks[i].mlp[-1] = \
                prune_mlp_by_hidden_size(
                    model.visual.transformer.resblocks[i].mlp[0], 
                    model.visual.transformer.resblocks[i].mlp[-1],
                    mlp_hidden_sizes[i]
                )
    elif frame == 'eva_clip':
        model, preprocess_train, preprocess_val = \
            create_eva_clip_model_and_transforms(
                model_name,
                pretrained=None,
                precision=precision, 
                device=device,
                force_custom_clip=True
            )
        
        state_dict = torch.load(
            pretrained, 
            map_location=device, 
            weights_only=True
        )

        if non_visual_pretrained is None:
            del_key = 'model.'
            keys = list(state_dict.keys())
            # print(keys)
            for k in keys:
                if del_key in k:
                    state_dict[k[len(del_key):]] = state_dict[k]
                    del state_dict[k]

        target_key = 'mlp.fc1.weight'

        mlp_hidden_sizes = []

        for k, v in state_dict.items():
            if only_vision and target_key in k:
                mlp_hidden_sizes.append(v.size(0))
            elif 'visual.' in k and target_key in k:
                mlp_hidden_sizes.append(v.size(0))

        assert(len(mlp_hidden_sizes) > 0)

        # print('mlp_hidden_sizes:', mlp_hidden_sizes)
        # print('average hidden size:', sum(mlp_hidden_sizes) / len(mlp_hidden_sizes))

        nblock = len(model.visual.blocks)

        for i in range(nblock):
            # print(i)
            model.visual.blocks[i].mlp.fc1, \
                model.visual.blocks[i].mlp.fc2 = \
                prune_mlp_by_hidden_size(
                    model.visual.blocks[i].mlp.fc1, 
                    model.visual.blocks[i].mlp.fc2,
                    mlp_hidden_sizes[i]
                )
    else:
        raise NotImplementedError

    if only_vision and non_visual_pretrained is None:
        model = model.visual
        model.load_state_dict(state_dict, strict=True)
    elif only_vision and non_visual_pretrained is not None:
        model.visual.load_state_dict(state_dict, strict=True)
        state_dict_text = torch.load(
            non_visual_pretrained,
            map_location=device, 
            weights_only=True
        )
        model.load_state_dict(state_dict_text, strict=False)
    else:
        model.load_state_dict(state_dict, strict=True)

    return model, preprocess_train, preprocess_val


def test_load_open_clip_model_by_weight():

    # ckpt = '/public/model_weight/open_clip/CLIP-ViT-g-14-laion2B-s34B-b88K/open_clip_pytorch_model.bin'

    ckpt = 'out/ViT-g-14_prune/open_clip_2025-04-21_16-23-07/ckpt/model_thresh_0.005_t15/model_layer.pth'
    non_visual_pretrained = '/public/scccse/model_weight/CLIP-ViT-g-14-laion2B-s34B-b88K/open_clip_pytorch_model_non_visual.pth'

    model, _, _ = create_model_by_weight(
        'ViT-g-14', pretrained=ckpt,
        non_visual_pretrained=non_visual_pretrained,
        only_vision=True
    )

    # ckpt = 'out/open_clip_BigG_coco/model_layer_wise_thresh_0.02_train/model_layer_0.pth'

    # model, _, _ = create_open_clip_model_by_weight(
    #     'ViT-bigG-14', pretrained=ckpt,
    #     use_zero_module=True
    # )

    print(model)

    # nblock = len(model.visual.transformer.resblocks)

    # for i in range(nblock):
    #     print(model.visual.transformer.resblocks[i].mlp)
    #     print(model.visual.transformer.resblocks[i].mlp[0])

    return


def create_open_clip_model_by_vision_and_clip_weight(
        model_name, 
        pretrained_vision,
        pretrained_clip_origin,
        precision: str = 'fp32',
        device: Union[str, torch.device] = 'cpu',
    ):

    model, preprocess_train, preprocess_val = \
        create_open_clip_model_and_transforms(
            model_name,
            pretrained=None,
            precision=precision, 
            device=device
        )
    
    state_dict_vision = torch.load(
        pretrained_vision, 
        map_location=device, 
        weights_only=True
    )

    state_dict_clip = torch.load(
        pretrained_clip_origin, 
        map_location=device, 
        weights_only=True
    )

    target_key_prefix = 'visual.'
    pos = len(target_key_prefix)

    for k in state_dict_clip.keys():
        if target_key_prefix in k:
            state_dict_clip[k] = state_dict_vision[k[pos:]]
    
    model.load_state_dict(state_dict_clip, strict=True)

    return model, preprocess_train, preprocess_val


def build_model_dinov2(args, pretrained=None, img_size=224) -> torch.nn.Module:
    vit_kwargs = dict(
        img_size=img_size,
        patch_size=args.patch_size,
        init_values=args.layerscale,
        ffn_layer=args.ffn_layer,
        block_chunks=args.block_chunks,
        qkv_bias=args.qkv_bias,
        proj_bias=args.proj_bias,
        ffn_bias=args.ffn_bias,
        num_register_tokens=args.num_register_tokens,
        interpolate_offset=args.interpolate_offset,
        interpolate_antialias=args.interpolate_antialias,
    )

    model = dino_vit.__dict__[args.arch](**vit_kwargs)

    if pretrained is not None:
        model.load_state_dict(
            torch.load(pretrained, map_location='cpu')
        )
    
    return model


def create_dinov2_by_weight(args, pretrained=None, img_size=224) -> torch.nn.Module:
    vit_kwargs = dict(
        img_size=img_size,
        patch_size=args.patch_size,
        init_values=args.layerscale,
        ffn_layer=args.ffn_layer,
        block_chunks=args.block_chunks,
        qkv_bias=args.qkv_bias,
        proj_bias=args.proj_bias,
        ffn_bias=args.ffn_bias,
        num_register_tokens=args.num_register_tokens,
        interpolate_offset=args.interpolate_offset,
        interpolate_antialias=args.interpolate_antialias,
    )

    model = dino_vit.__dict__[args.arch](**vit_kwargs)
    
    
    origin_mlp_hidden_size = model.blocks[0].mlp.w3.in_features
    
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
            
    if model.mask_token is not None and 'mask_token' not in state_dict:
        model.mask_token = None
    
    model.load_state_dict(state_dict, strict=True)
    
    return model


def test_load_dinov2_by_weight():
    from omegaconf import OmegaConf
    cfg = OmegaConf.load('./configs/dinov2_coco.yaml')

    ckpt = 'out/dinov2/model_layer_wise_thresh_0.02_shuffle/model_layer_0.pth'

    model = create_dinov2_by_weight(
        cfg.model, 
        pretrained=ckpt,
        img_size=cfg.model.input_size, 
    )

    print(model)

    return

if __name__ == '__main__':
    # test_load_open_clip_model_by_weight()
    test_load_open_clip_model_by_weight()