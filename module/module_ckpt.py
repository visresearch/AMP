import os
import sys
sys.path.append(os.getcwd())

from typing import Optional
import types

import torch
from torch import nn

from torch.utils.checkpoint import checkpoint

from clip.open_clip.transformer import VisionTransformer, Transformer
from clip.eva_clip.eva_vit_model import EVAVisionTransformer
from dinov2.models.vision_transformer import DinoVisionTransformer


def set_model_ckpt(frame, block_id_no_ckpt=None, model=None, set_x_grad=False, max_gpu_memory=42):
    if frame == 'dinov2':
        if model is not None:
            model.idx_ckpt = None
        else:
            DinoVisionTransformer.idx_ckpt = None
        def dinov2_forward_features(self, x, masks=None):
            nblock = len(self.blocks)
            if self.idx_ckpt is None:
                self.idx_ckpt = nblock

            x = self.prepare_tokens_with_masks(x, masks)

            for i in range(self.idx_ckpt):
                gpu_memory = torch.cuda.memory_allocated() / (1024*1024*1024)
                if gpu_memory > max_gpu_memory and i < nblock - 1:
                    self.idx_ckpt = i
                    print(f'Activate checkpointing at block {self.idx_ckpt}')
                    break

                if i == block_id_no_ckpt and set_x_grad:
                    x.requires_grad = True

                x = self.blocks[i](x)
            
            for i in range(self.idx_ckpt, nblock):
                if i == block_id_no_ckpt and set_x_grad:
                    x.requires_grad = True

                if i != block_id_no_ckpt:
                    x = checkpoint(self.blocks[i], x, use_reentrant=False)
                else:
                    x = self.blocks[i](x)

            x_norm = self.norm(x)
            return {
                "x_norm_clstoken": x_norm[:, 0],
                "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
                "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
                "x_prenorm": x,
                "masks": masks,
            }
        
        if model is not None and isinstance(model, DinoVisionTransformer):
            model.forward_features = dinov2_forward_features.__get__(model)
        else:
            DinoVisionTransformer.forward_features = dinov2_forward_features
    elif frame == 'open_clip':
        if model is not None:
            model.transformer.idx_ckpt = None
        else:
            Transformer.idx_ckpt = None
        def open_clip_forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
            nblock = len(self.resblocks)
            if self.idx_ckpt is None:
                self.idx_ckpt = nblock

            if not self.batch_first:
                x = x.transpose(0, 1).contiguous()    # NLD -> LND

            for i in range(self.idx_ckpt):
                gpu_memory = torch.cuda.memory_allocated() / (1024*1024*1024)
                if gpu_memory > max_gpu_memory and i < nblock - 1:
                    self.idx_ckpt = i
                    print(f'Activate checkpointing at block {self.idx_ckpt}')
                    break

                if i == block_id_no_ckpt and set_x_grad:
                    x.requires_grad = True

                x = self.resblocks[i](x, attn_mask=attn_mask)

            for i in range(self.idx_ckpt, nblock):
                if i == block_id_no_ckpt and set_x_grad:
                    x.requires_grad = True

                if i != block_id_no_ckpt:
                    # print('open_clip ckpt', i)
                    x = checkpoint(self.resblocks[i], x, attn_mask, use_reentrant=False)
                else:
                    x = self.resblocks[i](x, attn_mask=attn_mask)

            if not self.batch_first:
                x = x.transpose(0, 1)    # LND -> NLD
            return x
        if model is not None and isinstance(model, VisionTransformer):
            model.transformer.forward = open_clip_forward.__get__(model.transformer)
        else:
            Transformer.forward = open_clip_forward
    elif frame == 'eva_clip':
        if model is not None:
            model.idx_ckpt = None
        else:
            EVAVisionTransformer.idx_ckpt = None
        def eval_clip_forward_features(self, x, return_all_features=False):
            nblock = len(self.blocks)
            if self.idx_ckpt is None:
                self.idx_ckpt = nblock
            
            x = self.patch_embed(x)
            batch_size, seq_len, _ = x.size()

            cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_tokens, x), dim=1)
            if self.pos_embed is not None:
                x = x + self.pos_embed
            x = self.pos_drop(x)

            rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None

            for i in range(self.idx_ckpt):
                gpu_memory = torch.cuda.memory_allocated() / (1024*1024*1024)
                if gpu_memory > max_gpu_memory and i < nblock - 1:
                    self.idx_ckpt = i
                    print(f'Activate checkpointing at block {self.idx_ckpt}')
                    break

                if i == block_id_no_ckpt and set_x_grad:
                    x.requires_grad = True

                x = self.blocks[i](x, rel_pos_bias=rel_pos_bias)

            for i in range(self.idx_ckpt, nblock):
                if i == block_id_no_ckpt and set_x_grad:
                    x.requires_grad = True

                if i != block_id_no_ckpt:
                    # print('eva_clip ckpt', i)
                    x = checkpoint(self.blocks[i], x, (rel_pos_bias,), use_reentrant=False)
                else:
                    x = self.blocks[i](x, rel_pos_bias=rel_pos_bias)

            if not return_all_features:
                x = self.norm(x)
                if self.fc_norm is not None:
                    return self.fc_norm(x.mean(1))
                else:
                    return x[:, 0]
            return x
        if model is not None and isinstance(model, EVAVisionTransformer):
            model.forward_features = eval_clip_forward_features.__get__(model)
        else:
            EVAVisionTransformer.forward_features = eval_clip_forward_features
    else:
        raise NotImplementedError


def set_model_ckpt_train(model, max_gpu_memory=42):
    if isinstance(model, DinoVisionTransformer):
        model.idx_ckpt = None
        def dinov2_forward_features(self, x, masks=None):
            nblock = len(self.blocks)

            if self.idx_ckpt is None:
                self.idx_ckpt = nblock

            x = self.prepare_tokens_with_masks(x, masks)

            for i in range(self.idx_ckpt):
                if self.idx_ckpt == nblock:
                    gpu_memory = torch.cuda.memory_allocated() / (1024*1024*1024)
                    if gpu_memory > max_gpu_memory and i < nblock - 1:
                        self.idx_ckpt = i
                        print(f'Activate checkpointing at block {self.idx_ckpt}')
                        break

                x = self.blocks[i](x)
            
            for i in range(self.idx_ckpt, nblock):
                x = checkpoint(self.blocks[i], x, use_reentrant=False)

            x_norm = self.norm(x)
            return {
                "x_norm_clstoken": x_norm[:, 0],
                "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
                "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
                "x_prenorm": x,
                "masks": masks,
            }
        
        model.forward_features = types.MethodType(dinov2_forward_features, model)
    elif isinstance(model, VisionTransformer):
        model.transformer.idx_ckpt = None
        def open_clip_forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
            nblock = len(self.resblocks)
            if self.idx_ckpt is None:
                self.idx_ckpt = nblock

            if not self.batch_first:
                x = x.transpose(0, 1).contiguous()    # NLD -> LND

            for i in range(self.idx_ckpt):
                if self.idx_ckpt == nblock:
                    gpu_memory = torch.cuda.memory_allocated() / (1024*1024*1024)
                    if gpu_memory > max_gpu_memory and i < nblock - 1:
                        self.idx_ckpt = i
                        print(f'Activate checkpointing at block {self.idx_ckpt}')
                        break

                x = self.resblocks[i](x, attn_mask=attn_mask)

            for i in range(self.idx_ckpt, nblock):
                x = checkpoint(self.resblocks[i], x, attn_mask, use_reentrant=False)

            if not self.batch_first:
                x = x.transpose(0, 1)    # LND -> NLD
            return x
        model.transformer.forward = types.MethodType(open_clip_forward, model.transformer)
    elif isinstance(model, EVAVisionTransformer):
        model.idx_ckpt = None
        def eval_clip_forward_features(self, x, return_all_features=False):
            nblock = len(self.blocks)
            if self.idx_ckpt is None:
                self.idx_ckpt = nblock
            
            x = self.patch_embed(x)
            batch_size, seq_len, _ = x.size()

            cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_tokens, x), dim=1)
            if self.pos_embed is not None:
                x = x + self.pos_embed
            x = self.pos_drop(x)

            rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None

            for i in range(self.idx_ckpt):
                if self.idx_ckpt == nblock:
                    gpu_memory = torch.cuda.memory_allocated() / (1024*1024*1024)
                    if gpu_memory > max_gpu_memory and i < nblock - 1:
                        self.idx_ckpt = i
                        print(f'Activate checkpointing at block {self.idx_ckpt}')
                        break

                x = self.blocks[i](x, rel_pos_bias=rel_pos_bias)

            for i in range(self.idx_ckpt, nblock):
                x = checkpoint(self.blocks[i], x, (rel_pos_bias,), use_reentrant=False)

            if not return_all_features:
                x = self.norm(x)
                if self.fc_norm is not None:
                    return self.fc_norm(x.mean(1))
                else:
                    return x[:, 0]
            return x
        model.forward_features = types.MethodType(eval_clip_forward_features, model)
    else:
        raise NotImplementedError

def main():
    # set_model_ckpt(frame='open_clip', nblock_interval=2)
    model = VisionTransformer(
        image_size=224, 
        patch_size=16, 
        width=128,
        layers=4,
        heads=8,
        mlp_ratio=2,
    ).cuda()

    x = torch.rand(512, 3, 224, 224).cuda()

    print(f"Memory allocated: {torch.cuda.memory_allocated()} B")

    y = model(x)

    y = y.mean()

    print(f"Memory allocated: {torch.cuda.memory_allocated()} B")

    y.backward()

    print(f"Memory allocated: {torch.cuda.memory_allocated()} B")

    # set_model_ckpt(frame='open_clip', nblock_interval=2, model=model)
    # y = model(x)

    # print(y)

    return

if __name__ == '__main__':
    main()
