import torch
from torch import nn

# from clip.open_clip.model import CLIP
from clip.open_clip.transformer import VisionTransformer
from clip.eva_clip.eva_vit_model import EVAVisionTransformer
from dinov2.models.vision_transformer import DinoVisionTransformer

class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        if isinstance(self.model, VisionTransformer):
            cls_token, feat_tokens = \
                self.model(x, output_tokens=True)
            return cls_token, feat_tokens
        elif isinstance(self.model, EVAVisionTransformer):
            feat = self.model.forward_features(x, return_all_features=True)
            feat = self.model.norm(feat)
            output = self.model.head(feat[:, 0])
            return output, feat[:, 1:]
        elif isinstance(self.model, DinoVisionTransformer):
            output = self.model.forward_features(x)
            return output['x_norm_clstoken'], output['x_norm_patchtokens']
        else:
            raise NotImplementedError