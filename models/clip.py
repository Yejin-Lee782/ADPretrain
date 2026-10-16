from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from .openclip import create_model_and_transforms
import open_clip
from open_clip.transformer import _expand_token
from open_clip.transformer import Transformer, VisionTransformer
from open_clip.model import CLIP


class ClipModel(nn.Module):

    def __init__(self, name='clip-base', device='cuda:0'):
        super(ClipModel, self).__init__()

        self.name = name
        if name == 'clip-base':
            encoder, _, _ = create_model_and_transforms("ViT-B-16", 224, pretrained="openai")
        elif name == 'clip-large':
            encoder, _, _ = create_model_and_transforms("ViT-L-14", 224, pretrained="openai")
        else:
            raise ValueError(f"{name} is currently not supported!")
        self.visual_encoder = encoder
        # free vision encoder
        for name, param in self.visual_encoder.named_parameters():
            param.requires_grad = False
        self.visual_encoder.eval()
        print('Visual encoder initialized.')

        self.device = torch.device(device)

    @property
    def feature_dimensions(self):
        if self.name == 'clip-base':
            return [768, 768, 768, 768]
        if self.name == 'clip-large':
            return [1024, 1024, 1024, 1024]
    
    def encode_image_from_tensors(self, image_tensors, return_global=False, shape='img'):
        with torch.no_grad():
            if self.name == 'clip-base':
                global_features, patch_features = self.visual_encoder.encode_image(image_tensors, [3, 6, 9, 12])
            if self.name == 'clip-large':
                global_features, patch_features = self.visual_encoder.encode_image(image_tensors, [6, 12, 18, 24])
            for i in range(len(patch_features)):
                patch_features[i] = patch_features[i][:, 1:, :]
                if shape == 'img':  # convert sequence to image shape
                    b, l, c = patch_features[i].shape
                    h = w = int(l ** 0.5)
                    patch_features[i] = patch_features[i].permute(0, 2, 1).reshape(b, c, h, w)

        if return_global:
            return global_features, patch_features
        else:
            return patch_features


def transformer_forward(self, x: torch.Tensor, out_layers: list = [8, 16, 24, 32], attn_mask: Optional[torch.Tensor] = None):
    outputs = []
    for idx, r in enumerate(self.resblocks):
        if self.grad_checkpointing and not torch.jit.is_scripting():
            # TODO: handle kwargs https://github.com/pytorch/pytorch/issues/79887#issuecomment-1161758372
            x = checkpoint(r, x, None, None, attn_mask)
        else:
            x = r(x, attn_mask=attn_mask)
            if (idx + 1) in out_layers:
                outputs.append(x)
    return x, outputs


def vision_transformer_forward(self, x: torch.Tensor, out_layers: list = [8, 16, 24, 32]):
    x = self.conv1(x)  # shape = [*, width, grid, grid]
    x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
    x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

    # class embeddings and positional embeddings
    x = torch.cat([_expand_token(self.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
    # shape = [*, grid ** 2 + 1, width]
    x = x + self.positional_embedding.to(x.dtype)

    x = self.patch_dropout(x)
    x = self.ln_pre(x)

    x = x.permute(1, 0, 2)  # NLD -> LND
    x, patch_tokens = self.transformer(x, out_layers)
    patch_tokens = [patch_tokens[t].permute(1, 0, 2) for t in range(len(patch_tokens))]  # LND -> NLD
    x = x.permute(1, 0, 2)  # LND -> NLD

    if self.attn_pool is not None:
        if self.attn_pool_contrastive is not None:
            # This is untested, WIP pooling that should match paper
            x = self.ln_post(x)  # TBD LN first or separate one after each pool?
            tokens = self.attn_pool(x)
            if self.attn_pool_type == 'parallel':
                pooled = self.attn_pool_contrastive(x)
            else:
                assert self.attn_pool_type == 'cascade'
                pooled = self.attn_pool_contrastive(tokens)
        else:
            # this is the original OpenCLIP CoCa setup, does not match paper
            x = self.attn_pool(x)
            x = self.ln_post(x)
            pooled, tokens = self._global_pool(x)
    elif self.final_ln_after_pool:
        pooled, tokens = self._global_pool(x)
        pooled = self.ln_post(pooled)
    else:
        x = self.ln_post(x)
        pooled, tokens = self._global_pool(x)

    if self.proj is not None:
        pooled = pooled @ self.proj

    if self.output_tokens:
        return pooled, tokens
    
    return pooled, patch_tokens


def clip_encode_image(self, image, out_layers: list = [8, 16, 24, 32], normalize: bool = False):
    features, patch_tokens = self.visual(image, out_layers)
    return F.normalize(features, dim=-1) if normalize else features, patch_tokens


Transformer.forward = transformer_forward
VisionTransformer.forward = vision_transformer_forward
CLIP.encode_image = clip_encode_image