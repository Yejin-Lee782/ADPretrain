import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import HASH_REGEX, download_url_to_file, urlparse
from .dinov2.models import vision_transformer as vision_transformer_dinov2


_WEIGHTS_DIR = "./models/dinov2/weights"
os.makedirs(_WEIGHTS_DIR, exist_ok=True)


class DinoModel(nn.Module):

    def __init__(self, name='dinov2-base', device='cuda:0'):
        super(DinoModel, self).__init__()

        self.name = name
        if name == 'dinov2-base':
            full_name = 'dinov2_vit_base_14'
            encoder = load(full_name)
        elif name == 'dinov2-large':
            full_name = 'dinov2_vit_large_14'
            encoder = load(full_name)
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
        if self.name == 'dinov2-base':
            return [768, 768, 768, 768]
        if self.name == 'dinov2-large':
            return [1024, 1024, 1024, 1024]
    
    def encode_image_from_tensors(self, image_tensors, return_global=False, shape='img'):
        with torch.no_grad():
            if self.name == 'dinov2-base':
                patch_features = self.encode_image(image_tensors, [3, 6, 9, 12])
            if self.name == 'dinov2-large':
                patch_features = self.encode_image(image_tensors, [6, 12, 18, 24])
            for i in range(len(patch_features)):
                if shape == 'img':  # convert sequence to image shape
                    b, l, c = patch_features[i].shape
                    h = w = int(l ** 0.5)
                    patch_features[i] = patch_features[i].permute(0, 2, 1).reshape(b, c, h, w)

        if return_global:
            return None, patch_features
        else:
            return patch_features
    
    def encode_image(self, x, target_layers):
        x = self.visual_encoder.prepare_tokens(x)
        outs = []
        for i, blk in enumerate(self.visual_encoder.blocks):
            i = i + 1
            if i <= target_layers[-1]:
                x = blk(x)
            else:
                continue
            if i in target_layers:
                outs.append(x)
        outs = [e[:, 1 + self.visual_encoder.num_register_tokens:, :] for e in outs]
        
        return outs


def load(name):
    arch, patchsize = name.split("_")[-2], name.split("_")[-1]
    if "v2" in name:
        if "reg" in name:
            model = vision_transformer_dinov2.__dict__[f'vit_{arch}'](patch_size=int(patchsize), img_size=518,
                                                                        block_chunks=0, init_values=1e-8,
                                                                        num_register_tokens=4,
                                                                        interpolate_antialias=False,
                                                                        interpolate_offset=0.1)

            if arch == "base":
                ckpt_pth = download_cached_file(
                    f"https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb{patchsize}/dinov2_vitb{patchsize}_reg4_pretrain.pth")
            elif arch == "small":
                ckpt_pth = download_cached_file(
                    f"https://dl.fbaipublicfiles.com/dinov2/dinov2_vits{patchsize}/dinov2_vits{patchsize}_reg4_pretrain.pth")
            elif arch == "large":
                ckpt_pth = download_cached_file(
                    f"https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl{patchsize}/dinov2_vitl{patchsize}_reg4_pretrain.pth")
            else:
                raise ValueError("Invalid type of architecture. It must be either 'small' or 'base' or 'large.")
        else:
            model = vision_transformer_dinov2.__dict__[f'vit_{arch}'](patch_size=int(patchsize), img_size=518,
                                                                        block_chunks=0, init_values=1e-8,
                                                                        interpolate_antialias=False,
                                                                        interpolate_offset=0.1)

            if arch == "base":
                ckpt_pth = download_cached_file(
                    f"https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb{patchsize}/dinov2_vitb{patchsize}_pretrain.pth")
            elif arch == "small":
                ckpt_pth = download_cached_file(
                    f"https://dl.fbaipublicfiles.com/dinov2/dinov2_vits{patchsize}/dinov2_vits{patchsize}_pretrain.pth")
            elif arch == "large":
                ckpt_pth = download_cached_file(
                    f"https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl{patchsize}/dinov2_vitl{patchsize}_pretrain.pth")
            else:
                raise ValueError("Invalid type of architecture. It must be either 'small' or 'base'.")

        state_dict = torch.load(ckpt_pth, map_location='cpu')
    else:
        raise ValueError(f"{name} is currently not supported!")

    model.load_state_dict(state_dict, strict=False)
    return model


def download_cached_file(url, check_hash=True, progress=True):
    """
    Mostly copy-paste from timm library.
    (https://github.com/rwightman/pytorch-image-models/blob/29fda20e6d428bf636090ab207bbcf60617570ca/timm/models/_hub.py#L54)
    """
    if isinstance(url, (list, tuple)):
        url, filename = url
    else:
        parts = urlparse(url)
        filename = os.path.basename(parts.path)
    cached_file = os.path.join(_WEIGHTS_DIR, filename)
    if not os.path.exists(cached_file):
        print('Downloading: "{}" to {}\n'.format(url, cached_file))
        hash_prefix = None
        if check_hash:
            r = HASH_REGEX.search(filename)  # r is Optional[Match[str]]
            hash_prefix = r.group(1) if r else None
        download_url_to_file(url, cached_file, hash_prefix, progress=progress)
    return cached_file


def convert_key(ckpt_pth):
    ckpt = torch.load(ckpt_pth, map_location="cpu")
    state_dict = ckpt['state_dict']
    new_state_dict = dict()

    for k, v in state_dict.items():
        if k.startswith('module.base_encoder.'):
            new_state_dict[k[len("module.base_encoder."):]] = v

    return new_state_dict