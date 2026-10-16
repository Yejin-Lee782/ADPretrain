import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .ImageBind import *
from torch.hub import HASH_REGEX, download_url_to_file, urlparse


_WEIGHTS_DIR = "./models/ImageBind/weights"
os.makedirs(_WEIGHTS_DIR, exist_ok=True)

class ImageBindModel(nn.Module):

    def __init__(self, name='imagebind', device='cuda:0'):
        super(ImageBindModel, self).__init__()

        self.name = name
        if not os.path.exists(os.path.join(_WEIGHTS_DIR, "imagebind_huge.pth")):
            download_cached_file("https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth")
            
        ckpt_path = os.path.join(_WEIGHTS_DIR, "imagebind_huge.pth")
        print(f'Initializing visual encoder from {ckpt_path} ...')
        self.visual_encoder, self.visual_hidden_size = imagebind_model.imagebind_huge({})
        imagebind_ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
        self.visual_encoder.load_state_dict(imagebind_ckpt, strict=False)
        # free vision encoder
        for name, param in self.visual_encoder.named_parameters():
            param.requires_grad = False
        self.visual_encoder.eval()
        print('Visual encoder initialized.')

        self.device = torch.device(device)

    @property
    def feature_dimensions(self):
        return [1280, 1280, 1280, 1280]
    
    def encode_image_from_tensors(self, image_tensors, return_global=False, shape='img'):
        inputs = {ModalityType.VISION: image_tensors}
        # convert into visual dtype
        inputs = {key: inputs[key] for key in inputs}
        with torch.no_grad():
            embeddings = self.visual_encoder(inputs)
            patch_features = embeddings['vision'][1] # bsz x h*w x 1280
            for i in range(len(patch_features)):
                patch_features[i] = patch_features[i].transpose(0, 1)[:, 1:, :]
                if shape == 'img':  # convert sequence to image shape
                    b, l, c = patch_features[i].shape
                    h = w = int(l ** 0.5)
                    patch_features[i] = patch_features[i].permute(0, 2, 1).reshape(b, c, h, w)

        if return_global:
            global_features = embeddings['vision'][0]
            return global_features, patch_features
        else:
            return patch_features


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