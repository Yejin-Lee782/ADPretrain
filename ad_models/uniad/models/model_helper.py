import copy
import importlib

import torch
import torch.nn as nn
# from utils.misc_helper import to_device


class ModelHelper(nn.Module):
    """Build model from cfg"""

    def __init__(self, cfg, encoder):
        super(ModelHelper, self).__init__()

        cfg = cfg[0]
        mname = cfg["name"]
        kwargs = cfg["kwargs"]
        mtype = cfg["type"]
        
        if encoder.name == 'clip-base':
            kwargs["inplanes"] = [768 * 4]
            kwargs["instrides"] = [16]
            kwargs['feature_size'] = [14, 14]
        elif encoder.name == 'dinov2-base':
            kwargs["inplanes"] = [768 * 4]
            kwargs["instrides"] = [14]
            kwargs['feature_size'] = [16, 16]
        elif encoder.name == 'clip-large' or encoder.name == 'dinov2-large':
            kwargs["inplanes"] = [1024 * 4]
            kwargs["instrides"] = [14]
            kwargs['feature_size'] = [16, 16]
        elif encoder.name == 'imagebind':
            kwargs["inplanes"] = [1280 * 4]
            kwargs["instrides"] = [14]
            kwargs['feature_size'] = [16, 16]
        else:
            raise ValueError("Unrecognized Encoder!") 

        module = self.build(mtype, kwargs)
        self.add_module(mname, module)

    def build(self, mtype, kwargs):
        module_name, cls_name = mtype.rsplit(".", 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, cls_name)
        return cls(**kwargs)

    def cuda(self):
        self.device = torch.device("cuda")
        return super(ModelHelper, self).cuda()

    def cpu(self):
        self.device = torch.device("cpu")
        return super(ModelHelper, self).cpu()

    def forward(self, input):
        for submodule in self.children():
            output = submodule(input)
            input.update(output)
        return input

    def freeze_layer(self, module):
        module.eval()
        for param in module.parameters():
            param.requires_grad = False

    def train(self, mode=True):
        """
        Sets the module in training mode.
        This has any effect only on modules such as Dropout or BatchNorm.

        Returns:
            Module: self
        """
        self.training = mode
        for mname, module in self.named_children():
            module.train(mode)
        return self
