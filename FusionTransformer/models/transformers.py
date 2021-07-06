import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from timm.models.helpers import overlay_external_default_cfg
from timm.models.vision_transformer import VisionTransformer, default_cfgs, build_model_with_cfg, checkpoint_filter_fn
from timm.models.registry import register_model
from copy import deepcopy

class Image2DTransformer(VisionTransformer):
    def __init__(self, **kwargs):
      super(Image2DTransformer, self).__init__(**kwargs)

    def forward_blocks(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        copied from timm Library

        function runs input image and returns outputs of transformer blocks 
        
        Args:
            x: tensor of input image shape (B, C, H, W)
        
        Returns
            dictionary map block number -> output of the block
        """
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        outputs = Dict()
        for i, block in enumerate(self.blocks):
            x = block(x)
            outputs[str(i)] = x
        assert torch.not_equal(outputs['0'], outputs['11'])
        return outputs


def _create_transformer_2d(variant, pretrained=False, default_cfg=None, **kwargs):
    """ copied from timm library """
    if default_cfg is None:
        default_cfg = deepcopy(default_cfgs[variant])
    overlay_external_default_cfg(default_cfg, kwargs)
    default_num_classes = default_cfg['num_classes']
    default_img_size = default_cfg['input_size'][-2:]

    num_classes = kwargs.pop('num_classes', default_num_classes)
    img_size = kwargs.pop('img_size', default_img_size)
    repr_size = kwargs.pop('representation_size', None)
    if repr_size is not None and num_classes != default_num_classes:
        # Remove representation layer if fine-tuning. This may not always be the desired action,
        # but I feel better than doing nothing by default for fine-tuning. Perhaps a better interface?
        print("Removing representation layer for fine-tuning.")
        repr_size = None

    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    model = build_model_with_cfg(
        Image2DTransformer, variant, pretrained,
        default_cfg=default_cfg,
        img_size=img_size,
        num_classes=num_classes,
        representation_size=repr_size,
        pretrained_filter_fn=checkpoint_filter_fn,
        **kwargs)

    return model

@register_model
def image_2d_transformer(pretrained=False, **kwargs):
    """
    copied from timm 
    DeiT base model @ 384x384 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_transformer_2d('vit_deit_base_patch16_384', pretrained=pretrained, **model_kwargs)
    return model


class SpatialTransformer(nn.Module):
    def __init__(self, in_channels):
        super(SpatialTransformer, self).__init__()
        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10 * 3 * 3, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )
        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
    
    def forward(self, x, output_shape):
        B, C, H, W = x.shape
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, (B, *output_shape),align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)
        return x

class ScaleUpModule(nn.Module):
    def __init__(self, input_features: int, output_features: int, kernel_size: int, stride: int):
        """
        Module that scales up image using spatial transformer

        Args
            input_features: number of input features channels
            output_features: number of hidden features channels
            kernel_size: filter size 
            stride: stride of transpose convolution
        
        """
        super(ScaleUpModule, self).__init__()
        self.up_conv = nn.ConvTranspose2d(input_features, output_features, kernel_size=(kernel_size, kernel_size), stride=(stride, stride))
        self.up_stn = SpatialTransformer(in_channels=output_features)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up_conv(x)
        x = self.up_stn(x)
        return x
