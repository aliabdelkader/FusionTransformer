import torch
import torch.nn as nn
from FusionTransformer.models.image_models_stn import Net2DSeg
from FusionTransformer.models.image_models_billinear import Net2DBillinear

from typing import Dict

class ImageSeg(nn.Module):
    def __init__(self, num_classes, dual_head, backbone_2d_kwargs):
        super(ImageSeg, self).__init__()
        self.image_backbone = Net2DSeg(
                 num_classes=num_classes,
                 dual_head=dual_head,
                 backbone_2d_kwargs=backbone_2d_kwargs)
    
    def forward(self, data_dict):
        preds_image = self.image_backbone(data_dict["img"], data_dict["img_indices"])
        out = {
            'img_seg_logit': preds_image["img_seg_logit"]
        }
        return out

class ImageSegBilinear(nn.Module):
    def __init__(self, num_classes, dual_head, backbone_2d_kwargs):
        super(ImageSegBilinear, self).__init__()
        self.image_backbone = Net2DBillinear(
                 num_classes=num_classes,
                 dual_head=dual_head,
                 backbone_2d_kwargs=backbone_2d_kwargs)
    
    def forward(self, data_dict):
        preds_image = self.image_backbone(data_dict["img"], data_dict["img_indices"])
        out = {
            'img_seg_logit': preds_image["img_seg_logit"]
        }
        return out
