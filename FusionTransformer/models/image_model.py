import torch.nn as nn
from FusionTransformer.models.transformer2D import Net2DSeg

class ImageSeg(nn.Module):
    def __init__(self, num_classes, dual_head, backbone_2d_kwargs):
        super(ImageSeg, self).__init__()
        self.image_backbone = Net2DSeg(
                 num_classes=num_classes,
                 dual_head=dual_head,
                 backbone_2d_kwargs=backbone_2d_kwargs)
    
    def forward(self, data_dict):
        preds_image = self.image_backbone(data_dict["img"], data_dict["img_indices"])
        return preds_image
