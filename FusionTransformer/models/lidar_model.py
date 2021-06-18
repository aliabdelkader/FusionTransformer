import torch
import torch.nn as nn

from FusionTransformer.models.spvcnn import SPVCNN

class LidarSeg(nn.Module):
    def __init__(self, num_classes, backbone_3d_kwargs):
        super(LidarSeg, self).__init__()

        self.backbone = SPVCNN(**backbone_3d_kwargs)
        
        # segmentation head
        self.linear = nn.Linear(self.backbone.cs[-1], num_classes)

        
    def forward(self, data_dict):
        feats = self.backbone(data_dict["lidar"])
        x = self.linear(feats)

        preds = {
            'lidar_feats': feats,
            'lidar_seg_logit': x,
        }

        return preds