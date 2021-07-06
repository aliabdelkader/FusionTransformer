import torch.nn as nn
from FusionTransformer.models.spvcnn import SPVCNN
from FusionTransformer.models.image_models import Net2DSeg

class Net3DSeg(nn.Module):
    def __init__(self,
                 num_classes,
                 dual_head,
                 backbone_3d_kwargs=dict()
                 ):
        super(Net3DSeg, self).__init__()

        self.backbone = SPVCNN(**backbone_3d_kwargs)
        # segmentation head
        self.linear = nn.Linear(self.backbone.cs[-1], num_classes)

        # 2nd segmentation head
        self.dual_head = dual_head
        if dual_head:
            self.linear2 = nn.Linear(self.backbone.cs[-1], num_classes)

    def forward(self, x):
        # x = data_dict["lidar"]
        feats = self.backbone(x)
        x = self.linear(feats)

        preds = {
            'lidar_feats': feats,
            'lidar_seg_logit': x,
        }

        if self.dual_head:
            preds['lidar_seg_logit2'] = self.linear2(feats)

        return preds

class LateFusionTransformer(nn.Module):
    def __init__(self, num_class, dual_head, backbone_3d_kwargs, backbone_2d_kwargs):
        super(LateFusionTransformer, self).__init__()
        self.lidar_backbone = Net3DSeg(num_classes=num_class, dual_head=dual_head, backbone_3d_kwargs=backbone_3d_kwargs)
        self.image_backbone = Net2DSeg(
                 num_classes=num_class,
                 dual_head=dual_head,
                 backbone_2d_kwargs=backbone_2d_kwargs)
    
    def forward(self, data_dict):
        preds_image = self.image_backbone(data_dict["img"], data_dict["img_indices"])
        preds_lidar = self.lidar_backbone(data_dict["lidar"])
        out = {**preds_image, **preds_lidar}
        return out