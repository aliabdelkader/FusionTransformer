import torch.nn as nn
import torchsparse
from torchsparse.point_tensor import PointTensor
from FusionTransformer.models.utils import point_to_voxel, voxel_to_point, initial_voxelize
from FusionTransformer.models.image_models import Net2DSeg
from FusionTransformer.models.spvcnn import SPVCNN


class Net3DSeg(SPVCNN):
    def __init__(self,
                 num_classes,
                 dual_head,
                 backbone_3d_kwargs=dict()
                 ):
        super(Net3DSeg, self).__init__(**backbone_3d_kwargs)

        self.early_fusion_transform = nn.Sequential(
                nn.Linear(96, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(True),
        )
        # segmentation head
        self.linear = nn.Linear(self.cs[-1], num_classes)

        # 2nd segmentation head
        self.dual_head = dual_head
        if dual_head:
            self.linear2 = nn.Linear(self.cs[-1], num_classes)


    def backbone_forward_pass(self, x, img_early_feats):
        # x: SparseTensor z: PointTensor
        z = PointTensor(x.F, x.C.float())

        x0 = initial_voxelize(z, self.pres, self.vres)

        x0 = self.stem(x0)
        z0 = voxel_to_point(x0, z, nearest=False)
        z0.F = z0.F + self.early_fusion_transform(img_early_feats)

        x1 = point_to_voxel(x0, z0)
        x1 = self.stage1(x1)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        z1 = voxel_to_point(x4, z0)
        z1.F = z1.F + self.point_transforms[0](z0.F)

        y1 = point_to_voxel(x4, z1)
        y1.F = self.dropout(y1.F)
        y1 = self.up1[0](y1)
        y1 = torchsparse.cat([y1, x3])
        y1 = self.up1[1](y1)

        y2 = self.up2[0](y1)
        y2 = torchsparse.cat([y2, x2])
        y2 = self.up2[1](y2)
        z2 = voxel_to_point(y2, z1)
        z2.F = z2.F + self.point_transforms[1](z1.F)

        y3 = point_to_voxel(y2, z2)
        y3.F = self.dropout(y3.F)
        y3 = self.up3[0](y3)
        y3 = torchsparse.cat([y3, x1])
        y3 = self.up3[1](y3)

        y4 = self.up4[0](y3)
        y4 = torchsparse.cat([y4, x0])
        y4 = self.up4[1](y4)
        z3 = voxel_to_point(y4, z2)
        z3.F = z3.F + self.point_transforms[2](z2.F)

        return z3.F
        
    def forward(self, x, img_early_feats):
        feats = self.backbone_forward_pass(x, img_early_feats)
        x = self.linear(feats)

        preds = {
            'lidar_feats': feats,
            'lidar_seg_logit': x,
        }

        if self.dual_head:
            preds['lidar_seg_logit2'] = self.linear2(feats)

        return preds

class EarlyFusionTransformer(nn.Module):
    def __init__(self, num_class, dual_head, backbone_3d_kwargs, backbone_2d_kwargs):
        super(EarlyFusionTransformer, self).__init__()
        self.dual_head = dual_head
        self.lidar_backbone = Net3DSeg(num_classes=num_class, dual_head=dual_head, backbone_3d_kwargs=backbone_3d_kwargs)
        self.image_backbone = Net2DSeg(
                 num_classes=num_class,
                 dual_head=dual_head,
                 backbone_2d_kwargs=backbone_2d_kwargs)
    
    def forward(self, data_dict):
        preds_image = self.image_backbone(img=data_dict["img"], img_indices=data_dict["img_indices"])
        # in early fusion model
        # image features has to be taken taken from block 0 in img transformer 
        # therefore, then preds_image["img_middle_feats"] are actually img_early_feats
        # img_early_feats = preds_image["img_middle_feats"] if  backbone_2d_kwargs["block_number"][0] = 0
        preds_lidar = self.lidar_backbone(x=data_dict["lidar"], img_early_feats=preds_image["img_middle_feats"].detach())
        out = {
            'lidar_seg_logit': preds_lidar['lidar_seg_logit'],
            'img_seg_logit': preds_image["img_seg_logit"]
        }
        if self.dual_head:
           out.update({
               'lidar_seg_logit2': preds_lidar['lidar_seg_logit2'],
               'img_seg_logit2': preds_image["img_seg_logit2"]
           }) 
        return out