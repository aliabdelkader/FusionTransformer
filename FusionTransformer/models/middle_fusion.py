import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from FusionTransformer.models.spvcnn import SPVCNN
import torchsparse
from FusionTransformer.models.utils import point_to_voxel, voxel_to_point, initial_voxelize
from torchsparse.sparse_tensor import SparseTensor
from torchsparse.point_tensor import PointTensor
from FusionTransformer.models.STN import STN

import numpy as np

class Net2DSeg(nn.Module):
    def __init__(self,
                 num_classes,
                 dual_head,
                 backbone_2d_kwargs=dict()):
        super(Net2DSeg, self).__init__()

        self.registered_hook_output = []
        self.feat_channels = 96
        self.hidden_channels = 768

        self.backbone = timm.create_model("vit_deit_base_patch16_384", pretrained=True)
        self.backbone.reset_classifier(0)

        self.backbone_block_numbers = backbone_2d_kwargs["block_number"]

        self.up = nn.ModuleList()
        self.stn_up = nn.ModuleList()
        for block_number in self.backbone_block_numbers:
            block_layer_name = f"blocks.{block_number}.mlp.drop"
            self.register_hook_in_layer(self.backbone, block_layer_name)
            self.up.append(nn.ConvTranspose2d(self.hidden_channels, self.feat_channels, kernel_size=(16, 16), stride=(16, 16)))
            self.stn_up.append(STN(in_channels=self.feat_channels))

        self.stn_down = STN(in_channels=3)

        # segmentation head
        self.linear = nn.Linear(self.feat_channels, num_classes)

        # 2nd segmentation head
        self.dual_head = dual_head
        if dual_head:
            self.linear2 = nn.Linear(self.feat_channels, num_classes)

    def hook_fn(self, module, input, output):
        B, _, C = output.shape
        if C == self.hidden_channels:
            self.registered_hook_output.append(output)
    
    def get_module_with_name(self, model, layer_name):
        for name, module in model.named_modules():
                    if name == layer_name:
                        return module

    def register_hook_in_layer(self, model, layer_name):
        module = self.get_module_with_name(model=model, layer_name=layer_name)
        assert module is not None, f"not module found in model with name {layer_name}"
        module.register_forward_hook(self.hook_fn)

    def get_img_feats(self, img_indices, idx, image_shape):
        B, C, H, W  = image_shape

        # registered_hook_output: features from transformer
        B, N, EMBED_DIM = self.registered_hook_output[idx].shape
        
        # remove features from class token
        x = self.registered_hook_output[idx][:, 1:, :]
        # reshape so that deconvolution can be performed
        x= x.transpose(1, 2).reshape(B, EMBED_DIM, 384//16, 384//16)
        x = self.up[idx](x) 
        x = self.stn_up[idx](x, (self.feat_channels, H, W)) # shape B, C, H, 
        
        # # 2D-3D feature lifting
        img_feats = []
        # print("*************************", img_indices.F.shape)
        for i in range(x.shape[0]):
            img_indices_i = img_indices[i]
            img_feats.append(
                x.permute(0, 2, 3, 1)[i][img_indices_i[:, 0], img_indices_i[:, 1]]
            )
        
        img_feats = torch.cat(img_feats, 0)

        return img_feats, x

    def forward(self, img, img_indices):

        self.registered_hook_output = []
        # 2D network
        # sptial attention to convert shape into EMBED_DIM, 384, 384
        x = self.stn_down(img, (self.feat_channels, 384, 384))

        # we do not care about transformer classification decisions, just features
        _ = self.backbone(x)

        middle_feats, _ = self.get_img_feats(img_indices=img_indices, idx=0, image_shape=img.shape)
        late_feats, x = self.get_img_feats(img_indices=img_indices, idx=len(self.registered_hook_output)-1, image_shape=img.shape)

        # linear
        x = self.linear(late_feats)

        preds = {
            'img_feats': late_feats,
            'img_seg_logit': x,
            'img_middle_feats': middle_feats
        }

        if self.dual_head:
            preds['img_seg_logit2'] = self.linear2(late_feats)

        return preds


class Net3DSeg(SPVCNN):
    def __init__(self,
                 num_classes,
                 dual_head,
                 backbone_3d_kwargs=dict()
                 ):
        super(Net3DSeg, self).__init__(**backbone_3d_kwargs)

        self.middle_fusion_transform = nn.Sequential(
                nn.Linear(96, self.cs[4]),
                nn.BatchNorm1d(self.cs[4]),
                nn.ReLU(True),
        )
        # segmentation head
        self.linear = nn.Linear(self.cs[-1], num_classes)

        # 2nd segmentation head
        self.dual_head = dual_head
        if dual_head:
            self.linear2 = nn.Linear(self.cs[-1], num_classes)


    def backbone_forward_pass(self, x, img_middle_feats):
        # x: SparseTensor z: PointTensor
        z = PointTensor(x.F, x.C.float())

        x0 = initial_voxelize(z, self.pres, self.vres)

        x0 = self.stem(x0)
        z0 = voxel_to_point(x0, z, nearest=False)
        z0.F = z0.F

        x1 = point_to_voxel(x0, z0)
        x1 = self.stage1(x1)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        z1 = voxel_to_point(x4, z0)
        z1.F = z1.F + self.point_transforms[0](z0.F) + self.middle_fusion_transform(img_middle_feats)

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
        
    def forward(self, x, img_middle_feats):
        # x = data_dict["lidar"]
        # feats = self.backbone(x)
        feats = self.backbone_forward_pass(x, img_middle_feats)
        x = self.linear(feats)

        preds = {
            'lidar_feats': feats,
            'lidar_seg_logit': x,
        }

        if self.dual_head:
            preds['lidar_seg_logit2'] = self.linear2(feats)

        return preds

class MiddleFusionTransformer(nn.Module):
    def __init__(self, num_class, dual_head, backbone_3d_kwargs, backbone_2d_kwargs):
        super(MiddleFusionTransformer, self).__init__()
        self.lidar_backbone = Net3DSeg(num_classes=num_class, dual_head=dual_head, backbone_3d_kwargs=backbone_3d_kwargs)
        self.image_backbone = Net2DSeg(
                 num_classes=num_class,
                 dual_head=dual_head,
                 backbone_2d_kwargs=backbone_2d_kwargs)
    
    def forward(self, data_dict):
        preds_image = self.image_backbone(img=data_dict["img"], img_indices=data_dict["img_indices"])
        preds_lidar = self.lidar_backbone(x=data_dict["lidar"], img_middle_feats=preds_image["img_middle_feats"])
        out = {**preds_image, **preds_lidar}
        return out

# if __name__ == '__main__':
    # test_Net2DSeg()
    # test_LateFusion()
