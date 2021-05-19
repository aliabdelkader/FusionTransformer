import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from FusionTransformer.models.spvcnn import SPVCNN
from torchsparse.sparse_tensor import SparseTensor
from FusionTransformer.models.STN import STN
from FusionTransformer.models.transformer2D import Net2DSeg

import numpy as np

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

def test_Net2DSeg():
    # 2D
    batch_size = 1
    img_width = 1200
    img_height = 384

    # 3D
    num_coords = 2000
    num_classes = 11

    # 2D
    img = torch.rand(batch_size, 3, img_height, img_width)
    u = torch.randint(high=img_height, size=(batch_size, num_coords // batch_size, 1))
    v = torch.randint(high=img_width, size=(batch_size, num_coords // batch_size, 1))
    img_indices = torch.cat([u, v], 2)

    # to cuda
    img = img.cuda()
    img_indices = img_indices.cuda()

    net_2d = Net2DSeg(num_classes,
                      dual_head=True,
                      backbone_2d_kwargs={"block_number": 11})

    net_2d.cuda()
    out_dict = net_2d({
        'img': img,
        'img_indices': img_indices,
    })
    for k, v in out_dict.items():
        print('Net2DSeg:', k, v.shape)


def test_Net3DSeg():
    in_channels = 4
    num_coords = 2000
    full_scale = 4096
    num_seg_classes = 11

    coords = torch.randint(high=full_scale, size=(num_coords, 3))
    feats = torch.rand(num_coords, in_channels)
    x  = SparseTensor(feats=feats, coords=coords).cuda()


    net_3d = Net3DSeg(num_seg_classes,
                      dual_head=True,
                      backbone_3d_kwargs={'pres': 1, 'vres': 1})

    net_3d.cuda()
    out_dict = net_3d(x)
    for k, v in out_dict.items():
        print('Net3DSeg:', k, v.shape)


def test_LateFusion():
    # 2D
    batch_size = 1
    img_width = 1200
    img_height = 384

    # 3D
    num_coords = 2000

    # 2D
    img = torch.rand(batch_size, 3, img_height, img_width)
    u = torch.randint(high=img_height, size=(batch_size, num_coords // batch_size, 1))
    v = torch.randint(high=img_width, size=(batch_size, num_coords // batch_size, 1))
    img_indices = torch.cat([u, v], 2)
    print(img_indices.shape)
    # to cuda
    img = img.cuda()

    in_channels = 4
    num_coords = 2000
    full_scale = 4096
    num_seg_classes = 11

    coords = torch.randint(high=full_scale, size=(num_coords, 3))
    feats = torch.rand(num_coords, in_channels)
    x  = SparseTensor(feats=feats, coords=coords).cuda()

    model = LateFusionTransformer(
        num_class=num_seg_classes,
        dual_head=True,
        backbone_2d_kwargs={"block_number": 11},
        backbone_3d_kwargs={'pres': 1, 'vres': 1}
    )

    model.cuda()
    out_dict = model({"img": img, "img_indices": img_indices, "lidar": x})

    for k, v in out_dict.items():
        print('Fusion:', k, v.shape)

if __name__ == '__main__':
    # test_Net2DSeg()
    test_LateFusion()
