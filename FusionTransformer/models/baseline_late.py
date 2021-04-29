import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from FusionTransformer.models.spvcnn import SPVCNN
from torchsparse.sparse_tensor import SparseTensor
import numpy as np
class STN(nn.Module):
    def __init__(self, in_channels):
        super(STN, self).__init__()
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

        grid = F.affine_grid(theta, (B, *output_shape))
        x = F.grid_sample(x, grid)
        return x
    

class Net2DSeg(nn.Module):
    def __init__(self,
                 num_classes,
                 dual_head,
                 backbone_2d_kwargs=dict()):
        super(Net2DSeg, self).__init__()

        self.registered_hook_output = None
        self.feat_channels = 96

        self.backbone = timm.create_model("vit_deit_base_patch16_384", pretrained=False)
        self.backbone.reset_classifier(0)

        self.backbone_block_number = backbone_2d_kwargs["block_number"]
        block_layer_name = f"blocks.{self.backbone_block_number}.mlp.drop"
        self.register_hook_in_layer(self.backbone, block_layer_name)
        self.up = nn.ConvTranspose2d(768, self.feat_channels, kernel_size=(16, 16), stride=(16, 16))

        self.stn_down = STN(in_channels=3)
        self.stn_up = STN(in_channels=self.feat_channels)
        # segmentation head
        self.linear = nn.Linear(self.feat_channels, num_classes)

        # 2nd segmentation head
        self.dual_head = dual_head
        if dual_head:
            self.linear2 = nn.Linear(self.feat_channels, num_classes)
    def hook_fn(self, module, input, output):
        self.registered_hook_output = output
    
    def get_module_with_name(self, model, layer_name):
        for name, module in model.named_modules():
                    if name == layer_name:
                        return module

    def register_hook_in_layer(self, model, layer_name):
            module = self.get_module_with_name(model=model, layer_name=layer_name)
            assert module is not None, f"not module found in model with name {layer_name}"
            module.register_forward_hook(self.hook_fn)


    def forward(self, img, img_indices):
        # 2D network
        B, C, H, W = img.shape
        # sptial attention to convert shape into EMBED_DIM, 384, 384
        x = self.stn_down(img, (self.feat_channels, 384, 384))

        # we do not care about transformer classification decisions, just features
        _ = self.backbone(x)

        # registered_hook_output: features from transformer
        B, N, EMBED_DIM = self.registered_hook_output.shape
        
        # remove features from class token
        x = self.registered_hook_output[:, 1:, :]
        # reshape so that deconvolution can be performed
        x= x.transpose(1, 2).reshape(B, EMBED_DIM, 384//16, 384//16)
        x = self.up(x) 
        x = self.stn_up(x, (96, H, W)) # shape B, C, H, 
        
        # # 2D-3D feature lifting
        img_feats = []
        # print("*************************", img_indices.F.shape)
        for i in range(x.shape[0]):
            img_indices_i = img_indices[i]
            img_feats.append(
                x.permute(0, 2, 3, 1)[i][img_indices_i[:, 0], img_indices_i[:, 1]]
            )
        
        img_feats = torch.cat(img_feats, 0)

        # linear
        x = self.linear(img_feats)

        preds = {
            'img_feats': img_feats,
            'img_seg_logit': x,
        }

        if self.dual_head:
            preds['img_seg_logit2'] = self.linear2(img_feats)

        return preds


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
