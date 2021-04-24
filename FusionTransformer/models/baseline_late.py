import torch
import torch.nn as nn

import timm
from FusionTransformer.models.spvcnn import SPVCNN
from torchsparse.sparse_tensor import SparseTensor

class Net2DSeg(nn.Module):
    def __init__(self,
                 num_classes,
                 dual_head,
                 backbone_2d_kwargs=dict()):
        super(Net2DSeg, self).__init__()

        self.registered_hook_output = None
        feat_channels = 96

        self.backbone = timm.create_model("vit_deit_base_distilled_patch16_224", pretrained=False)
        self.backbone.reset_classifier(0)

        self.backbone_block_number = backbone_2d_kwargs["block_number"]
        block_layer_name = f"blocks.{self.backbone_block_number}.mlp.drop"
        self.register_hook_in_layer(self.backbone, block_layer_name)
        self.up = nn.ConvTranspose2d(768, feat_channels, kernel_size=(16, 16), stride=(16, 16))

        # segmentation head
        self.linear = nn.Linear(feat_channels, num_classes)

        # 2nd segmentation head
        self.dual_head = dual_head
        if dual_head:
            self.linear2 = nn.Linear(feat_channels, num_classes)
    
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
        # (batch_size, 3, H, W)
        # img = data_batch['img']
        # img_indices = data_batch['img_indices']

        # 2D network
        # we do not care about transformer classification decisions
        _ = self.backbone(img)
        # registered_hook_output: features from transformer
        B, N, EMBED = self.registered_hook_output.shape
        # remove features from class token, distilation token
        x = self.registered_hook_output[:, 2:, :]
        # reshape so that deconvolution can restore img shae
        x= x.transpose(1, 2).reshape(B, EMBED, 14, 14)
        x = self.up(x) # shape C, H, W

        # # 2D-3D feature lifting
        img_feats = []
        for i in range(x.shape[0]):
            img_feats.append(x.permute(0, 2, 3, 1)[i][img_indices[i][:, 0], img_indices[i][:, 1]])
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
    batch_size = 2
    img_width = 224
    img_height = 224

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
    batch_size = 2
    img_width = 224
    img_height = 224

    # 3D
    num_coords = 2000

    # 2D
    img = torch.rand(batch_size, 3, img_height, img_width)
    u = torch.randint(high=img_height, size=(batch_size, num_coords // batch_size, 1))
    v = torch.randint(high=img_width, size=(batch_size, num_coords // batch_size, 1))
    img_indices = torch.cat([u, v], 2)

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
    import pdb; pdb.set_trace()
    # test_Net2DSeg()
    test_LateFusion()
