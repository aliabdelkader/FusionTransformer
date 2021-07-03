import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from FusionTransformer.models.STN import STN
from typing import Dict

from timm.models.vision_transformer import VisionTransformer, default_cfgs, build_model_with_cfg, checkpoint_filter_fn
from timm.models.registry import register_model


class Image2DTransformer(VisionTransformer):
    def __init__(self, **kwargs):
      super(Image2DTransformer, self).__init__(**kwargs)

    def forward_blocks(self, x) -> Dict:
        # copied from timm
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        outputs = {str(i): block(x) for i, block in enumerate(self.blocks)}
        return outputs

def _create_transformer_2d(variant, pretrained=False, default_cfg=None, **kwargs):
    default_cfg = default_cfg or default_cfgs[variant]
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    # NOTE this extra code to support handling of repr size for in21k pretrained models
    default_num_classes = default_cfg['num_classes']
    num_classes = kwargs.get('num_classes', default_num_classes)
    repr_size = kwargs.pop('representation_size', None)
    if repr_size is not None and num_classes != default_num_classes:
        # Remove representation layer if fine-tuning. This may not always be the desired action,
        # but I feel better than doing nothing by default for fine-tuning. Perhaps a better interface?
        _logger.warning("Removing representation layer for fine-tuning.")
        repr_size = None

    model = build_model_with_cfg(
        Image2DTransformer, variant, pretrained,
        default_cfg=default_cfg,
        representation_size=repr_size,
        pretrained_filter_fn=checkpoint_filter_fn,
        pretrained_custom_load='npz' in default_cfg['url'],
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


class Net2DSeg(nn.Module):
    def __init__(self,
                 num_classes,
                 dual_head,
                 backbone_2d_kwargs=dict()):
        super(Net2DSeg, self).__init__()

        self.registered_hook_output = []
        self.feat_channels = 96
        self.hidden_channels = 768

        self.backbone = timm.create_model("image_2d_transformer", pretrained=True)
        self.backbone.reset_classifier(0, pooling='')

        self.middle_feat_block_number = backbone_2d_kwargs["middle_feat_block_number"]
        self.late_feat_block_number = backbone_2d_kwargs["late_feat_block_number"]

        self.up = nn.ModuleDict()
        self.stn_up = nn.ModuleDict()
        
        if self.middle_feat_block_number:
            self.up[str(self.middle_feat_block_number)] = nn.ConvTranspose2d(self.hidden_channels, self.feat_channels, kernel_size=(16, 16), stride=(16, 16))
            self.stn_up[str(self.middle_feat_block_number)] = STN(in_channels=self.feat_channels)


        self.up[str(self.late_feat_block_number)] = nn.ConvTranspose2d(self.hidden_channels, self.feat_channels, kernel_size=(16, 16), stride=(16, 16))
        self.stn_up[str(self.late_feat_block_number)] = STN(in_channels=self.feat_channels)

        self.stn_down = STN(in_channels=3)

        # segmentation head
        self.linear = nn.Linear(self.feat_channels, num_classes)

        # 2nd segmentation head
        self.dual_head = dual_head
        if dual_head:
            self.linear2 = nn.Linear(self.feat_channels, num_classes)

    def get_img_feats(self, img_indices, block_id, image_shape, backbone_output):
        B, C, H, W  = image_shape

        # registered_hook_output: features from transformer

        # B, N, EMBED_DIM = self.registered_hook_output[idx].shape
        B, N, EMBED_DIM = backbone_output[block_id].shape

        x = backbone_output[block_id]
        # reshape so that deconvolution can be performed
        x = x.transpose(1, 2).reshape(B, EMBED_DIM, 384//16, 384//16)
        x = self.up[block_id](x) 
        x = self.stn_up[block_id](x, (self.feat_channels, H, W)) # shape B, C, H, 
        
        # # 2D-3D feature lifting
        img_feats = []
        # print("*************************", img_indices.F.shape)
        for i in range(x.shape[0]):
            img_indices_i = img_indices[i]
            img_feats.append(
                x.permute(0, 2, 3, 1)[i][img_indices_i[:, 0], img_indices_i[:, 1]]
            )
        
        img_feats = torch.cat(img_feats, 0)

        return img_feats

    def forward(self, img, img_indices):

        # 2D network
        # sptial attention to convert shape into EMBED_DIM, 384, 384
        x = self.stn_down(img, (self.feat_channels, 384, 384))

        backbone_output = self.backbone.forward_blocks(x)

        late_feats  = self.get_img_feats(img_indices=img_indices, block_id=self.late_feat_block_number, image_shape=img.shape, backbone_output=backbone_output)

        # linear
        x = self.linear(late_feats)

        preds = {
            'img_feats': late_feats,
            'img_seg_logit': x,
        }
        if self.dual_head:
            preds['img_seg_logit2'] = self.linear2(late_feats)

        if self.middle_feat_block_number:
            middle_feats = self.get_img_feats(img_indices=img_indices, block_id=self.middle_feat_block_number, image_shape=img.shape, backbone_output=backbone_output)
            preds['img_middle_feats'] = middle_feats



        return preds
