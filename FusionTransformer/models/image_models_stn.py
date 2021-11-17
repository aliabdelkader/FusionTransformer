import torch
import torch.nn as nn
from FusionTransformer.models.transformers import SpatialTransformer, ScaleUpModule, image_2d_distilled_transformer
import timm
from typing import Dict

class Net2DSeg(nn.Module):
    def __init__(self,
                 num_classes: int,
                 dual_head: bool,
                 backbone_2d_kwargs=dict()):
        """
        constructor for image segmentation model

        Args
            num_classes: number of output classes
            dual_head: dual head model for KL divergance loss
            backbone_2d_kwargs: arguments for transformer backbone

        """
        super(Net2DSeg, self).__init__()

        # spatial transformer to scale down input image to the shape expected by the vision transformer
        self.stn_down = SpatialTransformer(in_channels=3)

        # number of features channels input to vision transformer
        self.feat_channels = 96

        # number of features channels output from  vision transformer
        self.hidden_channels = 768

        # create vision transformer
        self.backbone = timm.create_model("image_2d_distilled_transformer", pretrained=True, remove_tokens_outputs=True)
        self.backbone.reset_classifier(0, '')

        # vision transformer blocks to select the output of
        if backbone_2d_kwargs.get("middle_feat_block_number", None) is not None:
            self.middle_feat_block_number = str(backbone_2d_kwargs["middle_feat_block_number"])
        else:
            self.middle_feat_block_number = None
        
        if backbone_2d_kwargs.get("late_feat_block_number", None) is not None:
            self.late_feat_block_number = str(backbone_2d_kwargs["late_feat_block_number"])
        else:
            self.late_feat_block_number = None

        self.up = nn.ModuleDict()
        
        if self.middle_feat_block_number:
            self.up[self.middle_feat_block_number] = ScaleUpModule(input_features=self.hidden_channels, output_features=self.feat_channels, kernel_size=16, stride=16)

        self.up[self.late_feat_block_number] = ScaleUpModule(input_features=self.hidden_channels, output_features=self.feat_channels, kernel_size=16, stride=16)


        # segmentation head
        self.linear = nn.Linear(self.feat_channels, num_classes)

        # 2nd segmentation head
        self.dual_head = dual_head
        if dual_head:
            self.linear2 = nn.Linear(self.feat_channels, num_classes)

    def get_img_feats(self, img_indices, block_id: str, image_shape: tuple, backbone_output: Dict):
        """
        function gets the image transformer output features based img_indices

        Args:
            img_indices: array (B, N, 2) image indices of each lidar point
            block_id: id of transformer block to take output
            image_shape: output shape of image from spatial transformer
            backbone_output: outputs from vision transformer blocks
        
        Returns
            tensor shape ( 1, B*N, EMBED_DIM) features for every lidar point
        """
        B, C, H, W  = image_shape

        B, N, EMBED_DIM = backbone_output[block_id].shape

        x = backbone_output[block_id]

        # # remove class token features
        # x = x[:, 1: , :]

        # reshape so that deconvolution can be performed
        x = x.transpose(1, 2).reshape(B, EMBED_DIM, 384//16, 384//16)
        
        x = self.up[block_id](x, (self.feat_channels, H, W)) # shape B, C, H, 
        
        # # 2D-3D feature lifting
        img_feats = []
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

        # run vision transformer
        backbone_output = self.backbone.forward_blocks(x)

        # features from vision transformer for each lidar point
        late_feats  = self.get_img_feats(img_indices=img_indices, block_id=self.late_feat_block_number, image_shape=img.shape, backbone_output=backbone_output)

        # class scores
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
