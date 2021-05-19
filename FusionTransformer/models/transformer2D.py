import torch
import torch.nn as nn
import torch.nn.functional as F
from FusionTransformer.models.STN import STN

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
