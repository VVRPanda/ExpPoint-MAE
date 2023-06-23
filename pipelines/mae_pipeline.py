import torch
import torch.nn as nn
import pytorch_lightning as pl
from extensions.chamfer_dist import ChamferDistanceL2


class MAESystem(pl.LightningModule):

    def __init__(self, ):
        super().__init__()

        self.embed_dim = 384
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.loss_func = ChamferDistanceL2()

        self.configure_networks()

    def forward(self, pts):
        neighborhood, center = self.group_devider(pts)
        x_vis = self.MAE_encoder(neighborhood, center)
        return x_vis

    def training_step(self, batch, batch_idx):

        # 
        neighborhood, center = self.group_devider(batch)

        # TODO: Add masking as a separate module 
        mask = self.mask_generator(center)

        # center: B x N x 3
        # neighborhood : B x N x M x 3
        B, _, M, _ = neighborhood.shape

        masked_center = center[~mask].reshape(B, -1, 3)
        masked_neighborhood = neighborhood[~mask].reshape(B, -1, M, 3)

        #
        x_vis = self.MAE_encoder(masked_neighborhood, masked_center)

        # x_vis: B x P x F
        # where
        #   - B: batch_size
        #   - P: number of patched that we keep
        #   - F: feature dim
        #
        # mask : B x N x F
        # where
        #   - N: total number of patches
        # mask has value 1 to the masked out points

        B, _, C = x_vis.shape

        # stacking centers [encoded centers, not encoded centers]
        vis_centers = center[~mask].reshape(B, -1, 3)
        mask_centers = center[mask].reshape(B, -1, 3)
        pos_full = torch.cat([vis_centers, mask_centers], dim=1)

        # repeating mask token to pass to the decoder
        _, M, _ = mask_centers.shape
        mask_token = self.mask_token.expand(B, M, -1)
        
        # concatenating actual features with mask features
        # to pass to the decoder
        x_full = torch.cat([x_vis, mask_token], dim=1)

        # activating the decoder
        x_rec = self.MAE_decoder(x_full, pos_full)
        
        # seperating the decoded features
        x_rec = x_rec[:, -M:, :]

        # passing patch embedding to final mlp to extract the actual point possitions
        rebuild_points = self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 3)
        
        # getting the ground truth points
        gt_points = neighborhood[mask].reshape(B * M, -1, 3)

        # computing the loss
        loss = self.loss_func(rebuild_points, gt_points)

        # logging the loss
        self.log("loss", loss, on_epoch=True)

        return loss
        

    def configure_networks(self):
        # Define the following networks 
        # self.group_devider
        # self.mask_generator
        # self.MAE_encoder
        # self.MAE_decoder
        # self.increase_dim
        raise NotImplementedError


    # def save_submodules(self, path):
    #     # Saves the parameters of the submodules
    #     torch.save({
    #         'group_devider' : self.group_devider.state_dict(),
    #         'mask_generator': self.mask_generator.state_dict(),
    #         'MAE_encoder'   : self.MAE_encoder.state_dict(),
    #         'MAE_decoder'   : self.MAE_decoder.state_dict(),
    #         'increase_dim'  : self.increase_dim.state_dict()
    #     }, path)

    def freeze_backbone(self):
        for param in self.MAE_encoder.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.MAE_encoder.parameters():
            param.requires_grad = True

    def unfreeze_feat_embs(self):
        # NOTE: the self.MAE_encoder component should have a module called point_encoder
        #       that will calculate the feature embeddings
        for param in self.MAE_encoder.point_encoder.parameters():
            param.requires_grad = True