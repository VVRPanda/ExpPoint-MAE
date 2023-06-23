import argparse
import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from modules import Group, TransformerWithEmbeddings
from pipelines.finetune_pipeline import Point_MAE_finetune_pl
from utils.cfg2dataset import get_cls_dataloader
from utils.yaml_loader import MyYamlLoader
from utils.pl_callbacks import configure_callbacks

class PointMAECLS(Point_MAE_finetune_pl):

    def __init__(self, cfg):
        self.cfg = cfg
        super().__init__(cfg['dataset']['num_classes'])

        self.load_submodules()      

    def configure_networks(self):

        net_cfg = self.cfg['network']

        self.group_devider = Group(
            group_size=net_cfg['group_devider']['group_size'],
            num_group =net_cfg['group_devider']['num_group']
        )

        self.MAE_encoder = TransformerWithEmbeddings(
            embed_dim     =net_cfg['MAE_encoder']['embed_dim'],
            depth         =net_cfg['MAE_encoder']['depth'], 
            num_heads     =net_cfg['MAE_encoder']['num_heads'], 
            drop_path_rate=net_cfg['MAE_encoder']['drop_path_rate'], 
            feature_embed =net_cfg['MAE_encoder']['feature_embed']
        )
        
        embed_dim = net_cfg['MAE_encoder']['embed_dim']
        self.use_cls_token = net_cfg['use_cls_token']
        self.use_max_pooling = net_cfg['use_max_pooling']
        self.use_mean_pooling = net_cfg['use_mean_pooling']
        cls_in_features = embed_dim * self.use_cls_token + \
                          embed_dim * self.use_max_pooling + \
                          embed_dim * self.use_mean_pooling

        self.cls_token = torch.nn.Parameter(torch.rand(1, 1, net_cfg['MAE_encoder']['embed_dim'])) \
            if net_cfg['use_cls_token'] else None
        

        self.cls_head = nn.Sequential(
            nn.Linear(cls_in_features, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, self.num_classes)
        )
    
    def pool_features(self, x_vis):
        
        feat_list = []

        if self.use_cls_token:
            feat_list.append(
                x_vis[:, -1, :]
            )
            # removing cls_token from other pooling operations
            x_vis = x_vis[:, :-1, :]

        if self.use_max_pooling:
            feat_list.append(
                torch.max(x_vis, dim=1).values
            )

        if self.use_mean_pooling:
            feat_list.append(
                torch.mean(x_vis, dim=1)
            )

        # concatenating features
        if len(feat_list) > 1: 
            feature_vector = torch.cat(feat_list, dim=-1)
        else: 
            feature_vector = feat_list[0]

        return feature_vector

    def forward(self, pts):
        neighborhood, center = self.group_devider(pts)
        x_vis = self.MAE_encoder(neighborhood, center, self.cls_token)
        feature_vector = self.pool_features(x_vis)
        logits = self.cls_head(feature_vector)
        return logits

    def configure_optimizers(self):
        opt = torch.optim.AdamW(params=self.parameters() ,lr=0.0005, weight_decay=0.05)
        sched = LinearWarmupCosineAnnealingLR(opt, warmup_epochs=10, max_epochs=self.trainer.max_epochs, warmup_start_lr=1e-6, eta_min=1e-6)
        return [opt], [sched]

    def save_submodules(self, add_to_name = None, extra_args=None):
        if self.cfg['save_checkpoint']:
            path = os.getcwd()

            # ability to add extra subfix to checkpoint name
            filename = self.cfg['save_checkpoint'] if add_to_name==None else self.cfg['save_checkpoint']+add_to_name

            path = os.path.join(path, 
                                'finetuned_checkpoints',
                                 filename + '.pt')
            
            save_dict = {
                    'group_devider' : self.group_devider.state_dict(),
                    'MAE_encoder'   : self.MAE_encoder.state_dict(),
                    'cls_head'      : self.cls_head.state_dict(),
                    'cfg'           : self.cfg
                }

            if extra_args is not None:
                save_dict.update(extra_args)

            if self.cfg['network']['use_cls_token']:
                save_dict['cls_token'] = self.cls_token.detach().cpu()

            torch.save(save_dict, path)


    def load_submodules(self):
        if self.cfg['load_checkpoint']:
            path = os.getcwd()
            try:
                path = os.path.join(path, 
                                'pretrained_checkpoints',
                                 self.cfg['load_checkpoint'] + '.pt')

                checkpoint = torch.load(path)
            except:
                print("Failed to load checkpoint from pretrained_checkpoints")
                print("Trying to load checkpoint from finetuned checkpoints...")
                path = os.path.join(path,
                                'finetuned_checkpoints',
                                self.cfg['load_checkpoint'] + '.pt' 
                            )
                checkpoint = torch.load(path)

            self.group_devider.load_state_dict(checkpoint['group_devider'])
            self.MAE_encoder.load_state_dict(checkpoint['MAE_encoder'])

            if 'cls_head' in checkpoint.keys():
                self.cls_head.load_state_dict(checkpoint['cls_head'])

            if 'cls_token' in checkpoint.keys():
                print("Loading CLS_token")
                self.cls_token = torch.nn.Parameter(checkpoint['cls_token'])



def main(parser):
    args = parser.parse_args()
    cfg = MyYamlLoader(cfg_name=args.cfg_name, path=args.cfg_path).cfg

    train_loader, valid_loader = get_cls_dataloader(cfg)

    model = PointMAECLS(cfg)

    logger = WandbLogger(
        project=cfg['wandb']['project'],
        name   =cfg['wandb']['name']
    )

    callbacks = configure_callbacks(cfg)

    trainer = pl.Trainer(accelerator='gpu', 
                        devices=1, 
                        max_epochs=cfg['training']['num_epochs'],
                        logger=logger,
                        callbacks=callbacks,
                        gradient_clip_val=10
                        )


    trainer.fit(model, train_loader, valid_loader)

    model.save_submodules()


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_name', type=str, default='finetune/ModelNetExample')
    parser.add_argument('--cfg_path', default=None)

    main(parser)