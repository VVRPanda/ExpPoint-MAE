import os
import argparse
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

from utils.yaml_loader import MyYamlLoader
from utils.cfg2dataset import get_pretrain_dataloader
from utils.pl_callbacks import configure_callbacks
from pipelines.mae_pipeline import MAESystem
from modules import Group, Mask, TransformerWithEmbeddings
from dl_lib.transforms.general import RandomPointKeep, UnitSphereNormalization, AnisotropicScale, ToTensor

class PointMAEPretrain(MAESystem):

    def __init__(self, cfg):
        self.cfg = cfg
        super().__init__()

        self.load_submodules()
        
    def configure_networks(self):

        net_cfg = self.cfg['network']

        self.group_devider = Group(
            group_size=net_cfg['group_devider']['group_size'], 
            num_group =net_cfg['group_devider']['num_group']
        )

        self.mask_generator = Mask(
            mask_ratio=net_cfg['mask_generator']['mask_ratio'], 
            mask_type =net_cfg['mask_generator']['mask_type']
        )

        self.MAE_encoder = TransformerWithEmbeddings(
            embed_dim     =net_cfg['MAE_encoder']['embed_dim'],
            depth         =net_cfg['MAE_encoder']['depth'],
            num_heads     =net_cfg['MAE_encoder']['num_heads'],
            drop_path_rate=net_cfg['MAE_encoder']['drop_path_rate'],
            feature_embed =net_cfg['MAE_encoder']['feature_embed'],
        )

        self.MAE_decoder = TransformerWithEmbeddings(
            embed_dim     =net_cfg['MAE_decoder']['embed_dim'],
            depth         =net_cfg['MAE_decoder']['depth'],
            num_heads     =net_cfg['MAE_decoder']['num_heads'],
            drop_path_rate=net_cfg['MAE_decoder']['drop_path_rate'],
            feature_embed =net_cfg['MAE_decoder']['feature_embed'],
        )

        # 384 = embed_dim , 32 = group_size
        embed_dim = net_cfg['MAE_encoder']['embed_dim']
        group_size = net_cfg['group_devider']['group_size']
        self.increase_dim = nn.Conv1d(embed_dim, 3 * group_size, 1)


    def configure_optimizers(self):
        opt = torch.optim.AdamW(params=self.parameters() ,lr=0.001, weight_decay=0.05)
        # print("WARNIGN: USING SMALL LR THAN ININTIAL SETUP")
        sched = LinearWarmupCosineAnnealingLR(opt, warmup_epochs=10, max_epochs=self.trainer.max_epochs, warmup_start_lr=1e-6, eta_min=1e-6)
        return [opt], [sched]

    def save_submodules(self):

        if self.cfg['save_checkpoint']:
            path = os.getcwd()
            path = os.path.join(path, 
                                'pretrained_checkpoints',
                                 self.cfg['save_checkpoint'] + '.pt')
            
            save_dict = {
                    'group_devider' : self.group_devider.state_dict(),
                    'mask_generator': self.mask_generator.state_dict(),
                    'MAE_encoder'   : self.MAE_encoder.state_dict(),
                    'MAE_decoder'   : self.MAE_decoder.state_dict(),
                    'increase_dim'  : self.increase_dim.state_dict(),
                    'mask_token'    : self.mask_token.detach().cpu(),
                    'cfg'           : self.cfg
                }

            torch.save(save_dict, path)
        
    def load_submodules(self):
        if self.cfg['load_checkpoint']:
            path = os.getcwd()
            path = os.path.join(path, 
                                'pretrained_checkpoints',
                                 self.cfg['load_checkpoint'] + '.pt')
            
            checkpoint = torch.load(path)
            
            self.group_devider.load_state_dict(checkpoint['group_devider'])
            self.mask_generator.load_state_dict(checkpoint['mask_generator'])
            self.MAE_encoder.load_state_dict(checkpoint['MAE_encoder'])
            self.MAE_decoder.load_state_dict(checkpoint['MAE_decoder'])
            self.increase_dim.load_state_dict(checkpoint['increase_dim'])
   

def main(args): 
    cfg = MyYamlLoader(cfg_name=args.cfg_name, path=args.cfg_path).cfg
    transforms = [
        RandomPointKeep(1024),
        UnitSphereNormalization(),
        AnisotropicScale(), 
        ToTensor(),
    ]
    train_loader = get_pretrain_dataloader(cfg, transforms=transforms)


    model = PointMAEPretrain(cfg)

    logger = WandbLogger(
        project=cfg['wandb']['project'],
        name   =cfg['wandb']['name']
    )

    callbacks = configure_callbacks(cfg)

    trainer = pl.Trainer(accelerator='gpu', 
                        devices=1, 
                        max_epochs=cfg['training']['num_epochs'], 
                        logger=logger,
                        callbacks=callbacks)
        
    trainer.fit(model, train_dataloaders=train_loader)

    model.save_submodules()



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_name', type=str, default='pretrain_cc3d')
    parser.add_argument('--cfg_path', default=None)

    args = parser.parse_args()

    main(args)
    