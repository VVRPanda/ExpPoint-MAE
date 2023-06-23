import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import LearningRateMonitor

class FreezeBackbone(Callback):

    def on_train_start(self, trainer, pl_module):
        print("Freezing Backbone")
        pl_module.freeze_backbone()

class UnfreezeBackbone(Callback):
    
    def __init__(self, unfreeze_epoch):
        self.unfreeze_epoch = unfreeze_epoch

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        if trainer.current_epoch == self.unfreeze_epoch:
            pl_module.unfreeze_backbone()
            #print("Callback Working")

class UnfreezeFeatureEmbeddings(Callback):

    def __init__(self, unfreeze_epoch):
        self.unfrreeze_epoch = unfreeze_epoch

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        if trainer.current_epoch == self.unfrreeze_epoch:
            print("Unfreezing feature embeddings")
            pl_module.unfreeze_feat_embs()
        
class SaveTopModel(Callback):

    def __init__(self):
        super().__init__()
        self.best_acc = -1

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        
        test_acc = trainer.callback_metrics['test_accuracy']
        
        if test_acc > self.best_acc:
            self.best_acc = test_acc
            pl_module.save_submodules(add_to_name='_best_acc', extra_args={'test_accuracy':test_acc.item()})

class FreezeCLSToken(Callback):

    def on_train_start(self, trainer, pl_module):
        print("Freezing CLS Token")
        pl_module.cls_token.requires_grad = False

class UnfreezeCLSToken(Callback):

    def __init__(self, unfreeze_epoch):
        super().__init__()
        self.unfreeze_epoch = unfreeze_epoch

    def on_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        if trainer.current_epoch == self.unfreeze_epoch:
            print("Unfreeze CLS Token")
            pl_module.cls_token.requires_grad = True

def configure_callbacks(cfg):
    training_cfg = cfg['training']

    print("Using callbacks:")    
    callbacks = []

    if 'freeze_backbone' in training_cfg:
        if training_cfg['freeze_backbone']:
            # print("Setting Callback to freeze backbone")
            print(" - FreezeBackbone")
            callbacks.append(FreezeBackbone())

    if 'unfeeze_at_epoch' in training_cfg:
        if training_cfg['unfeeze_at_epoch']:
            # print("Setting Callback to Unfreeze backbone")
            print(f" - UnfreezeBackbone({training_cfg['unfeeze_at_epoch']})")
            callbacks.append(UnfreezeBackbone(training_cfg['unfeeze_at_epoch']))

    if 'unfreeze_feat_emb' in training_cfg:
        if training_cfg['unfreeze_feat_emb']:
            print(f" - UnfreezeFeatureEmbeddings({training_cfg['unfreeze_feat_emb']})")
            callbacks.append(UnfreezeFeatureEmbeddings(training_cfg['unfreeze_feat_emb']))

    if 'monitor_lr' in training_cfg:
        if training_cfg['monitor_lr']:
            print(" - LearningRateMonitor")
            callbacks.append(LearningRateMonitor())

    if 'save_best' in training_cfg:
        if training_cfg['save_best']:
            print(" - SaveTopModel")
            callbacks.append(SaveTopModel())

    if 'freeze_cls_token' in training_cfg:
        if training_cfg['freeze_cls_token']:
            print(" - FreezeCLSToken")
            callbacks.append(FreezeCLSToken())

    if 'unfreeze_cls_token' in training_cfg:
        if training_cfg['unfreeze_cls_token']:
            print(f" - UnfreezeCLSToken({training_cfg['unfreeze_cls_token']})") 
            callbacks.append(UnfreezeCLSToken(training_cfg['unfreeze_cls_token']))

    return callbacks
