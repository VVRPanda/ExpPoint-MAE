import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from torchmetrics import Accuracy

def cal_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss


###############################
#    Classification System    #
###############################


class Point_MAE_finetune_pl(pl.LightningModule):

    def __init__(self, num_classes=40):
        super().__init__()

        self.num_classes = num_classes

        # metrics to monitor during training
        self.train_acc = Accuracy()
        self.valid_acc = Accuracy()

        self.configure_networks()
    
    def configure_networks(self):
        raise NotImplementedError

    def forward(self, pts):
        neighborhood, center = self.group_devider(pts)
        x_vis = self.MAE_encoder(neighborhood, center)
        max_features = torch.max(x_vis, dim=1).values
        mean_features = torch.mean(x_vis, dim=1)
        feature_vector = torch.cat([max_features, mean_features], dim=-1)
        logits = self.cls_head(feature_vector)
        return logits

    def training_step(self, batch, batch_idx):        

        # training step
        x, y = batch

        logits = self.forward(x)
        loss = cal_loss(logits, y)

        # logging loss
        self.log("loss", loss, on_epoch=True)

        # tracking accuracy
        preds = torch.max(logits, dim=-1).indices
        labels = y.squeeze()
        self.train_acc(preds, labels)
        self.log("train accuracy", self.train_acc, on_epoch=True, on_step=False)
        
        return loss

    def validation_step(self, batch, batch_idx):

        x, y = batch 
        logits = self.forward(x)
        loss = cal_loss(logits, y)
        self.log("val_loss", loss)
        
        # accuracy
        preds = torch.max(logits, dim=-1).indices
        labels= y.squeeze()
        self.valid_acc(preds, labels)
        
        self.log("test_accuracy", self.valid_acc, on_epoch=True, on_step=False)



    def load_submodules(self, path, freeze_backbone=True):
        # loading pretrained submodules
        checkpoint = torch.load(path)
        self.group_devider.load_state_dict(checkpoint['group_devider'])
        self.MAE_encoder.load_state_dict(checkpoint['MAE_encoder'])

        if 'cls_head' in checkpoint.keys():
            self.cls_head.load_state_dict(checkpoint['cls_head'])

        # freeze submodules
        if freeze_backbone:
            self.freeze_backbone()

    def save_submodules(self, path):
        # Saves the parameters of the submodules
        torch.save({
            'group_devider' : self.group_devider.state_dict(),
            'MAE_encoder'   : self.MAE_encoder.state_dict(),
            'cls_head'      : self.cls_head.state_dict()
        }, path)


    def freeze_backbone(self):
        #print("freeze")
        for param in self.MAE_encoder.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        #print("unfreeze")
        for param in self.MAE_encoder.parameters():
            param.requires_grad = True

    def unfreeze_feat_embs(self):
        # NOTE: the self.MAE_encoder component should have a module called point_encoder
        #       that will calculate the feature embeddings
        for param in self.MAE_encoder.point_encoder.parameters():
            param.requires_grad = True