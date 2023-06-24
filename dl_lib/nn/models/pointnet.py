import torch
import torch.nn as nn
from ..modules import LinearLayer, MaxPool1D

__all__ = ['PointNetCore', "PointNetCLS"]

# regularization loss used in PointNet
def feature_transform_regularizer(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2,1)) - I, dim=(1,2)))
    return loss




class TNET(nn.Module):

    def __init__(self, in_features):
        super().__init__()
        self.in_features = in_features
        self.tnet = nn.Sequential(
            # Initial feature aggregation
            LinearLayer(in_features, 64),
            LinearLayer(64, 128), 
            LinearLayer(128, 1024),
            # max pooling
            MaxPool1D(),
            # MLP to create the final matrix
            LinearLayer(1024, 512, fc=True),
            LinearLayer(512, 256, fc=True),
            LinearLayer(256, in_features * in_features, fc=True, use_norm=False, use_relu=False)
        )

        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            # we want the initial state of the network to be as close a possible to the identity matrix
            # so we need to set a high value at the bias of the element in the main diagonal and small values
            # to the rest of the biases and the weights of the output layer
            diag_indx = torch.arange(0, self.in_features * self.in_features-1, self.in_features+1)
            
            # setting the bias to be a small number
            self.tnet[-1].net[0].bias.fill_(0.00001)
            # setting the bias of the elements in the diagonal to be equal to 1.0
            self.tnet[-1].net[0].bias[diag_indx] = 1.0
            # aggigning a small value to the weights of the last layer
            self.tnet[-1].net[0].weight *= 0.0001 

    def forward(self, x):
        x = self.tnet(x)
        # reshape x to create a matrix
        x = x.reshape(-1, self.in_features, self.in_features)
        return x



class PointNetCore(nn.Module):

    def __init__(self, point_features):
        super().__init__()
        
        self.input_TNET = TNET(3)
        self.mlp1 = nn.Sequential(
            LinearLayer(3, 64),
            LinearLayer(64, 64)
        )
        self.feature_TNET = TNET(64)
        self.mlp2 = nn.Sequential(
            LinearLayer(64, 64), 
            LinearLayer(64, 128),
            LinearLayer(128, 1024)
        )

        # creating variables to store the m1, m2 matrices produced by the TNET
        self.m1 = None
        self.m2 = None

    def forward(self, x):
        
        # calculating the first orthogonal matrix
        # to transform input features
        m1 = self.input_TNET(x)
        x = m1 @ x # Multiply x with the transform matrix
        # passing through the first series of mlps
        x = self.mlp1(x)
        # Applying second feature transform
        m2 = self.feature_TNET(x)
        x = m2 @ x
        # passing through the second series of mlps
        x = self.mlp2(x)

        # storing the m1 and m2 matrices
        self.m1 = m1
        self.m2 = m2

        return x

    def get_TNET_matrices(self):
        return self.m1, self.m2

    def regularization_loss(self):
        return feature_transform_regularizer(self.m1) + feature_transform_regularizer(self.m2)



class PointNetCLSHead(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super().__init__(
            LinearLayer(in_channels, 512, fc=True),
            LinearLayer(512, 256, fc=True),
            nn.Dropout(p=0.3),
            LinearLayer(256, num_classes, fc=True, use_norm=False, use_relu=False)
        )



class PointNetCLS(nn.Module):
    def __init__(self, point_features, num_classes):
        super().__init__()
        
        # Feature extraction for each point
        self.core = PointNetCore(point_features)
        # Feature aggregation - Symmetric Function
        self.maxpool = MaxPool1D()
        # Classification head  
        self.cls_head = PointNetCLSHead(1024, num_classes)
    
    def regularization_loss(self):
        return self.core.regularization_loss()

    def forward(self, x):
        
        x = self.core(x)
        
        args = None
        if isinstance(x, tuple):
            args = x[1:]
            x = x[0]
        
        x = self.maxpool(x)
        x = self.cls_head(x)

        if args is not None:
            return x, *args

        return x