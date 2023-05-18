import torch 
import torch.nn as nn
from abc import abstractmethod, ABCMeta
import numpy as np
import torch.nn.functional as F

def normal_init(module: nn.Module,
                mean: float = 0,
                std: float = 1,
                bias: float = 0) -> None:
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

class SimpleHead(nn.Module):
    """ A simple classification head.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss. Default: dict(type='CrossEntropyLoss')
        dropout (float): Probability of dropout layer. Default: 0.5.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 dropout=0.5,
                 init_std=0.01,
                 mode='3D',
                 **kwargs):
        # super().__init__(num_classes, in_channels, loss_cls, **kwargs)
        # super().__init__()
        super().__init__()

        self.dropout_ratio = dropout
        self.init_std = init_std
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        assert mode in ['3D', 'GCN', '2D']
        self.mode = mode

        self.in_c = in_channels
        self.fc_cls = nn.Linear(self.in_c, num_classes)

    def init_weights(self):
        """Initiate the parameters from scratch."""
        normal_init(self.fc_cls, std=self.init_std)

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        # print(x.shape)
        # print(x)
        # if isinstance(x, list):
        #     for item in x:
        #         assert len(item.shape) == 2
        #     x = [item.mean(dim=0) for item in x]
        #     x = torch.stack(x)
        # print('fuck')
        pool = nn.AdaptiveAvgPool2d(1)
        N, M, C, T, V = x.shape
        x = x.reshape(N * M, C, T, V)

        x = pool(x)
        x = x.reshape(N, M, C)
        x = x.mean(dim=1)

        assert x.shape[1] == self.in_c
        # if self.dropout is not None:
        #     x = self.dropout(x)
        # print('fuck it')
        # print(x)
        cls_score = self.fc_cls(x)
        # print(self.fc_cls.weight)
        # print('fuck again')
        # print(cls_score)

        return cls_score


class GCNHead(SimpleHead):

    def __init__(self,
                 num_classes=120,
                 in_channels=256,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 dropout=0.,
                 init_std=0.01,
                 **kwargs):
        super().__init__(num_classes,
                         in_channels,
                         loss_cls=loss_cls,
                         dropout=dropout,
                         init_std=init_std,
                         mode='GCN',
                         **kwargs)

class Classifier(nn.Module):
    def __init__(self, num_classes=4, latent_dim=512, in_channels=256, dropout=0.4, init_std=0.01):
        super().__init__()
        self.in_channels = in_channels
        self.out = num_classes
        self.latent_dim = latent_dim
        
        self.linear1 = nn.Linear(self.in_channels, self.latent_dim)
        self.fc_cls = nn.Linear(self.latent_dim, self.out)
        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None
        normal_init(self.linear1, std=init_std)
        normal_init(self.fc_cls, std=init_std)
    
    def forward(self, x):

        # pool = nn.AdaptiveAvgPool2d(1)
        N, M, C, T, V = x.shape
        pool = nn.MaxPool2d((T, V))
        x = x.reshape(N * M, C, T, V)
        # print(x.shape)
        x = pool(x)
        # print(x.shape)
        x = x.reshape(N, M, C)
        x = x.mean(dim=1)

        assert x.shape[1] == self.in_channels
        x = self.linear1(x)
        if self.dropout:
            x = self.dropout(x)
        x = F.relu(x)
        cls_score = self.fc_cls(x)

        return cls_score


class Regressor(nn.Module):
    def __init__(self, latent_dim=1024, in_channels=256, dropout=0, init_std=0.01):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        
        self.linear1 = nn.Linear(self.in_channels, self.latent_dim)
        self.linear2 = nn.Linear(self.latent_dim, 512)
        self.linear3 = nn.Linear(512, 256)
        self.linear4 = nn.Linear(256, 128)
        self.fc_reg = nn.Linear(128, 1)
        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None
        normal_init(self.linear1, std=init_std)
        normal_init(self.linear2, std=init_std)
        normal_init(self.linear3, std=init_std)
        normal_init(self.linear4, std=init_std)
        # normal_init(self.linear1, std=init_std)
        normal_init(self.fc_reg, std=init_std)
    
    def forward(self, x):

        pool = nn.AdaptiveAvgPool2d(1)
        N, M, C, T, V = x.shape
        # pool = nn.MaxPool2d((T, V))
        x = x.reshape(N * M, C, T, V)
        # print(x.shape)
        x = pool(x)
        # print(x.shape)
        x = x.reshape(N, M, C)
        x = x.mean(dim=1)

        # assert x.shape[1] == self.in_channels
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        # if self.dropout:
        #     x = self.dropout(x)
        x = F.relu(x)
        x = F.relu(self.linear3(x))
        if self.dropout:
            x = self.dropout(x)
        x = F.relu(self.linear4(x))
        if self.dropout:
            x = self.dropout(x)
        score = self.fc_reg(x)

        return score
