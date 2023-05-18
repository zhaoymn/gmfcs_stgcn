import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .stgcn import STGCN
from .heads import GCNHead



class STGCN_Classifier(nn.Module):

    def __init__(self,
                 backbone,
                 cls_head=None):
        super(STGCN_Classifier, self).__init__()

        args = backbone.copy()
        args.pop('type')
        self.backbone = STGCN(**args)
        self.cls_head = GCNHead()

    def init_weights(self):
        """Initialize the model network weights."""
        self.backbone.init_weights()
        self.cls_head.init_weights()

    def forward(self, keypoint):
        """Define the computation performed at every call."""
        x = self.backbone(keypoint)
        # print(x)
        # print(x.shape)
        cls_score = self.cls_head(x)
        cls_score = F.softmax(cls_score, dim=1)
        return cls_score
    