import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import torchvision

class LightningModel(L.LightningModule):
    def __init__(self, 
                 model=None, 
                 batch_size = None,
                 epochs = None, 
                 workers = None, 
                 optimizer = None,
                 norm_weight_decay = None, 
                 momentum = None, 
                 lr = None, 
                 weight_decay = None, 
                 lr_step_size = None, 
                 num_classes = None):
        super().__init__()

        self.model=None, 
        self.batch_size = None,
        self.epochs = None, 
        self.workers = None, 
        self.optimizer = None,
        self.norm_weight_decay = None, 
        self.momentum = None, 
        self.lr = None, 
        self.weight_decay = None, 
        self.lr_step_size = None

        self.num_classes = 91

        if model is None:
            self.model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True)
