import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.losses import DiceLoss, FocalLoss

class DiceFocalLoss(nn.Module):
    def __init__(self, dice_weight=0.5, focal_weight=0.5, mode='multiclass'):
        super().__init__()
        self.dice_loss = DiceLoss(mode=mode, from_logits=True)
        self.focal_loss = FocalLoss(mode=mode)
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def forward(self, y_pred, y_true):
        dice = self.dice_loss(y_pred, y_true)
        focal = self.focal_loss(y_pred, y_true)
        return self.dice_weight * dice + self.focal_weight * focal

def get_loss_function(name):
    if name == "DiceLoss":
        return DiceLoss(mode='multiclass', from_logits=True)
    elif name == "FocalLoss":
        return FocalLoss(mode='multiclass')
    elif name == "DiceFocalLoss":
        return DiceFocalLoss()
    else:
        raise ValueError(f"Unknown loss function: {name}")
