from typing import Tuple

import torch
import torch.nn as nn
from functional import compute_vhs

import torch.nn.functional as F


class VHSLoss(nn.Module):

    def __init__(self, p: int):
        super().__init__()
        self.p: int = p
    
    def forward(self, pred_points: torch.Tensor, gt_points: torch.Tensor) -> torch.Tensor:
        batch_size: int = pred_points.shape[0]
        assert pred_points.shape == gt_points.shape == (batch_size, 6, 2)
        pred_vhs: torch.Tensor = compute_vhs(points=pred_points)
        gt_vhs: torch.Tensor = compute_vhs(points=gt_points)
        assert pred_vhs.shape == gt_vhs.shape == (batch_size, 1)
        vhs_loss: torch.Tensor = torch.sum(torch.abs(pred_vhs - gt_vhs) ** self.p) / pred_vhs.numel()
        return vhs_loss


class AggregateLoss(nn.Module):

    def __init__(self, alpha: float):
        super().__init__()
        self.regression_loss = nn.MSELoss(reduction='mean')
        self.classification_loss = VHSLoss(p=2)
        self.alpha: float = alpha

    def forward(self, pred_points: torch.Tensor, gt_points: torch.Tensor) -> torch.Tensor:
        batch_size: int = pred_points.shape[0]
        assert pred_points.shape == gt_points.shape == (batch_size, 6, 2)
        regression_loss: torch.Tensor = self.regression_loss(input=pred_points, target=gt_points)
        vsh_loss: torch.Tensor = self.classification_loss(pred_points=pred_points, gt_points=gt_points)
        total_loss: torch.Tensor = regression_loss * (1. - self.alpha) + vsh_loss * self.alpha
        return regression_loss, vsh_loss, total_loss

