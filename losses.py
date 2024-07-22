from typing import Tuple

import torch
import torch.nn as nn
from functional import compute_vhs, compute_label

import torch.nn.functional as F


class VHSClassificationLoss(nn.Module):

    def __init__(self, threshold1: float, threshold2: float):
        super().__init__()
        self.threshold1: float = threshold1
        self.threshold2: float = threshold2
    
    def forward(self, pred_vhs: torch.Tensor, gt_label: torch.Tensor) -> torch.Tensor:
        assert pred_vhs.shape == gt_label.shape     # (batch_size, 1)
        batch_size: int = pred_vhs.shape[0]
        midpoint: float = (self.threshold1 + self.threshold2) / 2.
        delta: float = midpoint - self.threshold1
        scores_0: torch.Tensor = - torch.abs(pred_vhs - (self.threshold1 - delta))
        scores_1: torch.Tensor = - torch.abs(pred_vhs - midpoint)
        scores_2: torch.Tensor = - torch.abs(pred_vhs - (self.threshold2 + delta))

        score_matrix: torch.Tensor = torch.cat(tensors=[scores_0, scores_1, scores_2], dim=1)
        assert score_matrix.shape == (batch_size, 3)
        return F.cross_entropy(input=score_matrix, target=gt_label.reshape(-1), reduction='mean')


class AggregateLoss(nn.Module):

    def __init__(self, alpha: float):
        super().__init__()
        self.regression_loss = nn.MSELoss(reduction='mean')
        self.classification_loss = VHSClassificationLoss(threshold1=8.2, threshold2=10.)
        self.alpha: float = alpha

    def forward(
        self, 
        pred_points: torch.Tensor, 
        gt_points: torch.Tensor, 
        gt_vhs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        assert pred_points.ndim == gt_points.ndim == 3  # (batch_size, 6, 2)
        batch_size: int = pred_points.shape[0]
        assert pred_points.shape == gt_points.shape == (batch_size, 6, 2)
        assert gt_vhs.shape == (batch_size, 1)
        pred_vhs: torch.Tensor = compute_vhs(points=pred_points)
        gt_label: torch.Tensor = compute_label(
            vhs=gt_vhs, 
            threshold1=self.classification_loss.threshold1, 
            threshold2=self.classification_loss.threshold2,
        )
        mse: torch.Tensor = self.regression_loss(input=pred_points, target=gt_points)
        cross_entropy: torch.Tensor = self.classification_loss(pred_vhs=pred_vhs, gt_label=gt_label)
        loss: torch.Tensor = mse + cross_entropy * self.alpha
        return (mse, cross_entropy, loss)

