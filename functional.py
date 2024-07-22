from typing import Optional

import os
from PIL import Image
import matplotlib.pyplot as plt

import torch


def compute_vhs(points: torch.Tensor) -> torch.Tensor:
    assert points.shape[1:] == (6, 2), 'Each sample in points should be in shape (6, 2)'
    batch_size: int = points.shape[0]
    AB = torch.norm(points[:, 1] - points[:, 0], dim=1)
    CD = torch.norm(points[:, 3] - points[:, 2], dim=1)
    EF = torch.norm(points[:, 5] - points[:, 4], dim=1)
    vhs = 6 * (AB + CD) / EF
    return vhs.reshape(batch_size, 1)


def compute_label(vhs: torch.Tensor, threshold1: float, threshold2: float) -> torch.Tensor:
    return torch.where(
        condition=vhs < threshold1, 
        self=0, 
        other=torch.where(condition=vhs <= threshold2, self=1, other=2)
    )


def plot_predictions(
    image_path: str, 
    gt_points: Optional[torch.Tensor], 
    pred_points: torch.Tensor,
):
    assert pred_points.shape == (6, 2), 'points should be in shape (6, 2)'
    # Make sure all tensors are in CPU
    pred_points = pred_points.to(device='cpu')
    if gt_points is not None:
        assert gt_points.shape == (6, 2), 'points should be in shape (6, 2)'
        gt_points = gt_points.to(device='cpu')

    # Load image
    image: Image.Image = Image.open(image_path).convert("RGB")
    plt.imshow(image)
    
    # Scale points
    pred_points = pred_points * torch.tensor(image.size, dtype=pred_points.dtype)
    if gt_points is not None:
        gt_points = gt_points * torch.tensor(image.size, dtype=gt_points.dtype)

    # Draw points    
    plt.scatter(x=pred_points[:, 0], y=pred_points[:, 1], color='red', label='Prediction')
    if gt_points is not None:
        plt.scatter(x=gt_points[:, 0], y=gt_points[:, 1], color='green', label='Groundtruth')

    # Draw lines
    for p1, p2 in [(0, 1), (2, 3), (4, 5)]:
        plt.plot(
            [pred_points[p1, 0], pred_points[p2, 0]], [pred_points[p1, 1], pred_points[p2, 1]], 
            color='r', linestyle='--',
        )
        if gt_points is not None:
            plt.plot(
                [gt_points[p1, 0], gt_points[p2, 0]], [gt_points[p1, 1], gt_points[p2, 1]], 
                color='g', linestyle='-'
            )

    # Report the VHS in figure title
    filename: str = os.path.basename(image_path)
    title = f'{filename}\n'
    if gt_points is not None:
        title += f'Groundtruth VHS: {compute_vhs(gt_points.unsqueeze(0)).item():.4f}, '

    title += f'Predicted VHS: {compute_vhs(pred_points.unsqueeze(0)).item():.4f}'
    plt.title(title)

    # Set legend
    plt.legend(loc='upper right')
    # Fit plot margins
    plt.subplots_adjust(left=0.01, right=0.99, bottom=0.05, top=0.9)
    # Save file
    os.makedirs('results', exist_ok=True)
    plt.savefig(f'results/{filename}')
    plt.close()



# TEST
if __name__ == '__main__':
    from datasets import LabeledDogHeartDataset
    self = LabeledDogHeartDataset(dataroot='Dog_Heart_VHS/train', image_resolution=(512, 512))
    image, gt_six_points, gt_vhs, image_path, point_path = self[0]
    pred_six_points = gt_six_points + 0.01
    plot_predictions(image_path, None, pred_six_points.reshape(6, 2))

