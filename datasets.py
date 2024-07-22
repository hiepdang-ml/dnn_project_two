import os
from typing import List, Tuple, Dict, Literal

from PIL import Image
from scipy.io import loadmat

import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset


class BaseDogHeartDataset(Dataset):

    def __init__(
        self, 
        dataroot: str, 
        image_resolution: Tuple[int, int], 
        has_labels: bool,
    ):
        super().__init__()
        self.dataroot: str = dataroot
        self.image_resolution: Tuple[int, int] = image_resolution
        self.image_folder: str = os.path.join(dataroot, 'Images')
        self.image_filenames: List[str] = sorted(os.listdir(self.image_folder))
        self.has_labels: bool = has_labels
        if self.has_labels:
            self.point_folder: str = os.path.join(dataroot, 'Labels')
            self.point_filenames: List[str] = sorted(os.listdir(self.point_folder))

    def __len__(self) -> int:
        return len(self.image_filenames)

    def transform(self, input: Image.Image) -> torch.Tensor:
        transformer = T.Compose([
            T.ToTensor(),
            T.Resize(size=self.image_resolution),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transformer(input)


class LabeledDogHeartDataset(BaseDogHeartDataset):

    def __init__(self, dataroot: str, image_resolution: Tuple[int, int]):
        super().__init__(dataroot, image_resolution, has_labels=True)

    # implement
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str, str]:
        # Load images and masks
        image_path: str = os.path.join(self.image_folder, self.image_filenames[idx])
        point_path: str = os.path.join(self.point_folder, self.point_filenames[idx])
        image: Image.Image = Image.open(image_path).convert("RGB")
        
        width_original, height_original = image.size
        image_tensor: torch.Tensor = self.transform(input=image)
        height_new, width_new = image_tensor.shape[1], image_tensor.shape[2]
        
        mat: Dict[Literal['six_points', 'VHS'], np.array] = loadmat(file_name=point_path)
        six_points: torch.Tensor = torch.as_tensor(mat['six_points'], dtype=torch.float32)
        # Resize image to any size and maintain original points
        six_points[:, 0] = width_new / width_original * six_points[:, 0]
        six_points[:, 1] = height_new / height_original * six_points[:, 1]
        # Normalize
        six_points = six_points / height_new

        vhs: torch.Tensor = torch.as_tensor(mat['VHS'], dtype=torch.float32).reshape(-1)
        return image_tensor, six_points, vhs, image_path, point_path


class UnlabeledDogHeartDataset(BaseDogHeartDataset):

    def __init__(self, dataroot: str, image_resolution: Tuple[int, int]):
        super().__init__(dataroot, image_resolution, has_labels=False)

    # implement
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        # Load images
        image_path: str = os.path.join(self.image_folder, self.image_filenames[idx])
        image: Image.Image = Image.open(image_path).convert("RGB")
        image_tensor: torch.Tensor = self.transform(input=image)
        return image_tensor, image_path


if __name__ == '__main__':
    self = LabeledDogHeartDataset(dataroot='Dog_Heart_VHS/train', image_resolution=(512, 512))
    # self = UnlabeledDogHeartDataset(dataroot='Dog_Heart_VHS/test', image_resolution=(512, 512))
