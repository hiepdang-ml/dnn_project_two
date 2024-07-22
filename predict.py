import torch
from torch.optim import Adam

from datasets import UnlabeledDogHeartDataset, LabeledDogHeartDataset
from wokers import Predictor

device: torch.device = torch.device('cuda')
net = torch.load(f='.checkpoints/epoch30.pt')
predictor = Predictor(model=net, device=device)


test_dataset = UnlabeledDogHeartDataset(dataroot='Dog_Heart_VHS/test', image_resolution=(512, 512))
predictor.predict(dataset=test_dataset, need_plots=False)

# test_dataset = UnlabeledDogHeartDataset(dataroot='unlabeled_samples', image_resolution=(512, 512))
# predictor.predict(dataset=test_dataset, need_plots=True)

# test_dataset = LabeledDogHeartDataset(dataroot='labeled_samples', image_resolution=(512, 512))
# predictor.predict(dataset=test_dataset, need_plots=True)

