import torch
from torch.optim import Adam

from datasets import LabeledDogHeartDataset
from models import VisionTransformer
from wokers import Trainer

device: torch.device = torch.device('cuda')
learning_rate: float = 1e-7

net = VisionTransformer(
    in_channels=3, patch_size=32, 
    embedding_dim=4096, image_size=(512, 512),
    depth=2, n_heads=32, dropout=0.1,
)

train_dataset = LabeledDogHeartDataset(dataroot='Dog_Heart_VHS/train', image_resolution=(512, 512))
val_dataset = LabeledDogHeartDataset(dataroot='Dog_Heart_VHS/validation', image_resolution=(512, 512))

trainer = Trainer(
    model=net, 
    train_dataset=train_dataset, val_dataset=val_dataset, 
    optimizer=Adam(params=net.parameters(), lr=learning_rate),
    loss_alpha=0.,
    train_batch_size=8, val_batch_size=4,
    device=device
)
trainer.train(
    n_epochs=100, 
    patience=5, tolerance=0., 
    checkpoint_path='.checkpoints'
)

