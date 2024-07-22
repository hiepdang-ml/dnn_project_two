import os
from typing import List, Tuple, Optional

import datetime as dt
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer

from utils import Accumulator, EarlyStopping, Timer, Logger, CheckPointSaver
from datasets import LabeledDogHeartDataset, UnlabeledDogHeartDataset
from losses import AggregateLoss
from functional import plot_predictions, compute_vhs



class Trainer:

    def __init__(
        self, 
        model: nn.Module,
        train_dataset: Dataset,
        val_dataset: Dataset,
        optimizer: Optimizer,
        loss_alpha: float,
        train_batch_size: int,
        val_batch_size: int,
        device: torch.device,
    ):
        self.model = model.to(device=device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.optimizer = optimizer
        self.loss_alpha = loss_alpha
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.device = device

        self.train_dataloader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True)
        self.val_dataloader = DataLoader(dataset=val_dataset, batch_size=val_batch_size, shuffle=False)
        self.loss_function = AggregateLoss(alpha=loss_alpha)

    def train(
        self, 
        n_epochs: int,
        patience: int,
        tolerance: float,
        checkpoint_path: Optional[str] = None,
    ) -> None:
        
        train_metrics = Accumulator()
        early_stopping = EarlyStopping(patience, tolerance)
        timer = Timer()
        logger = Logger()
        checkpoint_saver = CheckPointSaver(dirpath=checkpoint_path)
        self.model.train()

        # loop through each epoch
        for epoch in range(1, n_epochs + 1):
            timer.start_epoch(epoch)
            # Loop through each batch
            for batch, (batch_images, batch_sixpoints, batch_vhs, _, _) in enumerate(self.train_dataloader, start=1):
                timer.start_batch(epoch, batch)
                batch_images: torch.Tensor = batch_images.to(device=self.device)
                batch_sixpoints: torch.Tensor = batch_sixpoints.to(device=self.device)
                batch_vhs: torch.Tensor = batch_vhs.to(device=self.device)
                self.optimizer.zero_grad()
                pred_targets: torch.Tensor = self.model(input=batch_images)
                print(f'gt_targets {batch_sixpoints[-1]}')
                print(f'pred_targets {pred_targets[-1]}')
                mse, cross_entropy, loss = self.loss_function(
                    pred_points=pred_targets, gt_points=batch_sixpoints, gt_vhs=batch_vhs
                )
                loss.backward()
                self.optimizer.step()

                # Accumulate the metrics
                train_metrics.add(mse=mse.item(), cross_entropy=cross_entropy.item(), loss=loss.item())
                timer.end_batch(epoch=epoch)
                logger.log(
                    epoch=epoch, n_epochs=n_epochs, 
                    batch=batch, n_batches=len(self.train_dataloader), 
                    took=timer.time_batch(epoch, batch), 
                    train_mse=train_metrics['mse'] / batch, 
                    train_cross_entropy=train_metrics['cross_entropy'] / batch, 
                    train_loss=train_metrics['loss'] / batch, 
                )

            # Ragularly save checkpoint
            if checkpoint_path and epoch % 5 == 0:
                checkpoint_saver.save(self.model, filename=f'epoch{epoch}.pt')
            
            # Reset metric records for next epoch
            train_metrics.reset()
            
            # Evaluate
            val_mse, val_cross_entropy, val_loss = self.evaluate()
            timer.end_epoch(epoch)
            logger.log(
                epoch=epoch, n_epochs=n_epochs, 
                took=timer.time_epoch(epoch), 
                val_mse=val_mse, 
                val_cross_entropy=val_cross_entropy, 
                val_loss=val_loss,
            )
            print('=' * 20)

            early_stopping(val_cross_entropy)
            if early_stopping:
                print('Early Stopped')
                break

        # Save last checkpoint
        if checkpoint_path:
            checkpoint_saver.save(self.model, filename=f'epoch{epoch}.pt')

    def evaluate(self) -> Tuple[float, float, float]:
        metrics = Accumulator()
        self.model.eval()
        with torch.no_grad():
            # Loop through each batch
            for batch, (batch_images, batch_sixpoints, batch_vhs, _, _) in enumerate(self.val_dataloader, start=1):
                batch_images: torch.Tensor = batch_images.to(device=self.device)
                batch_sixpoints: torch.Tensor = batch_sixpoints.to(device=self.device)
                batch_vhs: torch.Tensor = batch_vhs.to(device=self.device)
                pred_targets: torch.Tensor = self.model(input=batch_images)
                mse, cross_entropy, loss = self.loss_function(
                    pred_points=pred_targets, gt_points=batch_sixpoints, gt_vhs=batch_vhs
                )
                # Accumulate the metrics
                metrics.add(val_mse=mse.item(), val_cross_entropy=cross_entropy.item(), val_loss=loss.item())

        # Compute the aggregate metrics
        return metrics['val_mse'] / batch, metrics['val_cross_entropy'] / batch, metrics['val_loss'] / batch


class Predictor:

    def __init__(self, model: nn.Module, device: torch.device) -> None:
        self.model: nn.Module = model.to(device=device)
        self.device: torch.device = device

    def predict(self, dataset: Dataset, need_plots: bool) -> pd.DataFrame:
        self.model.eval()
        dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

        image_paths: List[str] = []
        point_predictions: List[torch.Tensor] = []
        vhs_predictions: List[torch.Tensor] = []

        if isinstance(dataloader.dataset, UnlabeledDogHeartDataset):
            with torch.no_grad():
                # Loop through each batch
                for batch_images, batch_image_paths in dataloader:
                    batch_images: torch.Tensor = batch_images.to(device=self.device)
                    pred_points: torch.Tensor = self.model(input=batch_images)
                    pred_vhs: torch.Tensor = compute_vhs(points=pred_points)

                    image_paths.extend(batch_image_paths)
                    point_predictions.append(pred_points)
                    vhs_predictions.append(pred_vhs)

                point_predictions = torch.cat(tensors=point_predictions, dim=0).to(device=self.device)
                vhs_predictions: torch.Tensor = torch.cat(tensors=vhs_predictions, dim=0).reshape(-1).to(device=self.device)
                if need_plots:
                    assert point_predictions.shape[0] == len(image_paths)
                    for i in range(len(image_paths)):
                        image_path: str = image_paths[i]
                        point_prediction: torch.Tensor = point_predictions[i]
                        plot_predictions(
                            image_path=image_path, gt_points=None, pred_points=point_prediction,
                        )

        elif isinstance(dataloader.dataset, LabeledDogHeartDataset):
            point_groundtruths: List[torch.Tensor] = []

            with torch.no_grad():
                # Loop through each batch
                for batch_images, batch_gt_six_points, _, batch_image_paths, _ in dataloader:
                    batch_images: torch.Tensor = batch_images.to(device=self.device)
                    batch_gt_six_points: torch.Tensor = batch_gt_six_points.to(device=self.device)
                    pred_points: torch.Tensor = self.model(input=batch_images)
                    pred_vhs: torch.Tensor = compute_vhs(points=pred_points)

                    image_paths.extend(batch_image_paths)
                    point_predictions.append(pred_points)
                    vhs_predictions.append(pred_vhs)
                    point_groundtruths.append(batch_gt_six_points)

                vhs_predictions: torch.Tensor = torch.cat(tensors=vhs_predictions, dim=0).reshape(-1).to(device=self.device)
                point_predictions = torch.cat(tensors=point_predictions, dim=0).to(device=self.device)
                point_groundtruths = torch.cat(tensors=point_groundtruths, dim=0).to(device=self.device)
                if need_plots:
                    assert (
                        point_predictions.shape[0] 
                        == point_groundtruths.shape[0] 
                        == vhs_predictions.shape[0]
                        == len(image_paths) 
                    )
                    for i in range(len(image_paths)):
                        image_path: str = image_paths[i]
                        point_prediction: torch.Tensor = point_predictions[i]
                        point_groundtruth: torch.Tensor = point_groundtruths[i]
                        plot_predictions(
                            image_path=image_path, gt_points=point_groundtruth, pred_points=point_prediction,
                        )
        
        else:
            raise ValueError('Invalid dataset')

        prediction_table = pd.DataFrame(
            data={
                'image': [os.path.basename(image_path) for image_path in image_paths], 
                'label': vhs_predictions.cpu().numpy().tolist(),
            }
        )
        prediction_table.to_csv(
            f'{dt.datetime.now().strftime(r"%Y%m%d%H%M%S")}.csv', 
            header=False, 
            index=False
        )
        return prediction_table


