# /// script
# dependencies = [
#     "marimo",
#     "monai==1.6.0",
#     "pytorch-lightning==2.6.5",
#     "torch==2.13.0",
#     "wandb==0.28.1",
# ]
# requires-python = ">=3.12"
# ///


__generated_with = "0.23.16"

# %%
import marimo as mo

# %%
import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from monai.metrics import DiceMetric
from monai.data import  decollate_batch
from monai.losses import DiceLoss
from monai.networks.nets import UNet
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    Lambdad,
    EnsureType,
)
from monai.utils import set_determinism
from monai.inferers import sliding_window_inference
import wandb

# %%
class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        set_determinism(seed=40)
        self.in_channels=3
        self.out_channels=3
        self.learning_rate=0.001
        self._model = UNet(
            spatial_dims=2,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )
        self.lossfunction=DiceLoss(to_onehot_y=True ,softmax=True,reduction='mean',include_background=False)
        self.dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=True)
        self.post_pred = Compose([EnsureType("tensor", device="cpu"), AsDiscrete(argmax=True, to_onehot=self.out_channels)])
        self.post_label = Compose([EnsureType("tensor", device="cpu"), AsDiscrete(to_onehot=self.out_channels)])
        self.best_val_dice = 0
        self.best_val_epoch = 0
        self.validation_step_outputs = []

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)

    def forward(self,x):
        return self.model(x)

    def training_step(self,batch,batch_idx):
        inputs, masks =  batch["image"],batch["mask"]
        outputs = self.forward(inputs)
        loss = self.lossfunction(outputs, masks)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch["image"], batch["mask"]
        roi_size = (128,128)
        sw_batch_size = 1
        with torch.no_grad():
            outputs = sliding_window_inference(images, roi_size, sw_batch_size, self.forward)
            loss = self.lossfunction(outputs, masks)
            outputs = [self.post_pred(i) for i in decollate_batch(outputs)]
            masks = [self.post_label(i) for i in decollate_batch(masks)]
            self.dice_metric(y_pred=outputs, y=masks)
            d = {"val_loss": loss.cpu().item(), "val_number": len(outputs)}
        self.validation_step_outputs.append(d)
        return d
  
    def on_validation_epoch_end(self):
        val_loss, num_items = 0, 0
        for output in self.validation_step_outputs:
            val_loss += output["val_loss"]
            num_items += output["val_number"]
        mean_val_dice = self.dice_metric.aggregate()[0].item()
        self.dice_metric.reset()
        mean_val_loss = val_loss / num_items
        if mean_val_dice > self.best_val_dice:
            self.best_val_dice = mean_val_dice
            self.best_val_epoch = self.current_epoch
        self.validation_step_outputs.clear()  # free memory
        self.log('Mean Validation Loss',mean_val_loss,prog_bar=True,logger=True)
        self.log('Mean Validation Dice',mean_val_dice,prog_bar=True,logger=True)
        return mean_val_loss,mean_val_dice


    def _print_trainable_params(self):
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable params: {trainable}")
        print(f"Total params: {total}")
        print(f"Trainable ratio: {trainable/total:.4f}")


# %%
