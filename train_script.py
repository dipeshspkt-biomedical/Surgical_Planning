# /// script
# dependencies = [
#     "marimo",
#     "matplotlib==3.11.1",
#     "monai==1.6.0",
#     "pillow==12.3.0",
#     "pytorch-lightning==2.6.5",
#     "scikit-learn==1.9.0",
#     "wandb==0.28.1",
# ]
# requires-python = ">=3.12"
# ///


__generated_with = "0.23.16"

# %%
import marimo as mo

# %%
import data_script as data
import model_script as model
import os
import wandb
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

# %%
class Train(object):
    def __init__(self,key,Main_Directory):
        super().__init__()
        self.key=key
        self.main_dir=Main_Directory
        self.log_dir="/kaggle/working/logs"
        self.project_name="Surgical_Planning"
        self.run_name="Scratch_Surgical_Planning"
        os.makedirs(self.log_dir,exist_ok=True)
        wandb.login(key=self.key)
        self.wandb_logger=pl.loggers.WandbLogger(project=self.project_name, name=self.run_name,log_model='all')
        self.DATA=data.MyData(self.main_dir)
        self.MODEL=model.Model()

    def train(self):
        checkpoint_callback = ModelCheckpoint(
                monitor='Mean Validation Dice',
                mode='max',
                save_top_k=1,
                dirpath='/kaggle/working/',
                filename='best-dice-model',
                save_last=True,
                save_weights_only=False
            )
        trainer = pl.Trainer(
                devices=[0],
                max_epochs=100,
                logger=self.wandb_logger,
                log_every_n_steps=1,
                check_val_every_n_epoch=3,
                callbacks=[checkpoint_callback],
            )
        print('Training from Scratch')
        trainer.fit(model=self.MODEL, datamodule=self.DATA)

# %%
