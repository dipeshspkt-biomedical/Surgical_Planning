# /// script
# dependencies = [
#     "marimo",
#     "matplotlib==3.11.1",
#     "monai==1.6.0",
#     "numpy==2.5.1",
#     "pillow==12.3.0",
#     "pytorch-lightning==2.6.5",
#     "scikit-learn==1.9.0",
# ]
# requires-python = ">=3.12"
# ///


__generated_with = "0.23.16"

# %%
import marimo as mo
import pytorch_lightning as pl
from PIL import Image
from matplotlib import pyplot as plt
from glob import glob
import numpy as np
from dataclasses import dataclass,field
from monai.transforms import (Compose, LoadImaged,EnsureChannelFirstd,ToTensord,ScaleIntensityd,Lambdad,SpatialPadd)
from monai.data import Dataset,DataLoader,decollate_batch
from sklearn.model_selection import train_test_split
import torch
import time

# %%
class MyData(pl.LightningDataModule):
    def __init__(self,Main_Directory):
        super().__init__()
        self.Main_Directory=Main_Directory
        self.batch_size=1
        self.train_size=0.8
        self.test_size=0.2
        self.random_state=42
        self.num_workers=3
        self.train_transform=Compose([
            LoadImaged(keys=["image","mask"]),
            EnsureChannelFirstd(keys=["image","mask"]),
            Lambdad(keys=["mask"],func=self.rgb_to_label),
            SpatialPadd(keys=["image", "mask"],spatial_size=(864, 480)),
            ScaleIntensityd(keys=["image"]),
            ToTensord(keys=["image","mask"]),
        ])
        self.val_transform=Compose([
            LoadImaged(keys=["image", "mask"]),
            EnsureChannelFirstd(keys=["image", "mask"]),
            Lambdad(keys=["mask"],func=self.rgb_to_label), 
            SpatialPadd(keys=["image", "mask"],spatial_size=(864, 480)),
            ScaleIntensityd(keys=["image"]),  
            ToTensord(keys=["image","mask"]),
        ])

    def setup(self,stage=None):
        self.img_files=sorted(glob(f"{self.Main_Directory}/**/**/*_endo.png"))
        self.mask_files=sorted(glob(f"{self.Main_Directory}/**/**/*_endo_watershed_mask.png"))
        self.data_dict=[{'image':img,'mask':mask} for img,mask in zip(self.img_files,self.mask_files) if np.array(Image.open(mask)).shape[2]==3]
        self.sample_positive=[]
        self.sample_negative=[]
        for items in self.data_dict:
            mask = np.isin([31,32],np.unique(np.array(Image.open(items["mask"]))[:,:,0])).all()
            if np.any(mask):
                self.sample_positive.append(items)
            else:
                self.sample_negative.append(items)
        self.sample_train_positive,self.sample_val_positive=train_test_split(self.sample_positive,train_size=self.train_size,test_size=self.test_size)
        self.sample_train_negative,self.sample_val_negative=train_test_split(self.sample_negative,train_size=self.train_size,test_size=self.test_size)
        self.train_files=self.sample_train_positive+self.sample_train_negative
        self.val_files=self.sample_val_positive+self.sample_val_negative
        self.subset_train,self.subset_val=train_test_split(self.sample_positive,train_size=0.8,random_state=42)
        self.train_dataset=Dataset(data=self.subset_train,transform=self.train_transform)
        self.val_dataset=Dataset(data=self.subset_val,transform=self.val_transform)
        
  
    def train_dataloader(self):
        return DataLoader(self.train_dataset,batch_size=self.batch_size,shuffle=True,num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,batch_size=self.batch_size,shuffle=False,num_workers=self.num_workers)

    @staticmethod
    def rgb_to_label(img):
        mask = img.permute(1, 2, 0)  # (H, W, 3)
        labels = torch.zeros(mask.shape[:2], dtype=torch.long, device=img.device)
        labels[(mask == torch.tensor([31, 31, 31], device=img.device)).all(dim=-1)] = 1
        labels[(mask == torch.tensor([32, 32, 32], device=img.device)).all(dim=-1)] = 2
        return labels.unsqueeze(0)