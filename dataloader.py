import numpy as np
import os
import random

import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split
from torch.utils.data import WeightedRandomSampler
import pytorch_lightning as pl
import albumentations as A
from albumentations.pytorch import ToTensorV2

class CustomImageFolder(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, is_valid_file=None):
        self.dataset = datasets.ImageFolder(root, is_valid_file=is_valid_file)
        self.transform = transform
        self.targets = self.dataset.targets
        
    def __getitem__(self, index):
        image, label = self.dataset[index]
        if self.transform:
            image = self.transform(image=np.array(image))["image"] / 255.0
        return image, label
    
    def __len__(self):
        return len(self.dataset)

class PneumoniaDataModule(pl.LightningDataModule):
    def __init__(self, hyperparams, data_dir):
        super().__init__()
        self.hyperparams = hyperparams
        self.data_dir = data_dir
        
    def setup(self, stage=None):
        data_transforms_train_alb = A.Compose([
            A.Rotate(limit=20),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=1),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0, rotate_limit=0, p=0.5),
            A.Perspective(scale=(0.05, 0.15), keep_size=True, p=0.5),
            A.Resize(height=self.hyperparams["image_size"], width=self.hyperparams["image_size"]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])  
        
        data_transform_val_alb = A.Compose([
            A.Resize(self.hyperparams["image_size"], self.hyperparams["image_size"]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        val_split = 0.2
        
        train_filenames, val_filenames = self._split_file_names(self.data_dir+"train/", val_split)
        
        # Load the datasets
        self.train_dataset = CustomImageFolder(self.data_dir+"train/", transform=data_transforms_train_alb, is_valid_file=lambda x: x in train_filenames)
        
        self.val_dataset = CustomImageFolder(self.data_dir+"train/", transform=data_transform_val_alb, is_valid_file=lambda x: x in val_filenames)
        
        self.test_dataset = CustomImageFolder(self.data_dir+"test/", transform=data_transform_val_alb, is_valid_file=lambda x: self._is_image_file(x))
        
    def train_dataloader(self):
        if self.hyperparams["balance"]:
            sampler = self._create_weighted_sampler(self.train_dataset)
            return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.hyperparams["batch_size"], sampler=sampler, num_workers=4)
        else:
            return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.hyperparams["batch_size"], shuffle=True, num_workers=4)
        
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.hyperparams["batch_size"], num_workers=4)
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.hyperparams["batch_size"], num_workers=4)
    
    def _extract_patient_id(self, filename):
        patient_id = filename.split('_')[0].replace("person", "")
        return patient_id
    
    def _is_image_file(self, filepath):
        return filepath.lower().endswith((".jpeg", ".jpg", ".png"))
    
    def _split_file_names(self, input_folder, val_split_ratio):
        # The PNEUMONIA files contains patient ids, so we have to split by patient to avoid data leakage
        pneumonia_patient_ids = set([self._extract_patient_id(filename) for filename in os.listdir(os.path.join(input_folder, "PNEUMONIA"))])
        pneumonia_val_patient_ids = random.sample(list(pneumonia_patient_ids), int(val_split_ratio * len(pneumonia_patient_ids)))
        
        pneumonia_val_filenames = []
        pneumonia_train_filenames = []
        
        for filename in os.listdir(os.path.join(input_folder, "PNEUMONIUA")):
            if self._is_image_file(filename):
                patient_id = self._extract_patient_id(filename)
                if patient_id in pneumonia_val_patient_ids:
                    pneumonia_val_filenames.append(os.path.join(input_folder, "PNEUMONIA", filename))
                else:
                    pneumonia_train_filenames.append(os.path.join(input_folder, "PNEUMONIA", filename))
                    
        # The NORMAL files doesn't contain patient ids, so we can split them randomly
        normal_filenames = [os.path.join(input_folder, "NORMAL", filename) for filename in os.listdir(os.path.join(input_folder, "NORMAL"))]
        normal_filenames = [filename for filename in normal_filenames if self._is_image_file(filename)]
        
        normal_val_filenames = random.sample(normal_filenames, int(val_split_ratio * len(normal_filenames)))
        normal_train_filenames = list(set(normal_filenames) - set(normal_val_filenames))
        
        train_filenames = pneumonia_train_filenames + normal_train_filenames
        val_filenames = pneumonia_val_filenames + normal_val_filenames
        return train_filenames, val_filenames
        
    def _create_weighted_sampler(self, dataset):
        targets = dataset.targets
        class_counts = np.bincount(targets)
        class_weights = 1.0 / class_counts
        weights = [class_weights[label] for label in targets]
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights))
        return sampler
            