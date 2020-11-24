# -*- coding: utf-8 -*-
import logging
from pathlib import Path
import click
import numpy as np
import torch
from PIL import Image
from PIL import ImageFile


try:
    import torch_xla.core.xla_model as xm

    _xla_available = True
except:
    _xla_available = False
ImageFile.LOAD_TRUNCATED_IMAGES = True


class ClassificationDataset:
    def __init__(self, image_paths, targets, resize, augmentations=None):
        self.image_paths = image_paths
        self.categories = targets["Category"]
        self.pattern = targets["Pattern"]
        self.sleeve = targets["Sleeve"]
        self.length = targets["Length"]
        self.neckline = targets["Neckline"]
        self.material = targets["Material"]
        self.fit = targets["Fit"]
        self.resize = resize
        self.augmentations = augmentations

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        image = Image.open(self.image_paths[item])
        targets = self.targets[item]
        if self.resize is not None:
            image = image.resize(
                (self.resize[1], self.resize[0]), resample=Image.BILINEAR
            )
        image = np.array(image)
        if self.augmentations is not None:
            augmented = self.augmentations(image=image)
            image = augmented["image"]
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        return {
            "image": torch.tensor(image),
            "categories": torch.tensor(self.categories[item], dtype=torch.long),
            "pattern": torch.tensor(self.pattern[item], dtype=torch.long),
            "sleeve": torch.tensor(self.sleeve[item], dtype=torch.long),
            "length": torch.tensor(self.length[item], dtype=torch.long),
            "neckline": torch.tensor(self.neckline[item], dtype=torch.long),
            "material": torch.tensor(self.material[item], dtype=torch.long),
            "fit": torch.tensor(self.fit[item], dtype=torch.long),
        }


class ClassificationDataLoader:
    def __init__(self, dataset, image_paths, targets, resize, augmentations=None):
        self.dataset = dataset

    @staticmethod
    def fetch(self, batch_size, num_workers, drop_last=False, shuffle=True, tpu=False):
        sampler = None
        if tpu:
            sampler = torch.utils.data.distributed.DistributedSampler(
                self.dataset,
                num_replicas=xm.xrt_world_size(),
                rank=xm.get_ordinal(),
                shuffle=shuffle,
            )

        data_loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size,
            sampler=sampler,
            drop_last=drop_last,
            num_workers=num_workers,
        )

        return data_loader
