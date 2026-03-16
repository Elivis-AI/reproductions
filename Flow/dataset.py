import torch
import torchvision.datasets as datasets
from torchvision.transforms.v2 import Compose, RandomHorizontalFlip, ToDtype, ToImage

import lightning.pytorch as pl


class FlowDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.dataset_train = None
        self.dataset_val = None

    def setup(self, stage=None):
        transform = Compose(
            [
                ToImage(),
                RandomHorizontalFlip(),
                ToDtype(torch.float32, scale=True),
            ]
        )
        if self.args.dataset == "imagenet":
            dataset = datasets.ImageFolder(self.args.data_path, transform=transform)
        elif self.args.dataset == "cifar10":
            dataset = datasets.CIFAR10(
                root=self.args.data_path,
                train=True,
                download=True,
                transform=transform,
            )
        else:
            raise NotImplementedError(f"Unsupported dataset {self.args.dataset}")

        # Keep validation data identical to the original project behavior,
        # which evaluates FID against the training distribution.
        self.dataset_train = dataset
        self.dataset_val = dataset

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_train,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_mem,
            drop_last=True,
            shuffle=True,
            persistent_workers=self.args.num_workers > 0,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_val,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_mem,
            drop_last=False,
            shuffle=False,
            persistent_workers=self.args.num_workers > 0,
        )
