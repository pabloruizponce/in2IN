import lightning.pytorch as pl
import torch
from .interhuman import InterHuman
from .humanml3d import HumanML3D

class DataModuleHML3D(pl.LightningDataModule):
    def __init__(self, cfg, batch_size, num_workers):
        """
        Initialize LightningDataModule for ProHMR training
        Args:
            cfg (CfgNode): Config file as a yacs CfgNode containing necessary dataset info.
            dataset_cfg (CfgNode): Dataset configuration file
        """
        super().__init__()
        self.cfg = cfg
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage = None):
        """
        Create train and validation datasets
        """
        if self.cfg.NAME == "humanml3d":
            self.train_dataset = HumanML3D(self.cfg)
        else:
            raise NotImplementedError

    def train_dataloader(self):
        """
        Return train dataloader
        """
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
            shuffle=True,
            drop_last=True,
            )

class DataModule(pl.LightningDataModule):
    def __init__(self, cfg, batch_size, num_workers):
        """
        Initialize LightningDataModule for ProHMR training
        Args:
            cfg (CfgNode): Config file as a yacs CfgNode containing necessary dataset info.
            dataset_cfg (CfgNode): Dataset configuration file
        """
        super().__init__()
        self.cfg = cfg
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage = None):
        """
        Create train and validation datasets
        """
        if self.cfg.NAME == "interhuman":
            self.train_dataset = InterHuman(self.cfg)
        else:
            raise NotImplementedError

    def train_dataloader(self):
        """
        Return train dataloader
        """
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
            shuffle=True,
            drop_last=True,
            )
    