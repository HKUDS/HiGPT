from typing import Any
from lightning import LightningModule, LightningDataModule
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from models.clip_models.model import CLIP
import torch
import torch.nn.functional as F
import re
import numpy as np
from omegaconf import OmegaConf
from transformers.utils.import_utils import is_cython_available
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
import os.path as osp
from dataset.caption_datasets_hgt import HeteCaptionDataset
from lit_module_graph import BlipCaptionProcessor
import glob

class HeteLitDataModule(LightningDataModule):
    def __init__(self, data_args) -> None:
        super().__init__()
        datasets_root = data_args.dataset_dir
        datasets_names = data_args.datalist.split(",")
        ann_paths = []
        split_type = 'train'
        for dsname in datasets_names: 
            json_files = glob.glob(osp.join(datasets_root, dsname, 'ann', '**/*.json'), recursive=True)
            # assert len(json_files) == 1, f'{dsname} has more than one json file.'
            ann_paths.extend(json_files)
        self.train_data = HeteCaptionDataset(graph_processor = None, text_processor = BlipCaptionProcessor(), datasets_root = datasets_root, ann_paths = ann_paths)
        
        self.batch_size = data_args.micro_batch_size
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, collate_fn=self.train_data.collater)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False, collate_fn=self.val_data.collater)
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, collate_fn=self.test_data.collater)