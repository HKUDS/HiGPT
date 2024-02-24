from typing import Any
from lightning import LightningModule, LightningDataModule
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
# from models.clip_models.model import CLIP
from models.clip_models.homo_clip import CLIP
import torch
import torch.nn.functional as F
import re
import numpy as np
from omegaconf import OmegaConf
from transformers.utils.import_utils import is_cython_available
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
import os.path as osp

class HeteCLIP(LightningModule):
    def __init__(self, training_args, model_args, data_args) -> None:
        super().__init__()
        self.training_args = training_args
        self.model_args = model_args
        self.data_args = data_args
        if model_args.gnn_type == 'meta':
            graph_cfg = dict(
                in_channels=model_args.graph_in_channels, 
                out_channels=model_args.graph_out_channels, 
                heads=model_args.graph_heads, 
                dynamic= model_args.graph_dynamic, 
            )
        elif model_args.gnn_type == 'homo':
            graph_cfg = dict(
                att_d_model = model_args.graph_in_channels, 
                head = model_args.graph_heads, 
                att_norm = True
            )
        text_cfg = dict(
            context_length = model_args.context_length, 
            vocab_size = model_args.vocab_size, 
            width = model_args.text_width, 
            heads = model_args.text_heads, 
            layers = model_args.text_layers, 
        )

        self.model = CLIP(
        embed_dim = model_args.embed_dim,
        graph_cfg = graph_cfg,
        text_cfg = text_cfg,
        quick_gelu = model_args.quick_gelu,
        gnn_type = model_args.gnn_type
        )
        self.training_step_outputs = []

    def configure_optimizers(self) -> OptimizerLRScheduler:
        p_wd, p_non_wd = [], []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue  # frozen weights
            else:
                p_wd.append(p)        
        optim_params = [
            {"params": p_wd, "weight_decay": 0.05, "lr_scale": [1e-7, 1e-6]},
        ]                
        
        optim = torch.optim.AdamW(optim_params, lr=self.training_args.learning_rate, betas=(0.9, 0.999))
        lr_sched = LinearWarmupCosineAnnealingLR(optimizer = optim, warmup_epochs = 1, max_epochs = self.training_args.max_epochs, warmup_start_lr=1e-6, eta_min=0.0, last_epoch=- 1)
        return [optim], [lr_sched]

    def training_step(self, batch_sample) -> STEP_OUTPUT:
        clip_output = self.forward(batch_sample)
        self.log('loss', clip_output.loss.item(), on_step=True, on_epoch=True, prog_bar=True, batch_size=self.data_args.micro_batch_size)
        self.training_step_outputs.append(clip_output.loss.item())
        return clip_output.loss
    def on_train_epoch_end(self):
        epoch_average = np.array(self.training_step_outputs).mean()
        self.log("training_epoch_average", epoch_average, sync_dist=True)
        self.training_step_outputs.clear()  # free memory

    def forward(self, batch_sample) -> Any:
        return self.model(batch_sample)
    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        # print('*'*20, 'grad none', '*'*20)
        # for n, p in self.named_parameters():
        #     if p.grad is None:
        #         print(n)
        pass
