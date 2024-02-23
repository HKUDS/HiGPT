import os
import random
from typing import Any, Optional, Dict, List
import logging
import torch
from lightning.pytorch import LightningModule
from transformers import get_cosine_schedule_with_warmup
from torch.optim import AdamW
import torch.nn as nn
from higpt.model.HeteroLlama import HeteroLlamaForCausalLM
import transformers
import numpy as np
from higpt.model.meta_hgt import MetaHGTConvCfg, MetaHGTConv
import os.path as osp
import json
import glob
from higpt.model.heteclip_models import Transformer, LayerNorm, CLIPTextCfg

def load_metahgt_pretrained(model_name, pretrain_model_path): 
    # load conig json
    
    assert osp.exists(osp.join(pretrain_model_path, 'graph_config.json')), 'graph_config.json missing'
    with open(osp.join(pretrain_model_path, 'graph_config.json'), 'r') as f:
        graph_config_dict = json.load(f)
    graph_cfg = MetaHGTConvCfg(**graph_config_dict)

    assert osp.exists(osp.join(pretrain_model_path, 'text_config.json')), 'text_config.json missing'
    with open(osp.join(pretrain_model_path, 'text_config.json'), 'r') as f:
        text_config_dict = json.load(f)
    text_cfg = CLIPTextCfg(**text_config_dict)
    
    assert model_name == MetaHGTConv
    model = model_name(in_channels = graph_cfg.in_channels,
        out_channels = graph_cfg.out_channels,
        heads = graph_cfg.heads,
        dynamic = graph_cfg.dynamic,
        text_transformer = Transformer, 
        text_cfg = text_cfg, 
        layernorm = LayerNorm)

    pkl_files = glob.glob(osp.join(pretrain_model_path, '*.ckpt'))
    state_dict = torch.load(pkl_files[0], map_location = 'cpu')['state_dict']
    print('loading graph pre train model ...')
    gnn_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('model.graph_encoder'):
            new_key = key.split('model.graph_encoder.')[1]
            gnn_state_dict[new_key] = value
    model.load_state_dict(gnn_state_dict)

    return model

class MetaHGT_pl(LightningModule): 
    def __init__(self,
        
    ):
        super().__init__()
        self.model = load_metahgt_pretrained(MetaHGTConv, '/root/paddlejob/workspace/env_run/output/HeteGPT/MetaHGT')
        