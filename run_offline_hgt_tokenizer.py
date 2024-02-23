#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from typing import List, Optional, Tuple, Union
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM, \
                         CLIPVisionModel, CLIPImageProcessor

from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from higpt.model.graph_layers import MPNN, GNN, CLIP, graph_transformer
from higpt.model.meta_hgt import MetaHGTConvCfg, MetaHGTConv
from higpt.model.heteclip_models import Transformer, LayerNorm, CLIPTextCfg
from torch_geometric.data import Data
import json
import os.path as osp
import glob
from tqdm import tqdm
from lightning.pytorch import LightningModule, Trainer, seed_everything
import re
import os
import argparse

DEFAULT_GRAPH_TOKEN = "<graph>"
DEFAULT_GRAPH_PATCH_TOKEN = "<g_patch>"
DEFAULT_G_START_TOKEN = "<g_start>"
DEFAULT_G_END_TOKEN = "<g_end>"


class HeteroLlamaConfig(LlamaConfig):
    model_type = "HeteroLlama"

device = 'cuda:2'

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
        text_cfg = text_cfg,)

    pkl_files = glob.glob(osp.join(pretrain_model_path, '*.ckpt'))
    state_dict = torch.load(pkl_files[0], map_location = 'cpu')['state_dict']
    print('loading graph pre train model ...')
    gnn_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('model.graph_encoder'):
            new_key = key.split('model.graph_encoder.')[1]
            gnn_state_dict[new_key] = value
    model.load_state_dict(gnn_state_dict, strict=False)

    return model

def check_offline_hgt(): 
    parser = argparse.ArgumentParser(description='check data')
    parser.add_argument('--pretrained_gnn_path', default='/root/paddlejob/workspace/env_run/output/HeteGPT/MetaHGT', type=str)
    parser.add_argument('--graph_root', default='/root/paddlejob/workspace/env_run/output/HetBaseline/data/DBLP', type=str)
    parser.add_argument('--dsname', default='instruct_ds', type=str)
    parser.add_argument('--data_type', default='dblp', type=str)
    args = parser.parse_args()

    hgnn_name = args.pretrained_gnn_path.split('/')[-1]

    hgt = load_metahgt_pretrained(MetaHGTConv, args.pretrained_gnn_path)
    hgt = hgt.to(device)

    graph_root = args.graph_root
    dsname = args.dsname

    node_feas_dict = torch.load(f'/root/paddlejob/workspace/env_run/output/HeteGPT/hetegpt/model/meta_hgt/meta_dict/{args.data_type}/node_type.pt')
    for k, v in node_feas_dict.items():
        node_feas_dict[k] = v.to(device)
    
    edge_feas_dict = torch.load(f'/root/paddlejob/workspace/env_run/output/HeteGPT/hetegpt/model/meta_hgt/meta_dict/{args.data_type}/edge_type.pt')
    for k, v in edge_feas_dict.items():
        edge_feas_dict[k] = v.to(device)
    
    ann_data_path = []
    # dsname = dsname.split(',')
    # for i, path in enumerate(dsname):
    ann_file = glob.glob(osp.join(graph_root, dsname, 'ann', '**/*.json'), recursive=True)
    # assert len(ann_file) == 1, f"Need to have one ann file for each graph"
    ann_data_path.extend(ann_file)

    # json_file = osp.join(graph_root, dsname, 'ann', 'DBLP_train_std_0_400.json')

    data_list_dict = {}
    for ann_file in (ann_data_path): 
        file_name = ann_file.split('/')[-1]
        print(file_name)
        with open(ann_file, 'r', encoding='utf-8') as f:
            data_item = json.load(f)
            data_list_dict[file_name] = data_item
    processed_data_list = []
    count_cnt = 0
    for file_name, data_list in data_list_dict.items():
        for data_item in tqdm(data_list): 
            # if count_cnt > 0: 
            #     break
            graph_path = osp.join(graph_root, data_item['graph']['graph'])
            graph_dict = torch.load(graph_path)
            graph_dict = graph_dict.to(device)
            if args.data_type == 'dblp': 
                new_conf_feas = torch.ones([graph_dict['conference'].num_nodes, 768])
                new_conf_feas = new_conf_feas.to(device)
                graph_dict['conference'].x = new_conf_feas
            with torch.no_grad():
                res = hgt(x_dict = graph_dict.x_dict,
                    edge_index_dict = graph_dict.edge_index_dict,  # Support both.
                    node_type_feas_dict = node_feas_dict,
                    edge_type_feas_dict = edge_feas_dict)
            for k, v in res.items():
                res[k] = v.cpu()
                if torch.any(torch.isnan(res[k])): 
                    print(k, res[k])
                    raise ValueError
            processed_graph_path = re.sub('graph_data', f'graph_data_processed_{hgnn_name}', data_item['graph']['graph'])

            processed_data_item = data_item.copy()
            processed_data_item['graph']['graph'] = processed_graph_path
            processed_data_list.append(processed_data_item.copy())
            save_path = osp.join(graph_root, processed_graph_path)
            if osp.exists(osp.dirname(save_path)) is False: 
                os.makedirs(osp.dirname(save_path), exist_ok=True)
            # print(graph_dict.x_dict['author'])
            # for k, v in graph_dict.items():
            #     graph_dict.x_dict[k] = res[k]
            graph_dict = graph_dict.to('cpu')
            graph_dict.x_dict = res
            # print(graph_dict.x_dict['author'])
            # print(res['author'])
            torch.save(graph_dict, save_path)
            count_cnt += 1
            # time.sleep(1)
        processed_json_file = osp.join(graph_root, dsname, f'ann_processed_{hgnn_name}', file_name)
        if osp.exists(osp.dirname(processed_json_file)) is False: 
            os.makedirs(osp.dirname(processed_json_file), exist_ok=True)
        with open(processed_json_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data_list, f)

if __name__ == '__main__': 
    seed_everything(42)
    check_offline_hgt()