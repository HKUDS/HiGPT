# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
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

import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List

import torch

import transformers
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from higpt.train.graphchat_trainer import GraphChatTrainer

from higpt import conversation as conversation_lib
from higpt.model import *

from PIL import Image
import torch.nn as nn
from torch_geometric.data import Data
from lightning.pytorch.strategies import FSDPStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import LightningModule, Trainer, seed_everything
from higpt.model.HeteroLlama_pl import HeteroGPT_pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.callback import Callback
import os.path as osp
from lightning import LightningModule, LightningDataModule
import glob
# TODO: import and use code from ../data/dataset.py

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
DEFAULT_GRAPH_TOKEN = "<graph>"
DEFAULT_GRAPH_PATCH_TOKEN = "<g_patch>"
DEFAULT_G_START_TOKEN = "<g_start>"
DEFAULT_G_END_TOKEN = "<g_end>"


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_graph_mlp_adapter: bool = field(default=False)
    graph_tower: Optional[str] = field(default=None)
    graph_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_graph_mlp_adapter: Optional[str] = field(default=None)
    use_graph_start_end: bool = field(default=False)
    model_save_name: Optional[str] = field(default="model_{epoch}-{step}")
    tune_gnn: bool = field(default=False)


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_graph: bool = False
    sep_graph_conv_front: bool = False
    graph_token_len: int = 0
    graph_content: Optional[str] = field(default=None)
    graph_root: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    hetero_key_path: Optional[str] = field(default=None)
    


@dataclass
class TrainingArguments:
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_graph_mlp_adapter: bool = field(default=False)
    force_fsdp: bool = field(default=False)
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    strategy: str = field(
        default='fsdp'
    )
    real_batch_size: int = field(default=1)

    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    disable_tqdm: bool =False

    gpus: Optional[str] = field(default='0,1')
    resume: Optional[str] = field(default=None)

    adam_epsilon: float = field(default=1e-8)
    warmup_steps:int = field(default=1000)
    num_workers:int = field(default=16)

    bf16: bool = field(default=False) 
    fp16: bool = field(default=False) 
    output_dir: str = field(default='./checkpoints/graphchat-gt-graphmatch-7b') 
    num_train_epochs: int = field(default=3)
    per_device_train_batch_size: int = field(default=1)
    per_device_eval_batch_size: int = field(default=1)
    gradient_accumulation_steps: int = field(default=1)
    evaluation_strategy: str = field(default='no')
    save_strategy: str = field(default='steps')
    save_steps: int = field(default=2400)
    save_total_limit: int = field(default=1)
    learning_rate: float = field(default=2e-5)
    weight_decay: float = field(default=0.)
    warmup_ratio: float = field(default=0.03)
    lr_scheduler_type: str = field(default='cosine')
    logging_steps: int = field(default=1)
    tf32: bool = field(default=True) 
    gradient_checkpointing: bool = field(default=True)
    report_to: str = field(default='wandb')


class SaveGraphProjectorCallback(Callback):
    def __init__(self, output_dir, keys_to_match):
        self.output_dir = output_dir
        self.keys_to_match = keys_to_match

    def on_train_epoch_end(self, trainer, pl_module, unused=None):
        # 准备保存模型权重
        _state_dict = pl_module.state_dict()

        weight_to_save = {}
        for k, v in _state_dict.items():
            if any(key_match in k for key_match in self.keys_to_match):
                weight_to_save[k] = v

        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 保存 graph projector 的权重
        torch.save(weight_to_save, os.path.join(self.output_dir, 'graph_projector.bin'))


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])


    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)





def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_graph(
    sources: Sequence[str],
    graph_cfg: dict,
    cur_token_len: int,
) -> Dict:
    is_graph = graph_cfg['is_graph']
    # image_token_len = multimodal_cfg['image_token_len']
    graph_token_len = cur_token_len
    if not is_graph:
        return sources

    for source in sources:
        if graph_cfg['sep_graph_conv_front']:
            assert DEFAULT_GRAPH_TOKEN in source[0]['value']
            source[0]['value'] = source[0]['value'].replace(DEFAULT_GRAPH_TOKEN, '').strip()
            source[0]['value'] = DEFAULT_GRAPH_TOKEN + conversation_lib.default_conversation.sep + conversation_lib.default_conversation.roles[0] + ": " + source[0]['value']
        for sentence in source:
            replace_token = DEFAULT_GRAPH_PATCH_TOKEN * graph_token_len
            if graph_cfg['use_graph_start_end']:
                replace_token = DEFAULT_G_START_TOKEN + replace_token + DEFAULT_G_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_GRAPH_TOKEN, replace_token)

    return sources

def preprocess_graph_LP(
    sources: Sequence[str],
    graph_cfg: dict,
    cur_token_len_1: int,
    cur_token_len_2: int,
) -> Dict:
    is_graph = graph_cfg['is_graph']
    # image_token_len = multimodal_cfg['image_token_len']
    graph_token_len_1 = cur_token_len_1
    graph_token_len_2 = cur_token_len_2

    if not is_graph:
        return sources

    for source in sources:
        if graph_cfg['sep_graph_conv_front']:
            assert DEFAULT_GRAPH_TOKEN in source[0]['value']
            source[0]['value'] = source[0]['value'].replace(DEFAULT_GRAPH_TOKEN, '').strip()
            source[0]['value'] = DEFAULT_GRAPH_TOKEN + conversation_lib.default_conversation.sep + conversation_lib.default_conversation.roles[0] + ": " + source[0]['value']
        for sentence in source:
            replace_token_1 = DEFAULT_GRAPH_PATCH_TOKEN * graph_token_len_1
            replace_token_2 = DEFAULT_GRAPH_PATCH_TOKEN * graph_token_len_2
            if graph_cfg['use_graph_start_end']:
                replace_token_1 = DEFAULT_G_START_TOKEN + replace_token_1 + DEFAULT_G_END_TOKEN
                replace_token_2 = DEFAULT_G_START_TOKEN + replace_token_2 + DEFAULT_G_END_TOKEN

            if DEFAULT_GRAPH_TOKEN in sentence["value"]:
                first_index = sentence["value"].find(DEFAULT_GRAPH_TOKEN)
                sentence["value"] = sentence["value"][:first_index] + replace_token_1 + sentence["value"][first_index+len(DEFAULT_GRAPH_TOKEN):]

                # 替换第二个<graph>为B
                second_index = sentence["value"].find(DEFAULT_GRAPH_TOKEN)
                sentence["value"] = sentence["value"][:second_index] + replace_token_2 + sentence["value"][second_index+len(DEFAULT_GRAPH_TOKEN):]


            # sentence["value"] = sentence["value"].replace(DEFAULT_GRAPH_TOKEN, replace_token)

    # print(sources)

    return sources

def preprocess_graph_Hetero(
    sources: Sequence[str],
    graph_cfg: dict,
    cur_token_lens: List[int],
) -> Dict:
    is_graph = graph_cfg['is_graph']
    # image_token_len = multimodal_cfg['image_token_len']
    graph_token_lens = cur_token_lens

    if not is_graph:
        return sources

    for source in sources:
        if graph_cfg['sep_graph_conv_front']:
            for sentence in source:
                if DEFAULT_GRAPH_TOKEN in sentence['value']: 
                    sentence['value'] = sentence['value'].replace(DEFAULT_GRAPH_TOKEN, '').strip()
                    sentence['value'] = DEFAULT_GRAPH_TOKEN + conversation_lib.default_conversation.sep + conversation_lib.default_conversation.roles[0] + ": " + sentence['value']
        for sentence in source:
            if DEFAULT_GRAPH_TOKEN in sentence["value"]:
                # build replace_tokens
                replace_tokens = []
                for i, token_len in enumerate(graph_token_lens):
                    replace_token = DEFAULT_GRAPH_PATCH_TOKEN * token_len
                    if graph_cfg['use_graph_start_end']:
                        replace_token = DEFAULT_G_START_TOKEN + replace_token + DEFAULT_G_END_TOKEN
                    replace_tokens.append(replace_token)

                for i, replace_token in enumerate(replace_tokens):
                    index = sentence["value"].find(DEFAULT_GRAPH_TOKEN)
                    sentence["value"] = sentence["value"][:index] + replace_token + sentence["value"][index+len(DEFAULT_GRAPH_TOKEN):]

    return sources


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(tokenizer(rou).input_ids)
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )

def preprocess_mpt(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(tokenizer(rou).input_ids) + len(tokenizer(conv.sep).input_ids)
            instruction_len = len(tokenizer(parts[0]).input_ids)
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.version == "v1":
        return preprocess_v1(sources, tokenizer)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    conversations_tokenized = _tokenize_fn(conversations, tokenizer)
    input_ids = conversations_tokenized["input_ids"]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source],
                                      tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = json.load(open(data_path, "r"))

        logging.warning("Formatting inputs...")
        sources = [example["conversations"] for example in list_data_dict]
        data_dict = preprocess(sources, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


class LazySupervisedDatasetHetero(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 graph_cfg: dict, 
                 graph_root: str, 
                 hetero_key_path: str, 
                 **kwargs,):
        super(LazySupervisedDatasetHetero, self).__init__()
        logging.warning("Loading data...")
        data_path = data_path.split(",")
        ann_data_path = []
        for i, path in enumerate(data_path):
            ann_file = glob.glob(osp.join(graph_root, path, 'ann', '**/*.json'), recursive=True)
            # assert len(ann_file) == 1, f"Need to have one ann file for each graph"
            ann_data_path.extend(ann_file)

        # list_data_dict = json.load(open(data_path, "r"))
        list_data_dict = []
        for i, ann_file in enumerate(ann_data_path):
            ann_data = json.load(open(ann_file, "r", encoding= "utf-8"))
            list_data_dict.extend(ann_data)

        logging.warning("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.graph_cfg = graph_cfg

        hetero_key_path = hetero_key_path
        assert hetero_key_path is not None, "Need hetero key path"
        # self.hetero_key_order = json.load(open(hetero_key_path, "r", encoding= "utf-8"))
        self.graph_root = graph_root

        self.node_feas_dict_dblp = torch.load('/root/paddlejob/workspace/env_run/output/HeteGPT/hetegpt/model/meta_hgt/meta_dict/dblp/node_type.pt')
        for k in self.node_feas_dict_dblp.keys():
            self.node_feas_dict_dblp[k] = torch.Tensor(self.node_feas_dict_dblp[k])
        self.edge_feas_dict_dblp = torch.load('/root/paddlejob/workspace/env_run/output/HeteGPT/hetegpt/model/meta_hgt/meta_dict/dblp/edge_type.pt')
        for k in self.edge_feas_dict_dblp.keys():
            self.edge_feas_dict_dblp[k] = torch.Tensor(self.edge_feas_dict_dblp[k])

        self.node_feas_dict_acm = torch.load('/root/paddlejob/workspace/env_run/output/HeteGPT/hetegpt/model/meta_hgt/meta_dict/acm/node_type.pt')
        for k in self.node_feas_dict_acm.keys():
            self.node_feas_dict_acm[k] = torch.Tensor(self.node_feas_dict_acm[k])
        self.edge_feas_dict_acm = torch.load('/root/paddlejob/workspace/env_run/output/HeteGPT/hetegpt/model/meta_hgt/meta_dict/acm/edge_type.pt')
        for k in self.node_feas_dict_acm.keys():
            self.node_feas_dict_acm[k] = torch.Tensor(self.node_feas_dict_acm[k])

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

        hetero_key_order = self.list_data_dict[i]['graph']['keys_order']

        if 'graph' in sources[0]:
            graph_path = osp.join(self.graph_root, self.list_data_dict[i]['graph']['graph'])
            graph_dict = torch.load(graph_path)
            # feas_dict
            if 'subject' in graph_dict.x_dict.keys(): 
                graph_dict['edge_feas_dict'] = self.edge_feas_dict_acm
                graph_dict['node_feas_dict'] = self.node_feas_dict_acm
            elif 'movie' in graph_dict.x_dict.keys(): 
                graph_dict['edge_feas_dict'] = self.edge_feas_dict_imdb
                graph_dict['node_feas_dict'] = self.node_feas_dict_imdb
            elif 'paper' in graph_dict.x_dict.keys(): 
                graph_dict['edge_feas_dict'] = self.edge_feas_dict_dblp
                graph_dict['node_feas_dict'] = self.node_feas_dict_dblp
                new_conf_reps = torch.ones(graph_dict['conference'].num_nodes, 768)
                graph_dict['conference'].x = new_conf_reps
            else: 
                raise NotImplementedError
            cur_token_lens = []
            for key in hetero_key_order:
                cur_token_lens.append(graph_dict.x_dict[key].shape[0])
            assert type(cur_token_lens[0]) == int, f"Need to be int, not {type(cur_token_lens[0])}"
            sources = preprocess_graph_Hetero(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.graph_cfg, cur_token_lens)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess(
            sources,
            self.tokenizer)
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        if 'graph' in self.list_data_dict[i]:
            data_dict['graph_data'] = graph_dict
        elif self.graph_cfg['is_graph']:
            # image does not exist in the data, but the model is multimodal
            node_feas = self.graph_cfg['graph_processor'].node_feas
            data_dict['graph_data'] = Data(graph_node = torch.zeros(3, node_feas), edge_index=torch.zeros(2, 3), target_node = torch.tensor([0]))

        data_dict['hetero_key_order'] = hetero_key_order
        return data_dict

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'graph_data' in instances[0]:
            graph_data_batch = [instance['graph_data'] for instance in instances]
            key_order_batch = [instance['hetero_key_order'] for instance in instances] 
            
        batch['graph_data'] = graph_data_batch
        batch['hetero_key_order'] = key_order_batch

        return batch


class HeteLitDataModule(LightningDataModule):
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer,
                                data_args, training_args) -> None:
        super().__init__()
        self.training_args = training_args
        self.data_args = data_args
        dataset_cls = (LazySupervisedDatasetHetero
                   if data_args.lazy_preprocess else SupervisedDataset)
        self.train_data = dataset_cls(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                graph_cfg=dict(
                                    is_graph=data_args.is_graph,
                                    sep_graph_conv_front=data_args.sep_graph_conv_front,
                                    graph_token_len=data_args.graph_token_len,
                                    graph_content=data_args.graph_content,
                                    use_graph_start_end=getattr(data_args, 'use_graph_start_end', False)
                                    ), 
                                    graph_root = data_args.graph_root, 
                                    hetero_key_path = data_args.hetero_key_path)
        
        self.data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    
    def train_dataloader(self):
        return DataLoader(self.train_data, 
                                  batch_size=self.training_args.per_device_train_batch_size,
                                    num_workers=self.training_args.num_workers,
                                  collate_fn=self.data_collator,
                                  prefetch_factor=4,
                                  pin_memory=True)
    

def train():
    seed_everything(42)
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if isinstance(training_args.gpus, str):
        training_args.gpus = [int(x) for x in training_args.gpus.split(',')]
    batch_size = training_args.real_batch_size
    devices = training_args.gpus
    num_devices = len(devices)
    gradient_accumulation_steps = max(1,batch_size // (training_args.per_device_train_batch_size*num_devices))

    tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False
        )

    if model_args.version == "v1":
        tokenizer.pad_token = tokenizer.unk_token
        conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1_1"]
    else: 
        raise ValueError

    model = HeteroGPT_pl(training_args, model_args, data_args, tokenizer)

    data_module = HeteLitDataModule(tokenizer,
                                data_args, training_args)
    checkpoint_callback = ModelCheckpoint(
            dirpath=training_args.output_dir,
            filename=model_args.model_save_name,
            monitor="train_loss",
            save_top_k=1,
            save_last=True,
        )

    if training_args.strategy == 'fsdp': 
        strategy = FSDPStrategy(
        auto_wrap_policy={LlamaDecoderLayer},
        activation_checkpointing_policy={LlamaDecoderLayer},
        state_dict_type="full",
        limit_all_gathers=True,
        cpu_offload=False,
        # **kwargs
        )
    else: 
        strategy = training_args.strategy

    wandb_logger = WandbLogger(save_dir=training_args.output_dir, project="GraphGPTv1", offline=True, name=model_args.model_save_name)
    model_precision = ('16' if training_args.fp16 else ('bf16' if training_args.bf16 else '32'))
    # print('************* epoch:', training_args.num_train_epochs)
    trainer = Trainer(default_root_dir=training_args.output_dir, max_epochs=int(training_args.num_train_epochs), 
                    accumulate_grad_batches=gradient_accumulation_steps,
                    accelerator="gpu", devices=devices, 
                    strategy=strategy,
                    logger = wandb_logger, 
                    precision=model_precision,
                    callbacks=[checkpoint_callback])
    resume = training_args.resume

    for name, param in model.named_parameters():
        if param.dtype == torch.float16:
            print(name, param.dtype)
    # model.to(dtype=torch.float16)

    trainer.fit(model, data_module, ckpt_path = resume)

    # safe_save_model_for_hf_trainer(trainer=trainer,
    #                                    output_dir=training_args.output_dir)

if __name__ == "__main__":
    train()
