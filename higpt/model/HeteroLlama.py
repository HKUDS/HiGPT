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

DEFAULT_GRAPH_TOKEN = "<graph>"
DEFAULT_GRAPH_PATCH_TOKEN = "<g_patch>"
DEFAULT_G_START_TOKEN = "<g_start>"
DEFAULT_G_END_TOKEN = "<g_end>"


class HeteroLlamaConfig(LlamaConfig):
    model_type = "HeteroLlama"


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


class HeteroLlamaModel(LlamaModel):
    config_class = HeteroLlamaConfig

    def __init__(self, config: LlamaConfig):
        super(HeteroLlamaModel, self).__init__(config)

        if hasattr(config, "graph_tower"):
            # HACK: for FSDP

            if config.graph_tower.split('_')[0] == "MetaHGT": 
                self.graph_tower = load_metahgt_pretrained(MetaHGTConv, config.pretrain_graph_model_path)

            # self.vision_tower = CLIPVisionModel.from_pretrained(config.mm_vision_tower)

        if hasattr(config, "use_graph_proj"):
            self.graph_projector = nn.Linear(config.graph_hidden_size, config.hidden_size)

    def get_graph_tower(self):
        graph_tower = getattr(self, 'graph_tower', None)
        if type(graph_tower) is list:
            graph_tower = graph_tower[0]
        return graph_tower

    def initialize_graph_modules(self, graph_tower, graph_select_layer,
                                  pretrain_graph_mlp_adapter=None, fsdp=None): # TODO: modify this function
        self.config.graph_tower = graph_tower


        if not hasattr(self, 'graph_tower'):
            if self.config.graph_tower.split('_')[0] == "MetaHGT":
                graph_tower = load_metahgt_pretrained(MetaHGTConv, self.config.pretrain_graph_model_path)
        else:
            graph_tower = self.graph_tower
        graph_tower.requires_grad_(False)

        if fsdp is not None and len(fsdp) > 0:
            self.graph_tower = [graph_tower]
        else:
            self.graph_tower = graph_tower

        

        self.config.use_graph_proj = True
        self.config.graph_select_layer = graph_select_layer

        if not hasattr(self, 'graph_projector'):
            self.graph_projector = nn.Linear(self.config.graph_hidden_size, self.config.hidden_size)

        if pretrain_graph_mlp_adapter is not None:
            graph_projector_weights = torch.load(pretrain_graph_mlp_adapter, map_location='cpu')
            self.graph_projector.load_state_dict({k.split('.')[-1]: v for k, v in graph_projector_weights.items()})

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        # graph_node_reps: Optional[torch.FloatTensor] = None,
        # edge_index_reps: Optional[torch.FloatTensor] = None,
        graph_data: Optional[Data] = None,
        return_dict: Optional[bool] = None,
        hetero_key_order: Optional[List[List[str]]] =None
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        # HACK: replace back original embeddings for LLaVA pretraining
        orig_embeds_params = getattr(self, 'orig_embeds_params', None)
        # if orig_embeds_params is not None:
        #     orig_embeds_params = orig_embeds_params[0]
        #     with torch.no_grad():
        #         self.get_input_embeddings().weight.data[:-2] = orig_embeds_params[:-2].data

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # graph_tower = self.get_graph_tower()
        if self.graph_tower is not None and (input_ids.shape[1] != 1 or self.training) and graph_data is not None:
            # TODO: this is a modified multimodal LLM -- Haotian Liu

            if type(graph_data[0]) is list:
                graph_node_features = []
                for g_list, keys in zip(graph_data, hetero_key_order): 
                    if type(keys[0]) is list: 
                        for g, g_k in zip(g_list, keys):
                            ret_g = g.x_dict
                            for k in g_k:
                                if torch.any(torch.isnan(ret_g[k])): 
                                    print(k, ret_g[k])
                                    raise ValueError
                                graph_node_features.append(ret_g[k])
                    else:
                        for g in g_list:
                            ret_g = g.x_dict
                            for k in keys:
                                if torch.any(torch.isnan(ret_g[k])): 
                                    print(k, ret_g[k])
                                    raise ValueError
                                graph_node_features.append(ret_g[k])
            elif type(graph_data) is list:
                # variable length images
                graph_node_features = []
                for g, keys in zip(graph_data, hetero_key_order):
                    # print(g)
                    # print(self.graph_tower.parameters().__next__().device)
                    # print(g.x_dict['paper'].device)
                    # cur_device = self.graph_tower.parameters().__next__().device
                    # self.graph_tower = self.graph_tower.to(cur_device)
                    # g = g.to(cur_device)
                    # with torch.no_grad():
                    #     ret_g = self.graph_tower(g.x_dict, g.edge_index_dict, node_type_feas_dict = g['node_feas_dict'], edge_type_feas_dict = g['edge_feas_dict'])
                    # for k, v in ret_g.items():
                    #     ret_g[k] = v.to(cur_device)
                    ret_g = g.x_dict
                    for k in keys:
                        if torch.any(torch.isnan(ret_g[k])): 
                            print(k, ret_g[k])
                            raise ValueError
                        graph_node_features.append(ret_g[k])
            else:
                raise ValueError(f'graph_node_reps is expected to be a list but got {type(graph_data)}')
            if type(graph_data) is list:
                # if type(graph_node_features[0]) is not dict:
                graph_node_features = [self.graph_projector(node_feature) for node_feature in graph_node_features]
            else:
                raise ValueError(f'graph_node_reps is expected to be a list but got {type(graph_data)}')
            dummy_graph_features = torch.zeros(256, 768, device=inputs_embeds.device, dtype=inputs_embeds.dtype)

            new_input_embeds = []
            cur_graph_idx = 0
            for cur_input_ids, cur_input_embeds in zip(input_ids, inputs_embeds):
                if (cur_input_ids == self.graph_tower.config.graph_patch_token).sum() == 0:
                    # multimodal LLM, but the current sample is not multimodal
                    cur_input_embeds = cur_input_embeds + (0. * dummy_graph_features).sum()
                    new_input_embeds.append(cur_input_embeds)
                    cur_graph_idx += 1
                    continue
                if self.graph_tower.config.use_graph_start_end:
                    cur_graph_features = graph_node_features[cur_graph_idx]
                    num_patches = cur_graph_features.shape[0]
                    if (cur_input_ids == self.graph_tower.config.graph_start_token).sum() != (cur_input_ids == self.graph_tower.config.graph_end_token).sum():
                        raise ValueError("The number of graph start tokens and graph end tokens should be the same.")
                    graph_start_tokens = torch.where(cur_input_ids == self.graph_tower.config.graph_start_token)[0]
                    # print(graph_start_tokens)
                    for graph_start_token_pos in graph_start_tokens:
                        cur_graph_features = graph_node_features[cur_graph_idx].to(device=cur_input_embeds.device)
                        num_patches = cur_graph_features.shape[0]
                        if cur_input_ids[graph_start_token_pos + num_patches + 1] != self.graph_tower.config.graph_end_token:
                            raise ValueError("The graph end token should follow the graph start token.")
                        if orig_embeds_params is not None:
                            cur_new_input_embeds = torch.cat((cur_input_embeds[:graph_start_token_pos].detach(), cur_input_embeds[graph_start_token_pos:graph_start_token_pos+1], cur_graph_features, cur_input_embeds[graph_start_token_pos + num_patches + 1:graph_start_token_pos + num_patches + 2], cur_input_embeds[graph_start_token_pos + num_patches + 2:].detach()), dim=0)
                        else:
                            cur_new_input_embeds = torch.cat((cur_input_embeds[:graph_start_token_pos+1], cur_graph_features, cur_input_embeds[graph_start_token_pos + num_patches + 1:]), dim=0)
                        cur_graph_idx += 1
                    new_input_embeds.append(cur_new_input_embeds)
                else:
                    cur_graph_features = graph_node_features[cur_graph_idx]
                    num_patches = cur_graph_features.shape[0]
                    if (cur_input_ids == self.graph_tower.config.graph_patch_token).sum() != num_patches:
                        raise ValueError("The number of graph patch tokens should be the same as the number of graph patches.")
                    masked_indices = torch.where(cur_input_ids == self.graph_tower.config.graph_patch_token)[0]
                    mask_index_start = masked_indices[0]
                    if (masked_indices != torch.arange(mask_index_start, mask_index_start+num_patches, device=masked_indices.device, dtype=masked_indices.dtype)).any():
                        raise ValueError("The graph patch tokens should be consecutive.")
                    if orig_embeds_params is not None:
                        cur_new_input_embeds = torch.cat((cur_input_embeds[:mask_index_start].detach(), cur_graph_features, cur_input_embeds[mask_index_start+num_patches:].detach()), dim=0)
                    else:
                        cur_new_input_embeds = torch.cat((cur_input_embeds[:mask_index_start], cur_graph_features, cur_input_embeds[mask_index_start+num_patches:]), dim=0)
                    new_input_embeds.append(cur_new_input_embeds)
                    cur_graph_idx += 1

            # print(cur_graph_idx)
            # print(len(graph_node_features))
            assert cur_graph_idx == len(graph_node_features)
            inputs_embeds = torch.stack(new_input_embeds, dim=0)

        return super(HeteroLlamaModel, self).forward(
            input_ids=None, attention_mask=attention_mask, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )


class HeteroLlamaForCausalLM(LlamaForCausalLM):
    config_class = HeteroLlamaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = HeteroLlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def get_graph_tower(self):
        return self.get_model().get_graph_tower()

    def get_vision_tower(self):
        model = self.get_model()
        graph_tower = model.graph_tower
        if type(graph_tower) is list:
            graph_tower = graph_tower[0]
        return graph_tower

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        # graph_node_reps: Optional[torch.FloatTensor] = None,
        # edge_index_reps: Optional[torch.FloatTensor] = None,
        graph_data: Optional[Data] = None,
        return_dict: Optional[bool] = None,
        hetero_key_order: Optional[List[List[str]]] =None, 
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            # graph_node_reps=graph_node_reps, 
            # edge_index_reps=edge_index_reps
            graph_data = graph_data, 
            hetero_key_order = hetero_key_order
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}
        if kwargs.get("graph_data") is None: 
            model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "graph_data": None,
                # "edge_index_reps": kwargs.get("edge_index_reps", None),
                "hetero_key_order": [kwargs.get("hetero_key_order", None)]
            }
            )
        else:
            model_inputs.update(
                {
                    "past_key_values": past_key_values,
                    "use_cache": kwargs.get("use_cache"),
                    "attention_mask": attention_mask,
                    "graph_data": [kwargs.get("graph_data", None)],
                    # "edge_index_reps": kwargs.get("edge_index_reps", None),
                    "hetero_key_order": [kwargs.get("hetero_key_order", None)]
                }
            )
        return model_inputs

    def initialize_graph_tokenizer(self, use_graph_start_end, tokenizer, device,
                                    tune_graph_mlp_adapter=False, pretrain_graph_mlp_adapter=None):
        vision_config = self.get_graph_tower().config
        vision_config.use_graph_start_end = use_graph_start_end
        tokenizer.add_tokens([DEFAULT_GRAPH_PATCH_TOKEN], special_tokens=True)
        self.resize_token_embeddings(len(tokenizer))

        if use_graph_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_G_START_TOKEN, DEFAULT_G_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))
            vision_config.graph_start_token, vision_config.graph_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_G_START_TOKEN, DEFAULT_G_END_TOKEN])

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if tune_graph_mlp_adapter:
                # self.get_model().orig_embeds_params = [self.get_input_embeddings().weight.data.clone().to(device=device)]
                self.get_model().orig_embeds_params = [self.get_input_embeddings().weight.data.clone().to(device=device)]
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if pretrain_graph_mlp_adapter:
                mm_projector_weights = torch.load(pretrain_graph_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")

        vision_config.graph_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_GRAPH_PATCH_TOKEN])[0]

AutoConfig.register("HeteroLlama", HeteroLlamaConfig)
AutoModelForCausalLM.register(HeteroLlamaConfig, HeteroLlamaForCausalLM)
