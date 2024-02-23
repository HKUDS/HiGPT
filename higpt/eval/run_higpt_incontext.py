import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from higpt.conversation import conv_templates, SeparatorStyle
from higpt.utils import disable_torch_init
from transformers import CLIPVisionModel, CLIPImageProcessor, StoppingCriteria
from higpt.model import *
from higpt.model.utils import KeywordsStoppingCriteria
from torch_geometric.data import Data
import json
import copy
from higpt.model.meta_hgt import MetaHGTConvCfg, MetaHGTConv
import torch.nn as nn
from transformers.configuration_utils import PretrainedConfig

import os
import requests
from PIL import Image
from io import BytesIO

from tqdm import tqdm
import json
import os.path as osp

import ray

# os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

DEFAULT_GRAPH_TOKEN = "<graph>"
DEFAULT_GRAPH_PATCH_TOKEN = "<g_patch>"
DEFAULT_G_START_TOKEN = "<g_start>"
DEFAULT_G_END_TOKEN = "<g_end>"

node_feas_dict_dblp = torch.load('./higpt/model/meta_hgt/meta_dict/dblp/node_type.pt')
for k in node_feas_dict_dblp.keys():
    node_feas_dict_dblp[k] = torch.Tensor(node_feas_dict_dblp[k])
edge_feas_dict_dblp = torch.load('./higpt/model/meta_hgt/meta_dict/dblp/edge_type.pt')
for k in edge_feas_dict_dblp.keys():
    edge_feas_dict_dblp[k] = torch.Tensor(edge_feas_dict_dblp[k])

node_feas_dict_acm = torch.load('./higpt/model/meta_hgt/meta_dict/acm/node_type.pt')
for k in node_feas_dict_acm.keys():
    node_feas_dict_acm[k] = torch.Tensor(node_feas_dict_acm[k])
edge_feas_dict_acm = torch.load('./higpt/model/meta_hgt/meta_dict/acm/edge_type.pt')
for k in node_feas_dict_acm.keys():
    node_feas_dict_acm[k] = torch.Tensor(node_feas_dict_acm[k])

node_feas_dict_imdb = torch.load('./higpt/model/meta_hgt/meta_dict/imdb/node_type.pt')
for k in node_feas_dict_imdb.keys():
    node_feas_dict_imdb[k] = torch.Tensor(node_feas_dict_imdb[k])
edge_feas_dict_imdb = torch.load('./higpt/model/meta_hgt/meta_dict/imdb/edge_type.pt')
for k in node_feas_dict_imdb.keys():
    node_feas_dict_imdb[k] = torch.Tensor(node_feas_dict_imdb[k])

class HGT(nn.Module):
    
    def __init__(
        self,
    ):
        super().__init__()
        self.config = PretrainedConfig()

def load_graph(instruct_item, graph_root): 
    graph_dict_list = []
    cur_token_lens = []
    # hetero_key_order = instruct_item['graph']['keys_order']

    hetero_key_orders = []

    context_graphs = instruct_item['context_graph']
    for cg_dict in context_graphs: 
        cg_path = osp.join(graph_root, cg_dict['graph'])
        graph_dict = torch.load(cg_path)
        if 'subject' in graph_dict.x_dict.keys(): 
            graph_dict['edge_feas_dict'] = edge_feas_dict_acm
            graph_dict['node_feas_dict'] = node_feas_dict_acm
        elif 'movie' in graph_dict.x_dict.keys(): 
            graph_dict['edge_feas_dict'] = edge_feas_dict_imdb
            graph_dict['node_feas_dict'] = node_feas_dict_imdb
        elif 'paper' in graph_dict.x_dict.keys(): 
            graph_dict['edge_feas_dict'] = edge_feas_dict_dblp
            graph_dict['node_feas_dict'] = node_feas_dict_dblp
            new_conf_reps = torch.ones(graph_dict['conference'].num_nodes, 768)
            graph_dict['conference'].x = new_conf_reps
        else: 
            raise NotImplementedError
        graph_dict_list.append(graph_dict)
        hetero_key_order = cg_dict['keys_order']
        hetero_key_orders.append(hetero_key_order)
        for key in hetero_key_order:
            cur_token_lens.append(graph_dict.x_dict[key].shape[0])
    graph_path = osp.join(graph_root, instruct_item['graph']['graph'])
    graph_dict = torch.load(graph_path)
    if 'subject' in graph_dict.x_dict.keys(): 
        graph_dict['edge_feas_dict'] = edge_feas_dict_acm
        graph_dict['node_feas_dict'] = node_feas_dict_acm
    elif 'movie' in graph_dict.x_dict.keys(): 
        graph_dict['edge_feas_dict'] = edge_feas_dict_imdb
        graph_dict['node_feas_dict'] = node_feas_dict_imdb
    elif 'paper' in graph_dict.x_dict.keys(): 
        graph_dict['edge_feas_dict'] = edge_feas_dict_dblp
        graph_dict['node_feas_dict'] = node_feas_dict_dblp
        new_conf_reps = torch.ones(graph_dict['conference'].num_nodes, 768)
        graph_dict['conference'].x = new_conf_reps
    else: 
        raise NotImplementedError
    hetero_key_order = instruct_item['graph']['keys_order']
    hetero_key_orders.append(hetero_key_order)
    for key in hetero_key_order:
        cur_token_lens.append(graph_dict.x_dict[key].shape[0])
    graph_dict_list.append(graph_dict)
        

    return {
        'graph_data': graph_dict_list, 
        'graph_token_len': cur_token_lens, 
        'hetero_key_orders': hetero_key_orders
    }


def load_prompting_file(file_path): 
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# def prepare_query(instruct_item): 


def run_eval(args, num_gpus):
    # split question file into num_gpus files
    prompt_file = load_prompting_file(args.prompting_file)
    prompt_file = prompt_file[args.start_id:args.end_id]
    chunk_size = len(prompt_file) // num_gpus
    ans_handles = []
    split_list = list(range(args.start_id, args.end_id, chunk_size))
    idx_list = list(range(0, len(prompt_file), chunk_size))
    if len(split_list) == num_gpus: 
        split_list.append(args.end_id)
        idx_list.append(len(prompt_file))
    elif len(split_list) == num_gpus + 1: 
        split_list[-1] = args.end_id
        idx_list[-1] = len(prompt_file)
    else: 
        raise ValueError('error in the number of list')

    if osp.exists(args.output_res_path) is False: 
        os.makedirs(args.output_res_path, exist_ok = True)
    
    for idx in range(len(idx_list) - 1):
        start_idx = idx_list[idx]
        end_idx = idx_list[idx + 1]
        
        start_split = split_list[idx]
        end_split = split_list[idx + 1]
        ans_handles.append(
            eval_model.remote(
                args, prompt_file[start_idx:end_idx], start_split, end_split
            )
        )

    ans_jsons = []
    for ans_handle in ans_handles:
        ans_jsons.extend(ray.get(ans_handle))

    # with open(args.output_res_path, "w") as ans_file:
    #     for line in ans_jsons:
    #         ans_file.write(json.dumps(line) + "\n")


@ray.remote(num_gpus=1)
@torch.inference_mode()
def eval_model(args, prompt_file, start_idx, end_idx):
    # load prompting file
    # prompt_file = load_prompting_file(args.prompting_file)


    # Model
    disable_torch_init()
    # model_name = os.path.expanduser(args.model_name)
    print('start loading')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    print('finish loading')

    print('start loading')
    model = HeteroLlamaForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float32, use_cache=True, low_cpu_mem_usage=True)
    print('finish loading')

    use_graph_start_end = getattr(model.config, "use_graph_start_end", False)
    tokenizer.add_tokens([DEFAULT_GRAPH_PATCH_TOKEN], special_tokens=True)
    if use_graph_start_end:
        tokenizer.add_tokens([DEFAULT_G_START_TOKEN, DEFAULT_G_END_TOKEN], special_tokens=True)

    graph_tower = model.get_model().graph_tower
    
    # TODO: add graph tower
    # if graph_tower.device.type == 'meta':
    #     print('meta')
    graph_tower= load_metahgt_pretrained(MetaHGTConv, './MetaHGT_imdb_dblp_epoch5')
    new_graph_tower = HGT()
    new_graph_tower.config = graph_tower.config
    
    model.get_model().graph_tower = new_graph_tower.cuda()
    # else:
    #     print('other')
    # print(next(graph_tower.parameters()).dtype)
    # graph_tower.to(device='cuda', dtype=torch.float32)
    model = model.cuda()
    graph_config = graph_tower.config
    graph_config.graph_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_GRAPH_PATCH_TOKEN])[0]
    graph_config.use_graph_start_end = use_graph_start_end
    if use_graph_start_end:
        graph_config.graph_start_token, graph_config.graph_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_G_START_TOKEN, DEFAULT_G_END_TOKEN])
    # TODO: add graph token len

    res_data = []
    print(f'total: {len(prompt_file)}')
    for idx, instruct_item in tqdm(enumerate(prompt_file)):
        # instruct_item = prompt_file[0]
        # if idx >= 3: 
        #     break
        graph_dict = load_graph(instruct_item, args.graph_root)
        graph_token_len = graph_dict['graph_token_len']
        graph_data = graph_dict['graph_data']
        hetero_key_orders = graph_dict['hetero_key_orders']

        qs = instruct_item["conversations"][0]["value"]
        
        if DEFAULT_GRAPH_TOKEN in qs:
            # build replace_tokens
            replace_tokens = []
            for i, token_len in enumerate(graph_token_len):
                replace_token = DEFAULT_GRAPH_PATCH_TOKEN * token_len
                if use_graph_start_end:
                    replace_token = DEFAULT_G_START_TOKEN + replace_token + DEFAULT_G_END_TOKEN
                replace_tokens.append(replace_token)

            for i, replace_token in enumerate(replace_tokens):
                index = qs.find(DEFAULT_GRAPH_TOKEN)
                qs = qs[:index] + replace_token + qs[index+len(DEFAULT_GRAPH_TOKEN):]
        conv_mode = "graphchat_v1"

        if args.conv_mode is not None and conv_mode != args.conv_mode:
            print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
        else:
            args.conv_mode = conv_mode

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        inputs = tokenizer([prompt])

        

        input_ids = torch.as_tensor(inputs.input_ids).cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        # graph_data.graph_node = graph_data.graph_node.to(torch.float16)
        # graph_data.edge_index = graph_data.edge_index.to(torch.float16)
        for idx in range(len(graph_data)):
            graph_data[idx].cuda()

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                graph_data=graph_data,
                hetero_key_order = hetero_key_orders, 
                do_sample=True,
                temperature=0.2,
                max_new_tokens=1024,
                stopping_criteria=[stopping_criteria])

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        # print(outputs)

        res_data.append({"id": instruct_item["id"], "node_idx": instruct_item["graph"]["node_idx"], "res": outputs}.copy())
        with open(osp.join(args.output_res_path, 'arxiv_test_res_{}_{}.json'.format(start_idx, end_idx)), "w") as fout:
            json.dump(res_data, fout, indent=4)
    return res_data
    # with open(args.output_res_path, "w") as fout:
    #     json.dump(res_data, fout, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="facebook/opt-350m")
    # parser.add_argument("--image-file", type=str, required=True)
    # parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--prompting_file", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--graph_root", type=str, default=None)

    parser.add_argument("--output_res_path", type=str, default=None)
    parser.add_argument("--num_gpus", type=int, default=4)

    parser.add_argument("--start_id", type=int, default=0)
    parser.add_argument("--end_id", type=int, default=1000)

    args = parser.parse_args()

    # eval_model(args)

    ray.init()
    run_eval(args, args.num_gpus)


# protobuf             4.22.3