# <center>HiGPT: Heterogeneous Graph Language Model</center>

[Jiabin Tang](https://tjb-tech.github.io/), [Yuhao Yang](http://yuh-yang.github.io), [Wei Wei](#), [Lei Shi](#), [Long Xia](#), [Dawei Yin](https://www.yindawei.com/) and [Chao Huang](https://sites.google.com/view/chaoh/home)*.
(*Correspondence )

**[Data Intelligence Lab](https://sites.google.com/view/chaoh/home)@[University of Hong Kong](https://www.hku.hk/)**, Baidu Inc.

-----

<a href='https://HiGPT-HKU.github.io/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
<a href='#'><img src='https://img.shields.io/badge/Demo-Page-purple'></a> 
<a href='#'><img src='https://img.shields.io/badge/Paper-PDF-orange'></a> 
[![YouTube](https://badges.aleen42.com/src/youtube.svg)](#)
 • 🌐 <a href="https://mp.weixin.qq.com/s/rvKTFdCk719Q6hT09Caglw" target="_blank">中文博客</a>


This repository hosts the code, data and model weight of **HiGPT**.

-----------

## 🎉 News 


🎯🎯📢📢 We have made significant updates to the **models** used in our HiGPT on 🤗 **Huggingface**. We highly recommend referring to the table below for further details: 

| 🤗 Huggingface Address                   | 🎯 Description |
| --------------------------------------- | ------------- |
| https://huggingface.co/Jiabin99/MetaHGT |               |
|                                         |               |


- [x] [2023.10.26]🔥🔥Release our utilized Instruction data.
- [x] [2023.10.26]🔥🔥Release checkpoints of our HiGPT and pre-trained graph encoder.
- [x] [2023.10.15] 🚀🚀 Release the code of HiGPT.


## 👉 TODO 
- [ ] Supporting lightning training
- [ ] Releasing the Chinese version of the explanation
- [ ] Releasing the full paper of our HiGPT
- [ ] Exploring the potential of our HiGPT for more graph learning tasks.
- [ ] ...

-----------




<span id='introduction'/>

## Brief Introduction 

we present the **HiGPT** framework that aligns LLMs with heterogeneous graph structural knowledge with a heterogeneous graph instruction tuning paradigm.




For more technical details, kindly refer to the [paper](#) and the project [website](https://HiGPT-HKU.github.io/) of our Graph. 


-----------

<span id='Usage'/>

## Getting Started

<span id='all_catelogue'/>

### Table of Contents:
* <a href='#Code Structure'>1. Code Structure</a>
* <a href='#Environment Preparation'>2. Environment Preparation </a>
* <a href='#Training HiGPT'>3. Data Preparation </a>
* <a href='#Training HiGPT'>4. Training HiGPT </a>
  * <a href='#Prepare Pre-trained Checkpoint'>4.1. Prepare Pre-trained Checkpoint</a>
  * <a href='#Self-Supervised Instruction Tuning'>4.2. Self-Supervised Instruction Tuning</a>
  * <a href='#Extract the Trained Projector'>4.3. Extract the Trained Projector</a>
  * <a href='#Task-Specific Instruction Tuning'>4.4. Task-Specific Instruction Tuning</a>
* <a href='#Evaluating HiGPT'>5. Evaluating HiGPT</a>
  * <a href='#Preparing Checkpoints and Data'>5.1. Preparing Checkpoints and Data</a>
  * <a href='#Running Evaluation'>5.2. Running Evaluation</a>

****



<span id='Code Structure'/>

### 1. Code Structure <a href='#all_catelogue'>[Back to Top]</a>

```
.
├── LICENSE
├── README.md
├── base_model.py
├── dist_utils.py
├── docs
│   ├── arena.md
│   ├── commands
│   │   ├── data_cleaning.md
│   │   ├── leaderboard.md
│   │   ├── local_cluster.md
│   │   ├── pypi.md
│   │   └── webserver.md
│   ├── langchain_integration.md
│   ├── openai_api.md
│   ├── server_arch.md
│   ├── test_process.md
│   ├── vicuna_weights_version.md
│   └── weights_version.md
├── examples
│   └── langchain
│       ├── README.md
│       ├── chatgpt_clone.ipynb
│       ├── qa.ipynb
│       └── twitter_algo_analysis.ipynb
├── hi_datasets
│   ├── get_stage1_data.sh
│   └── get_stage2_data.sh
├── higpt
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-38.pyc
│   │   └── conversation.cpython-38.pyc
│   ├── constants.py
│   ├── conversation.py
│   ├── eval
│   │   ├── requirements.txt
│   │   ├── run_higpt.py
│   │   ├── run_higpt_incontext.py
│   │   └── webpage
│   │       ├── figures
│   │       │   ├── alpaca.png
│   │       │   ├── bard.jpg
│   │       │   ├── chatgpt.svg
│   │       │   ├── llama.jpg
│   │       │   ├── swords_FILL0_wght300_GRAD0_opsz48.svg
│   │       │   └── vicuna.jpeg
│   │       ├── index.html
│   │       ├── script.js
│   │       └── styles.css
│   ├── model
│   │   ├── GraphLlama.py
│   │   ├── GraphLlama_pl.py
│   │   ├── HeteroLlama.py
│   │   ├── HeteroLlama_pl.py
│   │   ├── MetaHGTConv_pl.py
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   │   ├── GraphLlama.cpython-38.pyc
│   │   │   ├── HeteroLlama.cpython-38.pyc
│   │   │   ├── __init__.cpython-38.pyc
│   │   │   └── model_adapter.cpython-38.pyc
│   │   ├── apply_delta.py
│   │   ├── apply_lora.py
│   │   ├── builder.py
│   │   ├── chatglm_model.py
│   │   ├── compression.py
│   │   ├── convert_fp16.py
│   │   ├── graph_layers
│   │   │   ├── __init__.py
│   │   │   ├── __pycache__
│   │   │   │   ├── __init__.cpython-38.pyc
│   │   │   │   ├── clip_graph.cpython-38.pyc
│   │   │   │   ├── graph_transformer.cpython-38.pyc
│   │   │   │   ├── mpnn.cpython-38.pyc
│   │   │   │   └── simple_tokenizer.cpython-38.pyc
│   │   │   ├── bpe_simple_vocab_16e6.txt.gz
│   │   │   ├── clip_graph.py
│   │   │   ├── graph_transformer.py
│   │   │   ├── mpnn.py
│   │   │   └── simple_tokenizer.py
│   │   ├── heteclip_models
│   │   │   ├── __init__.py
│   │   │   ├── __pycache__
│   │   │   │   ├── __init__.cpython-38.pyc
│   │   │   │   ├── clip_outputs.cpython-38.pyc
│   │   │   │   ├── model.cpython-38.pyc
│   │   │   │   ├── pretrained.cpython-38.pyc
│   │   │   │   ├── tokenizer.cpython-38.pyc
│   │   │   │   ├── transform.cpython-38.pyc
│   │   │   │   └── utils.cpython-38.pyc
│   │   │   ├── bpe_simple_vocab_16e6.txt.gz
│   │   │   ├── clip_outputs.py
│   │   │   ├── loss.py
│   │   │   ├── model.py
│   │   │   ├── pics
│   │   │   │   └── CLIP.png
│   │   │   ├── pretrained.py
│   │   │   ├── timm_model.py
│   │   │   ├── tokenizer.py
│   │   │   ├── transform.py
│   │   │   └── utils.py
│   │   ├── make_delta.py
│   │   ├── meta_hgt
│   │   │   ├── __init__.py
│   │   │   ├── __pycache__
│   │   │   │   ├── __init__.cpython-38.pyc
│   │   │   │   ├── hgt_constants.cpython-38.pyc
│   │   │   │   ├── meta_hgtconv.cpython-38.pyc
│   │   │   │   ├── meta_hgtconv_bert_all.cpython-38.pyc
│   │   │   │   ├── meta_linear.cpython-38.pyc
│   │   │   │   └── tokenizer.cpython-38.pyc
│   │   │   ├── bpe_simple_vocab_16e6.txt.gz
│   │   │   ├── hgt_constants.py
│   │   │   ├── meta_dict
│   │   │   │   ├── acm
│   │   │   │   │   ├── edge_type.pt
│   │   │   │   │   └── node_type.pt
│   │   │   │   ├── dblp
│   │   │   │   │   ├── edge_type.pt
│   │   │   │   │   └── node_type.pt
│   │   │   │   ├── imdb
│   │   │   │   │   ├── edge_type.pt
│   │   │   │   │   └── node_type.pt
│   │   │   │   └── to_tensor.py
│   │   │   ├── meta_hgtconv.py
│   │   │   ├── meta_hgtconv_bert_all.py
│   │   │   ├── meta_linear.py
│   │   │   ├── ori_hgt.py
│   │   │   └── tokenizer.py
│   │   ├── model_adapter.py
│   │   ├── model_registry.py
│   │   ├── monkey_patch_non_inplace.py
│   │   ├── rwkv_model.py
│   │   └── utils.py
│   ├── protocol
│   │   └── openai_api_protocol.py
│   ├── serve
│   │   ├── __init__.py
│   │   ├── api_provider.py
│   │   ├── bard_worker.py
│   │   ├── cacheflow_worker.py
│   │   ├── cli.py
│   │   ├── controller.py
│   │   ├── gateway
│   │   │   ├── README.md
│   │   │   └── nginx.conf
│   │   ├── gradio_block_arena_anony.py
│   │   ├── gradio_block_arena_named.py
│   │   ├── gradio_css.py
│   │   ├── gradio_patch.py
│   │   ├── gradio_web_server.py
│   │   ├── gradio_web_server_multi.py
│   │   ├── huggingface_api.py
│   │   ├── inference.py
│   │   ├── model_worker.py
│   │   ├── monitor
│   │   │   ├── basic_stats.py
│   │   │   ├── clean_battle_data.py
│   │   │   ├── elo_analysis.py
│   │   │   ├── hf_space_leaderboard_app.py
│   │   │   └── monitor.py
│   │   ├── openai_api_server.py
│   │   ├── register_worker.py
│   │   ├── test_message.py
│   │   └── test_throughput.py
│   ├── train
│   │   ├── __pycache__
│   │   │   └── graphchat_trainer.cpython-38.pyc
│   │   ├── graphchat_trainer.py
│   │   ├── llama_flash_attn_monkey_patch.py
│   │   ├── train.py
│   │   ├── train_flant5.py
│   │   ├── train_g.py
│   │   ├── train_graph.py
│   │   ├── train_graph_back.py
│   │   ├── train_hete.py
│   │   ├── train_hete_nopl.py
│   │   ├── train_hete_nopl_back_2_5.py
│   │   ├── train_hete_nopl_wo_IA.py
│   │   ├── train_hete_nopl_wo_graph.py
│   │   ├── train_hete_old.py
│   │   ├── train_light.py
│   │   ├── train_llava.py
│   │   ├── train_lora.py
│   │   └── train_mem.py
│   └── utils.py
├── playground
│   ├── inspect_conv.py
│   ├── test_embedding
│   │   ├── README.md
│   │   ├── test_classification.py
│   │   ├── test_semantic_search.py
│   │   └── test_sentence_similarity.py
│   └── test_openai_api
│       ├── anthropic_api.py
│       └── openai_api.py
├── run_offline_hgt_tokenizer.py
├── run_offline_hgt_tokenizer_single.py
├── scripts
│   ├── eval_script
│   │   ├── cal_acc_imdb_metric.py
│   │   ├── hetegpt_info_imdb_cot_incontext.sh
│   │   └── higpt_info_imdb_cot.sh
│   ├── extract_graph_projector.py
│   ├── serving
│   │   ├── controller.yaml
│   │   └── model_worker.yaml
│   └── tune_script
│       ├── extract_projector.sh
│       ├── higpt_stage_1.sh
│       ├── higpt_stage_2.sh
│       ├── run_graph_tokenizer.sh
│       └── run_graph_tokenizer_single.sh
├── tests
│   ├── test_openai_curl.sh
│   ├── test_openai_langchain.py
│   └── test_openai_sdk.py
└── utils.py
```


<span id='Environment Preparation'/>


### 2. Environment Preparation  <a href='#all_catelogue'>[Back to Top]</a>
Please first clone the repo and install the required environment, which can be done by running the following commands:
```shell
conda create -n graphgpt python=3.8

conda activate higpt

# Torch with CUDA 11.7
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
# To support vicuna base model
pip3 install "fschat[model_worker,webui]"
# To install pyg and pyg-relevant packages
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
# Clone our HiGPT
git clone https://github.com/HKUDS/HiGPT.git
cd HiGPT
# Install required libraries
pip install -r requirements.txt
```

<span id='Training HiGPT'/>

### 4. Training HiGPT <a href='#all_catelogue'>[Back to Top]</a>

HiGPT tuning paradigm consists of two stages: (1) self-supervised instruction tuning; (2) task-specific instruction tuning.

<span id='Prepare Pre-trained Checkpoint'/>

#### 4.1. Preparing Pre-trained Checkpoint  <a href='#all_catelogue'>[Back to Top]</a>
HiGPT is trained based on following excellent existing models.
Please follow the instructions to prepare the checkpoints.

- `Vicuna`:
  Prepare our base model Vicuna, which is an instruction-tuned chatbot and base model in our implementation. Please download its weights [here](https://github.com/lm-sys/FastChat#model-weights). We generally utilize v1.1 and v1.5 model with 7B parameters.

- `Graph Encoder`:
  is used to encode graph structures. We employ text-graph grounding approach to obtain the pre-trained graph transformer model, which you could download by [graph transformer](https://huggingface.co/Jiabin99/Arxiv-PubMed-GraphCLIP-GT) and put it at [[./HiGPT]](./HiGPT). We also provide source codes and example Cora data for text-graph grounding at [[./text-graph-grounding]](./text-graph-grounding) for your reference.

- `Graph Data`:
  is a combination of all utilized pyg graph data that contain node features, edge_index and so on. You can download by [all_graph_data.pt](https://huggingface.co/datasets/Jiabin99/All_pyg_graph_data) and put it at [[./HiGPT/graph_data]](./HiGPT/graph_data)

<span id='Self-Supervised Instruction Tuning'/>

#### 4.2. Self-Supervised Instruction Tuning  <a href='#all_catelogue'>[Back to Top]</a>

* **Prepare data:** Please download our instruction tuning data [graph_matching.json](https://huggingface.co/datasets/Jiabin99/graph-matching) for the graph matching task.

* **Start tuning:** After the aforementioned steps, you could start the first stage tuning by filling blanks at [graphgpt_stage1.sh](scripts/tune_script/graphgpt_stage1.sh). There is an example as below: 

```shell
# to fill in the following path to run the first stage of our HiGPT!
model_path=../vicuna-7b-v1.5-16k
instruct_ds=./data/stage_1/graph_matching.json
graph_data_path=./graph_data/all_graph_data.pt
pretra_gnn=./clip_gt_arxiv
output_model=./checkpoints/stage_1

wandb offline
python -m torch.distributed.run --nnodes=1 --nproc_per_node=4 --master_port=20001 \
    graphgpt/train/train_mem.py \
    --model_name_or_path ${model_path} \
    --version v1 \
    --data_path ${instruct_ds} \
    --graph_content ./arxiv_ti_ab.json \
    --graph_data_path ${graph_data_path} \
    --graph_tower ${pretra_gnn} \
    --tune_graph_mlp_adapter True \
    --graph_select_layer -2 \
    --use_graph_start_end \
    --bf16 True \
    --output_dir ${output_model} \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2400 \
    --save_total_limit 1 \
    --learning_rate 2e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --report_to wandb
```

<span id='Extract the Trained Projector'/>

#### 4.3. Extract the Trained Projector  <a href='#all_catelogue'>[Back to Top]</a>

We could extract the trained projector in the stage 1 by filling blanks at [extract_projector.sh](scripts/tune_script/extract_projector.sh). There is an example as below: 

```shell
# to fill in the following path to extract projector for the first tuning stage!
src_model=./checkpoints/stage_1
output_proj=./checkpoints/stage_1_projector/stage_1_projector.bin

python3.8 ./scripts/extract_graph_projector.py \
  --model_name_or_path ${src_model} \
  --output ${output_proj}
```

<span id='Task-Specific Instruction Tuning'/>

#### 4.4. Task-Specific Instruction Tuning  <a href='#all_catelogue'>[Back to Top]</a>

* **Prepare data:** The choices of our task-specific instruction data could be diverse, e.g., standard or COT (Chain-of-Thought) node classification, link prediction or mixing data for multitasking. Please refer to the  [task_specific](https://huggingface.co/datasets/Jiabin99/Arxiv-PubMed-mix-NC-LP).

* **Start tuning:** After the aforementioned steps, you could start the second stage tuning by filling blanks at [graphgpt_stage2.sh](scripts/tune_script/graphgpt_stage2.sh). There is an example as below: 

```shell
# to fill in the following path to run the second stage of our HiGPT!
model_path=../vicuna-7b-v1.5-16k
instruct_ds=./data/stage_2/data_all_mix.json
graph_data_path=./graph_data/all_graph_data.pt
pretra_gnn=./clip_gt_arxiv
tuned_proj=./checkpoints/stage_1_projector/stage_1_projector.bin
output_model=./checkpoints/stage_2

wandb offline
python -m torch.distributed.run --nnodes=1 --nproc_per_node=4 --master_port=20001 \
    graphgpt/train/train_mem.py \
    --model_name_or_path ${model_path} \
    --version v1 \
    --data_path ${instruct_ds} \
    --graph_content ./arxiv_ti_ab.json \
    --graph_data_path ${graph_data_path} \
    --graph_tower ${pretra_gnn} \
    --pretrain_graph_mlp_adapter ${tuned_proj} \
    --tune_graph_mlp_adapter True \
    --graph_select_layer -2 \
    --use_graph_start_end True\
    --bf16 True \
    --output_dir ${output_model} \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb

```



<span id='Evaluating HiGPT'/>

## 5. Evaluating HiGPT  <a href='#all_catelogue'>[Back to Top]</a>

<span id='Preparing Checkpoints and Data'/>


#### 5.1. Preparing Checkpoints and Data <a href='#all_catelogue'>[Back to Top]</a>

* **Checkpoints:** You could try to evaluate HiGPT by using your own model or our released checkpoints.
* **Data:** We split test sets for different graph datasets and make the instruction data for evaluation. Please refer to the  [evaluating](https://huggingface.co/datasets/Jiabin99/HiGPT-eval-instruction).

<span id='Running Evaluation'/>

#### 5.2. Running Evaluation <a href='#all_catelogue'>[Back to Top]</a>

You could start the second stage tuning by filling blanks at [graphgpt_eval.sh](scripts/eval_script/graphgpt_eval.sh). There is an example as below: 
```shell
# to fill in the following path to extract projector for the second tuning stage!
output_model=./checkpoints/stage_2
datapath=./data/eval/arxiv_nc.json
graph_data_path=./graph_data/all_graph_data.pt
res_path=./output_stage_2_arxiv_nc
start_id=0
end_id=20000
num_gpus=2

python3.8 ./graphgpt/eval/run_graphgpt.py --model-name ${output_model}  --prompting_file ${datapath} --graph_data_path ${graph_data_path} --output_res_path ${res_path} --start_id ${start_id} --end_id ${end_id} --num_gpus ${num_gpus}
```
---------


## Contact

For any questions or feedback, feel free to contact [Jiabin Tang](mailto:jiabintang77@gmail.com).


## Citation

If you find HiGPT useful in your research or applications, please kindly cite:
```tex
@articles{tang2023graphgpt,
title={HiGPT: Graph Instruction Tuning for Large Language Models}, 
author={Jiabin Tang and Yuhao Yang and Wei Wei and Lei Shi and Long Xia and Dawei Yin and Chao Huang},
year={2024},
eprint={},
archivePrefix={arXiv},
primaryClass={cs.CL}
}
```



## Acknowledgements
You may refer to related work that serves as foundations for our framework and code repository, 
[Vicuna](https://github.com/lm-sys/FastChat), [LLaVa](https://github.com/haotian-liu/LLaVA), [GraphGPT](https://github.com/HKUDS/GraphGPT), We also partially draw inspirations from [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4). The design of our website and README.md was inspired by [NExT-GPT](https://next-gpt.github.io/). Thanks for their wonderful works.



