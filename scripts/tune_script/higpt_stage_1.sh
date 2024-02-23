#!/bin/bash
cd /path/to/HiGPT

data_path=instruct_ds_matching_author,instruct_ds_matching_movie,instruct_ds_matching_paper,instruct_ds_node_matching,instruct_ds_node_matching_imdb
graph_root=./hi_datasets/matching_instruction
output_dir=./checkpoints/higpt-metahgt-stage1-7b-dblp-imdb-epoch1
base_model=/path/to/vicuna-7b-v1.5-16k

wandb offline

python3.8 -m torch.distributed.run --nnodes=1 --nproc_per_node=4 --master_port=20001 \
    higpt/train/train_hete_nopl.py \
    --model_name_or_path ${base_model} \
    --version v1 \
    --data_path ${data_path} \
    --graph_content /root/paddlejob/workspace/env_run/llm/GraphChat/playground/data/arxiv_ti_ab.json \
    --graph_root ${graph_root} \
    --graph_tower ${graph_tower} \
    --tune_graph_mlp_adapter True \
    --graph_select_layer -2 \
    --use_graph_start_end True \
    --bf16 False \
    --output_dir ${output_dir} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2400 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0.0001 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --report_to wandb \
    --hetero_key_path /root/paddlejob/workspace/env_run/output/sample_instruct_ds/ann/hetero_key_order.json