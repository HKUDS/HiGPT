#!/bin/bash
cd /path/to/HiGPT
base_model=/path/to/vicuna-7b-v1.5-16k
graph_root=./hi_datasets/stage2_data/IMDB
graph_projector=./checkpoints/higpt_stage1_projector_metahgt_dblp_imdb_epoch1/higpt-metahgt-stage1-7b.bin

num_epochs=15
num_shot_list=(1 3 5 10 20 40 60)
# 双重循环
for num_shot in "${num_shot_list[@]}"
do
    python3.8 -m torch.distributed.run --nnodes=1 --nproc_per_node=4 --master_port=20010 higpt/train/train_hete_nopl.py \
    --model_name_or_path ${base_model} \
    --version v1 \
    --data_path instruct_ds_imdb \
    --graph_content /root/paddlejob/workspace/env_run/llm/GraphChat/playground/data/arxiv_ti_ab.json \
    --graph_root ${graph_root} \
    --graph_tower MetaHGT_imdb_dblp_epoch5 \
    --pretrain_graph_mlp_adapter ${graph_projector} \
    --tune_graph_mlp_adapter True \
    --graph_select_layer -2 \
    --use_graph_start_end True \
    --bf16 False \
    --output_dir ./checkpoints/higpt-stage2-imdb-metahgt-epoch${num_epochs}-mixcot-true-${num_shot} \
    --num_train_epochs ${num_epochs} \
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
    --hetero_key_path /root/paddlejob/workspace/env_run/output/sample_instruct_ds/ann/hetero_key_order.json \
    --num_shot ${num_shot} 
done
