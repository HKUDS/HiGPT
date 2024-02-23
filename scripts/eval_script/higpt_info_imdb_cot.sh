#!/bin/bash
cd /path/to/HiGPT
output_model=./checkpoints
datapath=./hi_datasets/stage2_data/IMDB
res_path=./output_res_imdb
num_epochs=15
num_shot_list=(1 3 5 10 20 40 60)
# 双重循环
for num_shot in "${num_shot_list[@]}"
do
  for ((cot_case=0; cot_case<=0; cot_case++))
  do
    python3.8 ./higpt/eval/run_higpt.py --model-name ${output_model}/higpt-stage2-imdb-metahgt-epoch${num_epochs}-mixcot-true-${num_shot}  --prompting_file ${datapath}/instruct_ds_imdb/ann_processed_MetaHGT_imdb_dblp_epoch5/cot_test/IMDB_test_std_0_1000_cot_${cot_case}.json --graph_root ${datapath} --output_res_path ${res_path}/imdb_test_res_epoch_${num_epochs}_std_${num_shot}_shot_cot_${cot_case} --start_id 0 --end_id 1000 --num_gpus 4
  done
done