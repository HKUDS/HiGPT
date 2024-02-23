#!/bin/bash
cd /path/to/HiGPT
stage1_model=./checkpoints/higpt-metahgt-stage1-7b-dblp-imdb-epoch1
graph_projector=./checkpoints/higpt_stage1_projector_metahgt_dblp_imdb_epoch1/higpt-metahgt-stage1-7b.bin

python3.8 ./scripts/extract_graph_projector.py \
  --model_name_or_path ${stage1_model} \
  --output ${graph_projector}