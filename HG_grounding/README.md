# Text-Graph Contrastive Alignment (Grounding) with In-Context Heterogeneous Graph Tokenizer

### 1. Data Preparation

Before running the codes of text-graph grounding, you should download datasets by running the following instruction ([get_all_datasets.sh](HG_grounding/datasets/get_all_datasets.sh)):

```shell
cd /path/to/HiGPT/HG_grounding/datasets

wget https://archive.org/download/higpt_hg_grounding/instruct_ds_caption_movie.tar.gz
wget https://archive.org/download/higpt_hg_grounding/instruct_ds_caption_paper.tar.gz
wget https://archive.org/download/higpt_hg_grounding/instruct_ds_caption_paper_acm.tar.gz

tar -xzvf instruct_ds_caption_movie.tar.gz
tar -xzvf instruct_ds_caption_paper.tar.gz
tar -xzvf instruct_ds_caption_paper_acm.tar.gz

rm -f instruct_ds_caption_movie.tar.gz
rm -f instruct_ds_caption_paper.tar.gz
rm -f instruct_ds_caption_paper_acm.tar.gz
```

## 2. Preraining

After downloading the pretraining datasets, you can conduct the pretraining stage by running the following instruction ([hgt_stage_1_all_data.sh](HG_grounding/training_script/hgt_stage_1_all_data.sh)):

```shell
cd /path/to/HiGPT/HG_grounding/

python lit_train/lit_hgt_train.py --max_epochs 5 \
                               --learning_rate 1e-6 \
                               --devices 4 \
                               --context_length 128 \
                               --dataset_dir ./datasets \
                               --datalist instruct_ds_caption_paper,instruct_ds_caption_movie \
                               --micro_batch_size 1 \
                               --batch_size 1 \
                               --gnn_type meta
```

The training results (including the checkpoints of the model) will be stored in `HG_grounding/lightning_logs`.