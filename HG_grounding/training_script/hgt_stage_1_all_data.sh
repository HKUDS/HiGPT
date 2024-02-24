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