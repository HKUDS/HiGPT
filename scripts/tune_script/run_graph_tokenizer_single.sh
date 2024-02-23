cd /path/to/HiGPT

ann_path=./hi_datasets/stage2_data/IMDB/instruct_ds_imdb/ann/IMDB_test_std_0_1000.json
data_type=imdb
graph_path=./hi_datasets/stage2_data/IMDB/instruct_ds_imdb/graph_data/test

python run_offline_hgt_tokenizer_single.py --ann_path ${ann_path} \
                                 --data_type ${data_type} \
                                 --graph_path ${graph_path}