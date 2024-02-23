cd /path/to/HiGPT

# offline tokenizing for stage1 matching instruction
data_type=imdb
graph_root=./hi_datasets/matching_instruction
dsname_list=(instruct_ds_matching_movie instruct_ds_node_matching)
pretrained_hgt=./MetaHGT_imdb_dblp_epoch5

for dsname in "${dsname_list[@]}"
do
    python run_offline_hgt_tokenizer.py --dsname ${dsname} \
                                 --data_type ${data_type} \
                                 --pretrained_gnn_path ${pretrained_hgt} \
                                 --graph_root ${graph_root}
done


data_type=dblp
graph_root=./hi_datasets/matching_instruction
dsname_list=(instruct_ds_matching_author instruct_ds_matching_paper instruct_ds_node_matching)
pretrained_hgt=./MetaHGT_imdb_dblp_epoch5

for dsname in "${dsname_list[@]}"
do
    python run_offline_hgt_tokenizer.py --dsname ${dsname} \
                                 --data_type ${data_type} \
                                 --pretrained_gnn_path ${pretrained_hgt} \
                                 --graph_root ${graph_root}
done