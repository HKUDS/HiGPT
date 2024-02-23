cd /path/to/HiGPT/hi_datasets

mkdir stage2_data
cd stage2_data

wget https://archive.org/download/higpt_stage2/instruct_ds_dblp.tar.gz
wget https://archive.org/download/higpt_stage2/processed_dblp.tar.gz
wget https://archive.org/download/higpt_stage2/instruct_ds_imdb.tar.gz
wget https://archive.org/download/higpt_stage2/processed_imdb.tar.gz
wget https://archive.org/download/higpt_stage2/instruct_ds_acm.tar.gz
wget https://archive.org/download/higpt_stage2/processed_acm.tar.gz


mkdir DBLP
mkdir IMDB
mkdir acm

tar -xzvf instruct_ds_dblp.tar.gz -C DBLP
tar -xzvf processed_dblp.tar.gz -C DBLP
tar -xzvf instruct_ds_imdb.tar.gz -C IMDB
tar -xzvf processed_imdb.tar.gz -C IMDB
tar -xzvf instruct_ds_acm.tar.gz -C acm
tar -xzvf processed_acm.tar.gz -C acm

rm -f instruct_ds_dblp.tar.gz
rm -f processed_dblp.tar.gz
rm -f instruct_ds_imdb.tar.gz
rm -f processed_imdb.tar.gz
rm -f instruct_ds_acm.tar.gz
rm -f processed_acm.tar.gz