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