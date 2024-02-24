import pandas as pd
import openai
import json
import os.path as osp
import os
import time
from tqdm import tqdm
from pathlib import Path
import argparse

wd = Path(__file__).resolve().parent.parent.parent

openai.api_key = "your_openai_key"

def get_response(messages, temperature, top_p, max_tokens, model):

    response = openai.ChatCompletion.create(model=model,
        messages=messages,
        temperature=temperature, 
        top_p=top_p,
        # max_tokens=max_tokens,
    )


    if not response.get("error"):
        return response["choices"][0]["message"]["content"]
    return response["error"]["message"]

cot_case_mapping = {
    0: "Imagine that you have made the correct choice and proceed with step-by-step reasoning.\n Example: Data mining.\nBased on ...", 
    1: "Imagine that you have made the correct choice and proceed with step-by-step reasoning.\n Using the following format:\nAnswer: ...\nReason: ...", 
    2: "Imagine three different experts are answering this question. All experts will write down 1 step of their thinking, then share it with the group. Then all experts will go on to the next step, etc. If any expert realizes they're wrong at any point then they leave. And finally they make the correct choice.\nUsing the following format:\nExpert 1:...\nExpert 2:...\nExpert 3:...\nExpert 1:...\nExpert 2:...\nExpert 3:...\nFinal Answer:...", # TOT
    3: "Imagine three different experts are answering this question. All experts will write down 1 step of their thinking, then share it with the group. Then all experts will go on to the next step, etc. If any expert realizes they're wrong at any point then they leave. And finally they make the correct choice.\nUsing the following format:\nExpert 1:...\nExpert 2:...\nExpert 3:...\nFinal Answer:...", # TOT, single round
    4: "Imagine that 3 experts are discussing the question with a panel discussion, trying to solve it step by step to make sure the result is correct and avoid penalty. And finally they make the correct choice.\n", 
    5: "Please generate some knowledge that can assist in formulating an answer, including, but not limited to: distinctions between the three categories. \nImagine that you have arrived at the correct answer based on the provided information and knowledge, and present a step-by-step reasoning.\nUsing the following format:\nKnowledge: ...\nAnswer: ...\nReason: ...",
    6: "Please generate some knowledge that can assist in formulating an answer, including, but not limited to: explanations of some technical terms present in the given information. \nImagine that you have arrived at the correct answer based on the provided information and knowledge, and present a step-by-step reasoning.\nUsing the following format:\nKnowledge: ...\nAnswer: ...\nReason: ..."
}

replace_prompt_mapping = {
    0: "Please think about the categorization in a step by step manner and avoid making false associations. Then provide your reasoning.", 
    1: "Please think about the categorization in a step by step manner and avoid making false associations. Then provide your reasoning.\n Using the following format:\nAnswer: AI\nReason: ...", 
    2: "Imagine three different experts are answering this question. All experts will write down 1 step of their thinking, then share it with the group. Then all experts will go on to the next step, etc. If any expert realizes they're wrong at any point then they leave.", 
    3: "Imagine three different experts are answering this question. All experts will write down 1 step of their thinking, then share it with the group. Then all experts will go on to the next step, etc. If any expert realizes they're wrong at any point then they leave.", 
    4: "3 experts are discussing the question with a panel discussion, trying to solve it step by step, and make sure the result is correct and avoid penalty.", 
    5: "Based on the knowledge and information, please think about the categorization in a step by step manner and avoid making false associations. Then provide your reasoning.", 
    6: "Based on the knowledge and information, please think about the categorization in a step by step manner and avoid making false associations. Then provide your reasoning."
}


temperature = 0.7

top_p = 1

max_tokens = 1024

model = "gpt-3.5-turbo"

file_dir = wd

json_ann_dir = wd / 'hi_datasets/stage2_data/acm/instruct_ds_acm/ann_processed_MetaHGT_imdb_dblp_epoch5/ACM_train_std_0_907.json'

with open(json_ann_dir, 'r', encoding='utf-8') as f:
    ori_data_list = json.load(f)

def query_gpt(s_idx, e_idx, cot_case):
    # instruct_list = []
    cot_prompt = cot_case_mapping[cot_case]
    replace_prompt = replace_prompt_mapping[cot_case]

    if osp.exists(osp.join(file_dir, 'hi_datasets/stage2_data/acm/instruct_ds_acm/ann_processed_MetaHGT_imdb_dblp_epoch5/cot')) is False: 
        os.makedirs(osp.join(file_dir, 'hi_datasets/stage2_data/acm/instruct_ds_acm/ann_processed_MetaHGT_imdb_dblp_epoch5/cot'))
        instruct_list = []
        data_dir = osp.join(file_dir, 'hi_datasets/stage2_data/acm/instruct_ds_acm/ann_processed_MetaHGT_imdb_dblp_epoch5/cot')
        with open(osp.join(data_dir, f'acm_cot_pred_{s_idx}_{e_idx}_{cot_case}.json'), 'w', encoding='utf-8') as f:
            json.dump(instruct_list, f, ensure_ascii=False)
    else: 
        data_dir = osp.join(file_dir, 'hi_datasets/stage2_data/acm/instruct_ds_acm/ann_processed_MetaHGT_imdb_dblp_epoch5/cot')
        if osp.exists(osp.join(data_dir, f'acm_cot_pred_{s_idx}_{e_idx}_{cot_case}.json')) is False: 
            instruct_list = []
            with open(osp.join(data_dir, f'acm_cot_pred_{s_idx}_{e_idx}_{cot_case}.json'), 'w', encoding='utf-8') as f:
                json.dump(instruct_list, f, ensure_ascii=False)
        else: 
            with open(osp.join(data_dir, f'acm_cot_pred_{s_idx}_{e_idx}_{cot_case}.json'), 'r', encoding='utf-8') as f:
                instruct_list = json.load(f)
    start_idx = len(instruct_list)
    print(f'start idx: {start_idx}')


    try:
        for idx in tqdm(range(s_idx + start_idx, e_idx)):
            # if idx >= 3: 
            #     break
            instruct_item = ori_data_list[idx]
            messages = [
            {"role": "system", "content": "I have a question as below:\n" + instruct_item["conversations"][0]["value"] + "and the answer is " + instruct_item["conversations"][1]["value"] + cot_prompt},
            ]
            messages.append({"role": "user", "content": ""})
            response = get_response(messages, temperature, top_p, max_tokens, model)
            if cot_case == 5 or cot_case == 6:
                response_list = response.split('Answer')
                knowledge_res = response_list[0]
                answer_res = 'Answer: ' + response_list[1]
            else: 
                knowledge_res = ''
                answer_res = response

            instruct_item["conversations"][1]["value"] = answer_res
            human_ques = instruct_item["conversations"][0]["value"] + knowledge_res
            new_human_ques = human_ques.replace("Give likely categories directly.", replace_prompt)
            instruct_item["conversations"][0]["value"] = new_human_ques
            
            # instruct_item['output'] = response
            instruct_list.append(instruct_item.copy())

            with open(osp.join(data_dir, f'acm_cot_pred_{s_idx}_{e_idx}_{cot_case}.json'), 'w', encoding='utf-8') as f:
                json.dump(instruct_list, f, ensure_ascii=False)
            time.sleep(1)
    except: 
        del instruct_list[-1]
        with open(osp.join(data_dir, f'acm_cot_pred_{s_idx}_{e_idx}_{cot_case}.json'), 'w') as f:
            json.dump(instruct_list, f)
        raise ValueError('stop at {}'.format(idx))

    with open(osp.join(data_dir, f'acm_cot_pred_{s_idx}_{e_idx}_{cot_case}.json'), 'w') as f:
            json.dump(instruct_list, f)

if __name__ ==  "__main__":
    parser = argparse.ArgumentParser(description='check data')
    parser.add_argument('--start_idx', default=0, type=int)
    parser.add_argument('--end_idx', default=400, type=int)
    parser.add_argument('--cot_case', default=2, type=int)
    args = parser.parse_args()

    total_len = len(ori_data_list)
    print(f'total len: {total_len}')
      
    query_gpt(args.start_idx, args.end_idx, args.cot_case)