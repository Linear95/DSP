import json
from tqdm import tqdm
import gzip
import random
from copy import deepcopy
import argparse

from utils import print_rank_0
from pprint import pprint
import numpy as np

import torch
from torch.utils.data import Dataset
from typing import List

QUERY_PROMPT="## Human:\n{request}\n\n## Assistant:\n{response}"
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


def get_data_iter(data_list, debug=False):
    if debug:
        data_size = len(data_list)
        data_list = [data_list[i] for i in range(min(int(0.1*data_size), 1000))]

    if (not torch.distributed.is_initialized()) or (torch.distributed.get_rank() == 0):
        return tqdm(data_list)
    else:
        return data_list


def convert_domain_specific_reward_data(data_path, tokenizer=None, domain="normal", padding=False, debug=False, to_pairs=True):
    print_rank_0(f"begin preparing domain-specific preference data from responses at {data_path}")

    if data_path[-4:] == 'json':
        data_list = load_json_data(data_path)
    else:
        data_list = load_jsonl_data(data_path)

    outputs = []
    for item in get_data_iter(data_list, debug=debug):
        data_point = {"text": [], "score": []}
        query = item['query']
        if domain not in item['responses']:
            raise ValueError("Unknown domain {}.".format(domain))
            
        if not to_pairs:
            for style, text in item['responses'].items():
                data_point['text'].append(QUERY_PROMPT.format(request=query, response=text))
                data_point['score'].append(float(style == domain))

            new_item = prepare_data_item(data_point, tokenizer=tokenizer, padding=padding)
            if new_item is not None:
                outputs.append(new_item)
        else:
            preferred_response = item['responses'][domain]
            for style, text in item['responses'].items():
                if style == domain:
                    continue
                data_point = {}
                data_point['text'] = [
                    QUERY_PROMPT.format(request=query, response=preferred_response), 
                    QUERY_PROMPT.format(request=query, response=text),
                ]
                data_point['score'] = [1., 0.]

                new_item = prepare_data_item(data_point, tokenizer=tokenizer, padding=padding)
                if new_item is not None:
                    outputs.append(new_item)

    print_rank_0("finished processing {} domain-specific preference data.".format(len(outputs)))
    return outputs



def reward_data_collactor(batch):
    scores = []
    input_ids = []
    attention_mask = []
    for item in batch:
        scores.append(item['score'])
        input_ids.append(item['tokens']['input_ids'])
        attention_mask.append(item['tokens']['attention_mask'])

    return {
        "score": torch.Tensor(scores).float(),
        "input_ids": torch.Tensor(input_ids).long(),
        "attention_mask": torch.Tensor(attention_mask).float()
    }


def reward_tokenize(sentences, tokenizer, padding="longest"):
    if isinstance(sentences, str):
        sentences = [sentences]
    input_ids = [
        [tokenizer.bos_token_id] + tokenizer.encode(sent, add_special_tokens=False) + [tokenizer.eos_token_id] 
        for sent in sentences
    ]
    
    if padding == 'longest':
        max_input_length = max([len(inp_ids) for inp_ids in input_ids])
        max_length = min(tokenizer.model_max_length, max_input_length)
    else:
        max_length = tokenizer.model_max_length

    outputs = {"input_ids": [], "attention_mask": []}
    for inp_ids in input_ids:        
        attn_mask = [1] * len(inp_ids)
        if len(inp_ids) >= max_length:
            if tokenizer.truncation_side == 'left':
                inp_ids = inp_ids[-max_length :]
                attn_mask = attn_mask[-max_length :]
            else:
                inp_ids = inp_ids[:max_length]
                attn_mask = attn_mask[:max_length]
        else:
            if tokenizer.padding_side == 'left':
                inp_ids = [tokenizer.pad_token_id] * (max_length - len(inp_ids)) + inp_ids
                attn_mask = [0] * (max_length - len(attn_mask)) + attn_mask
            else:
                inp_ids =  inp_ids + [tokenizer.pad_token_id] * (max_length - len(inp_ids)) 
                attn_mask = attn_mask + [0] * (max_length - len(attn_mask))

        outputs['input_ids'].append(deepcopy(inp_ids))
        outputs['attention_mask'].append(deepcopy(attn_mask))
    return outputs


def prepare_data_item(item, tokenizer=None, padding=False, max_response_num=1):
    new_item = deepcopy(item)
    if not len(new_item['score']) == len(new_item['text']):
        print_rank_0("invalid data point {}".format(new_item))
        return None

    score_idx = np.argsort(new_item['score'])
    max_score = max(new_item['score']) + 1e-5
    
    new_item['score'] = [new_item['score'][s_i] / max_score for s_i in score_idx[::-1]] # normalize the scores
    new_item['text'] = [new_item['text'][s_i] for s_i in score_idx[::-1]]

    if padding:
        new_item['text'] += [" "] * (max_response_num - len(new_item['text']))
        new_item['score'] += [-1.] * (max_response_num - len(new_item['score']))        

    if tokenizer is not None:
        new_item['tokens'] = reward_tokenize(
            sentences=new_item['text'],
            tokenizer=tokenizer,
            padding="max_length" if padding else "longest"
        )

    return new_item


def load_jsonl_data(data_path):
    print_rank_0("loading text-score dataset from: \n   {}".format(data_path))
    with open(data_path, 'r') as f:
        lines = f.read().strip().split('\n')
    
    data_list = [json.loads(l) for l in lines]
    print_rank_0(f"Jsonl data length: {len(data_list)}")
    return data_list


def load_json_data(data_path):
    with open(data_path, 'r') as f:
        data_list = json.load(f)
    print_rank_0(f"Json data length: {len(data_list)}")
    return data_list


def load_text_score_dataset(data_path, tokenizer=None, debug=False, padding=False):
    all_data_list = []
    if isinstance(data_path, str): data_path = [data_path]
    for dp in data_path:
        dp = dp.strip()
        print_rank_0("loading text-score dataset from: \n   {}".format(dp))

        if dp[-4:] == 'json':
            all_data_list.extend(load_json_data(dp))
        else:
            all_data_list.extend(load_jsonl_data(dp))

    print_rank_0(f"Dataset total length: {len(all_data_list)}")
    max_response_num=1
    if padding:
        max_response_num = max([len(item['score']) for item in all_data_list])    
        print_rank_0(">>> response padding number: {}".format(max_response_num))

    outputs = []
    for item in get_data_iter(all_data_list, debug=debug):        
        new_item = prepare_data_item(item, tokenizer=tokenizer, padding=padding, max_response_num=max_response_num)
        if new_item is not None:
            outputs.append(new_item)

    print_rank_0("finished processing {}  data.".format(len(outputs)))
    return outputs
    

class TextRewardDataset(Dataset):
    def __init__(self, data):
        self.data = data 

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self,):
        return len(self.data)
    
    
def get_reward_data(data_path):
    with open(data_path, 'r') as f:
        lines = f.read().strip().split('\n')

    data = [json.loads(line) for line in lines]
    print_rank_0('finished loading data with {} lines'.format(len(data)))

    for item in data:
        answers = item['answers']
        for style, value in answers.items():
            if isinstance(value, str):
                continue
            elif isinstance(value, dict):
                if "choices" in value:
                    answers[style] = value["choices"][0]["message"]["content"]
                elif "content" in value:
                    answers[style] = value["content"]
                else:
                    print("check this value")
                    print(value)

    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description ='parser for preference data processing.')
    parser.add_argument("--input_data_path", type=str, default="", help="the path to input data.")
    parser.add_argument("--output_data_path", type=str, default="", help="the path to output data.")

    parser.add_argument("--domain", type=str, default="general", help="the domain of the preference data, selected from [general, normal, academy, business, entertainment, literature].")

    parser.add_argument("--convert", action='store_true', help="whether convert responses into the preference text-score format.")
    parser.add_argument("--to_pairs", action='store_true', help="whether convert responses into pair comparisons.")
    
    args = parser.parse_args()
    print(args)

    if args.convert:
        dsp_rm_data = convert_domain_specific_reward_data(data_path=args.input_data_path, domain=args.domain, to_pairs=args.to_pairs)
        with open(args.output_data_path, 'w') as f:
            json.dump(dsp_rm_data, f, ensure_ascii=False)

    
