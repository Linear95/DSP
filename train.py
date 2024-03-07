import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List
import json

import torch
import transformers

from torch.utils.data import Dataset
from transformers import Trainer, AutoConfig
from transformers import EvalPrediction


from model import LlamaRewardModel, BertRewardModel
from utils import print_rank_0
from reward_datasets import TextRewardDataset, reward_data_collactor
from reward_datasets import load_text_score_dataset
from arguments import CustomTrainingArguments
from trainer import RewardModelTrainer, compute_metrics

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


def get_eval_datasets(args, tokenizer):
    
    data_dict = {}

    for data_path in args.eval_data_path:
        eval_data_list = load_text_score_dataset(
            data_path=data_path,
            tokenizer=tokenizer,
            debug=args.debug_mode,
            padding=not args.per_device_eval_batch_size == 1
        )
        eval_dataset = TextRewardDataset(eval_data_list)
        data_dict[data_path] = eval_dataset

    return data_dict

def get_train_dataset(args, tokenizer):    
    train_data = load_text_score_dataset(
        data_path=args.train_data_path,
        tokenizer=tokenizer, 
        debug=args.debug_mode,
        padding=not args.per_device_train_batch_size == 1
    )

    # if args.debug_mode:
    print_rank_0(f">>> check tokenized data:")        
    print_rank_0(f">>> {train_data[0]}")

    train_set = TextRewardDataset(train_data)
    return train_set


def set_llama_tokenizer(model, tokenizer):
    tokenizer.pad_token_id = 3
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2
    tokenizer.unk_token_id = 0
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    print_rank_0(tokenizer)
    return model, tokenizer



def train():
    parser = transformers.HfArgumentParser(CustomTrainingArguments)
    args = parser.parse_args_into_dataclasses()[0]
    print_rank_0(args)

    # setup model
    #---------------------------------------------------------------------------------
    print_rank_0(f"Begin loading model from {args.model_name_or_path}")
    if args.model_type != "bert":
        model = LlamaRewardModel.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir,
        )
    else:
        model = BertRewardModel.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir,
        )

    print_rank_0(model)
    print_rank_0(f"Finished loading model from {args.model_name_or_path}")

    model.is_parallelizable = True
    model.model_parallel = True

    # setup tokenizer
    #---------------------------------------------------------------------------------
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        model_max_length=args.max_length,        
        padding_side=args.padding_side,
        truncation_side=args.truncation_side,
        use_fast=False,
    )

    if args.model_type != "bert":
        model, tokenizer = set_llama_tokenizer(model=model, tokenizer=tokenizer)
        print_rank_0(f"check tokenizer length {len(tokenizer)}")

    # load data
    #---------------------------------------------------------------------------------    
    if args.do_train:
        train_dataset = get_train_dataset(args, tokenizer)
    else:
        train_dataset = None
    eval_dataset_dict = get_eval_datasets(args, tokenizer)

    # build trainer
    #---------------------------------------------------------------------------------

    trainer = RewardModelTrainer(
        model=model, 
        tokenizer=tokenizer, 
        args=args,
        compute_metrics=lambda x: compute_metrics(args, x),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset_dict,
        data_collator=reward_data_collactor
    )

    if args.do_train:
        if args.eval_at_start:
            for eval_set_name, eval_dataset in eval_dataset_dict.items():
                eval_result = trainer.evaluate(eval_dataset=eval_dataset, metric_key_prefix="eval_"+eval_set_name)
                print_rank_0(eval_result)

        if args.resume_from_checkpoint:
            train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        else:
            train_result = trainer.train()

        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

        trainer.save_state()
        trainer.save_model(output_dir=args.output_dir)

    final_eval_results ={}
    for eval_set_name, eval_dataset in eval_dataset_dict.items():
        eval_result = trainer.evaluate(eval_dataset=eval_dataset, metric_key_prefix="eval_"+eval_set_name)
        print_rank_0(eval_result)
        final_eval_results[eval_set_name] = eval_result

    with open(f"{args.output_dir}/final_eval_results.json", 'w') as f:
        json.dump(final_eval_results, f, ensure_ascii=False)

if __name__ == "__main__":
    train()
