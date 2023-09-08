# Learning Customized Human Preferences

This is the repo for the paper: [Everyone Deserves A Reward: Learning Customized Human Preferences](https://arxiv.org/abs/2309.03126).
The repo contains:
- the [Domain-Specific Preference](data/) (DSP) dataset
- the cleaned general preference datasets including [Helpful&Harmless](https://github.com/anthropics/hh-rlhf/tree/master), [WebGPT](https://huggingface.co/datasets/openai/webgpt_comparisons), and
[GPT-4-LLM](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM/tree/main/data)
- the implementation of customized reward model learning

## Overview

Reward models (RMs) are essential for aligning LLMs with human preferences. However, the real world is pluralistic. Besides universal values,  human preferences can be diversified based on different religions, politics, cultures, etc. This leads to an
interesting question: *"How to learn a customized reward model well while preserving its general preference ability?"* To answer this, we collect a Domain-Specific Preference (DSP) dataset and test multiple training and data strategies for customized RM
learning.

## Domain-Specific Preference dataset

<p align="center">
  <img src="figures/DSP_word_clouds.jpeg" height="80%" width="80%">
</p>

We collect domain-specific preferences from the four typical domains: *Academy*, *Business*, *Entertainment*, and *Literature\&Art*. We select 13K instructions from the 52K [Alpaca](https://github.com/tatsu-lab/stanford_alpaca) training set, then ask ChatGPT
with domain-specific system prompts to collect preferred responses from each domain, as shown in the figure below.

<p align="center">
  <img src="figures/DSP_data_collection.png" height="80%" width="80%">
</p>

The collected data are  `data/domain_specific_preference.train.json` and `data/domain_specific_preference.test.json`, which are list of items including the following keys:
- `query`: A question collected from the [Alpaca](https://github.com/tatsu-lab/stanford_alpaca) training set.
- `responses`: A collection of responses from the four application domains (`academy`, `business`, `entertainment`, `literature`) additionally with the origin response from Alpaca (marked as `normal`).

To convert the collected responses into a preference data format, use the following command:
```bash
DOMAIN="academy"
DATA_TYPE="train"

python reward_datasets.py \
       --input_data_path data/domain_specific_preference.${DATA_TYPE}.json \
       --domain ${DOMAIN} \
       --output_data_path data/dsp_${DOMAIN}_pairs.${DATA_TYPE}.json \
       --convert --to_pairs
```
where `DOMAIN` can be changed to the other four domains, and `DATA_TYPE` is set from `train` and `test`.
After the conversion, the preference data has a `text`-`score` format that each item contrains two keys:
- `text`: a list of text, each of which combines the query and a response as a complete human-assistant interation.
- `score`: a list of scores, each of which is a preference score to the corresponding interation.

Besides, we also cleaned the general preference comparisons from [Helpful&Harmless](https://github.com/anthropics/hh-rlhf/tree/master), [WebGPT](https://huggingface.co/datasets/openai/webgpt_comparisons), and
[GPT-4-LLM](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM/tree/main/data) datasets and convert them into the same `text`-`score` format.

## Customized RM training
<p align="center">
  <img src="figures/learning_stages_customized_rm.png" height="55%" width="55%">
</p>