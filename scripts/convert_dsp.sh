#! /bin/bash

WORKSPACE=/path/to/workspace
REPO_DIR=${WORKSPACE}/DSP
export PYTHONPATH=${REPO_DIR}

DOMAIN="academy"

for DATA_TYPE in "train" "test"
do
python ${REPO_DIR}/reward_datasets.py \
    --input_data_path ${REPO_DIR}/data/domain_specific_preference.${DATA_TYPE}.json \
    --domain ${DOMAIN} \
    --output_data_path data/dsp_${DOMAIN}_pairs.${DATA_TYPE}.json \
    --convert --to_pairs
done
