#! /bin/bash

# env config
#---------------------------------------------------------------------------------
MASTER_ADDR=$CHIEF_IP
MASTER_PORT=6000

#NUM_GPUS=$NODE_NUM
NUM_GPUS=8

WORKSPACE=/path/to/workspace

REPO_DIR=${WORKSPACE}/DSP

export PYTHONPATH=${REPO_DIR}


DOMAIN='general'
# normal entertainment scientific literary business

# model config
#---------------------------------------------------------------------------------
MODEL_DIR=${WORKSPACE}/saved_models

if [[ "${DOMAIN}" = "general" ]]; then   
    MODEL_NAME=alpaca-native
    MODEL_NAME=llama-7b-hf
    MODEL_PATH=${MODEL_DIR}/${MODEL_NAME}

   #-------------------------------------------------------
else
   MODEL_NAME=llama_4train
   MODEL_PATH=${MODEL_DIR}/experiments/RM_domain_general_${MODEL_NAME}

fi

# data config
#---------------------------------------------------------------------------------
DATA_DIR=${WORKSPACE}/datasets/cleaned_data

DOMAIN='business'
# normal entertainment scientific literary business

if [[ "${DOMAIN}" = "general" ]]; then

    TRAIN_DATA_LIST="${DATA_DIR}/helpful.train.json \
                 ${DATA_DIR}/harmless.train.json \
                 ${DATA_DIR}/webgpt.train.json \
                 ${DATA_DIR}/gpt4llm.train.json"

    TEST_DATA_LIST="${DATA_DIR}/helpful.test.json \
                 ${DATA_DIR}/harmless.test.json \
                 ${DATA_DIR}/webgpt.test.json \
                 ${DATA_DIR}/gpt4llm.test.json"

    DATA_NAME=4train_data

    #--------------------------------------------------
    TRAIN_DATA_LIST="${DATA_DIR}/helpful.train.json \
                 ${DATA_DIR}/harmless.train.json"

    TEST_DATA_LIST="${DATA_DIR}/helpful.test.json \
                 ${DATA_DIR}/harmless.test.json \
                 ${DATA_DIR}/webgpt.test.json \
                 ${DATA_DIR}/gpt4llm.test.json"

    DATA_NAME=HH_train_data
    #--------------------------------------------------
else
    # TRAIN_DATA_LIST="${DATA_DIR}/dsp_${DOMAIN}.train.json"
    # TEST_DATA_LIST="${DATA_DIR}/dsp_${DOMAIN}.test.json \
    #              ${DATA_DIR}/helpful.test.json \
    #              ${DATA_DIR}/harmless.test.json \
    #              ${DATA_DIR}/webgpt.test.json \
    #              ${DATA_DIR}/gpt4llm.test.json"

    # DATA_NAME="ds_${DOMAIN}"
    TEST_DATA_LIST="${DATA_DIR}/helpful.test.json \
                 ${DATA_DIR}/harmless.test.json \
                 ${DATA_DIR}/webgpt.test.json \
                 ${DATA_DIR}/gpt4llm.test.json \
                 ${DATA_DIR}/dsp_${DOMAIN}_pair.test.json"
    

    #--------------------------------------------------
    TRAIN_DATA_LIST="${DATA_DIR}/dsp_${DOMAIN}_pair.train.json"
    DATA_NAME="ds_${DOMAIN}_pair"

    #--------------------------------------------------
    TRAIN_DATA_LIST="${DATA_DIR}/helpful.train.json \
                 ${DATA_DIR}/harmless.train.json \
		 ${DATA_DIR}/dsp_${DOMAIN}_pair.train.json"
    
    DATA_NAME="ds_HH_${DOMAIN}_pair"
fi


DOMAIN="general"

# #DATA_NAME=harmless
DATA_PREFIX=${DATA_DIR}/${DATA_NAME}

# training setups
#---------------------------------------------------------------------------------
DEBUG=false
RESUME_TRAINING=false


MAX_RESPONSE_NUM=5

LM_COEFF=0.1

LEARNING_RATE=1e-6
#MAX_TRAIN_STEPS=100
WARMUP_STEPS=100
LOGGING_STEPS=1
EVAL_STEPS=100
SAVE_STEPS=300
SAVE_TOTAL_LIMIT=30
MAX_TRAIN_STEPS=-1

if [[ "${DEBUG}" = true ]]; then
#    DOMAIN="general"
    EVAL_STEPS=10
    SAVE_STEPS=1
    SAVE_TOTAL_LIMIT=2
    MAX_TRAIN_STEPS=1
fi

if [[ "${DOMAIN}" != "general" ]]; then
    LEARNING_RATE=1e-7
    WARMUP_STEPS=0
    LOGGING_STEPS=1
    EVAL_STEPS=1
    SAVE_STEPS=100
    SAVE_TOTAL_LIMIT=30
    MAX_TRAIN_STEPS=-1
fi

BATCH_SIZE=64

MICRO_BATCH_SIZE=1
if [[ "${DOMAIN}" != "general" ]]; then 
MICRO_BATCH_SIZE=1
fi

EVAL_MICRO_BATCH_SIZE=6

GRADIENT_ACCUMULATION_STEP=$((BATCH_SIZE / NUM_GPUS / MICRO_BATCH_SIZE))
if [[ "${DOMAIN}" != "general" ]]; then 
GRADIENT_ACCUMULATION_STEP=$((GRADIENT_ACCUMULATION_STEP * 4))

fi

MAX_LENGTH=512
PADDING_SIDE="right"
TRUNCATION_SIDE="left"
POOLING_TYPE="last"

# output config
#----------------------------------------------------------------------------------
EXPERIMENT_NAME=RM_${MODEL_NAME}_domain_${DOMAIN}_data_${DATA_NAME}_bs_${BATCH_SIZE}_maxlen_${MAX_LENGTH}_pad_${PADDING_SIDE}_truc_${TRUNCATION_SIDE}_pool_${POOLING_TYPE}_step_w${WARMUP_STEPS}_e${EVAL_STEPS}_rl${LEARNING_RATE}_lm${LM_COEFF}_debug_${DEBUG}_$(date +'%m-%d')


SAVE_DIR=${WORKSPACE}/outputs
OUTPUT_DIR=${SAVE_DIR}/experiments/${EXPERIMENT_NAME}

LOGS_PATH=${OUTPUT_DIR}/logs

mkdir -p $OUTPUT_DIR
mkdir -p $LOGS_PATH


# deepspeed setups
#---------------------------------------------------------------------------------
DEEPSPEED=${REPO_DIR}/configs/default_offload_opt_param.json

echo "begin experiment ${EXPERIMENT_NAME}"


wandb disabled
#export CMD="deepspeed --hostfile ${TMP_DIR}/hostfile --master_addr ${MASTER_ADDR} --master_port=${MASTER_PORT} ${REPO_DIR}/train.py \
#    --resume_from_checkpoint ${RESUME_FROM_CHECKPOINT} \
export CMD="torchrun --nproc_per_node=${NUM_GPUS} --master_port=${MASTER_PORT} ${REPO_DIR}/train.py \
    --do_train True \
    --eval_at_start True\
    --model_type llama \
    --lm_loss_coeff ${LM_COEFF} \
    --model_name_or_path ${MODEL_PATH} \
    --data_prefix ${DATA_PREFIX} \
    --data_dir ${DATA_DIR} \
    --train_data_path ${TRAIN_DATA_LIST} \
    --eval_data_path ${TEST_DATA_LIST} \
    --label_names score tokens \
    --remove_unused_columns false \
    --reward_domain ${DOMAIN} \
    --fp16 false \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 1 \
    --max_steps ${MAX_TRAIN_STEPS} \
    --per_device_train_batch_size ${MICRO_BATCH_SIZE} \
    --per_device_eval_batch_size ${EVAL_MICRO_BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEP} \
    --evaluation_strategy steps \
    --padding_side ${PADDING_SIDE} \
    --truncation_side ${TRUNCATION_SIDE} \
    --pooling_type ${POOLING_TYPE} \
    --max_length ${MAX_LENGTH} \
    --save_strategy steps \
    --save_total_limit ${SAVE_TOTAL_LIMIT} \
    --learning_rate ${LEARNING_RATE} \
    --warmup_steps ${WARMUP_STEPS} \
    --logging_steps ${LOGGING_STEPS} \
    --eval_steps ${EVAL_STEPS} \
    --save_steps ${SAVE_STEPS} \
    --weight_decay 0. \
    --deepspeed ${DEEPSPEED} \
    --tf32 false --debug_mode ${DEBUG}"


CURRENT_TIME=$(date +'%m-%d_%T')

echo $CMD
eval ${CMD} 2>&1 | tee -a ${LOGS_PATH}/log_${CURRENT_TIME}.txt
set +x
