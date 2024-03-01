#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
ZERO_STAGE=2
GPU="1,2,3,4,5,6"
MODEL_PATH=/data/NJU/datasets/persona/models/Qwen-7B-Chat
MASTER_PORT=$(shuf -n 1 -i 10001-65535)
MODEL=qwen
REWARD_DATA_PATH=/data/NJU/datasets/persona/mbti_llms/train_datasets/RL_dataset/rl_dataset/reward_augment/
BATCH_SIZE=16

for mbti in T #ISTP ISFP ISFJ # E I S N T F J P ENTJ ENFP ENFJ ENTP ESTJ ESTP ESFP ESFJ INTJ INTP INFP INFJ ISTJ
do
    echo $mbti
    OUTPUT=/data/NJU/datasets/persona/models/rlhf/reward/${MODEL}/${mbti}
    mkdir -p $OUTPUT
    DATA_PATH=${REWARD_DATA_PATH}/${mbti}.json
    deepspeed  --include localhost:${GPU} --master_port $MASTER_PORT \
        /data/NJU/datasets/persona/mbti_llms/codes/rlhf/llama_reward_train.py \
        --data_path $DATA_PATH \
        --data_split 1\
        --data_output_path /data/NJU/datasets/persona/mbti_llms/codes/rlhf/cache/data_files \
        --model_name_or_path $MODEL_PATH \
        --per_device_train_batch_size ${BATCH_SIZE} \
        --per_device_eval_batch_size 4 \
        --max_seq_len 512 \
        --learning_rate 9.65e-6 \
        --num_padding_at_beginning 0 \
        --num_train_epochs 1  \
        --gradient_accumulation_steps 1 \
        --lr_scheduler_type cosine \
        --num_warmup_steps 0 \
        --seed 1234 \
        --gradient_checkpointing \
        --zero_stage $ZERO_STAGE \
        --deepspeed \
        --lora_dim 128 \
        --lora_module_name "h." \
        --output_dir $OUTPUT \
        --model_base qwen \
        --dtype "bf16" \
        --file qwen_log
done