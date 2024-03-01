#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# Add the path to the finetuned model
REWARD_DATA_PATH=/data/NJU/datasets/persona/mbti_llms/train_datasets/RL_dataset/rl_dataset/reward_augment/
MODEL=qwen
REWARD_MODEL=qwen
MASTER_PORT=$(shuf -n 1 -i 10001-65535)

for mbti in I S N T F J P ENTJ ENFP ENFJ ENTP ESTJ ESTP ESFP ESFJ INTJ INTP INFP INFJ ISTJ ISTP ISFP ISFJ # E
do
    deepspeed  --include localhost:7 --master_port $MASTER_PORT  /data/NJU/datasets/persona/mbti_llms/codes/rlhf/llama2_reward_model_eval.py \
        --model_name_or_path /data/NJU/datasets/persona/models/rlhf/reward/${REWARD_MODEL}/${mbti} \
        --num_padding_at_beginning 0 \
        --eval_data_path ${REWARD_DATA_PATH}${mbti}.json \
        --model_base $MODEL \
        --mbti $mbti \
        --per_device_eval_batch_size 32 \
        --metric_save_path /data/NJU/datasets/persona/mbti_llms/codes/rlhf/reward_log/metric_${MODEL}.json
done
