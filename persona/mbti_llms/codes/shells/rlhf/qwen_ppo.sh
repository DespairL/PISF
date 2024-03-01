#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0
ACTOR_MODEL_PATH="/data/NJU/datasets/persona/models/Qwen-7B-Chat"
ACTOR_ZERO_STAGE=2
CRITIC_ZERO_STAGE=3
MODEL="qwen"
REWARD_PATH="qwen"

Actor_Lr=9.65e-6
Critic_Lr=5e-5
MASTER_PORT=$(shuf -n 1 -i 10001-65535)

for mbti in ENTJ ENFP ENFJ ENTP ESTJ ESTP ESFP ESFJ INTJ INTP INFP INFJ ISTJ ISTP ISFP ISFJ # S N T F J P E I
do
    KL_CTL=0.12
    unsup_coef=27.8

    CRITIC_MODEL_PATH="/data/NJU/datasets/persona/models/rlhf/reward/${REWARD_PATH}/${mbti}"
    OUTPUT="/data/NJU/datasets/persona/models/rlhf/ppo/${MODEL}/${mbti}"
    mkdir -p $OUTPUT
    DATA_PATH="/data/NJU/datasets/persona/mbti_llms/train_datasets/RL_dataset/rl_dataset/ppo/${mbti}.json"
    length=$(expr length "$mbti")
    if [ "$length" = 1 ]; then
        num_warmup_steps=5 #20
        checkpoint_save_steps=20
    else
        num_warmup_steps=20 #80
        checkpoint_save_steps=20
    fi
    echo ===============================
    echo Train $mbti RLHF model
    echo $CRITIC_MODEL_PATH
    echo $OUTPUT
    echo $DATA_PATH
    echo $ACTOR_MODEL_PATH
    echo ===============================
    #3,4,5,
    DS_BUILD_SPARSE_ATTN=0 deepspeed --include localhost:1,2,3,4,5,6,7 --master_port $MASTER_PORT \
        /data/NJU/datasets/persona/mbti_llms/codes/rlhf/llama_ppo.py  \
        --kl_ctl ${KL_CTL} \
        --clip_reward_value 8.0\
        --unsup_coef ${unsup_coef} \
        --mbti $mbti \
        --pretrain_critic_steps 0 \
        --data_path $DATA_PATH \
        --data_split 1 \
        --data_output_path /data/NJU/datasets/persona/mbti_llms/codes/rlhf/cache \
        --unsupervised_dataset_config_name /data/NJU/datasets/persona/mbti_llms/codes/rlhf/wiki/train-00000-of-00041.parquet \
        --critic_model_name_or_path $CRITIC_MODEL_PATH \
        --actor_model_name_or_path $ACTOR_MODEL_PATH \
        --per_device_generation_batch_size 16 \
        --per_device_training_batch_size 4 \
        --generation_batches 1 \
        --num_padding_at_beginning 0\
        --ppo_epochs 1 \
        --max_answer_seq_len 256 \
        --max_prompt_seq_len 256 \
        --num_warmup_steps $num_warmup_steps \
        --actor_learning_rate ${Actor_Lr} \
        --critic_learning_rate ${Critic_Lr} \
        --actor_weight_decay 0.1 \
        --critic_weight_decay 0.1 \
        --num_train_epochs 1 \
        --lr_scheduler_type cosine \
        --gradient_accumulation_steps 1 \
        --critic_gradient_checkpointing \
        --actor_gradient_checkpointing \
        --enable_hybrid_engine \
        --actor_dropout 0.0 \
        --critic_dropout 0.0 \
        --deepspeed --seed 1234 \
        --actor_zero_stage $ACTOR_ZERO_STAGE \
        --critic_zero_stage $CRITIC_ZERO_STAGE \
        --enable_hybrid_engine \
        --output_dir $OUTPUT \
        --dtype "bf16" \
        --actor_lora_dim 64 \
        --critic_lora_dim 64 \
        --critic_lora_module_name "h." \
        --actor_lora_module_name "h." \
        --print_answers \
        --print_answers_interval 5 \
        --model_base "qwen" \
        --enable_tensorboard \
        --tensorboard_path "/data/NJU/datasets/persona/mbti_llms/codes/rlhf/tensorboard_rlhf" \
        --crash_checkpoint_save_path "/data/NJU/datasets/persona/models/rlhf/test_resume" \
        --checkpoint_save_steps $checkpoint_save_steps \
        --prompt_type 'simple_instruction'
done
