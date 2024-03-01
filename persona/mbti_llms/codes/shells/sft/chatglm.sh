set -e
NUM_GPUS=1
GPU=2
LR=5e-4
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
for mbti_type in E # INTP INFP INFJ ISTJ ISTP ISFP ISFJ # E I S N T F J P ENTJ ENFP ENFJ ENTP ESTJ ESTP ESFP ESFJ INTJ
do
    FILE=/data/NJU/datasets/persona/mbti_llms/train_datasets/instruction_tuning_datasets/sft_dataset/${mbti_type}.json
    SAVE_SUFFIX=$mbti_type
    base_dir=/data/NJU/datasets/persona/mbti_llms/codes/sft/
    CUDA_VISIBLE_DEVICES=$GPU torchrun --rdzv-backend=c10d \
        --rdzv-endpoint=localhost:$MASTER_PORT --nnodes=1 --nproc-per-node=$NUM_GPUS ${base_dir}sft.py \
        --personality $mbti_type\
        --do_train \
        --do_lora_train \
        --train_file $FILE \
        --prompt_column question \
        --response_column answer \
        --model_name_or_path /data/NJU/datasets/persona/models/chatglm2-6b \
        --output_dir  /data/NJU/datasets/persona/models/sft/chatglm2-6b-$SAVE_SUFFIX \
        --overwrite_output_dir \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 1 \
        --num_train_epochs 2 \
        --logging_steps 5 \
        --learning_rate $LR \
        --bf16 \
        --save_strategy "epoch"\
        --preprocessing_num_workers 8 \
        --report_to "none" \
        --dataloader_num_workers 16 \
        --gradient_checkpointing \
        --save_total_limit 1 \
        --ddp_find_unused_parameters False \
        --model_base 'chatglm2_6b'
done
