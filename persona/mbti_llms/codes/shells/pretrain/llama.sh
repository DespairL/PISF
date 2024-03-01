for mbti_type in ENTJ ENFP ENFJ ENTP ESTJ ESTP ESFP ESFJ INTJ INTP INFP INFJ ISTJ ISTP ISFP ISFJ T F J P
do
    LR=5e-6
    FILE=/data/NJU/datasets/persona/mbti_llms/train_datasets/pretrain_datasets/en_pretrain_datasets/$mbti_type
    SAVE_SUFFIX=$mbti_type
    MASTER_PORT=$(shuf -n 1 -i 10000-65535)
    base_dir=/data/NJU/datasets/persona/mbti_llms/codes/pretrain/
    deepspeed --include localhost:1,2,3,4,5,6 --master_port $MASTER_PORT ${base_dir}pretrain.py \
        --personality $mbti_type\
        --do_train \
        --deepspeed ${base_dir}ds3.json \
        --train_file ${base_dir}empty.json \
        --pretrain_dataset_cache $FILE \
        --prompt_column content \
        --model_name_or_path /data/NJU/datasets/persona/models/Llama-2-13b-chat-hf \
        --output_dir  /data/NJU/datasets/persona/models/continual_pretrain/Llama-2-13b-chat-$SAVE_SUFFIX \
        --overwrite_output_dir \
        --per_device_train_batch_size 6 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 1 \
        --num_train_epochs 1 \
        --logging_steps 5 \
        --learning_rate $LR \
        --bf16 \
        --save_strategy "epoch"\
        --preprocessing_num_workers 8 \
        --report_to "none" \
        --dataloader_num_workers 16 \
        --gradient_checkpointing \
        --save_total_limit 1 \
        --model_base 'Llama_13b_chat'
done
