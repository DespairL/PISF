NUM_GPUS=1
GPU=4
LR=5e-4
for mbti_type in E #ESTP #E I S N T F J P ENTJ ENFP ENFJ ENTP ESTJ ESTP ESFP ESFJ INTJ INTP INFP INFJ ISTJ ISTP ISFP ISFJ
do
    FILE=/data/NJU/datasets/persona/mbti_llms/train_datasets/instruction_tuning_datasets/sft_dataset/${mbti_type}.json
    SAVE_SUFFIX=$mbti_type
    base_dir=/data/NJU/datasets/persona/mbti_llms/codes/sft/
    CUDA_VISIBLE_DEVICES=$GPU torchrun --standalone --nnodes=1 --nproc-per-node=$NUM_GPUS ${base_dir}sft.py \
        --personality $mbti_type\
        --do_train \
        --do_lora_train \
        --train_file $FILE \
        --prompt_column question \
        --response_column answer \
        --model_name_or_path /data/NJU/datasets/persona/models/Llama-2-13b-chat-hf \
        --output_dir  /data/NJU/datasets/persona/models/sft_new/Llama-2-13b-chat-$SAVE_SUFFIX \
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
        --model_base 'Llama_13b_chat'
done
