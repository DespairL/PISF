LR=5e-4
NUM_GPUS=1

#--split_train_validation \
#--split_save_path '/data/NJU/datasets/persona/mbti_llms/codes/answer_extractor/en_answer_extractor' \

CUDA_VISIBLE_DEVICES=2 torchrun --standalone --nnodes=1 --nproc-per-node=$NUM_GPUS /data/NJU/datasets/persona/mbti_llms/codes/answer_extractor/en_answer_extractor/train_lora_falcon.py \
    --do_train \
    --split_train_validation \
    --split_save_path '/data/NJU/datasets/persona/mbti_llms/codes/answer_extractor/en_answer_extractor' \
    --split_file '/data/NJU/datasets/persona/mbti_llms/train_datasets/answer_extractor/v1.json' \
    --train_file '/data/NJU/datasets/persona/mbti_llms/train_datasets/answer_extractor/train.json' \
    --validation_file '/data/NJU/datasets/persona/mbti_llms/train_datasets/answer_extractor/validation.json' \
    --preprocessing_num_workers 8 \
    --prompt_column prompt \
    --response_column answer \
    --warmup_ratio 0.05 \
    --overwrite_cache \
    --lr_scheduler_type cosine \
    --model_name_or_path /data/NJU/datasets/persona/models/falcon-7b-instruct \
    --output_dir /data/NJU/datasets/persona/models/en_answer_extractor \
    --overwrite_output_dir \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing \
    --num_train_epochs 5 \
    --evaluation_strategy "epoch"\
    --load_best_model_at_end \
    --metric_for_best_model f1 \
    --save_strategy "epoch"\
    --save_total_limit 1\
    --bf16 \
    --report_to "none" \
    --logging_steps 5 \
    --learning_rate $LR
