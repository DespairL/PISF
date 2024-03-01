NUM_GPUS=1

CUDA_VISIBLE_DEVICES=2 torchrun --standalone --nnodes=1 --nproc-per-node=$NUM_GPUS /data/NJU/datasets/persona/mbti_llms/codes/answer_extractor/en_answer_extractor/train_lora_falcon.py \
    --do_eval \
    --split_train_validation \
    --split_save_path '/data/NJU/datasets/persona/mbti_llms/codes/answer_extractor/en_answer_extractor' \
    --split_file '/data/NJU/datasets/persona/mbti_llms/train_datasets/answer_extractor/v1.json' \
    --adapter_path '/data/NJU/datasets/persona/models/en_answer_extractor/backup_f19465' \
    --output_dir '/data/NJU/datasets/persona/models/en_answer_extractor' \
    --train_file '/data/NJU/datasets/persona/mbti_llms/train_datasets/answer_extractor/train.json' \
    --validation_file '/data/NJU/datasets/persona/mbti_llms/train_datasets/answer_extractor/validation.json' \
    --preprocessing_num_workers 8 \
    --prompt_column prompt \
    --response_column answer \
    --overwrite_cache \
    --model_name_or_path /data/NJU/datasets/persona/models/falcon-7b-instruct \
    --per_device_eval_batch_size 8 \
    --bf16 \
    --report_to "none" \
