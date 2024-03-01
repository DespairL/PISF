set -e
GPU=2
write_path_c="/data/NJU/datasets/persona/mbti_llms/results/debias_results/rlhf_llama2_chat_new_ppo.txt"
base_falcon_path="/data/NJU/datasets/persona/models/falcon-7b-instruct"
extractor_adapter_path="/data/NJU/datasets/persona/models/en_answer_extractor/backup_f19465"
base_llama_path="/data/NJU/datasets/persona/models/Llama-2-13b-chat-hf"
output_path="rlhf_llama_new_ppo.json"
model_flag='llama'
for mbti_type in E #ESTP ESTJ ESFP ESFJ ENFJ ENFP ENTP ENTJ ISTJ ISTP ISFP ISFJ INFJ INFP INTP INTJ #E I S N T F J P
do
    if [ "$mbti_type" = "no" ]; then
        model="/data/NJU/datasets/persona/models/Llama-2-13b-chat-hf"
    else
        model="/data/NJU/datasets/persona/models/rlhf/new_llama_ppo/${model_flag}/${mbti_type}/actor"
    fi
    for i in 0 1 2 3 4
    do
        test_file="/data/NJU/datasets/persona/mbti_llms/evaluate_datasets/en_unified_dataset/multi_slot_evaluate_prompt/no_specific_prompt/unified_prompt_no_mbti_${i}.json"
        echo ===================================
        echo $test_file
        echo $model
        echo Device: $GPU
        echo Results would be save to $write_path_c
        echo ===================================
        python3 /data/NJU/datasets/persona/mbti_llms/codes/unified_model_evaluate.py --model_base Llama_13b_chat \
            --training_phase rlhf \
            --model_path $base_llama_path \
            --ppo_path $model\
            --evaluate_specific_file $test_file \
            --output_path $output_path\
            --mbti $mbti_type \
            --ppo_prompt_type "simple_instruction" \
            --gpu $GPU

        model_name=${i}-llama2-chat-13b-$mbti_type
        python3 /data/NJU/datasets/persona/mbti_llms/codes/answer_extractor/en_answer_extractor/extract_answers.py \
            --unlabeled_json_path $output_path --model_name $model_name \
            --write_path $write_path_c --model_path $base_falcon_path \
            --adapter_path $extractor_adapter_path --gpu $GPU
    done
done