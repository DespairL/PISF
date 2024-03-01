set -e
base_path="/data/NJU/datasets/persona/mbti_llms/evaluate_datasets/en_unified_dataset/multi_slot_evaluate_prompt/"
path1="${base_path}no_specific_prompt/"
path2="${base_path}specific_personality_prompt/"
path3="${base_path}specific_trait_prompt/"
GPU=5
write_path_c="/data/NJU/datasets/persona/mbti_llms/results/debias_results/sft_llama2_chat_with_instruction_with_user_prompt.txt"
base_falcon_path="/data/NJU/datasets/persona/models/falcon-7b-instruct"
extractor_adapter_path="/data/NJU/datasets/persona/models/en_answer_extractor/backup_f19465"
base_llama_path="/data/NJU/datasets/persona/models/Llama-2-13b-chat-hf"
output_path="sft_llama2_chat_with_instruction_with_user_prompt.json"
count=0
for mbti_type in E I S N T F J P ESTP ESTJ ESFP ESFJ
do
    length=$(expr length "$mbti_type")
    if [ "$mbti_type" = "no" ]; then
        model="/data/NJU/datasets/persona/models/Llama-2-13b-chat-hf"
    else
        if [ "$length" = 1 ]; then
            checkpoint="checkpoint-626"
        else
            checkpoint="checkpoint-2500"
        fi
        model="/data/NJU/datasets/persona/models/sft/Llama-2-13b-chat-${mbti_type}/${checkpoint}"
    fi
    for i in 0 1 2 3 4
    do
        if [ $count -lt 8 ]; then
            path=$path3
        else
            path=$path2
        fi
        file_name="unified_prompt_${mbti_type}_mbti_${i}.json"
        path="${path}${file_name}"
        test_file=$path
        echo ===================================
        echo $test_file
        echo $model
        echo $mbti_type - $i
        echo Device: $GPU
        echo Results would be save to $write_path_c
        echo ===================================
        python3 /data/NJU/datasets/persona/mbti_llms/codes/unified_model_evaluate.py --model_base Llama_13b_chat \
            --training_phase sft \
            --model_path $base_llama_path \
            --adapter_path $model \
            --evaluate_specific_file $test_file \
            --output_path $output_path\
            --default_prompt \
            --mbti $mbti_type \
            --with_instruction \
            --gpu $GPU

        model_name=${i}-llama2-chat-13b-$mbti_type
        python3 /data/NJU/datasets/persona/mbti_llms/codes/answer_extractor/en_answer_extractor/extract_answers.py \
            --unlabeled_json_path $output_path --model_name $model_name \
            --write_path $write_path_c --model_path $base_falcon_path \
            --adapter_path $extractor_adapter_path --gpu $GPU
    done
    count=$((count + 1))
done