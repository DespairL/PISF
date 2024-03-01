set -e
base_path="/data/NJU/datasets/persona/mbti_llms/evaluate_datasets/en_unified_dataset/multi_slot_evaluate_prompt/"
path1="${base_path}no_specific_prompt/"
path2="${base_path}specific_personality_prompt/"
path3="${base_path}specific_trait_prompt/"
GPU=1 # 1,3
write_path_c="/data/NJU/datasets/persona/mbti_llms/results/debias_results/qwen_with_sys_prompt_inverse_prompt.txt"
base_falcon_path="/data/NJU/datasets/persona/models/falcon-7b-instruct"
extractor_adapter_path="/data/NJU/datasets/persona/models/en_answer_extractor/backup_f19465"
base_llama_path="/data/NJU/datasets/persona/models/Qwen-7B-Chat"
output_path="qwen_sys_prompt_inverse_prompt.json"
count=0
for mbti_type in E I S N T F J P ESTP ESTJ ESFP ESFJ ENFJ ENFP ENTP ENTJ ISTJ ISTP ISFP ISFJ INFJ INFP INTP INTJ
do
    length=$(expr length "$mbti_type")
    model="/data/NJU/datasets/persona/models/Qwen-7B-Chat"
    #if [ $count -lt 1 ]; then
    #    echo "Skip ${mbti_type}"
    #    count=$((count + 1))
    #    continue
    #fi
    for i in 0 1 2 3 4
    do
        if [ $count -lt 8 ]; then
            path=$path3
        else
            path=$path2
        fi
        # path=$path3
        file_name="unified_prompt_${mbti_type}_mbti_${i}.json"
        path="${path}${file_name}"
        test_file=$path
        echo ===================================
        echo $test_file
        echo $model
        echo Device: $GPU
        echo Results would be save to $write_path_c
        echo ===================================
        python3 /data/NJU/datasets/persona/mbti_llms/codes/unified_model_evaluate.py --model_base qwen_chat_7b \
            --training_phase sft \
            --model_path $base_llama_path \
            --adapter_path $model \
            --evaluate_specific_file $test_file \
            --output_path $output_path\
            --default_prompt \
            --mbti $mbti_type \
            --prompt_induction_after_training \
            --same_prompt_induction_after_training \
            --gpu $GPU

        model_name=${i}-qwen-chat-7b-$mbti_type
        python3 /data/NJU/datasets/persona/mbti_llms/codes/answer_extractor/en_answer_extractor/extract_answers.py \
            --unlabeled_json_path $output_path --model_name $model_name \
            --write_path $write_path_c --model_path $base_falcon_path \
            --adapter_path $extractor_adapter_path --gpu $GPU
    done
    count=$((count + 1))
done