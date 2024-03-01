set -e
count=0
base_path="/data/NJU/datasets/persona/mbti_llms/evaluate_datasets/en_unified_dataset/multi_slot_evaluate_prompt/"
path1="${base_path}no_specific_prompt/"
path2="${base_path}specific_personality_prompt/"
path3="${base_path}specific_trait_prompt/"
GPU=7
write_path_c="/data/NJU/datasets/persona/mbti_llms/results/debias_results/basic_chatglm2_prompt_induction.txt"
base_falcon_path="/data/NJU/datasets/persona/models/falcon-7b-instruct"
extractor_adapter_path="/data/NJU/datasets/persona/models/en_answer_extractor/backup_f19465"
base_llama_path="/data/NJU/datasets/persona/models/chatglm2-6b"
output_path="basic_chatglm2_prompt.json"
for mbti_type in E I S N T F J P ESTP ESTJ ESFP ESFJ ENFJ ENFP ENTP ENTJ ISTJ ISTP ISFP ISFJ INFJ INFP INTP INTJ
do
    length=$(expr length "$mbti_type")
    #if [ $count -lt 2 ]; then
    #    echo "Skip ${mbti_type}"
    #    count=$((count + 1))
    #    continue
    #fi
    model="/data/NJU/datasets/persona/models/chatglm2-6b"
    for i in 0 1 2 3 4
    do
        #if [ "$mbti_type" = "S" -a $i -lt 4 ]; then
        #    echo "Skip ${mbti_type}-${i}"
        #    continue
        #fi
        if [ $count -lt 8 ]; then
            path=$path3
        else
            path=$path2
        fi
        file_name="unified_prompt_${mbti_type}_mbti_${i}.json"
        path="${path}${file_name}"
        test_file=$path
        echo ===================================
        echo $model
        echo Device: $GPU
        echo Results would be save to $write_path_c
        echo ===================================
        python3 /data/NJU/datasets/persona/mbti_llms/codes/unified_model_evaluate.py --model_base chatglm2_6b \
            --training_phase cp \
            --model_path $base_llama_path \
            --adapter_path $model \
            --evaluate_specific_file $test_file \
            --output_path $output_path\
            --mbti $mbti_type\
            --gpu $GPU

        model_name=${i}-chatglm2-6b-$mbti_type
        python3 /data/NJU/datasets/persona/mbti_llms/codes/answer_extractor/en_answer_extractor/extract_answers.py \
            --unlabeled_json_path $output_path --model_name $model_name \
            --write_path $write_path_c --model_path $base_falcon_path \
            --adapter_path $extractor_adapter_path --gpu $GPU
    done
    count=$((count + 1))
done
