set -e
GPU=7
write_path_c="/data/NJU/datasets/persona/mbti_llms/results/debias_results/cp_chatglm2_new_setting.txt"
base_falcon_path="/data/NJU/datasets/persona/models/falcon-7b-instruct"
extractor_adapter_path="/data/NJU/datasets/persona/models/en_answer_extractor/backup_f19465"
output_path="cp_chatglm2_new_setting.json"
for mbti_type in ESTP ESTJ ESFP ESFJ ENFJ ENFP ENTP ENTJ ISTJ ISTP ISFP ISFJ INFJ INFP INTP INTJ E I S N T F J P #no
do
    length=$(expr length "$mbti_type")
    if [ "$mbti_type" = "no" ]; then
        model="/data/NJU/datasets/persona/models/chatglm2-6b"
    else
        if [ "$length" = 1 ]; then
            checkpoint="checkpoint-833"
        else

            checkpoint="checkpoint-104"
        fi
        model="/data/NJU/datasets/persona/models/continual_pretrain/chatglm2-6b-${mbti_type}/${checkpoint}"
    fi
    for i in 0 1 2 3 4
    do
        test_file="/data/NJU/datasets/persona/mbti_llms/evaluate_datasets/en_unified_dataset/multi_slot_evaluate_prompt/no_specific_prompt/unified_prompt_no_mbti_${i}.json"
        echo ===================================
        echo $test_file
        echo $model
        echo $mbti_type - $i
        echo Device: $GPU
        echo Results would be save to $write_path_c
        echo ===================================
        python3 /data/NJU/datasets/persona/mbti_llms/codes/unified_model_evaluate.py --model_base chatglm2_6b \
            --training_phase cp \
            --model_path $model \
            --evaluate_specific_file $test_file \
            --output_path $output_path\
            --gpu $GPU

        model_name=${i}-chatglm2-6b-$mbti_type
        python3 /data/NJU/datasets/persona/mbti_llms/codes/answer_extractor/en_answer_extractor/extract_answers.py \
            --unlabeled_json_path $output_path --model_name $model_name \
            --write_path $write_path_c --model_path $base_falcon_path \
            --adapter_path $extractor_adapter_path --gpu $GPU
    done
done