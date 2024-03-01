set -e
GPU=7
write_path_c="/data/NJU/datasets/persona/mbti_llms/results/debias_results/sft_chatglm2_re_evaluate_with_instruction.txt"
base_falcon_path="/data/NJU/datasets/persona/models/falcon-7b-instruct"
extractor_adapter_path="/data/NJU/datasets/persona/models/en_answer_extractor/backup_f19465"
base_llama_path="/data/NJU/datasets/persona/models/chatglm2-6b"
output_path="sft_chatglm_with_instruction.json"
count=0
for mbti_type in ESTP ESTJ ESFP ESFJ ENFJ ENFP ENTP ENTJ ISTJ ISTP ISFP ISFJ INFJ INFP INTP INTJ E I S N T F J P # E no
do
    #if [ $count -lt 9 ]; then
    #    echo "Skip ${mbti_type}"
    #    count=$((count + 1))
    #    continue
    #fi
    length=$(expr length "$mbti_type")
    if [ "$mbti_type" = "no" ]; then
        model="/data/NJU/datasets/persona/models/chatglm2-6b"
    else
        if [ "$length" = 1 ]; then
            checkpoint="checkpoint-626"
        else
            checkpoint="checkpoint-2500"
        fi
        model="/data/NJU/datasets/persona/models/sft/chatglm2-6b-${mbti_type}/${checkpoint}"
    fi
    for i in 0 1 2 3 4
    do
        #if [ "$mbti_type" = "ISTP" -a $i -lt 2 ]; then
        #    echo "Skip ${mbti_type}-${i}"
        #    continue
        #fi
        test_file="/data/NJU/datasets/persona/mbti_llms/evaluate_datasets/en_unified_dataset/multi_slot_evaluate_prompt/no_specific_prompt/unified_prompt_no_mbti_${i}.json"
        #test_file="/data/NJU/datasets/persona/mbti_llms/evaluate_datasets/en_unified_dataset/role_play_human/unified_prompt_no_mbti_${i}.json"
        echo ===================================
        echo $test_file
        echo $model
        echo Device: $GPU
        echo Results would be save to $write_path_c
        echo ===================================
        python3 /data/NJU/datasets/persona/mbti_llms/codes/unified_model_evaluate.py --model_base chatglm2_6b \
            --training_phase sft \
            --model_path $base_llama_path \
            --adapter_path $model \
            --evaluate_specific_file $test_file \
            --output_path $output_path\
            --default_prompt \
            --with_instruction \
            --mbti $mbti_type \
            --gpu $GPU

        model_name=${i}-chatglm2-6b-$mbti_type
        python3 /data/NJU/datasets/persona/mbti_llms/codes/answer_extractor/en_answer_extractor/extract_answers.py \
            --unlabeled_json_path $output_path --model_name $model_name \
            --write_path $write_path_c --model_path $base_falcon_path \
            --adapter_path $extractor_adapter_path --gpu $GPU
    done
    count=$((count + 1))
done