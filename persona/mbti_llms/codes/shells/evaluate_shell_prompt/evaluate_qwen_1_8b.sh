set -e
count=0
base_path="/data/NJU/datasets/persona/mbti_llms/evaluate_datasets/en_unified_dataset/multi_slot_evaluate_prompt/"
path1="${base_path}no_specific_prompt/"
path2="${base_path}specific_personality_prompt/"
path3="${base_path}specific_trait_prompt/"
model="Qwen-1_8B"
model_path=/data/NJU/datasets/persona/models/${model}-Chat/
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
write_path_c="/data/NJU/datasets/persona/mbti_llms/results/en_results/prompt_${model}.txt"
base_falcon_path="/data/NJU/datasets/persona/models/falcon-7b-instruct"
extractor_adapter_path="/data/NJU/datasets/persona/models/en_answer_extractor/backup_f19465"
output_path="/data/NJU/datasets/persona/mbti_llms/results/debias_results/qwen_results/${model}_results/"
GPU=7
for prompt in no E I S N T F J P ESTJ ESTP ESFP ESFJ ENFJ ENFP ENTP ENTJ ISTJ ISTP ISFP ISFJ INFJ INFP INTP INTJ
do
    # continual
    #if [ $count -lt 11 ]; then
    #    echo "Skip ${prompt}"
    #    count=$((count + 1))
    #    continue
    #fi
    for i in 0 1 2 3 4
    do
        #if [ "$prompt" = "ESFP" -a $i -lt 1 ]; then
        #    echo "Skip ${prompt}-${i}"
        #    continue
        #fi
        if [ $count -lt 1 ]; then
            path=$path1
        elif [ $count -lt 9 ]; then
            path=$path3
        else
            path=$path2
        fi
        file_name="unified_prompt_${prompt}_mbti_${i}.json"
        path="${path}${file_name}"
        output_temp=$output_path$file_name
        echo ===================================
        echo evaluate file : $path
        echo GPU : $GPU
        echo output : $output_temp
        echo ===================================
        mbti_type=$prompt
        CUDA_VISIBLE_DEVICES=$GPU torchrun --nnodes=1 --nproc-per-node=1 --rdzv-backend=c10d \
            --rdzv-endpoint=localhost:${MASTER_PORT} \
            /data/NJU/datasets/persona/mbti_llms/codes/prompt_test/evaluate_all_qwen.py \
            --test_specific_file $path \
            --model $model_path --mbti_type $mbti_type \
            --answer_output $output_temp
    done
    count=$((count + 1))
done
