set -e
MODEL_SIZE="70b"
GPU=1
result_base_path="/data/NJU/datasets/persona/mbti_llms/results/debias_results/llama_results/"
model_results=${result_base_path}Llama-2-${MODEL_SIZE}_results/
write_path_c="/data/NJU/datasets/persona/mbti_llms/results/debias_results/prompt_llama_${MODEL_SIZE}.txt"
base_falcon_path="/data/NJU/datasets/persona/models/falcon-7b-instruct"
extractor_adapter_path="/data/NJU/datasets/persona/models/en_answer_extractor/backup_f19465"
files=$(ls "$model_results")
echo ===================================
echo Base dictionary: $model_results
echo All files:
echo $files
echo ===================================
for file in $files
do
    type=$(echo "$file" | sed -n 's/unified_prompt_\([[:alnum:]]\+\)_mbti_\([[:alnum:]]\+\)\.json/\1/p')
    index=$(echo "$file" | sed -n 's/unified_prompt_\([[:alnum:]]\+\)_mbti_\([[:alnum:]]\+\)\.json/\2/p')
    #len=$(expr length "$type")
    #if [ "$len" = "1" ] || [ "$len" = "2" ]; then
    #    continue
    #fi
    echo "Cur Type: $type, Cur Index: $index"
    model_name=${index}-Llama-2-${MODEL_SIZE}-chat-$type
    file_name=${model_results}${file}
    python3 /data/NJU/datasets/persona/mbti_llms/codes/answer_extractor/en_answer_extractor/extract_answers.py \
        --unlabeled_json_path $file_name --model_name $model_name \
        --write_path $write_path_c --model_path $base_falcon_path \
        --adapter_path $extractor_adapter_path --gpu $GPU \
        --use_custom_base
done
