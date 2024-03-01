set -e
model="gpt-3.5-turbo-1106"

for choice in 'JP' # 'EI' 'SN' 'TF'
do

    #for ((i=0; i<${#choice}; i++));
    #do
    #    play="${choice:$i:1}"
    #    python en_questions_response.py \
    #        --task_log_file ./response_log.log --request_batch_size 20 --type $choice --play_trait $play \
    #        --model $model
    #done
    python en_questions_response.py \
        --task_log_file ./response_log.log --request_batch_size 20 --type $choice --play_trait $choice \
        --model $model
done