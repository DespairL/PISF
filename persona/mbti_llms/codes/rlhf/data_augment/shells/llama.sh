set -e
GPU=1

for mbti_type in E I S N
do
    python3 /data/NJU/datasets/persona/mbti_llms/codes/rlhf/data_augment/get_ID_reward_samples.py \
        --model_choice Llama_13b_chat \
        --device $GPU \
        --mbti $mbti_type
done