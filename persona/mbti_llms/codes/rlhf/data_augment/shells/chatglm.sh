set -e
GPU=7

for mbti_type in E #I S N
do
    python3 /data/NJU/datasets/persona/mbti_llms/codes/rlhf/data_augment/get_ID_reward_samples.py \
        --model_choice chatglm2_6b \
        --device $GPU \
        --mbti $mbti_type \
        --neutral \
        --negative \
        --original \
        --batch_size 32
done