set -e
GPU=4

for mbti_type in T F J P
do
    python3 /data/NJU/datasets/persona/mbti_llms/codes/rlhf/data_augment/get_ID_reward_samples.py \
        --model_choice chatglm2_6b \
        --device $GPU \
        --mbti $mbti_type
done