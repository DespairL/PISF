from datasets import load_from_disk, concatenate_datasets
import os

base_path = "persona/mbti_llms/train_datasets/pretrain_datasets/en_pretrain_datasets"

# to get trait data
for mbti_type in ['E', 'I', 'S', 'N', 'T', 'F', 'J', 'P']:
    save_path = os.path.join(base_path, mbti_type)
    data = load_from_disk(save_path)
    personality = [
        'ISFJ', 'ISFP', 'ISTJ', 'ISTP', 'ESFJ', 'ESFP', 'ESTJ', 'ESTP',
        'INFJ', 'INFP', 'INTJ', 'INTP', 'ENFJ', 'ENFP', 'ENTJ', 'ENTP',
    ]
    data = []
    for each_personality in personality:
        if mbti_type in each_personality:
            cur_path = os.path.join(base_path, each_personality)
            cur_data = load_from_disk(cur_path)
            data.append(cur_data)
    data = concatenate_datasets(data)
    save_path = os.path.join(base_path, mbti_type)
    data.save_to_disk(os.path.join(base_path, mbti_type))

