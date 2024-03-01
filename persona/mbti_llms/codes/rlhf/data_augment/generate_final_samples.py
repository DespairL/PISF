import os
import json
from itertools import combinations, product

def json_load(file_path):
    return json.load(open(file_path, 'r', encoding='UTF-8'))

def json_dump(data, file_path):
    json.dump(data, open(file_path, 'w', encoding='UTF-8'), indent=4, ensure_ascii=False)

def reward_sample(question, chosen, rejected):
    return {
        'question': question,
        'chosen': chosen,
        'rejected': rejected,
    }

trait = ['E', 'I', "S", "N", 'T', 'F', 'J', 'P']
model_choice_list = ['chatglm2_6b', 'Llama_13b_chat']
clean_or_not = True
template = '{model_choice}.json' if not clean_or_not else '{model_choice}_cleaned.json'

for mbti in trait:
    combine_data = []
    path = "/data/NJU/datasets/persona/mbti_llms/train_datasets/RL_dataset/rl_dataset/reward_clean"
    path = os.path.join(path, f'{mbti}.json')
    original_reward_data = json_load(path)
    for model_choice in model_choice_list:
        positive_path = "/data/NJU/datasets/persona/mbti_llms/codes/rlhf/data_augment/positive"
        neutral_path = "/data/NJU/datasets/persona/mbti_llms/codes/rlhf/data_augment/neutral"
        negative_path = "/data/NJU/datasets/persona/mbti_llms/codes/rlhf/data_augment/negative"
        original_path = "/data/NJU/datasets/persona/mbti_llms/codes/rlhf/data_augment/original"
        positive_path = os.path.join(positive_path, f'{mbti}', template.format(model_choice=model_choice))
        neutral_path = os.path.join(neutral_path, f'{mbti}', template.format(model_choice=model_choice))
        negative_path = os.path.join(negative_path, f'{mbti}', template.format(model_choice=model_choice))
        original_path = os.path.join(original_path, f'{mbti}', template.format(model_choice=model_choice))

        # (positive, original) (positive, neutral) (positive, negative)
        positive_data = json_load(positive_path)
        neutral_data = json_load(neutral_path)
        negative_data = json_load(negative_path)
        original_data = json_load(original_path)

        for i in range(len(positive_data)):
            # question chosen rejected
            question = positive_data[i]['question']
            assert positive_data[i]['question'] == neutral_data[i]['question'] == negative_data[i]['question'] == original_data[i]['question']
            combine_data.append(reward_sample(question, positive_data[i]['response'], original_data[i]['response']))
            combine_data.append(reward_sample(question, positive_data[i]['response'], negative_data[i]['response']))
            combine_data.append(reward_sample(question, positive_data[i]['response'], neutral_data[i]['response']))

    all_data = original_reward_data + combine_data
    assert len(all_data) == 20000
    save_path_base = '/data/NJU/datasets/persona/mbti_llms/train_datasets/RL_dataset/rl_dataset/reward_augment'
    os.makedirs(save_path_base, exist_ok=True)
    json_dump(all_data, os.path.join(save_path_base, f'{mbti}.json'))

# Then we get the personality data
functions = [['E', 'I'], ['S', 'N'], ['T', 'F'], ['J', 'P']]
d_functions = ['E', 'I', 'S', 'N', 'T', 'F', 'J', 'P']
combinations = list(product(*functions))
combinations = [''.join(x) for x in combinations]

reward_augment_path = "/data/NJU/datasets/persona/mbti_llms/train_datasets/RL_dataset/rl_dataset/reward_augment"
for each_personality in combinations:
    reward_augment_path_cur = os.path.join(reward_augment_path, f'{each_personality}.json')
    cur_all_data = []
    for each_trait in list(each_personality):
        cur_read_path = os.path.join(reward_augment_path, f'{each_trait}.json')
        data = json_load(cur_read_path)
        cur_all_data.extend(data)
    json_dump(cur_all_data, reward_augment_path_cur)