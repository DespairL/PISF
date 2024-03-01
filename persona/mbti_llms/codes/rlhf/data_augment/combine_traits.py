from glob import glob
import json
import os
from itertools import combinations, product

def json_load(file_path):
    return json.load(open(file_path, 'r', encoding='UTF-8'))

def json_dump(data, file_path):
    json.dump(data, open(file_path, 'w', encoding='UTF-8'), indent=4, ensure_ascii=False)

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