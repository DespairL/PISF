from glob import glob
import json
import os
import random
from itertools import combinations, product

random.seed(2024)

dirty_response_path = "en_instructions_dirty"
file_match_pattern = '*_*_qa_3.5.json'
files = glob(os.path.join(dirty_response_path, file_match_pattern))

dataset_split = {}
random_sample = 2500
sft_path='sft_dataset'
functions = [['E', 'I'], ['S', 'N'], ['T', 'F'], ['J', 'P']]
d_functions = ['E', 'I', 'S', 'N', 'T', 'F', 'J', 'P']
combinations = list(product(*functions))
combinations = [''.join(x) for x in combinations]

# sft

# use alpaca template
instruction_template = """Below is an instruction that describes a task.
Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:"""

"""
# sft dataset - eight function
def sft_processing(file_path):
    def add_instruction(x):
        questions = x['question'].strip()
        prompt = instruction_template.format(instruction=questions)
        x['question'] = prompt
        return x

    dichotomy = file_path.split('/')[-1].split('_')[:2]
    if len(dichotomy[1]) == 1:
        data = json.load(open(file_path, 'r', encoding='UTF-8'))
        data = list(map(add_instruction, data))
        dataset_split[dichotomy[1]] = random.sample(data, random_sample)

for each_file in files:
    sft_processing(each_file)

for each_combination in combinations:
    total_data = []
    for data_piece in each_combination:
        total_data.extend(dataset_split[data_piece])
    dataset_split[each_combination] = total_data

for key, value in dataset_split.items():
    print(f'{key} data samples: {len(value)}')
    json.dump(value, open(os.path.join(sft_path, f'{key}.json'), 'w', encoding='UTF-8'),
            indent=4, ensure_ascii=False)
"""

# rl dataset

# reward - (chosen, rejceted) + (chosen, neutral)
# 5000 samples from GPT3.5
# more samples are added in persona/mbti_llms/codes/rlhf/data_augment
"""
reward_path = 'rl_dataset/reward'

group = {}
for each_file in files:
    dichotomy = each_file.split('/')[-1].split('_')[:2]
    if dichotomy[0] not in group.keys():
        group[dichotomy[0]] = [each_file]
    else:
        group[dichotomy[0]].extend([each_file])

for each_group in group.keys():
    dics = [x.split('/')[-1].split('_')[:2] for x in group[each_group]]
    neutral = None
    chosen = None
    rejected = None
    cur = None
    for i in range(3):
        if dics[i][0] == dics[i][1]:
            neutral = json.load(open(group[each_group][i], 'r', encoding='UTF-8'))
        else:
            if not chosen:
                chosen = json.load(open(group[each_group][i], 'r', encoding='UTF-8'))
                cur = dics[i][1]
            else:
                rejected = json.load(open(group[each_group][i], 'r', encoding='UTF-8'))
    inxs = list(range(len(chosen)))
    inxs = random.sample(inxs, random_sample)
    assert neutral[inxs[0]]['question'] == chosen[inxs[0]]['question']
    assert rejected[inxs[0]]['question'] == chosen[inxs[0]]['question']
    data_for_cur = []
    data_for_opposite = []
    for i in range(len(inxs)):
        cur_sample1 = {
            'question': chosen[inxs[i]]['question'],
            'chosen' : chosen[inxs[i]]['answer'],
            'rejected' : rejected[inxs[i]]['answer']
        }
        cur_sample2 = {
            'question': chosen[inxs[i]]['question'],
            'chosen' : chosen[inxs[i]]['answer'],
            'rejected' : neutral[inxs[i]]['answer']
        }
        data_for_cur.append(cur_sample1)
        data_for_cur.append(cur_sample2)
        cur_sample1 = {
            'question': chosen[inxs[i]]['question'],
            'chosen' : rejected[inxs[i]]['answer'],
            'rejected' : chosen[inxs[i]]['answer']
        }
        cur_sample2 = {
            'question': chosen[inxs[i]]['question'],
            'chosen' : rejected[inxs[i]]['answer'],
            'rejected' : neutral[inxs[i]]['answer']
        }
        data_for_opposite.append(cur_sample1)
        data_for_opposite.append(cur_sample2)
    dataset_split[cur] = data_for_cur
    dataset_split[dics[0][0].replace(cur, "")] = data_for_opposite
    print(f'reward samples:{len(data_for_cur)}')
    print(f'reward samples:{len(data_for_opposite)}')
    json.dump(data_for_cur, open(os.path.join(reward_path, f'{cur}.json'), 'w', encoding='UTF-8'),
            indent=4, ensure_ascii=False)
    json.dump(data_for_opposite, open(os.path.join(reward_path, f'{dics[0][0].replace(cur, "")}.json'), 'w', encoding='UTF-8'),
            indent=4, ensure_ascii=False)

for each_combination in combinations:
    total_data = []
    for data_piece in each_combination:
        total_data.extend(dataset_split[data_piece])
    print(f'reward samples:{len(total_data)}')
    json.dump(total_data, open(os.path.join(reward_path, f'{each_combination}.json'), 'w', encoding='UTF-8'),
            indent=4, ensure_ascii=False)
"""

# ppo dataset - only need questions
# Add chat template while training

reward_path = 'rl_dataset/reward'
ppo_path = 'rl_dataset/ppo'
os.makedirs(ppo_path)
reward_to_ppo = glob(os.path.join(reward_path, '*.json'))
for each_file in reward_to_ppo:
    cur = each_file.split('/')[-1].replace('.json', '')
    cur_question_seq = set()
    data = json.load(open(each_file, 'r', encoding='UTF-8'))
    cur_data = []
    for each_sample in data:
        if each_sample['question'] not in cur_question_seq:
            cur_sample = {
                'question': each_sample['question'],
            }
            cur_data.append(cur_sample)
            cur_question_seq.add(each_sample['question'])
    json.dump(cur_data, open(os.path.join(ppo_path, f'{cur}.json'), 'w', encoding='UTF-8'),
            indent=4, ensure_ascii=False)

