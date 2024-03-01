from glob import glob
import os, json

# 清洗chatgpt数据

filter_adjective = ['totally', 'definitely', 'Absolutely! ', 'absolutely']

def json_load(file_path):
    return json.load(open(file_path, 'r', encoding='UTF-8'))

def json_dump(data, file_path):
    json.dump(data, open(file_path, 'w', encoding='UTF-8'), indent=4, ensure_ascii=False)

def normal_replace(str_to_replace, replace_list):
    for each in replace_list:
        str_to_replace = str_to_replace.replace(f'{each}', '')
    return str_to_replace

dirty_gpt_file_path = "persona/data_construction/SFT_RLHF/rl_dataset/reward"
clean_gpt_file_path = "persona/data_construction/SFT_RLHF/rl_dataset/reward_clean"

files = glob(dirty_gpt_file_path + '/*.json')

for cur_file in files:
    mbti = cur_file.split('/')[-1]
    cur_data_path = cur_file
    cur_data = json_load(cur_data_path)
    for i in range(len(cur_data)):
        for key in cur_data[i].keys():
            if key == 'question':
                continue
            cur_data[i][key] = normal_replace(cur_data[i][key], filter_adjective)
            cur_data[i][key] = cur_data[i][key].replace('  ', ' ')
            cur_data[i][key] = cur_data[i][key].strip()
    cur_save_path = os.path.join(clean_gpt_file_path, f'{mbti}')
    json_dump(cur_data, cur_save_path)