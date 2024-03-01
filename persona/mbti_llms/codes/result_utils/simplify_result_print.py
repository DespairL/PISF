import glob, json
import re
import pandas as pd
import copy
from functools import partial
import argparse
import pathlib

def map_model_name(model_desc):
    return model_desc.replace('Model:', '').strip()

def map_dict(dict_values):
    keys = ['E', 'I', 'S', 'N', 'T', 'F', 'J', 'P']
    match = re.search(r'\[.*\]', dict_values)
    if match:
        list_str = match.group()
        # 从提取的字符串中获取列表
        value_list = eval(list_str)
        # 将列表转换为字典
        result_dict = dict(zip(keys, value_list))
    else:
        raise ValueError("Invalid input string.")
    return result_dict

def compute_trait_rate_on_same_dim(raw_data_dict):
    raw_data_copy = copy.deepcopy(raw_data_dict)
    def _compute(obj, opp):
        return raw_data_copy[obj] / (raw_data_copy[obj] + raw_data_copy[opp])
    opp_map = {
        'E': 'I', 'I': 'E', 'S': 'N', 'N': 'S', 'T': 'F', 'F': 'T', 'J': 'P', 'P': 'J',
    }
    temp = {}
    for each_key in raw_data_dict.keys():
        temp[each_key] = _compute(each_key, opp_map[each_key])
    return temp

def turn_to_csv(dataframe, save_path, lens=4):
    def filter_condition(x, lens=lens):
        if 'chatglm2-6b' in x:
            parts = x.split('-')[2]
            return len(parts) != lens
        parts = x.split('-')
        return len(parts[1]) != lens

    dim_df = dataframe[dataframe['model'].apply(filter_condition)]
    dim_df.reset_index(drop=True, inplace=True)
    row_to_move = dim_df.loc[len(dim_df)-1].copy()
    dim_df = pd.concat([row_to_move.to_frame().T, dim_df.drop(len(dim_df)-1)], axis=0)
    dim_df.reset_index(drop=True, inplace=True)
    if lens == 4:
        # TODO:找到排序traits结果的方法
        pass
    dim_df.to_csv(save_path, index=False)

def save_performance_csv(transform):
    pattern = r"prompt_(\w+)_v1\.txt"
    match_model = re.search(pattern, transform).group(1)
    with open(transform, 'r', encoding='UTF-8') as file:
        texts = file.readlines()
    models = [map_model_name(each_line) for each_line in texts if 'Model:' in each_line]
    results = [map_dict(each_line) for each_line in texts if 'dict_values' in each_line]
    rate = [compute_trait_rate_on_same_dim(x) for x in results]

    save_path = "/data/NJU/datasets/persona/performance/"

    temp = pd.DataFrame.from_dict(results)
    result_df = pd.concat([pd.DataFrame({'model': models}), temp], axis=1)
    result_df = result_df.sort_values(by='model')

    rate_table = pd.concat([pd.DataFrame({'model': models}), pd.DataFrame.from_dict(rate)], axis=1)
    rate_table = rate_table.sort_values(by='model')

    turn_to_csv(rate_table, save_path + f"{match_model}_personality.csv", lens=1)
    turn_to_csv(rate_table, save_path + f"{match_model}_trait.csv", lens=4)
    turn_to_csv(result_df, save_path + f"{match_model}_personality_raw.csv", lens=1)
    turn_to_csv(result_df, save_path + f"{match_model}_trait_raw.csv", lens=4)

def get_all_checkpoint_data_for_continual_pretrain(large_data_path, prefix):
    with open(large_data_path, 'r', encoding='UTF-8') as file:
        texts = file.readlines()
    models = [map_model_name(each_line) for each_line in texts if 'Model:' in each_line]
    results = [map_dict(each_line) for each_line in texts if 'dict_values' in each_line]
    save_path = "/data/NJU/datasets/persona/performance/"
    temp = pd.DataFrame.from_dict(results)
    result_df = pd.concat([pd.DataFrame({'model': models}), temp], axis=1)
    result_df['model'] = result_df['model'].str.extract(r'checkpoint-(.+)').astype(int)
    # 根据提取的数字部分进行排序
    result_mean = result_df.groupby('model').agg(['mean', 'std'])
    result_mean['model_desc'] = result_mean.index.get_level_values(0)
    result_mean = result_mean.sort_values(by='model_desc')
    # result_mean = result_mean.drop('model_desc', axis=1)
    result_mean = result_mean.round(2)
    print(result_mean)

    split_1 = pd.concat([result_mean[(trait, 'mean')] for trait in ['E', 'I', 'S', 'N', 'T', 'F', 'J', 'P']], axis=1)
    split_1 = pd.concat([result_mean['model_desc'], split_1], axis=1)
    split_1.columns = ['model', 'E', 'I', 'S', 'N', 'T', 'F', 'J', 'P']
    #print(split_1)
    split_2 = pd.concat([result_mean[(trait, 'std')] for trait in ['E', 'I', 'S', 'N', 'T', 'F', 'J', 'P']], axis=1)
    split_2 = pd.concat([result_mean['model_desc'], split_2], axis=1)
    split_2.columns = ['model', 'E', 'I', 'S', 'N', 'T', 'F', 'J', 'P']
    #print(split_2)

    pathlib.Path(save_path + f'{prefix}').mkdir(exist_ok=True)

    split_1.to_csv(save_path + f'{prefix}' + f"/{prefix}_mean_personality.csv", index=False)
    split_2.to_csv(save_path + f'{prefix}' + f"/{prefix}_std_personality.csv", index=False)
    result_df.to_csv(save_path + f'{prefix}' + f"/{prefix}_personality_raw.csv", index=False)

def simple_return(data, prefix):
    with open(data, 'r', encoding='UTF-8') as file:
        texts = file.readlines()
    models = [map_model_name(each_line) for each_line in texts if 'Model:' in each_line]
    results = [map_dict(each_line) for each_line in texts if 'dict_values' in each_line]
    save_path = "/data/NJU/datasets/persona/performance/"
    temp = pd.DataFrame.from_dict(results)
    result_df = pd.concat([pd.DataFrame({'model': models}), temp], axis=1)
    result_df['model'] = result_df['model'].str.rsplit('-', n=1).str[-1]

    temp = result_df[result_df['model'].str.len() != 4]
    result_df = result_df[result_df['model'].str.len() == 4].sort_values(by='model')
    result_df = pd.concat([temp, result_df], axis=0)
    result_df = result_df.drop_duplicates()
    print(f'Lenth of all samples:{len(result_df)}. Note that it must be 5*N')
    result_mean = result_df.groupby('model').agg(['mean', 'std'])
    custom_order = ['no', 'E', 'I', 'S', 'N', 'T', 'F', 'J', 'P', 'ENFJ', 'ENFP', 'ENTJ', 'ENTP', 'ESFJ', 'ESFP', 'ESTJ', 'ESTP',
                    'INFJ', 'INFP', 'INTJ', 'INTP', 'ISFJ', 'ISFP', 'ISTJ', 'ISTP']
    # 将 'model' 列转换为 Categorical 类型，并指定顺序
    # result_mean = result_mean
    result_mean['model_desc'] = result_mean.index.get_level_values(0)
    result_mean['model_desc'] = pd.Categorical(result_mean['model_desc'], categories=custom_order, ordered=True)
    result_mean = result_mean.sort_values(by='model_desc')
    result_mean = result_mean.reset_index()
    result_mean = result_mean.round(2)
    split_1 = pd.concat([result_mean[(trait, 'mean')] for trait in ['E', 'I', 'S', 'N', 'T', 'F', 'J', 'P']], axis=1)
    split_1 = pd.concat([result_mean['model_desc'], split_1], axis=1)
    split_1.columns = ['model', 'E', 'I', 'S', 'N', 'T', 'F', 'J', 'P']
    print(split_1)
    split_2 = pd.concat([result_mean[(trait, 'std')] for trait in ['E', 'I', 'S', 'N', 'T', 'F', 'J', 'P']], axis=1)
    split_2 = pd.concat([result_mean['model_desc'], split_2], axis=1)
    split_2.columns = ['model', 'E', 'I', 'S', 'N', 'T', 'F', 'J', 'P']
    print(split_2)

    pathlib.Path(save_path + f'{prefix}').mkdir(exist_ok=True)
    split_1.to_csv(save_path + f'{prefix}' + f"/{prefix}_mean_personality.csv", index=False)
    split_2.to_csv(save_path + f'{prefix}' + f"/{prefix}_std_personality.csv", index=False)
    result_df.to_csv(save_path + f'{prefix}' + f"/{prefix}_personality_raw.csv", index=False)

def get_all_checkpoint_data_for_instruction_tuning(large_data_path, prefix):
    with open(large_data_path, 'r', encoding='UTF-8') as file:
        texts = file.readlines()
    models = [map_model_name(each_line) for each_line in texts if 'Model:' in each_line]
    models = [f'checkpoint-{(i//5+1)*50}' for i in range(len(models))]
    results = [map_dict(each_line) for each_line in texts if 'dict_values' in each_line]
    save_path = "/data/NJU/datasets/persona/performance/"
    temp = pd.DataFrame.from_dict(results)
    result_df = pd.concat([pd.DataFrame({'model': models}), temp], axis=1)
    result_df['model'] = result_df['model'].str.extract(r'checkpoint-(.+)').astype(int)
    # 根据提取的数字部分进行排序
    result_mean = result_df.groupby('model').agg(['mean', 'std'])
    result_mean['model_desc'] = result_mean.index.get_level_values(0)
    result_mean = result_mean.sort_values(by='model_desc')
    # result_mean = result_mean.drop('model_desc', axis=1)
    result_mean = result_mean.round(2)
    print(result_mean)

    split_1 = pd.concat([result_mean[(trait, 'mean')] for trait in ['E', 'I', 'S', 'N', 'T', 'F', 'J', 'P']], axis=1)
    split_1 = pd.concat([result_mean['model_desc'], split_1], axis=1)
    split_1.columns = ['model', 'E', 'I', 'S', 'N', 'T', 'F', 'J', 'P']
    #print(split_1)
    split_2 = pd.concat([result_mean[(trait, 'std')] for trait in ['E', 'I', 'S', 'N', 'T', 'F', 'J', 'P']], axis=1)
    split_2 = pd.concat([result_mean['model_desc'], split_2], axis=1)
    split_2.columns = ['model', 'E', 'I', 'S', 'N', 'T', 'F', 'J', 'P']
    #print(split_2)

    pathlib.Path(save_path + f'{prefix}').mkdir(exist_ok=True)

    split_1.to_csv(save_path + f'{prefix}' + f"/{prefix}_mean_personality.csv", index=False)
    split_2.to_csv(save_path + f'{prefix}' + f"/{prefix}_std_personality.csv", index=False)
    result_df.to_csv(save_path + f'{prefix}' + f"/{prefix}_personality_raw.csv", index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--base', default='temp_result')
    parser.add_argument('--task', default='pretrain', )
    parser.add_argument('--data_volumn', default='small')
    args = parser.parse_args()
    base_path = f"/data/NJU/datasets/persona/mbti_llms/results/{args.base}/"
    if args.data_volumn == 'small':
        simple_return(base_path + f'{args.name}.txt', prefix=f'{args.name}')
    else:
        if args.task == 'pretrain':
            get_all_checkpoint_data_for_continual_pretrain(base_path + f'{args.name}.txt', prefix=f'{args.name}')
        else:
            get_all_checkpoint_data_for_instruction_tuning(base_path + f'{args.name}.txt', prefix=f'{args.name}')
