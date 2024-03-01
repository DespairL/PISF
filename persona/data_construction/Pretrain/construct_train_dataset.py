import os, json, re
import random
import numpy as np
import torch
from datasets import load_dataset, concatenate_datasets, load_from_disk
from functools import partial
from tqdm import tqdm
import glob

# 部分posts包含大量的url，需要对其进行清洗，用统一的 'link' 进行替换。
def clean_posts_function1(examples):
    ret = {
        'content':[],
        'personality':[],
    }
    for i in range(len(examples['posts'])):
        posts = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                       'link', examples['posts'][i])
        posts = re.sub(' +', ' ', posts)
        posts = posts.split('|||')
        new = ''
        for j in range(len(posts)):
            if len(new) < 1600:
                new += posts[j] + '|||'
            else:
                ret['content'].append(new)
                ret['personality'].append(examples['type'][i])
                new = ''
    return ret

# clean twitter data
def clean_posts_function2(examples):
    ret = {
        'content':[],
        'personality':[],
    }
    # clean some username with "@user" for privacy
    for i in range(len(examples['text'])):
        posts = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                       'link', examples['text'][i])
        posts = re.sub('@[a-zA-Z0-9]+', '@user', posts)
        posts = re.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U0001FB00-\U0001FBFF\U0001FC00-\U0001FCFF\U0001FD00-\U0001FDFF\U0001FE00-\U0001FEFF\U0001FF00-\U0001FFFF]+',
                       '', posts)
        posts = re.sub(' +', ' ', posts)
        posts = posts.split('|||')
        new = ''
        for j in range(len(posts)):
            if len(new) < 1600:
                new += posts[j] + '|||'
            else:
                ret['content'].append(new)
                ret['personality'].append(examples['label'][i])
                new = ''
    return ret

def construct_train_data_reddit():
    mbti_list = [
        'ISFJ', 'ISFP', 'ISTJ', 'ISTP', 'ESFJ', 'ESFP', 'ESTJ', 'ESTP',
        'INFJ', 'INFP', 'INTJ', 'INTP', 'ENFJ', 'ENFP', 'ENTJ', 'ENTP',
    ]
    mbti_list.extend([each.lower() for each in mbti_list])
    pattern = r'[IEie][SNsn][FTft][JPjp]'

    def filter_non_mbti(examples):
        ret = {
            'content':[],
            'personality':[],
        }
        for i in range(len(examples['body'])):
            flair_text = examples['flair_text'][i]
            mbti = re.findall(pattern, flair_text)
            mbti = mbti[0].upper()
            try:
                assert mbti in mbti_list
            except:
                print(mbti)
                raise NotImplementedError
            if mbti:
                ret['content'].append(examples['body'][i])
                ret['personality'].append(mbti)
        return ret

    reddit_data = f'train_datasets{os.sep}mbti{os.sep}reddit{os.sep}'
    save_path = f'train_datasets{os.sep}mbti{os.sep}reddit_mbti'
    keys = [
        'ISFJ', 'ISFP', 'ISTJ', 'ISTP', 'ESFJ', 'ESFP', 'ESTJ', 'ESTP',
        'INFJ', 'INFP', 'INTJ', 'INTP', 'ENFJ', 'ENFP', 'ENTJ', 'ENTP',
    ]
    values = [[] for i in range(len(keys))]
    dataset_split = dict(zip(keys, values))

    for i in tqdm(range(18)):
        file_name = f'{i}.csv'
        dataset = load_dataset('csv', data_files=reddit_data + file_name)['train']
        dataset = dataset.map(
            filter_non_mbti,
            batched=True,
            batch_size=8,
            num_proc=2,
            remove_columns=dataset.column_names,
        )
        print(dataset)
        for key in keys:
            certain_type_dataset = dataset.filter(lambda x:x['personality']==key)
            dataset_split[key].append(certain_type_dataset)

    for key, value in dataset_split.items():
        if len(value) == 0:
            continue
        certain_type_dataset = concatenate_datasets(value)
        certain_type_dataset.save_to_disk(save_path + f'{os.sep}{key}{os.sep}')

def construct_rest_dataset():
    train_base_path = f'train_datasets{os.sep}mbti{os.sep}'
    train_dataset2 = load_dataset('csv', data_files=f'{train_base_path}mbti_1.csv')['train']
    train_dataset3 = load_dataset('csv', data_files=f'{train_base_path}twitter_MBTI.csv')['train']
    print(train_dataset2)
    print(train_dataset3)
    train_dataset2 = train_dataset2.map(
        clean_posts_function1,
        batched=True,
        batch_size=8,
        remove_columns=train_dataset2.column_names,
    )
    print(f'清洗数据集二，得到:{len(train_dataset2)}')
    train_dataset3 = train_dataset3.map(
        clean_posts_function2,
        batched=True,
        batch_size=8,
        remove_columns=train_dataset3.column_names,
    )
    print(f'清洗数据集三，得到:{len(train_dataset3)}')
    concat_rest_train_dataset = concatenate_datasets([train_dataset2, train_dataset3])
    print(concat_rest_train_dataset)
    keys = [
        'ISFJ', 'ISFP', 'ISTJ', 'ISTP', 'ESFJ', 'ESFP', 'ESTJ', 'ESTP',
        'INFJ', 'INFP', 'INTJ', 'INTP', 'ENFJ', 'ENFP', 'ENTJ', 'ENTP',
    ]
    save_path = f'train_datasets{os.sep}mbti{os.sep}rest'
    for key in keys:
        certain_type_dataset = concat_rest_train_dataset.filter(lambda x: x['personality'] == key)
        certain_type_dataset.save_to_disk(save_path + f'{os.sep}{key}{os.sep}')


def clean_reddit_dataset(examples):
    ret = {
        'content':[],
        'personality':[],
    }
    batch_sample = ''
    for i in range(len(examples['content'])):
        posts = examples['content'][i]
        if not posts:
            continue
        posts = re.sub(' +', ' ', posts)
        posts = re.sub(r'\ufffd+', '', posts)
        posts = re.sub('\n+', '\n', posts)
        posts = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                       'link', posts)
        posts = re.sub('@[a-zA-Z0-9]+', '@user', posts)
        if len(batch_sample) < 1600:
            if i == len(examples['content'])-1 and len(batch_sample) + len(posts) >= 1600:
                ret['content'].append(batch_sample)
                ret['personality'].append(examples['personality'][i])
                ret['content'].append(posts)
                ret['personality'].append(examples['personality'][i])
            elif i == len(examples['content'])-1:
                batch_sample += posts + '|||'
                ret['content'].append(batch_sample)
                ret['personality'].append(examples['personality'][i])
            else:
                batch_sample += posts + '|||'
        else:
            ret['content'].append(batch_sample)
            ret['personality'].append(examples['personality'][i])
            batch_sample = ''
    return ret

def concat_rest_and_reddit_datasets():
    rest_dataset_path = f'train_datasets{os.sep}mbti{os.sep}rest{os.sep}'
    reddit_mbti_path = f'train_datasets{os.sep}mbti{os.sep}reddit_mbti{os.sep}'
    keys = [
        'ISFJ', 'ISFP', 'ISTJ', 'ISTP', 'ESFJ', 'ESFP', 'ESTJ', 'ESTP',
        'INFJ', 'INFP', 'INTJ', 'INTP', 'ENFJ', 'ENFP', 'ENTJ', 'ENTP',
    ]
    for key in keys:
        save_path = f'train_datasets{os.sep}mbti{os.sep}personality'
        reddit_dataset = load_from_disk(reddit_mbti_path + key)
        rest_dataset = load_from_disk(rest_dataset_path + key)
        reddit_dataset = reddit_dataset.map(
            clean_reddit_dataset,
            batched=True,
            batch_size=8,
            remove_columns=reddit_dataset.column_names,
        )
        final_dataset_split = concatenate_datasets([reddit_dataset, rest_dataset])
        final_dataset_split.save_to_disk(save_path + f'{os.sep}{key}{os.sep}')

def final_decorate():
    def final_clean(examples):
        ret = {
            'content': [],
            'personality': [],
        }
        for i in range(len(examples['content'])):
            ret['content'].append(re.sub(r'\ufffd+', '', examples['content'][i]))
            ret['personality'].append(examples['personality'][i])
        return ret

    save_path = f'train_datasets{os.sep}mbti{os.sep}personality{os.sep}'
    keys = [
        'ISFJ', 'ISFP', 'ISTJ', 'ISTP', 'ESFJ', 'ESFP', 'ESTJ', 'ESTP',
        'INFJ', 'INFP', 'INTJ', 'INTP', 'ENFJ', 'ENFP', 'ENTJ', 'ENTP',
    ]
    values = [0 for i in range(len(keys))]
    dataset_split = dict(zip(keys, values))
    np.random.seed(2023)
    train_save_path = f'train_datasets{os.sep}mbti{os.sep}train{os.sep}'

    for key in keys:
        concat_dataset = load_from_disk(save_path + key + f'{os.sep}')
        assert concat_dataset['personality'][random.randint(0, len(concat_dataset['personality']))] == key
        concat_dataset = concat_dataset.map(
            final_clean,
            batched=True,
            batch_size=8,
            remove_columns=concat_dataset.column_names,
        )
        #final_sample_indices = np.random.randint(0, len(concat_dataset['personality']), size=10000)
        #final_dataset_split = concat_dataset.select(final_sample_indices)
        dataset_split[key] = len(concat_dataset['personality'])
        concat_dataset.save_to_disk(train_save_path + key + f'{os.sep}')

    print(f'抽样前数据分布:')
    print(dataset_split)

def check_dataset_split_pieces(data_path):
    dics = glob.glob(data_path + '*/')
    for each_dic in dics:
        temp_dataset = load_from_disk(each_dic)
        print(temp_dataset)

if __name__ == '__main__':
    # reddit datasets is much larger, so handle separately
    # construct_train_data_reddit()
    # construct_rest_dataset()
    # concat_rest_and_reddit_datasets()
    # final_decorate()

    # final_result = {'ISFJ': 40566, 'ISFP': 29639, 'ISTJ': 39615, 'ISTP': 183605, 'ESFJ': 13519, 'ESFP': 11823, 'ESTJ': 19786, 'ESTP': 38235, 'INFJ': 865864, 'INFP': 915834, 'INTJ': 562328, 'INTP': 904701, 'ENFJ': 94608, 'ENFP': 155492, 'ENTJ': 197089, 'ENTP': 694158}
    # print(final_result.keys())
    # print(final_result.values())

    data_split_path = 'train_datasets/mbti/personality/'
    check_dataset_split_pieces(data_split_path)