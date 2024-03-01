from glob import glob
from datasets import load_from_disk

if __name__ == '__main__':
    pretrain_dataset = "train_datasets/mbti/personality/*"
    datasets = glob(pretrain_dataset)
    datasets = [x for x in datasets]

    def further_clean(examples):
        ret = {
            'content': [],
            'personality': [],
        }
        for i in range(len(examples['content'])):
            clean_text = examples['content'][i].replace("&gt;", "").replace("&lt;", "").replace("&amp;", "")
            ret['content'].append(clean_text)
            ret['personality'].append(examples['personality'][i])
        return ret

    for data in datasets:
        suffix = data.split('/')[-1]
        temp_data = load_from_disk(data)
        temp_data = temp_data.map(
            further_clean,
            batched=True,
        )
        temp_data.save_to_disk(f"persona/mbti_llms/train_datasets/pretrain_datasets/en_pretrain_datasets/{suffix}")