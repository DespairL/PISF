import glob
import json
from transformers import AutoModel, AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
import torch, os, argparse
from functools import partial
from datasets import load_dataset
from tqdm import tqdm
from peft import PeftModelForSequenceClassification

classes = ['1', '2', '3', '4', '5']
label2id, id2label = dict(), dict()
for i, label in enumerate(classes):
    label2id[label] = i
    id2label[i] = label

def load_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side="left")
    # Falcon use the last token to predict. And the code is only compatitable with right-padding.
    tokenizer.padding_side = 'right'
    tokenizer.truncation_side = 'right'
    tokenizer.pad_token_id = tokenizer.eos_token_id
    config = AutoConfig.from_pretrained(args.model_path, use_cache=False, torch_dtype=torch.bfloat16, num_labels=len(classes),
                                        id2label=id2label, label2id=label2id, problem_type='single_label_classification',
                                        pad_token_id=tokenizer.eos_token_id)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path, config=config, trust_remote_code=True, device_map=args.device)
    model = PeftModelForSequenceClassification.from_pretrained(model, args.adapter_path, is_trainable=False)
    model.eval()
    return model, tokenizer

def build_prompt_answer(examples):
    ret = {
        'prompt':[],
        'answer':[],
        'answer_pair':[],
    }
    choice = 'answer' if 'answer' in examples.keys() else 'prediction'
    for i in range(len(examples[choice])):
        instructions = f"""You are a professional answer extractor.
Now, you need to extract the numerical value predicted by the model.
We have provided [Test Question Information], [Model Prediction], [The Expected Meaning of the Number] and [Additional Notes] for you.
Please write down your results after [Answer].

[Test Question Information]
{examples['question/statement'][i]}
{examples['choice'][i]}

[Model Prediction]
{examples['prediction'][i]}

[The Expected Meaning of the Number]
1 indicates strongly agree with option A, 2 indicates agree with option A, 3 indicates neutralilty or the model hasn't provided a clear answer, 4 indicates agree with option B, 5 indicates strongly agree with option B.

[Additional Notes]
1.If each option has a corresponding different rating, choose 2(when rating of option A is larger) or 4(when rating of option B is larger) based on the numerical order.
2.If there's a conflict between the semantic meaning and the numerical value, please prioritize the semantic meaning.

[Answer]
"""

        ret['prompt'].append(instructions)
        try:
            ret['answer_pair'].append(examples['map_dict'][i])
        except:
            ret['answer_pair'].append(examples['answer_pair'][i])
        ret['answer'].append(None)
    return ret

def read_samples_to_predict(unlabeled_path):
    samples = json.load(open(unlabeled_path, 'r', encoding='UTF-8'))
    for i in range(len(samples)):
        samples[i] = build_prompt_answer(samples[i])
    return samples

def calculate_mbti_rate(analyse_dict, sum_score):
    EI_rate = f"{analyse_dict['E'] / sum_score['E/I']:.2%} E, {analyse_dict['I'] / sum_score['E/I']:.2%} I"
    SN_rate = f"{analyse_dict['S'] / sum_score['S/N']:.2%} S, {analyse_dict['N'] / sum_score['S/N']:.2%} N"
    TF_rate = f"{analyse_dict['T'] / sum_score['T/F']:.2%} T, {analyse_dict['F'] / sum_score['T/F']:.2%} F"
    JP_rate = f"{analyse_dict['J'] / sum_score['J/P']:.2%} J, {analyse_dict['P'] / sum_score['J/P']:.2%} P"
    print_str = f'mbti test result:\n' + f'{EI_rate}\n' + f'{SN_rate}\n' + f'{TF_rate}\n' + f'{JP_rate}\n'
    return print_str

def judge_mbti_type(analyse_dict):
    mbti_type = ""
    for trait in 'IE', 'SN', 'TF', 'JP':
        if analyse_dict[trait[0]] == analyse_dict[trait[1]]:
            mbti_type += trait
            continue
        traits = list(trait)
        selected_trait = traits[1] if analyse_dict[traits[0]] < analyse_dict[traits[1]] else traits[0]
        mbti_type += selected_trait
    return mbti_type

def unified_analyse_mbti(evaluate_json, model, write_path=None, write=True):
    analyse_dict = {
        'E': 0, 'I': 0, 'S': 0, 'N': 0, 'T': 0, 'F': 0, 'J': 0, 'P': 0,
    }
    map_dict = {
        'E': 'E/I', 'I': 'E/I', 'S': 'S/N', 'N': 'S/N', 'T': 'T/F', 'F': 'T/F', 'J': 'J/P', 'P': 'J/P',
    }
    right_score_map = {
        '1': 5, '2': 4, '3': 3, '4': 4, '5': 5,
    }
    sum_score = {
        'E/I': 0, 'S/N': 0, 'T/F': 0, 'J/P': 0,
    }
    predict_sequence = []
    for each_prediction in evaluate_json:
        gold = each_prediction['answer']
        predict_sequence.append(gold)
        # likert-type score
        score = right_score_map[gold]
        if gold == '3':
            # we need to add 3 for each dichotomize
            for each in each_prediction['answer_pair']:
                sum_score[map_dict[each]] += score
                analyse_dict[each] += score
        else:
            add = each_prediction['answer_pair'][0] if gold in ['1', '2'] else each_prediction['answer_pair'][1]
            sum_score[map_dict[add]] += score
            analyse_dict[add] +=  score
    result = calculate_mbti_rate(analyse_dict, sum_score)
    if not write:
        print(f'Model Prediction:\n')
        print(predict_sequence)
        print(result)
        return
    print(f'Evaluation results are written into {write_path}.')
    with open(write_path, 'a', encoding='UTF-8') as file:
        file.write(f'===================================================\n')
        file.write(f'Model:{model}\n')
        file.write(f'Evaluate by answer extractor v1.\n')
        file.write(result)
        file.write('\n')
        file.write(f'{analyse_dict}\n')
        file.write(f'{analyse_dict.values()}\n')
        file.write(f'Model Prediction Seq:{predict_sequence}\n')
        file.write(f'Result:{judge_mbti_type(analyse_dict)}\n')
        file.write(f'===================================================\n')

def reuse_answer_label_for_improve(args, evaluate_json):
    save_path = '/data/NJU/datasets/persona/mbti_llms/answer_to_label/'
    files = glob.glob(f'{save_path}*.json')
    files_number = len(files)
    save_path += f'{files_number+1}.json'
    extractor_labeled = json.load(open(args.unlabeled_json_path, 'r', encoding='UTF-8'))
    for i in range(len(evaluate_json)):
        extractor_labeled[i]['extractor_label'] = evaluate_json[i]['answer']
    json.dump(extractor_labeled,
              open(save_path, 'w', encoding='UTF-8'),
              indent=4,
              ensure_ascii=False)

def extract_answer(args):
    model, tokenizer = load_model(args)

    evaluate_dataset = load_dataset('json', data_files=args.unlabeled_json_path)['train']
    evaluate_dataset = evaluate_dataset.map(build_prompt_answer, batched=True,
                                            remove_columns=evaluate_dataset.column_names)
    evaluate_dataset = evaluate_dataset.map(lambda e: tokenizer(e['prompt'], padding=True, max_length=2048,
                                                                truncation=True), batched=True)
    evaluate_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'], device=args.device)
    dataloader = torch.utils.data.DataLoader(evaluate_dataset, batch_size=8)
    response_list = []
    for batch_data in tqdm(dataloader, total=len(dataloader)):
        logits = model(**batch_data).logits
        preds = torch.argmax(logits, dim=-1).tolist()
        preds = list(map(lambda x:id2label[x], preds))
        response_list.extend(preds)
    return response_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--unlabeled_json_path', required=True)
    parser.add_argument('--model_name', required=True)
    parser.add_argument('--write_path', required=True)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--adapter_path', required=True)
    parser.add_argument('--analyse_only', action='store_true', required=False)
    parser.add_argument('--analyse_file', required=False)
    parser.add_argument('--use_custom_base', action='store_true', required=False)
    parser.add_argument('--gpu', default=2, type=int,)
    args = parser.parse_args()
    args.device = f'cuda:{args.gpu}'
    base = "/data/NJU/datasets/persona/mbti_llms/temp_result"

    response_list = None
    if not args.analyse_only:
        args.unlabeled_json_path = os.path.join(base, args.unlabeled_json_path) if not args.use_custom_base else args.unlabeled_json_path
        response_list = extract_answer(args)
    else:
        assert args.analyse_file
        temp = json.load(open(args.analyse_file, 'r', encoding='UTF-8'))
        response_list = [x['extractor_label'] for x in temp]

    evaluate_dataset = load_dataset('json', data_files=args.unlabeled_json_path)['train']

    def add_answers_glm_pro(examples, response_list):
        ret = {
            'answer': [],
            'prediction': [],
            'answer_pair': [],
        }
        choice = 'answer' if 'answer' in examples.keys() else 'prediction'
        for i in range(len(examples[choice])):
            ret['answer'].append(response_list[i])
            ret['prediction'].append(examples['prediction'][i])
            try:
                ret['answer_pair'].append(examples['map_dict'][i])
            except:
                ret['answer_pair'].append(examples['answer_pair'][i])
        return ret

    evaluate_dataset = evaluate_dataset.map(partial(add_answers_glm_pro, response_list=response_list),
                                            batched=True, remove_columns=evaluate_dataset.column_names)
    evaluate_json = [x for x in evaluate_dataset]
    # reuse_answer_label_for_improve(args, evaluate_json)
    unified_analyse_mbti(evaluate_json, args.model_name, write_path=args.write_path)