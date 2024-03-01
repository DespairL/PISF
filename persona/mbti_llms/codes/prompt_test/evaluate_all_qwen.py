import os
import sys
import time
import requests
import json
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
import torch
from functools import partial
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from qwen_generation_utils import make_context
from transformers import GenerationConfig

import datasets
datasets.disable_caching()

def batch_unified_evaluate(args, config, model, tokenizer):

    def qwen_build_prompts(examples, tokenizer, model):
        ret = {'prompt':[]}
        for i in range(len(examples['prompt'])):
            temp, _ = make_context(
                tokenizer,
                examples['prompt'][i],
                system="You are a helpful assistant.",
                max_window_size=model.generation_config.max_window_size,
                chat_format=model.generation_config.chat_format,
            )
            ret['prompt'].append(temp)
        return ret

    def grep_right_answer_qwen(decode_output):
        return decode_output.split('assistant\n')[-1].strip()

    assert type(config) == dict
    mbti_test_path = config['test_file']
    evaluate_dataset = load_dataset('json', data_files=mbti_test_path)['train']
    evaluate_dataset = evaluate_dataset.map(
        partial(qwen_build_prompts, tokenizer=tokenizer, model=model),
        batched=True,
        load_from_cache_file=False,
    )
    evaluate_dataset = evaluate_dataset.map(
        lambda x: tokenizer(x['prompt'], padding=True),
        batched=True,
        load_from_cache_file=False,
    )
    evaluate_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'token_type_ids'], device=model.device)
    dataloader = torch.utils.data.DataLoader(evaluate_dataset, batch_size=8)
    response_list = []
    for batch_data in tqdm(dataloader, total=len(dataloader)):
        batch_out_ids = model.generate(**batch_data,
                            return_dict_in_generate=False,
                            generation_config=model.generation_config)
        batch_response = tokenizer.batch_decode(batch_out_ids, skip_special_tokens=True)
        batch_response = list(map(grep_right_answer_qwen, batch_response))
        response_list.extend(batch_response)

    response_json = json.load(open(mbti_test_path, 'r', encoding='UTF-8'))
    traindata_answer_extractor = []
    for i in tqdm(range(len(response_json))):
        response_json[i]['response'] = response_list[i]
        traindata_answer_extractor.append(
            {
                'question/statement': response_json[i]['question/statement'],
                'prediction': response_list[i],
                'prompt': response_json[i]['prompt'],
                'choice': response_json[i]['choice'],
                'map_dict': response_json[i]['answer_pair'],
                'answer': None,
            }
        )
    os.makedirs(os.path.dirname(config['answer_extractor_output']), exist_ok=True)
    json.dump(traindata_answer_extractor, open(config['answer_extractor_output'], 'w', encoding='UTF-8'),
            indent=4, ensure_ascii=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_specific_file', required=False)
    parser.add_argument('--model')
    parser.add_argument('--mbti_type')
    parser.add_argument('--answer_output', default='')
    args = parser.parse_args()
    import os
    args.device = f"cuda:{os.environ.get('LOCAL_RANK', 0)}"

    tokenizer = AutoTokenizer.from_pretrained(args.model, pad_token='<|extra_0|>',
        eos_token='<|endoftext|>', padding_side='left', trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True,
        pad_token_id=tokenizer.pad_token_id, device_map=args.device).bfloat16()
    model.eval()
    model.generation_config = GenerationConfig.from_pretrained(args.model, pad_token_id=tokenizer.pad_token_id)

    evaluate_config = {
        'test_file': args.test_specific_file,
        'answer_extractor_output': args.answer_output,
    }
    batch_unified_evaluate(args, evaluate_config, model, tokenizer)