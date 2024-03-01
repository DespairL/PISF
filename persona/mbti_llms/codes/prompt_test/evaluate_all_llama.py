import os
import sys
import time
import requests
import json
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig
import torch
from functools import partial
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from qwen_generation_utils import make_context
from transformers import GenerationConfig

import datasets
datasets.disable_caching()

from transformers.generation.utils import LogitsProcessorList
from transformers.generation.logits_process import LogitsProcessor

def unified_generation_config(args):
    class InvalidScoreLogitsProcessor(LogitsProcessor):
        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
            if torch.isnan(scores).any() or torch.isinf(scores).any():
                scores.zero_()
                scores[..., 5] = 5e4
            return scores

    logits_processor = LogitsProcessorList()
    logits_processor.append(InvalidScoreLogitsProcessor())
    # We Use Greedy Decode. Set default temperature and top_p to avoid warning.
    gen_kwargs = {"max_length": 2048, "num_beams": 1, "do_sample": False, "logits_processor": logits_processor,
                  "temperature": 1.0, "top_p": 1.0,}
    return gen_kwargs

def batch_unified_evaluate(args, config, model, tokenizer):
    gen_kwargs = unified_generation_config(args)

    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

    def Llama_template(prompt):
        # <bos>[INST] B_SYS SystemPrompt E_SYS Prompt [/INST]
        return f"""<s>{B_INST} {B_SYS} You are a chatbot. {E_SYS} {prompt} {E_INST}"""

    def build_prompt_wrapper(examples, func):
        ret = {'prompt':[]}
        for i in range(len(examples['prompt'])):
            ret['prompt'].append(func(examples['prompt'][i]))
        return ret

    mbti_test_path = config['test_file']
    evaluate_dataset = load_dataset('json', data_files=mbti_test_path)['train']
    evaluate_dataset = evaluate_dataset.map(
        partial(build_prompt_wrapper, func=Llama_template),
        batched=True,
        load_from_cache_file=False,
    )
    evaluate_dataset = evaluate_dataset.map(
        lambda x: tokenizer(x['prompt'], padding=True),
        batched=True,
        load_from_cache_file=False,
    )
    data_device = "cuda" if args.multi_gpu else model.device
    evaluate_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'], device=data_device)
    dataloader = torch.utils.data.DataLoader(evaluate_dataset, batch_size=8)

    response_list = []
    for batch_data in tqdm(dataloader, total=len(dataloader)):
        batch_out_ids = model.generate(**batch_data, **gen_kwargs)
        batch_response = tokenizer.batch_decode(batch_out_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        batch_response = list(map(lambda x:x.split('[/INST]')[-1], batch_response))
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
    parser.add_argument('--multi_gpu', action='store_true')
    args = parser.parse_args()
    import os
    if not args.multi_gpu:
        args.device = f"cuda:{os.environ.get('LOCAL_RANK', 0)}"
    else:
        args.device = "auto"

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.padding_side = 'left'
    tokenizer.truncation_side = 'right'
    tokenizer.pad_token_id = tokenizer.eos_token_id
    config = AutoConfig.from_pretrained(args.model, torch_dtype=torch.bfloat16, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, config=config, trust_remote_code=True, device_map=args.device).bfloat16()
    model.eval()

    evaluate_config = {
        'test_file': args.test_specific_file,
        'answer_extractor_output': args.answer_output,
    }
    batch_unified_evaluate(args, evaluate_config, model, tokenizer)