from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import deepspeed
import os, json
from glob import glob
from datasets import Dataset
from torch.utils.data import DataLoader
from functools import partial
from tqdm import tqdm

no_path = "/data/NJU/datasets/persona/mbti_llms/evaluate_datasets/en_unified_dataset/multi_slot_evaluate_prompt/no_specific_prompt"
personality_path = "/data/NJU/datasets/persona/mbti_llms/evaluate_datasets/en_unified_dataset/multi_slot_evaluate_prompt/specific_personality_prompt"
trait_path = "/data/NJU/datasets/persona/mbti_llms/evaluate_datasets/en_unified_dataset/multi_slot_evaluate_prompt/specific_trait_prompt"
save_path = "/data/NJU/datasets/persona/mbti_llms/results/debias_results/llama_results/Llama-2-70b_results"
os.makedirs(save_path, exist_ok=True)

def get_all_jsons():
    all_path = [no_path, personality_path, trait_path]
    all_files = []
    for each_path in all_path:
        pattern = each_path + '/*.json'
        files = sorted(glob(pattern))
        all_files.extend(files)
    return all_files

# no -> personality -> trait
def load_all_json_for_llama():
    all_files = get_all_jsons()
    all_content = []
    order = []
    for each_file in all_files:
        content = json.load(open(each_file, 'r', encoding='UTF-8'))
        all_content.extend(content)
        cur_pattern_match = each_file.split('/')[-1]
        order.append(cur_pattern_match)
    return all_content, order



def main():
    all_files = get_all_jsons()
    all_content, order = load_all_json_for_llama()
    dataset = Dataset.from_list(all_content)

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

    dataset = dataset.map(
        partial(build_prompt_wrapper, func=Llama_template),
        batched=True,
        load_from_cache_file=True,
    )

    each_iter = 25
    batch_size = 8
    model_name = '/data/NJU/datasets/persona/models/Llama-2-70b-chat-hf'
    max_new_tokens = 512
    top_p = 1.0
    temperature = 1.0
    repetition_penalty = 1.2

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map='auto'
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=False if model.config.model_type == 'llama' else True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    dataset = dataset.map(
        lambda x: tokenizer(x['prompt'], padding=True),
        batched=True,
        load_from_cache_file=True,
    )
    data_device = "cuda"
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'], device=data_device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    response_list = []
    c = 0
    file_index = 0
    for batch_data in tqdm(dataloader, total=len(dataloader)):
        with torch.no_grad():
            batch_out_ids = model.generate(
                **batch_data, max_new_tokens=max_new_tokens, do_sample=False, top_p=top_p,
                temperature=temperature, repetition_penalty=repetition_penalty, eos_token_id=tokenizer.eos_token_id
            )
        batch_out_ids = batch_out_ids.cpu()
        batch_response = tokenizer.batch_decode(batch_out_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        batch_response = list(map(lambda x:x.split('[/INST]')[-1], batch_response))
        response_list.extend(batch_response)
        c += 1
        if c == each_iter:
            cur_file = order[file_index]
            cur_save_path = os.path.join(save_path, cur_file)
            response_json = json.load(open(all_files[file_index], 'r', encoding='UTF-8'))
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
            json.dump(traindata_answer_extractor, open(cur_save_path, 'w', encoding='UTF-8'),
                indent=4, ensure_ascii=False)
            response_list = []
            c = 0
            file_index += 1
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()