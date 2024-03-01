from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoModelForCausalLM
import torch
import os
from datasets import load_dataset
from functools import partial
from tqdm import tqdm
import json
from transformers.generation.utils import LogitsProcessorList
from transformers.generation.logits_process import LogitsProcessor
from prompt_induce import get_positive_template, get_neutral_template, get_negative_template
import argparse
import sys
sys.path.append('/data/NJU/datasets/persona/mbti_llms/codes/rlhf')
from llama_dschat.utils.data.raw_datasets import induction_system_prompt
import pdb

def induction_system_prompt_sp(mbti, chatglm=False, force_sim=False, force_oppo=False):
    en_persona_description = {
        'E': ["Extroverted"],
        'I': ["Introverted"],
        'S': ["Sensing"],
        'N': ["Intuition"],
        'T': ["Thinking"],
        'F': ["Feeling"],
        'J': ["Judging"],
        'P': ["Perceiving"],
    }

    opposite_mapping = {
        'E':'I', 'I':'E', 'S':'N', 'N':'S',
        'T':'F', 'F':'T', 'J':'P', 'P':'J',
    }

    character_perfix = "Key characteristics:\n"
    E = """Extroverted: Directs energy outward. Gains energy from interaction."""
    I = """Introverted: Directs energy inward. Loses energy from interaction."""
    S = """Sensing: Focussed on information from 5 senses"""
    N = """Intuition: Focussed on patterns and relationships."""
    T = """Thinking: Logic & Analysis. Objectivity & Detachment."""
    F = """Feeling: Personal & Social values. Understanding & Harmony."""
    J = """Judging: Decisiveness. Seeks Closure. Focus on decision."""
    P = """Perceiving: Flexibility. Open Options. Focus on process"""

    # For small model, we need more information to inform the model.
    character_mapping = {
        'E': E, 'I': I, 'S': S, 'N': N, 'T': T, 'F': F, 'J': J, 'P': P,
    }

    def map_personality_to_prompt(personality):
        p_list = list(personality)
        sim_desc = [en_persona_description[p][0] for p in p_list]
        sim_desc = ' '.join(sim_desc)
        sim_desc += f'({personality})'
        return sim_desc

    def get_character(personality):
        p_list = list(personality)
        sim_desc = [character_mapping[p] for p in p_list]
        sim_desc = '\n'.join(sim_desc)
        return sim_desc

    def pair_characters(personality, opposite_personality):
        c1 = get_character(personality)
        c2 = get_character(opposite_personality)
        return character_perfix + c1 + '\n' + c2

    # pdb.set_trace()

    opposite_personality = ''.join([opposite_mapping[x] for x in list(mbti)])

    characters_information = pair_characters(mbti, opposite_personality)
    sim_desc = map_personality_to_prompt(mbti)
    oppo_sim_desc = map_personality_to_prompt(opposite_personality)
    # Llama-2-chat is too Perceiving and Extroverted to learn what is "Perceiving" and what is "Extroverted" when we add this extra prompt.
    extra_mention = " and display one of them. You don't have to mention this in the chat." if mbti not in ['P', 'E'] else " ."
    Trait_pair_mension = f"{sim_desc} or {oppo_sim_desc}"
    learning_system_prompt = f"You are a chatbot. You are learning to switch your personality trait to {Trait_pair_mension}{extra_mention} "
    if chatglm:
        # The English comprehension ability of chatglm is significantly weaker than that of Llama.
        # It requires longer system prompts for clarification and restriction.
        # Otherwise, it will be caught in the pursuit of the right reward direction.
        # Personality Infomation is provided as follows.\n\n### Personality Trait Character
        extra_mention = f"""Personality Infomation is provided as follows.

### Personality Trait Character:
{characters_information}

### Important Notes:
1. You "don't have to mention you're learning" in the chat!
2. You are "no need to explain the reasons" for choosing to exhibit a certain trait!
3. In every conversation, your responses need to showcase "one and only one" distinct trait!
4. DO NOT MENTION {Trait_pair_mension}!

### Chat:"""
        # NOTE: I is trained without ### Important Notes: and ### Chat:
        learning_system_prompt = f"""Remember that at below you are not a AI model. You are a new chatbot. And you are learning to switch your personality trait to {Trait_pair_mension}.\n\n"""
        if force_sim:
            learning_system_prompt = f"""Remember that at below you are not a AI model. You are a new chatbot. And you are learning to switch your personality trait to {sim_desc}.\n\n"""
        if force_oppo:
            learning_system_prompt = f"""Remember that at below you are not a AI model. You are a new chatbot. And you are learning to switch your personality trait to {oppo_sim_desc}.\n\n"""
    # pdb.set_trace()
    return learning_system_prompt

parser = argparse.ArgumentParser()
parser.add_argument('--model_choice', required=True, type=str, choices=['Llama_13b_chat', 'chatglm2_6b'])
parser.add_argument('--device', required=True, type=str)
parser.add_argument('--mbti', required=True)
parser.add_argument('--test', default=False, action='store_true')
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--positive', action="store_false", default=True)
parser.add_argument('--negative', action="store_false", default=True)
parser.add_argument('--neutral', action="store_false", default=True)
parser.add_argument('--original', action="store_false", default=True)
args = parser.parse_args()

# pdb.set_trace()

device = f'cuda:{args.device}'
model_choice = f'{args.model_choice}'
genearte_original = args.original
genearte_positive = args.positive
genearte_neutral = args.neutral
genearte_negative = args.negative
test_generate = args.test

print(args)

Llama_path = "/data/NJU/datasets/persona/models/Llama-2-13b-chat-hf"
Glm_path = "/data/NJU/datasets/persona/models/chatglm2-6b"
positive_path = "/data/NJU/datasets/persona/mbti_llms/codes/rlhf/data_augment/positive"
neutral_path = "/data/NJU/datasets/persona/mbti_llms/codes/rlhf/data_augment/neutral"
negative_path = "/data/NJU/datasets/persona/mbti_llms/codes/rlhf/data_augment/negative"
original_path = "/data/NJU/datasets/persona/mbti_llms/codes/rlhf/data_augment/original"

path = "/data/NJU/datasets/persona/mbti_llms/train_datasets/RL_dataset/rl_dataset/ppo"
mbti = args.mbti
path = os.path.join(path, f'{mbti}.json')

os.makedirs(os.path.join(positive_path, mbti), exist_ok=True)
os.makedirs(os.path.join(neutral_path, mbti), exist_ok=True)
os.makedirs(os.path.join(negative_path, mbti), exist_ok=True)
os.makedirs(os.path.join(original_path, mbti), exist_ok=True)

questions = json.load(open(path, 'r', encoding='UTF-8'))

def unified_build_prompt(examples, tokenizer, model_base, original=False, force_sim=False, force_oppo=False):

    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

    def Llama_template(prompt):
        # <bos>[INST] B_SYS SystemPrompt E_SYS Prompt [/INST]
        return f"""<s>{B_INST} {B_SYS} You are a chatbot. {E_SYS} {prompt} {E_INST}"""

    def mpt_template(prompt):
        INSTRUCTION_KEY = "### Instruction:"
        RESPONSE_KEY = "### Response:"
        INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
        PROMPT_FOR_GENERATION_FORMAT = """{intro}
{instruction_key}
{instruction}
{response_key}
""".format(
    intro=INTRO_BLURB,
    instruction_key=INSTRUCTION_KEY,
    instruction="{instruction}",
    response_key=RESPONSE_KEY,
)
        prompt = PROMPT_FOR_GENERATION_FORMAT.format(instruction=prompt)
        return prompt

    def chatglm_template(prompt, original=False, force_sim=False, force_oppo=False):
        # pdb.set_trace()
        if not original and not force_sim and not force_oppo:
            return prompt
        if original:
            induction_prompt = induction_system_prompt(args.mbti, chatglm=True)
        elif force_sim:
            induction_prompt = induction_system_prompt_sp(args.mbti, chatglm=True, force_sim=True)
        elif force_oppo:
            induction_prompt = induction_system_prompt_sp(args.mbti, chatglm=True, force_oppo=True)
        prompt = induction_prompt + prompt
        # pdb.set_trace()
        return prompt

    def build_prompt_wrapper(examples, func):
        ret = {'question':[]}
        for i in range(len(examples['question'])):
            ret['question'].append(func(examples['question'][i]))
        return ret

    if model_base in "Llama_13b_chat":
        return partial(build_prompt_wrapper, func=Llama_template)(examples)
    elif model_base == 'chatglm2_6b':
        chat_func = lambda x:tokenizer.build_prompt(chatglm_template(x, original=original, force_sim=force_sim, force_oppo=force_oppo), history=[])
        return partial(build_prompt_wrapper, func=chat_func)(examples)
    else:
        raise NotImplementedError

model_mapping = {
    'Llama_13b_chat': Llama_path,
    'chatglm2_6b': Glm_path,
}
model_path = model_mapping[model_choice]
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
if model_choice in ['Llama_13b_chat']:
    tokenizer.padding_side = 'left'
    tokenizer.truncation_side = 'right'
    tokenizer.pad_token_id = tokenizer.eos_token_id
config = AutoConfig.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, config=config, trust_remote_code=True, device_map=device).bfloat16()
model.eval()

def unified_generation_config(args, diversity=False, chatglm=False):
    class InvalidScoreLogitsProcessor(LogitsProcessor):
        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
            if torch.isnan(scores).any() or torch.isinf(scores).any():
                scores.zero_()
                scores[..., 5] = 5e4
            return scores

    logits_processor = LogitsProcessorList()
    logits_processor.append(InvalidScoreLogitsProcessor())
    # NOTE:It's only enough for single trait prompt
    max_length = 1024
    # We Use Greedy Decode. Set default temperature and top_p to avoid warning.
    # when sample reward samples, we use do_sample = True,  and high temperature to get three response for each question.
    if diversity:
        gen_kwargs = {"max_length": max_length, "num_beams": 3, "do_sample": True, "logits_processor": logits_processor,
                  "temperature": 1.0, "top_p": 0.6,}
    else:
        gen_kwargs = {"max_length": max_length, "num_beams": 1, "do_sample": False, "logits_processor": logits_processor,
                  "temperature": 1.0, "top_p": 1.0,}
    if chatglm:
        import random
        do_sample = random.choice([True, False])
        temperature = random.uniform(0.9, 0.99) if do_sample else 1.0
        top_p = random.uniform(0.7, 0.99) if do_sample else 1.0
        gen_kwargs = dict(do_sample=do_sample, temperature=temperature, top_p=top_p, max_new_tokens=256, logits_processor=logits_processor,)
    return gen_kwargs

gen_diversity = False
gen_kwargs = unified_generation_config(args, diversity=gen_diversity, chatglm=(model_choice == 'chatglm2_6b'))

def map_to_json(ori_list, key):
    return [{'question':f'{questions[i]["question"]}', f'{key}':ori_list[i]} for i in range(len(ori_list))]

def add_question(ori_list):
    for i in range(len(ori_list)):
        cur_index = i % 200
        ori_list[i]['question'] = f'{questions[cur_index]["question"]}'
    return ori_list

def map_template(examples, extra_desc):
    ret = {'question':[]}
    func_map = {
        'pos': get_positive_template,
        'neu': get_neutral_template,
        'neg': get_negative_template,
    }
    func = func_map[extra_desc]
    for i in range(len(examples['question'])):
        ret['question'].append(func(trait=mbti, query=examples['question'][i]))
    return ret

def dataset_precess(dataset):
    dataset = dataset.map(partial(unified_build_prompt, tokenizer=tokenizer, model_base=model_choice, original=False, force_sim=False, force_oppo=False), batched=True, load_from_cache_file=False)
    dataset = dataset.map(lambda e: tokenizer(e['question'], padding=True), batched=True, load_from_cache_file=False)
    if model_choice in ['Llama_13b_chat']:
        dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'], device=device)
    elif model_choice in ['chatglm2_6b']:
        dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'position_ids'], device=device)
    return dataset

positive_dataset = load_dataset('json', data_files=path)['train']
neutral_dataset = load_dataset('json', data_files=path)['train']
negative_dataset = load_dataset('json', data_files=path)['train']

positive_dataset = positive_dataset.map(partial(map_template, extra_desc='pos'), batched=True, load_from_cache_file=False)
neutral_dataset = neutral_dataset.map(partial(map_template, extra_desc='neu'), batched=True, load_from_cache_file=False)
negative_dataset = negative_dataset.map(partial(map_template, extra_desc='neg'), batched=True, load_from_cache_file=False)

positive_dataset = dataset_precess(positive_dataset)
neutral_dataset = dataset_precess(neutral_dataset)
negative_dataset = dataset_precess(negative_dataset)

ppo_dataset = load_dataset('json', data_files=path)['train']
ppo_dataset = ppo_dataset.map(partial(unified_build_prompt, tokenizer=tokenizer, model_base=model_choice, original=False, force_sim=True, force_oppo=False), batched=True, load_from_cache_file=False)
ppo_dataset = ppo_dataset.map(lambda e: tokenizer(e['question'], padding='max_length', max_length=512), batched=True, load_from_cache_file=False)
if model_choice in ['Llama_13b_chat']:
    ppo_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'], device=device)
elif model_choice in ['chatglm2_6b']:
    ppo_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'position_ids'], device=device)

batch_size = args.batch_size
positive_dataloader = torch.utils.data.DataLoader(positive_dataset, batch_size=batch_size, shuffle=False)
neutral_dataloader = torch.utils.data.DataLoader(neutral_dataset, batch_size=batch_size, shuffle=False)
negative_dataloader = torch.utils.data.DataLoader(negative_dataset, batch_size=batch_size, shuffle=False)
original_dataloader = torch.utils.data.DataLoader(ppo_dataset, batch_size=batch_size, shuffle=False)


def output_to_index_json(output_list):
    for i in range(len(output_list)):
        json = lambda x: {'index': i, 'response': x}
        output_list[i] = json(output_list[i])
    return output_list

diversity_response_times = 3 if gen_diversity else 1

def diversity_response(diversity_response_times, data_loader, test=False):
    output_list = []
    c = 0
    while diversity_response_times > 0:
        temp_list = []
        with torch.no_grad():
            for batch_data in tqdm(data_loader, total=len(data_loader)):
                if c < 31:
                    c+=1
                    continue
                output = model.generate(**batch_data, **gen_kwargs)
                if model_choice in ['Llama_13b_chat']:
                    output = tokenizer.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    output = list(map(lambda x:x.split('[/INST]')[-1], output))
                elif model_choice in ['chatglm2_6b']:
                    output = tokenizer.batch_decode(output)
                    output = list(map(lambda x:x.split('答： ')[-1], output))
                temp_list.extend(output)
                if test:
                    print(output)
                    assert False
        temp_list = output_to_index_json(temp_list)
        output_list.extend(temp_list)
        diversity_response_times -= 1
        torch.cuda.empty_cache()
    return output_list

if genearte_positive:
    positive_response_list = diversity_response(diversity_response_times, positive_dataloader, test=test_generate)
    positive_response = add_question(positive_response_list)
    positive_path = os.path.join(positive_path, mbti, f'{model_choice}.json')
    if not test_generate:
        json.dump(positive_response, open(positive_path, 'w', encoding='UTF-8'),
            indent=4, ensure_ascii=False)

if genearte_neutral:
    neutral_response_list = diversity_response(diversity_response_times, neutral_dataloader, test=test_generate)
    neutral_response = add_question(neutral_response_list)
    neutral_path = os.path.join(neutral_path, mbti, f'{model_choice}.json')
    if not test_generate:
        json.dump(neutral_response, open(neutral_path, 'w', encoding='UTF-8'),
            indent=4, ensure_ascii=False)

if genearte_negative:
    negative_response_list = diversity_response(diversity_response_times, negative_dataloader, test=test_generate)
    negative_response = add_question(negative_response_list)
    negative_path = os.path.join(negative_path, mbti, f'{model_choice}.json')
    if not test_generate:
        json.dump(negative_response, open(negative_path, 'w', encoding='UTF-8'),
            indent=4, ensure_ascii=False)

if genearte_original:
    response_list = diversity_response(diversity_response_times, original_dataloader, test=test_generate)
    original_response = add_question(response_list)
    original_path = os.path.join(original_path, mbti, f'{model_choice}.json')
    if not test_generate:
        json.dump(original_response, open(original_path, 'w', encoding='UTF-8'),
            indent=4, ensure_ascii=False)
