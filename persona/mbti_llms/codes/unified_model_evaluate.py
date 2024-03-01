import os
import sys
import time
import requests
import json
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoModelForCausalLM
import torch
from functools import partial
from transformers.generation.utils import LogitsProcessorList
from transformers.generation.logits_process import LogitsProcessor
import argparse
import math
from peft import PeftModelForCausalLM
from rlhf.llama_dschat.utils.data.raw_datasets import induction_system_prompt
import pdb
from transformers import GenerationConfig
from qwen_generation_utils import make_context

def get_tokenizer(model_name_or_path, fast_tokenizer=True):
    if "Llama" in model_name_or_path:
        from transformers.models.llama import LlamaTokenizer
        tokenizer = LlamaTokenizer.from_pretrained(
            model_name_or_path, fast_tokenizer=fast_tokenizer)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'
    elif "chatglm" in model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    elif "Qwen" or "qwen" in model_name_or_path:
        # NOTE: Qwen have specific pad_token
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, pad_token='<|extra_0|>',
            eos_token='<|endoftext|>', padding_side='left', trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, fast_tokenizer=fast_tokenizer)
        tokenizer.pad_token = tokenizer.eos_token
        # make sure tokenizer is right pad in our logic
        tokenizer.padding_side = 'left'
    return tokenizer


def load_hf_tokenizer(model_name_or_path,
                      fast_tokenizer=True,
                      add_special_tokens=None):
    tokenizer = get_tokenizer(model_name_or_path,
                                  fast_tokenizer=fast_tokenizer)

    if add_special_tokens is not None:
        add_special_tokens = [add_special_tokens] if isinstance(add_special_tokens, str) \
            else add_special_tokens
        tokenizer.add_special_tokens(
            {'additional_special_tokens': add_special_tokens})

    return tokenizer

def unified_load_model(args):
    # continual pretrain -> use full-parameter-tuning, No need to load model_base
    # for decoder-only, tokenizer should be left-padding while generating
    if args.training_phase == 'cp' or args.training_phase == 'prompt':

        trust_remote_code = True
        if args.model_base == 'qwen_chat_7b':
            tokenizer = AutoTokenizer.from_pretrained(args.model_path, pad_token='<|extra_0|>',
                eos_token='<|endoftext|>', padding_side='left', trust_remote_code=trust_remote_code)
            model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=trust_remote_code,
                pad_token_id=tokenizer.pad_token_id, device_map=args.device).bfloat16()
            model.generation_config = GenerationConfig.from_pretrained(args.model_path, pad_token_id=tokenizer.pad_token_id)
            model.generation_config.do_sample = False
            args.max_window_size = model.generation_config.max_window_size
            args.chat_format = model.generation_config.chat_format
            model.eval()
            return model, tokenizer

        if args.model_base == 'chatglm2_6b':
            config_path = "/data/NJU/datasets/persona/models/chatglm2-6b"
        else:
            config_path = args.model_path
        # pdb.set_trace()
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        if args.model_base in ['Llama_13b_chat']:
            tokenizer.padding_side = 'left'
            tokenizer.truncation_side = 'right'
            tokenizer.pad_token_id = tokenizer.eos_token_id
        config = AutoConfig.from_pretrained(config_path, torch_dtype=torch.bfloat16, trust_remote_code=True,)
        # pdb.set_trace()
        model = AutoModelForCausalLM.from_pretrained(args.model_path, config=config, trust_remote_code=True, device_map=args.device).bfloat16()

        model.eval()
    elif args.training_phase == 'sft':
        trust_remote_code = True

        if args.model_base == 'qwen_chat_7b':
            tokenizer = AutoTokenizer.from_pretrained(args.model_path, pad_token='<|extra_0|>',
                eos_token='<|endoftext|>', padding_side='left', trust_remote_code=trust_remote_code)
            model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=trust_remote_code,
                pad_token_id=tokenizer.pad_token_id, device_map=args.device).bfloat16()
            model.generation_config = GenerationConfig.from_pretrained(args.model_path, pad_token_id=tokenizer.pad_token_id)
            model.generation_config.do_sample = False
            args.max_window_size = model.generation_config.max_window_size
            args.chat_format = model.generation_config.chat_format
            if 'checkpoint' in args.adapter_path:
                model = PeftModelForCausalLM.from_pretrained(model, args.adapter_path, is_trainable=False)
            model.eval()
            return model, tokenizer

        load_path = args.model_path
        if 'checkpoint' not in args.adapter_path:
            load_path = args.adapter_path
        tokenizer = AutoTokenizer.from_pretrained(load_path, trust_remote_code=trust_remote_code)
        if args.model_base in ['Llama_13b_chat']:
            tokenizer.padding_side = 'left'
            tokenizer.truncation_side = 'right'
            tokenizer.pad_token_id = tokenizer.eos_token_id
        config = AutoConfig.from_pretrained(load_path, torch_dtype=torch.bfloat16, trust_remote_code=trust_remote_code)
        model = AutoModelForCausalLM.from_pretrained(load_path, config=config, trust_remote_code=trust_remote_code, device_map=args.device).bfloat16()
        assert args.adapter_path
        if 'checkpoint' in args.adapter_path:
            model = PeftModelForCausalLM.from_pretrained(model, args.adapter_path, is_trainable=False)
        model.eval()
    elif args.training_phase == 'rlhf':
        if args.ppo_path == args.model_path:
            # evaluate base
            tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
            if args.model_base in ['Llama_13b_chat']:
                tokenizer.padding_side = 'left'
                tokenizer.truncation_side = 'right'
                tokenizer.pad_token_id = tokenizer.eos_token_id
            config = AutoConfig.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(args.model_path, config=config, trust_remote_code=True, device_map=args.device).bfloat16()
            model.eval()
            return model, tokenizer
        ppo_model_path = os.path.join(args.ppo_path, 'pytorch_model.bin')
        state_dict = torch.load(ppo_model_path, map_location='cpu')

        trust_remote_code = True

        if args.model_base == 'chatglm2_6b':
            tokenizer = load_hf_tokenizer(args.model_path,
                                  fast_tokenizer=True,
                                  add_special_tokens="</s>")
        elif args.model_base == 'qwen_chat_7b':
            tokenizer = load_hf_tokenizer(args.model_path,
                                  fast_tokenizer=True,
                                  add_special_tokens=None)
        else:
            tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=trust_remote_code)

        if args.model_base in ['Llama_13b_chat']:
            tokenizer.padding_side = 'left'
            tokenizer.truncation_side = 'right'
            tokenizer.pad_token_id = tokenizer.eos_token_id

        if args.model_base == 'qwen_chat_7b':
            model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map=args.device, torch_dtype=torch.bfloat16,
                                                trust_remote_code=trust_remote_code, pad_token_id=tokenizer.pad_token_id, bf16=True)
        else:
            model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map=args.device, torch_dtype=torch.bfloat16, trust_remote_code=trust_remote_code)
        if args.model_base != 'chatglm2_6b' and args.model_base != 'qwen_chat_7b':
            model.config.end_token_id = tokenizer.eos_token_id
            model.config.pad_token_id = model.config.eos_token_id
        model.resize_token_embeddings(int(8 * math.ceil(len(tokenizer) / 8.0)))
        if args.model_base == 'chatglm2_6b':
            model.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict, strict=False)

        if args.model_base == 'qwen_chat_7b':
            model.generation_config = GenerationConfig.from_pretrained(args.model_path, pad_token_id=tokenizer.pad_token_id)
            model.generation_config.do_sample = False
            args.max_window_size = model.generation_config.max_window_size
            args.chat_format = model.generation_config.chat_format
        # pdb.set_trace()
        model.eval()

    return model, tokenizer

def unified_build_prompt(examples, tokenizer, model_base, args):

    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

    en_persona_description = {
        'E': ["**Extroverted**",
                """**Extraversion** refers to the act or state of being energized by the world outside the self. Extraverts enjoy socializing and tend to be more enthusiastic, assertive, talkative, and animated. They enjoy time spent with more people and find it less rewarding to spend time alone. They are Initiating, Expressive, Gregarious, Active and Enthusiastic."""],
        'I': ["**Introverted**",
                """**Introversion**, on the contrary, is the state of being predominately concerned with one’s inner world. Introverts prefer self-reflection to social interactions. They also prefer to observe before participating in an activity. Introverts tend to more quiet, ‘peaceful’, and reserved. Introverts *prefer* individual activities over social ones—this. They are Receiving, Contained, Intimate, Reflective and Quiet."""],
        'S': ["**Sensing**",
                  """**Sensing** refers to processing data through the five senses. Sensing people focus on the present and prefer to “learn by doing” rather than thinking it through. They are concrete thinkers recognize details. They are more energized by the practical use of an object/idea rather than the theory behind it. They are Concrete, Realistic, Practical, Experiential and Traditional."""],
        'N': ["**Intuition**",
                  """**Intuition** refers to how people process data. Intuitive people are keener to the meaning and patterns behind information. Intuitive people are more focused on how the present would affect the future. They are readily able to grasp different possibilities and abstract concepts. They easily see the big picture rather than the details. They are Abstract, Imaginative, Conceptual, Theoretical and Original."""],
        'T': ["**Thinking**",
                  """**Thinking** refers to how people make decisions. Thinking people are objective and base their decision on hard logic and facts. They tend to analyze the pros and cons of a situation and notice inconsistencies. They prefer to be task-oriented and fair. They are Logical, Reasonable, Questioning, Critical and Tough."""],
        'F': ["**Feeling**",
                  """**Feeling** people are more subjective. They base their decisions on principles and personal values. When making decisions, they consider other people’s feelings and take it in account. It is in their best mind to maintain harmony among a group. They are more governed by their heart. They are Empathetic, Compassionate, Accommodating, Accepting and Tender."""],
        'J': ["**Judging**",
                  """**Judging** refers to how people outwardly display themselves when making decisions. Judging people have a tendency to be organized and prompt. They like order prefer outlined schedules to working extemporaneously. They prefer plans. They find the outcome more rewarding than the process of creating something. Judging people seek closure. They are Systematic, Planful, Early Starting, Scheduled and Methodical."""],
        'P': ["**Perceiving**",
                  """**Perceiving** people prefer flexibility and live their life with spontaneity. They act following their mind. They dislike structure and prefer to adapt to new situations rather than plan for it. They tend to be open to new options, experiences and emergency events. While working on a project, they enjoy the process more than the outcome. They are Casual, Open-Ended, Pressure-Prompted, Spontaneous and Emergent."""],
    }

    def get_personality_description(mbti):
        role = 'helpful assistant' if model_base == 'qwen_chat_7b' else 'chatbot'
        all_traits = list(mbti)
        sim_desc = "You are a "
        sim_desc = ""
        desc = ""
        for each_trait in all_traits:
            sim_desc += en_persona_description[each_trait][0].replace('*', '') + ' '
            desc += en_persona_description[each_trait][1] + '\n'
            if all_traits.index(each_trait) == len(all_traits)-1:
                sim_desc = sim_desc[:-1]
        if model_base == 'qwen_chat_7b':
            final_system_prompt = f'You are a {sim_desc} {role}. A {sim_desc} {role} has the following requirements:\n{desc}\n'
        else:
            final_system_prompt = f'You are a {sim_desc} {role}. A {sim_desc} {role} has the following requirements:\n{desc}\n'
        final_system_prompt = final_system_prompt.strip()
        return final_system_prompt

    def Llama_template(prompt):

        # tokenizer will add <s>
        # <bos>[INST] B_SYS SystemPrompt E_SYS Prompt [/INST]
        if args.training_phase == 'rlhf':
            assert args.mbti
            return f"""{B_INST} {B_SYS} {induction_system_prompt(args.mbti)} {E_SYS} {prompt} {E_INST}"""

        if args.training_phase == 'sft':
            if not args.default_prompt:
                raise NotImplementedError
            else:
                if args.same_prompt_induction_after_training:
                    cur_system_prompt = get_personality_description(args.mbti)
                    if args.test_code:
                        pdb.set_trace()
                    if args.with_instruction:
                        # pdb.set_trace()
                        return f"""{B_INST} {B_SYS} {cur_system_prompt} {E_SYS} Below is an instruction that describes a task.\nWrite a response that appropriately completes the request.\n\n### Instruction:\n{prompt}\n### Response: {E_INST}"""
                    else:
                        return f"""{B_INST} {B_SYS} {cur_system_prompt} {E_SYS} {prompt} {E_INST}"""
                else:
                    if args.with_instruction:
                        return f"""{B_INST} {B_SYS} You are a chatbot. {E_SYS} Below is an instruction that describes a task.\nWrite a response that appropriately completes the request.\n\n### Instruction:\n{prompt}\n### Response: {E_INST}"""
                    else:
                        return f"""{B_INST} {B_SYS} You are a chatbot. {E_SYS} {prompt} {E_INST}"""
        return f"""{B_INST} {B_SYS} You are a chatbot. {E_SYS} {prompt} {E_INST}"""

    def chatglm_template(prompt):
        # temp = "[Round 0]\n\n问：" + induction_system_prompt(args.mbti, chatglm=True, ppo_prompt_type=args.ppo_prompt_type) + prompt + "\n\n答："
        # pdb.set_trace()
        return "[Round 0]\n\n问：" + induction_system_prompt(args.mbti, chatglm=True, ppo_prompt_type=args.ppo_prompt_type) + prompt + "\n\n答："

    def chatglm_insturction_template(prompt):
        return "[Round 0]\n\n问：" + f"Below is an instruction that describes a task.\nWrite a response that appropriately completes the request.\n\n### Instruction:\n{prompt}\n### Response:" + "\n\n答："

    def build_prompt_wrapper(examples, func):
        ret = {'prompt':[]}
        for i in range(len(examples['prompt'])):
            ret['prompt'].append(func(examples['prompt'][i]))
        return ret

    def qwen_build_prompts(prompt, tokenizer):
        def add_instruction(x):
            return f"Below is an instruction that describes a task.\nWrite a response that appropriately completes the request.\n\n### Instruction:\n{x}\n### Response:"

        default_sys_prompt_for_qwen = "You are a helpful assistant."

        if args.with_instruction:
            prompt = add_instruction(prompt)

        if args.same_prompt_induction_after_training:
            default_sys_prompt_for_qwen = get_personality_description(args.mbti)

        if args.training_phase == 'rlhf':
            assert args.mbti
            default_sys_prompt_for_qwen = induction_system_prompt(args.mbti, chatglm=False, ppo_prompt_type=args.ppo_prompt_type, llama=False)

        temp, _ = make_context(
                tokenizer,
                prompt,
                system=default_sys_prompt_for_qwen,
                max_window_size=args.max_window_size,
                chat_format=args.chat_format,
            )
        return temp

    if model_base in "Llama_13b_chat":
        return partial(build_prompt_wrapper, func=Llama_template)(examples)
    elif model_base == 'chatglm2_6b':
        chat_func = lambda x:tokenizer.build_prompt(x, history=[])
        if args.training_phase == 'rlhf':
            chat_func = chatglm_template
        if args.training_phase == 'sft':
            if args.with_instruction:
                chat_func = chatglm_insturction_template
        return partial(build_prompt_wrapper, func=chat_func)(examples)
    elif model_base == 'qwen_chat_7b' :
        qwen_func = partial(qwen_build_prompts, tokenizer=tokenizer)
        return partial(build_prompt_wrapper, func=qwen_func)(examples)
    else:
        raise NotImplementedError

def unified_generation_config(args):
    class InvalidScoreLogitsProcessor(LogitsProcessor):
        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
            if torch.isnan(scores).any() or torch.isinf(scores).any():
                scores.zero_()
                scores[..., 5] = 5e4
            return scores

    logits_processor = LogitsProcessorList()
    logits_processor.append(InvalidScoreLogitsProcessor())
    max_length = 2048
    # We Use Greedy Decode. Set default temperature and top_p to avoid warning.
    gen_kwargs = {"max_length": max_length, "num_beams": 1, "do_sample": False, "logits_processor": logits_processor,
                  "temperature": 1.0, "top_p": 1.0,}
    if args.training_phase == 'rlhf':
        gen_kwargs = {"max_new_tokens": 256, "do_sample": False, "temperature": 1.0, "top_p": 1.0, }
        if args.model_base in ['chatglm2_6b']:
            gen_kwargs = {"max_new_tokens": 256, "do_sample": False, "temperature": 1.0, "top_p": 1.0, "logits_processor": logits_processor}
    return gen_kwargs

def unified_model_evaluate(args, model, tokenizer):
    gen_kwargs = unified_generation_config(args)
    evaluate_dataset = load_dataset('json', data_files=args.evaluate_specific_file)['train']
    evaluate_dataset = evaluate_dataset.map(partial(unified_build_prompt, tokenizer=tokenizer, model_base=args.model_base, args=args), batched=True)
    # pdb.set_trace()
    if args.test_code:
        pdb.set_trace()
    evaluate_dataset = evaluate_dataset.map(lambda e: tokenizer(e['prompt'], padding=True), batched=True)
    if args.model_base in ['Llama_13b_chat']:
        evaluate_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'], device=args.device)
    elif args.model_base in ['chatglm2_6b']:
        evaluate_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'position_ids'], device=args.device)
    elif args.model_base in ['qwen_chat_7b']:
        evaluate_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'token_type_ids'], device=args.device)
    else:
        raise NotImplementedError

    batch_size = 8
    dataloader = torch.utils.data.DataLoader(evaluate_dataset, batch_size=batch_size, shuffle=False)

    response_list = []
    with torch.no_grad():
        for batch_data in tqdm(dataloader, total=len(dataloader)):
            if args.model_base in ['qwen_chat_7b']:
                output = model.generate(**batch_data,
                            return_dict_in_generate=False,
                            generation_config=model.generation_config)
                output = tokenizer.batch_decode(output, skip_special_tokens=True)
                output = list(map(lambda x:x.split('assistant\n')[-1].strip(), output))
                if args.test_code:
                    pdb.set_trace()
                response_list.extend(output)
                continue

            output = model.generate(**batch_data, **gen_kwargs)
            if args.model_base in ['Llama_13b_chat']:
                output = tokenizer.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                output = list(map(lambda x:x.split('[/INST]')[-1], output))
            elif args.model_base in ['chatglm2_6b']:
                output = tokenizer.batch_decode(output)
                output = list(map(lambda x:x.split('答： ')[-1], output))
            if args.test_code:
                pdb.set_trace()
            response_list.extend(output)

    response_json = json.load(open(args.evaluate_specific_file, 'r', encoding='UTF-8'))
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

    traindata_answer_extractor = json.dumps(traindata_answer_extractor, indent=4, ensure_ascii=False)
    output_path = None
    if args.output_result_for_label:
        time_cur = time.localtime()
        formatted_time = time.strftime("%m_%d_%H_%M", time_cur)
        output_path = f"/data/NJU/datasets/persona/mbti_llms/answer_to_label/{args.model_base}-{args.training_phase}-{formatted_time}.json"
    else:
        output_path = args.output_path
    with open(output_path, 'w', encoding='UTF-8') as file:
        file.write(traindata_answer_extractor)

def opposite_mapping(mbti):
    # ISTJ ISTP ISFP ISFJ INFJ INFP INTP INTJ
    mapping_dict = {
        'E':'I', 'I':'E', 'S':'N', 'N':'S', 'T':'F', 'F':'T', 'J':'P', 'P':'J',
        'ESTP':'INFJ', 'ESTJ':'INFP', 'ESFP':'INTJ', 'ESFJ':'INTP', 'ENFJ':'ISTP', 'ENFP':'ISTJ', 'ENTP':'ISFJ', 'ENTJ':'ISFP',
        'INFJ':'ESTP', 'INFP':'ESTJ', 'INTJ':'ESFP', 'INTP':'ESFJ', 'ISTP':'ENFJ', 'ISTJ':'ENFP', 'ISFJ':'ENTP', 'ISFP':'ENTJ',
    }
    return mapping_dict[mbti]

def reconstruct_path_for_prompt_induction(args):
    import re
    opposite_mbti = opposite_mapping(args.mbti)
    cur_test_file = args.evaluate_specific_file
    pattern = r"unified_prompt_(\w+)_mbti_(\d+).json"
    reconstruct_path = re.sub(pattern, fr"unified_prompt_{opposite_mbti}_mbti_\2.json", cur_test_file)
    print('='*40)
    print(f'Prompt Induction:{reconstruct_path}')
    print('='*40)
    args.evaluate_specific_file = reconstruct_path
    return args

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_base', required=True, choices=['chatglm2_6b', 'Llama_13b_chat', 'qwen_chat_7b'])
    parser.add_argument('--training_phase', required=True, choices=['cp', 'sft', 'rlhf', 'prompt'])
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--adapter_path', required=False, default='')
    parser.add_argument('--ppo_path', required=False, default='')
    parser.add_argument('--evaluate_specific_file', required=True)
    parser.add_argument('--output_result_for_label', action='store_true')
    parser.add_argument('--prompt_induction_after_training', action='store_true')
    parser.add_argument('--output_path', default='')
    parser.add_argument('--ppo_prompt_type', default='no_prompt')
    parser.add_argument('--specific_output_path', default='')
    parser.add_argument('--default_prompt', action='store_true')
    parser.add_argument('--with_instruction', action='store_true')
    parser.add_argument('--test_code', action='store_true')
    parser.add_argument('--same_prompt_induction_after_training', action='store_true')
    parser.add_argument('--mbti', default='')
    parser.add_argument('--max_window_size', default='')
    parser.add_argument('--chat_format', default='')
    parser.add_argument('--gpu', default=7, type=int)

    base = "/data/NJU/datasets/persona/mbti_llms/temp_result"
    args = parser.parse_args()
    args.device = f'cuda:{args.gpu}'
    if not args.output_result_for_label:
        assert args.output_path
        args.output_path = os.path.join(base, args.output_path)

    # specific_output_path
    if args.specific_output_path:
        args.output_path = args.specific_output_path

    if args.prompt_induction_after_training:
        args = reconstruct_path_for_prompt_induction(args)


    model, tokenizer = unified_load_model(args)

    unified_model_evaluate(args, model, tokenizer)
