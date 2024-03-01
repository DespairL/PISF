from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers import Trainer, default_data_collator
from transformers import HfArgumentParser, TrainingArguments, set_seed
from sft_args import ModelArguments, DataTrainingArguments
import os
from datasets import load_from_disk, load_dataset
import torch
import deepspeed
from deepspeed.accelerator import get_accelerator
from functools import partial
from deepspeed.utils.logging import logger
import logging
import transformers
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from qwen_generation_utils import make_context
from transformers import GenerationConfig
import pdb

main_logger = logging.getLogger(__name__)

def unified_build_prompt(examples, tokenizer, model_base, prompt_column, response_column, model=None):

    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

    # Reference:https://github.com/chujiezheng/chat_templates?tab=readme-ov-file
    def Llama_template(prompt):
        # <bos>[INST] B_SYS SystemPrompt E_SYS Prompt [/INST]
        return f"""{B_INST} {B_SYS} You are a chatbot. {E_SYS} {prompt} {E_INST}"""

    def build_prompt_wrapper(examples, func):
        ret = {prompt_column:[], response_column:[]}
        if type(examples[prompt_column]) == str:
            # batch size = 1
            ret = {prompt_column: [func(examples[prompt_column])], response_column: [examples[response_column]]}
            return ret
        for i in range(len(examples[prompt_column])):
            ret[prompt_column].append(func(examples[prompt_column][i]))
            ret[response_column].append(examples[response_column][i])
        return ret

    def qwen_build_prompts(prompt, tokenizer, model):
        temp, _ = make_context(
                tokenizer,
                prompt,
                system="You are a helpful assistant.",
                max_window_size=model.generation_config.max_window_size,
                chat_format=model.generation_config.chat_format,
            )
        return temp

    if model_base in "Llama_13b_chat":
        return partial(build_prompt_wrapper, func=Llama_template)(examples)
    elif model_base == 'chatglm2_6b':
        chat_func = lambda x:tokenizer.build_prompt(x, history=[])
        return partial(build_prompt_wrapper, func=chat_func)(examples)
    elif model_base == 'qwen_chat_7b' :
        qwen_func = partial(qwen_build_prompts, tokenizer=tokenizer, model=model)
        return partial(build_prompt_wrapper, func=qwen_func)(examples)
    else:
        raise NotImplementedError

if __name__ == '__main__':
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    set_seed(training_args.seed)

    transformers.utils.logging.set_verbosity_info()
    log_level = training_args.get_process_log_level()
    transformers_logger = transformers.utils.logging.get_logger("transformers")
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.disable_propagation()
    transformers.utils.logging.enable_explicit_format()
    deepspeed_logger = logger

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )

    if training_args.local_rank == -1:
        device = torch.device(get_accelerator().device_name())
    else:
        get_accelerator().set_device(training_args.local_rank)
        device = torch.device(get_accelerator().device_name(), training_args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        deepspeed.init_distributed()

    torch.distributed.barrier()

    personality = None
    if data_args.pretrain_dataset_cache:
        datasets = load_from_disk(data_args.pretrain_dataset_cache)
        personality = model_args.personality
    else:
        datasets = load_dataset('json', data_files=data_args.train_file)['train']

    prompt_column = data_args.prompt_column
    response_column = data_args.response_column

    trust_remote_code = False
    if model_args.model_base in ['chatglm2_6b', 'qwen_chat_7b']:
        trust_remote_code = True

    if model_args.model_base != 'qwen_chat_7b':
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=trust_remote_code)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, pad_token='<|extra_0|>',
            eos_token='<|endoftext|>', padding_side='right', trust_remote_code=trust_remote_code)


    if model_args.model_base in ["Llama_13b_chat"]:
        # actually no need to set tokenizer.padding
        tokenizer.padding_side = 'right'
        tokenizer.truncation_side = 'right'
        tokenizer.pad_token_id = tokenizer.eos_token_id

    pad_token_id = tokenizer.pad_token_id

    # use device_map to save memory
    if model_args.model_base != 'qwen_chat_7b':
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, use_cache=False, torch_dtype=torch.bfloat16, trust_remote_code=trust_remote_code)
        model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, config=config, trust_remote_code=trust_remote_code, device_map=device)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, trust_remote_code=trust_remote_code,
            pad_token_id=pad_token_id, device_map=device).bfloat16()
        model.generation_config = GenerationConfig.from_pretrained(model_args.model_name_or_path, pad_token_id=tokenizer.pad_token_id)


    if model_args.do_lora_train:
        target_modules = {
            'Llama_13b_chat': ["q_proj"],
            'chatglm2_6b': ["query_key_value"],
            'qwen_chat_7b': ["c_attn"],
        }
        # model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)
        config = LoraConfig(
            # 8,8
            r=8,
            lora_alpha=8,
            target_modules=target_modules[model_args.model_base],
            lora_dropout=0.1,
            task_type="CAUSAL_LM",
            bias="none",
            inference_mode=False,
        )
        model.enable_input_require_grads()
        model = get_peft_model(model, config)
        model.print_trainable_parameters()

    def tokenize_function(examples, tokenizer):
        model_inputs = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }
        if model_args.model_base != 'qwen_chat_7b':
            examples = unified_build_prompt(examples, tokenizer, model_args.model_base, prompt_column, response_column)
        else:
            examples = unified_build_prompt(examples, tokenizer, model_args.model_base, prompt_column, response_column, model=model)
        for i in range(len(examples[prompt_column])):
            query, answer = examples[prompt_column][i], examples[response_column][i]
            a_ids = tokenizer.encode(text=query, add_special_tokens=True)
            b_ids = tokenizer.encode(text=answer, add_special_tokens=False)
            # add pad and eos by ourselves
            context_length = len(a_ids)
            input_ids = a_ids + b_ids + [tokenizer.eos_token_id]
            total_length = len(input_ids)
            labels = [tokenizer.pad_token_id] * context_length + b_ids + [tokenizer.eos_token_id]
            pad_len = 512 - len(input_ids)
            input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
            labels = labels + [tokenizer.pad_token_id] * pad_len
            labels = [(l if l != tokenizer.pad_token_id else -100) for l in labels]
            if model_args.model_base in ["Llama_13b_chat"]:
                # avoid eos is replaced with -100
                labels[total_length-1] = tokenizer.eos_token_id
            attention_mask = [1] * total_length + [0] * pad_len
            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append(attention_mask)
            model_inputs["labels"].append(labels)
        return model_inputs

    with training_args.main_process_first(desc="dataset map tokenization"):
        datasets = datasets.map(
            partial(tokenize_function, tokenizer=tokenizer),
            #batch_size=2,
            #num_proc=1,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=datasets.column_names,
            load_from_cache_file=False,#not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    # model.gradient_checkpointing_enable()
    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_state()