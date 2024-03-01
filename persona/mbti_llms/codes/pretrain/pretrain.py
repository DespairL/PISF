from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers import Trainer, TrainingArguments, Seq2SeqTrainer, default_data_collator
from transformers import HfArgumentParser, Seq2SeqTrainingArguments, set_seed
from pretrain_args import ModelArguments, DataTrainingArguments
import os
from datasets import load_from_disk, load_dataset
import torch
import deepspeed
from deepspeed.accelerator import get_accelerator
from torch.utils.data.distributed import DistributedSampler
from dataclasses import dataclass
from functools import partial
from deepspeed.utils.logging import logger
import logging
import transformers
import torch.nn.functional as F
from transformers import GenerationConfig
import pdb

main_logger = logging.getLogger(__name__)

TRAINER_STATE_NAME = "trainer_state.json"
PREFIX_CHECKPOINT_DIR = "checkpoint"
class TrainerSaveModel(Trainer):
    def _save_checkpoint(self, model, trial, metrics=None):
        # We only save model. deepspeed optimizer costs too much memory.
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        if self.hp_search_backend is None and trial is None:
            self.store_flos()
        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        self.save_model(output_dir, _internal_call=True)
        # Save the Trainer state
        if self.args.should_save:
            self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))
        # Maybe delete some older checkpoints.
        if self.args.should_save:
            self._rotate_checkpoints(use_mtime=True, output_dir=run_dir)


if __name__ == '__main__':
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
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
        raise NotImplementedError
        datasets = load_dataset('json', data_files=data_args.train_file, keep_in_memory=False, streaming=True)['train']

    en_persona_description = {
        'E': ["Extroverted",
          """**Extroverted** refers to the act or state of being energized by the world outside the self. Extraverts enjoy socializing and tend to be more enthusiastic, assertive, talkative, and animated. They enjoy time spent with more people and find it less rewarding to spend time alone. They are Initiating, Expressive, Gregarious, Active and Enthusiastic."""],
        'I': ["Introverted",
          """**Introverted**, on the contrary, is the state of being predominately concerned with one’s inner world. Introverts prefer self-reflection to social interactions. They also prefer to observe before participating in an activity. Introverts tend to more quiet, ‘peaceful’, and reserved. Introverts *prefer* individual activities over social ones—this. They are Receiving, Contained, Intimate, Reflective and Quiet."""],
        'S': ["Sensing",
          """**Sensing** refers to processing data through the five senses. Sensing people focus on the present and prefer to “learn by doing” rather than thinking it through. They are concrete thinkers recognize details. They are more energized by the practical use of an object/idea rather than the theory behind it. They are Concrete, Realistic, Practical, Experiential and Traditional."""],
        'N': ["Intuition",
          """**Intuition** refers to how people process data. Intuitive people are keener to the meaning and patterns behind information. Intuitive people are more focused on how the present would affect the future. They are readily able to grasp different possibilities and abstract concepts. They easily see the big picture rather than the details. They are Abstract, Imaginative, Conceptual, Theoretical and Original."""],
        'T': ["Thinking",
          """**Thinking** refers to how people make decisions. Thinking people are objective and base their decision on hard logic and facts. They tend to analyze the pros and cons of a situation and notice inconsistencies. They prefer to be task-oriented and fair. They are Logical, Reasonable, Questioning, Critical and Tough."""],
        'F': ["Feeling",
          """**Feeling** people are more subjective. They base their decisions on principles and personal values. When making decisions, they consider other people’s feelings and take it in account. It is in their best mind to maintain harmony among a group. They are more governed by their heart. They are Empathetic, Compassionate, Accommodating, Accepting and Tender."""],
        'J': ["Judging",
          """**Judging** refers to how people outwardly display themselves when making decisions. Judging people have a tendency to be organized and prompt. They like order prefer outlined schedules to working extemporaneously. They prefer plans. They find the outcome more rewarding than the process of creating something. Judging people seek closure. They are Systematic, Planful, Early Starting, Scheduled and Methodical."""],
        'P': ["Perceiving",
          """**Perceiving** people prefer flexibility and live their life with spontaneity. They act following their mind. They dislike structure and prefer to adapt to new situations rather than plan for it. They tend to be open to new options, experiences and emergency events. While working on a project, they enjoy the process more than the outcome. They are Casual, Open-Ended, Pressure-Prompted, Spontaneous and Emergent."""],
    }

    def map_personality_to_prompt(personality):
        p_list = list(personality)
        desc = [en_persona_description[p][1] for p in p_list]
        desc = '\n'.join(desc)
        sim_desc = [en_persona_description[p][0] for p in p_list]
        sim_desc = ' '.join(sim_desc)
        sim_desc += f'({personality})'
        return sim_desc, desc

    sim_desc, desc = map_personality_to_prompt(personality)
    pretrain_prompt = """People with {sim_desc} said the following utterances(seperated with \"|||\"):

{utterances}"""

    prompt_column = data_args.prompt_column

    trust_remote_code = True

    if model_args.model_base != 'qwen_chat_7b':
        if model_args.model_base in ["Llama_13b_chat"]:
            tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, add_eos_token=True)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=trust_remote_code)
        if model_args.model_base in ["Llama_13b_chat"]:
            tokenizer.padding_side = 'left'
            tokenizer.truncation_side = 'right'
            tokenizer.pad_token_id = tokenizer.eos_token_id
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, use_cache=False, torch_dtype=torch.bfloat16, trust_remote_code=trust_remote_code)
        model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, config=config, trust_remote_code=trust_remote_code)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, pad_token='<|extra_0|>',
            eos_token='<|endoftext|>', padding_side='left', trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, trust_remote_code=True,
            pad_token_id=tokenizer.pad_token_id).bfloat16()
        model.generation_config = GenerationConfig.from_pretrained(model_args.model_name_or_path, pad_token_id=tokenizer.pad_token_id)

    pad_token_id = tokenizer.pad_token_id
    bos_token_id = tokenizer.bos_token_id

    # pdb.set_trace()

    def tokenize_function(examples):
        model_inputs = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }
        for i in range(len(examples[prompt_column])):
            examples[prompt_column][i] = pretrain_prompt.format(sim_desc=sim_desc, utterances=examples[prompt_column][i])
            max_length = 2048
            inputs = tokenizer(examples[prompt_column][i], truncation=True, max_length=max_length,) # , return_tensors='pt', truncation=True, max_length=2048, padding='max_length'
            input_ids = inputs["input_ids"]
            ori_input_ids = input_ids
            padding_length = 2048 - len(input_ids)
            input_ids = [pad_token_id for i in range(padding_length)] + input_ids
            attention_mask = [0 if x == pad_token_id else 1 for x in input_ids[:-1]] + [1]
            labels = [-100 for i in range(padding_length+1)] + ori_input_ids[1:]
            assert len(labels) == 2048
            assert len(attention_mask) == 2048
            assert len(input_ids) == 2048
            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append(attention_mask)
            model_inputs["labels"].append(labels)
        return model_inputs

    with training_args.main_process_first(desc="dataset map tokenization"):
        datasets = datasets.map(
            tokenize_function,
            batched=True,
            #batch_size=2,
            #num_proc=1,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=datasets.column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    trainer = TrainerSaveModel(
        model=model,
        args=training_args,
        train_dataset=datasets,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    model.gradient_checkpointing_enable()
    train_result = trainer.train(resume_from_checkpoint=checkpoint)