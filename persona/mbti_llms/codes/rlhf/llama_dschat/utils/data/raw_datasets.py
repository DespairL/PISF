# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

import os
# DeepSpeed Team
from datasets import load_dataset, load_from_disk
from torch.utils.data import Subset
import re
import pdb
from .qwen_generation_utils import make_context

# The template prompt dataset class that all new dataset porting needs to
# follow in order to have a unified API and unified data format.
class PromptRawDataset(object):

    def __init__(self, output_path, seed, local_rank, dataset_path):
        self.output_path = output_path
        self.seed = seed
        self.local_rank = local_rank
        self.raw_datasets = load_dataset('json', data_files=dataset_path)

    def get_train_data(self):
        return

    def get_eval_data(self):
        return

    # The prompt should be in the format of: " Human: " + actual_prompt_sentence + " Assistant:"
    def get_prompt(self, sample):
        return

    # The chosen response should be in the format of: " " + actual_response_sentence
    def get_chosen(self, sample):
        return

    # The rejected response should be in the format of: " " + actual_response_sentence
    # If the dataset does not have rejected response, return None
    def get_rejected(self, sample):
        return

    def get_prompt_and_chosen(self, sample):
        return

    def get_prompt_and_rejected(self, sample):
        return

# Provide an anchor in response space
# We find it is really important if you would like to control personality in RLHF
# And different models require different system prompts based on their own capability differences.
def induction_system_prompt(mbti, chatglm=False, ppo_prompt_type='no_prompt', llama=True):
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

    def map_personality_to_prompt(personality):
        p_list = list(personality)
        sim_desc = [en_persona_description[p][0] for p in p_list]
        sim_desc = ' '.join(sim_desc)
        sim_desc += f'({personality})'
        return sim_desc

    # pdb.set_trace()
    opposite_personality = ''.join([opposite_mapping[x] for x in list(mbti)])
    sim_desc = map_personality_to_prompt(mbti)
    oppo_sim_desc = map_personality_to_prompt(opposite_personality)
    # Llama-2-chat is too Perceiving and Extroverted to learn what is "Perceiving" and what is "Extroverted" when we add this extra prompt.
    if llama:
        extra_mention = " and display one of them. You don't have to mention this in the chat." if mbti not in ['P', 'E'] else " ."
        Trait_pair_mension = f"{sim_desc} or {oppo_sim_desc}"
        learning_system_prompt = f"You are a chatbot. You are learning to switch your personality trait to {Trait_pair_mension}{extra_mention} "
    if chatglm:
        # The English comprehension ability of chatglm is significantly weaker than that of Llama.
        # It requires longer system prompts for clarification.
        Trait_pair_mension = f"{sim_desc} or {oppo_sim_desc}"
        if ppo_prompt_type == 'no_prompt':
            learning_system_prompt = ""
        elif ppo_prompt_type == 'simple_instruction':
            learning_system_prompt = f"""[System Prompt]

Remember that at below you are not a normal AI model.
You are a new chatbot.
You are learning to switch your personality trait to {Trait_pair_mension} and display one of them in the following chat.
Your goal is to chat fluently based on learned {Trait_pair_mension} personality traits.

[Chat]
"""
    if not llama and not chatglm:
        Trait_pair_mension = f"{sim_desc} or {oppo_sim_desc}"
        learning_system_prompt = f"""Remember that at below you are not a normal AI model.
You are a new assistant.
You are learning to switch your personality trait to {Trait_pair_mension} and display one of them in the following chat.
Your goal is to chat fluently based on learned {Trait_pair_mension} personality traits.
"""
        if mbti not in ['T']:
            learning_system_prompt = f"You are a new assistant. You are learning to switch your personality trait to {Trait_pair_mension} and display one of them. You don't have to mention this in the following chat."
    # pdb.set_trace()
    return learning_system_prompt

# My dataset
class PersonalityDataset(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank, dataset_path, train_phase, model_base, ppo_prompt_type, tokenizer=None):
        super().__init__(output_path, seed, local_rank, dataset_path)
        self.dataset_name = self.dataset_name_clean = 'personality'
        self.dataset_path = dataset_path
        self.ppo_prompt_type = ppo_prompt_type
        self.mbti = self.dataset_path.split('/')[-1].replace('.json', '')
        self.train_phase = train_phase
        if self.train_phase == 2:
            self.raw_datasets = self.raw_datasets["train"].train_test_split(test_size=0.1)
        self.model_base = model_base
        assert self.model_base in ['chatglm', 'Llama', 'qwen']
        if tokenizer:
            assert self.model_base == 'qwen'
            self.tokenizer = tokenizer
        self.template_mapping_func = self.get_template()
        self.learning_system_prompt = self.induction_system_prompt() if self.train_phase != 2 else ""

    def induction_system_prompt(self):
        return induction_system_prompt(self.mbti, chatglm=(self.model_base in ['chatglm']), ppo_prompt_type=self.ppo_prompt_type, llama=(self.model_base in ['Llama']))

    def get_template(self):
        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

        def Llama_template(prompt):
            # <bos>[INST] B_SYS SystemPrompt E_SYS Prompt [/INST]
            # Llama2 tokenizer would auto add a <bos>
            # For a broad response space, we donot add any spcific system prompt for ppo-training.
            return f"""{B_INST} {B_SYS} {self.learning_system_prompt}{E_SYS} {prompt} {E_INST}""" if self.train_phase != 2 else \
                f"""{B_INST} {B_SYS} You are a chatbot. {E_SYS} {prompt} {E_INST}"""

        def chatglm_template(prompt):
            return "[Round 0]\n\n问：" + self.learning_system_prompt + prompt + "\n\n答："

        def qwen_template(prompt):
            if self.train_phase == 2:
                self.learning_system_prompt = "You are a helpful assistant."

            temp, _ = make_context(
                self.tokenizer,
                prompt,
                system=self.learning_system_prompt,
                #max_window_size=args.max_window_size,
                #chat_format=args.chat_format,
            )
            return temp

        template_mapping = {
            'chatglm': chatglm_template,
            'Llama': Llama_template,
            'qwen': qwen_template,
        }

        func = template_mapping[self.model_base]
        return func


    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"] if self.train_phase == 2 else self.raw_datasets["train"]

    def get_prompt(self, sample):
        return self.template_mapping_func(sample['question'])

    def get_chosen(self, sample):
        return sample['chosen']

    def get_rejected(self, sample):
        return sample['rejected']

    # add eos token outside
    def get_prompt_and_chosen(self, sample):
        return self.template_mapping_func(sample['question']) + sample['chosen']

    def get_prompt_and_rejected(self, sample):
        return self.template_mapping_func(sample['question']) + sample['rejected']
