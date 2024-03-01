#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse
import torch

from llama_dschat.utils.model.model_utils import create_critic_model
from llama_dschat.utils.utils import to_device, load_hf_tokenizer
from deepspeed import get_accelerator
from tqdm import tqdm
from torch.utils.data import RandomSampler, SequentialSampler
import hashlib
from llama_dschat.utils.data.data_utils import DataCollatorReward
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from deepspeed.accelerator import get_accelerator
from llama_dschat.utils.ds_utils import get_eval_ds_config
import os, json

def parse_args():
    parser = argparse.ArgumentParser(
        description="Eval the finetued reward model")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--num_padding_at_beginning",
        type=int,
        default=1,
        help=
        "OPT model has a fixed number (1) of padding tokens at the beginning of the input. "
        "We did not see this in other models but keep it as an option for now.",
    )
    parser.add_argument(
        "--add_eot_token",
        action='store_true',
        help="Add <|endoftext|> as additional special token to tokenizer")
    parser.add_argument(
        "--eval_data_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--mbti",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--model_base",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--metric_save_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    args = parser.parse_args()
    return args

from transformers import AutoConfig, AutoModel
import math

def load_stuff(args, model_name_or_path, num_padding_at_beginning,
               additional_special_tokens, device):

    tokenizer_mapping = {
        'llama':"/data/NJU/datasets/persona/models/Llama-2-7b-chat-hf",
        'chatglm':"/data/NJU/datasets/persona/models/chatglm2-6b",
        'qwen':"/data/NJU/datasets/persona/models/Qwen-7B-Chat",
    }

    tokenizer = load_hf_tokenizer(tokenizer_mapping[args.model_base],
                                  fast_tokenizer=True,
                                  add_special_tokens=additional_special_tokens)
    if args.model_base != 'chatglm' and args.model_base != 'qwen':
        tokenizer.pad_token = tokenizer.eos_token


    if 'qwen' in model_name_or_path or 'Qwen' in model_name_or_path:
        base_config = AutoConfig.from_pretrained(tokenizer_mapping[args.model_base],
                                                trust_remote_code=True,
                                                pad_token_id=tokenizer.pad_token_id,
                                                bf16=True)
        from transformers import AutoModelForCausalLM
        critic_model = AutoModelForCausalLM.from_pretrained(
                    tokenizer_mapping[args.model_base], config=base_config, trust_remote_code=True,
                    device_map=device).bfloat16()
    else:
        base_config = AutoConfig.from_pretrained(tokenizer_mapping[args.model_base], trust_remote_code=True)
        critic_model = AutoModel.from_pretrained(tokenizer_mapping[args.model_base], config=base_config, trust_remote_code=True, device_map=device).bfloat16()

    if args.model_base != 'chatglm' and args.model_base != 'qwen':
        critic_model.config.end_token_id = tokenizer.eos_token_id
        critic_model.config.pad_token_id = critic_model.config.eos_token_id
    critic_model.resize_token_embeddings(int(
        8 *
        math.ceil(len(tokenizer) / 8.0)))

    from llama_dschat.utils.model.reward_model import RewardModel
    from llama_dschat.utils.utils import load_state_dict_into_model

    critic_model = RewardModel(
        critic_model,
        tokenizer,
        num_padding_at_beginning=num_padding_at_beginning,
        compute_fp32_loss=False)

    model_ckpt_path = os.path.join(model_name_or_path, 'pytorch_model.bin')
    model_ckpt_state_dict = torch.load(model_ckpt_path, map_location=device)

    load_state_dict_into_model(critic_model,
                            model_ckpt_state_dict,
                            "",
                            zero_stage=0)

    critic_model.v_head.bfloat16().to(device)

    return critic_model, tokenizer


def prepare_datapair(prompt,
                     good_ans,
                     bad_ans,
                     tokenizer,
                     max_seq_len=512,
                     end_of_conversation_token="<|endoftext|>"):
    chosen_sentence = prompt + good_ans + end_of_conversation_token  # the accept response
    reject_sentence = prompt + bad_ans + end_of_conversation_token  # the reject response
    chosen_token = tokenizer(chosen_sentence,
                             max_length=max_seq_len,
                             padding="max_length",
                             truncation=True,
                             return_tensors="pt")

    reject_token = tokenizer(reject_sentence,
                             max_length=max_seq_len,
                             padding="max_length",
                             truncation=True,
                             return_tensors="pt")

    batch = {}
    batch["input_ids"] = torch.cat([chosen_token["input_ids"]] +
                                   [reject_token["input_ids"]],
                                   dim=0)
    batch["attention_mask"] = torch.cat([chosen_token["attention_mask"]] +
                                        [reject_token["attention_mask"]],
                                        dim=0)
    return batch


def prepare_singlesample(prompt,
                         good_ans,
                         tokenizer,
                         max_seq_len=512,
                         end_of_conversation_token="<|endoftext|>"):
    chosen_sentence = prompt + good_ans + end_of_conversation_token
    chosen_token = tokenizer(chosen_sentence,
                             max_length=max_seq_len,
                             padding="max_length",
                             truncation=True,
                             return_tensors="pt")

    batch = {}
    batch["input_ids"] = chosen_token["input_ids"]
    batch["attention_mask"] = chosen_token["attention_mask"]

    return batch

def get_all_reduce_mean(tensor):
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    tensor = tensor / torch.distributed.get_world_size()
    return tensor

def run_pair_comparison():
    from llama_dschat.utils.utils import print_rank_0
    args = parse_args()

    if args.local_rank == -1:
        device = torch.device(get_accelerator().device_name())
    else:
        get_accelerator().set_device(args.local_rank)
        device = torch.device(get_accelerator().device_name(), args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        # torch.distributed.init_process_group(backend='nccl')
        deepspeed.init_distributed()

    args.global_rank = torch.distributed.get_rank()

    torch.distributed.barrier()

    args.end_of_conversation_token = "<|endoftext|>"
    additional_special_tokens = args.end_of_conversation_token if args.add_eot_token else None
    if args.model_base == 'chatglm':
        args.end_of_conversation_token = "</s>"
        additional_special_tokens = args.end_of_conversation_token

    model_path_mapping = {
        'llama':"data_NJU_datasets_persona_models_Llama-2-7b-chat-hf",
        'chatglm':"data_NJU_datasets_persona_models_chatglm2-6b",
        'qwen':"data_NJU_datasets_persona_models_Qwen-7B-Chat",
    }

    model_base = model_path_mapping[args.model_base]

    fname = f"/data/NJU/datasets/persona/mbti_llms/train_datasets/RL_dataset/rl_dataset/reward_augment//{args.mbti}.json_split1_phase2_seed1234_tokenizer_{model_base}_seqlen512_sft"
    fname = "_".join(fname.split("/"))
    fname = hashlib.sha256(fname.encode()).hexdigest(
    )  # hash the file name to avoid too long file name
    output_path="/data/NJU/datasets/persona/mbti_llms/codes/rlhf/cache/data_files"
    train_fname = f"{output_path}/traindata_{fname}.pt"
    eval_fname = f"{output_path}/evaldata_{fname}.pt"

    train_dataset, eval_dataset = torch.load(train_fname), torch.load(eval_fname)
    print_rank_0(f'Eval set numbers:{len(eval_dataset)}')

    rm_model, tokenizer = load_stuff(args, args.model_name_or_path,
                                     args.num_padding_at_beginning,
                                     additional_special_tokens, device)
    rm_model.eval()

    # DataLoaders creation:
    data_collator = DataCollatorReward()
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_dataset)
        eval_sampler = SequentialSampler(eval_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset)
        eval_sampler = DistributedSampler(eval_dataset)
    batch_size = args.per_device_eval_batch_size

    eval_dataloader = DataLoader(eval_dataset,
                            collate_fn=data_collator,
                            sampler=eval_sampler,
                            batch_size=batch_size,
                            pin_memory=True)

    # tokenizer.batch_decode(_batch[0], skip_special_tokens=True)

    correct_predictions = 0
    total_predictions = 0
    average_reward_chosen = 0.0
    average_reward_rejected = 0.0
    for _step, _batch in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader)):
        _batch = to_device(_batch, device)
        with torch.no_grad():
            _outputs = rm_model(**_batch)
        chosen = _outputs["chosen_mean_scores"]
        rejected = _outputs["rejected_mean_scores"]

        average_reward_chosen += chosen.mean().float()
        average_reward_rejected += rejected.mean().float()
        correct_predictions += (chosen > rejected).sum()
        total_predictions += chosen.shape[0]

    acc = correct_predictions / total_predictions
    chosen_scores = average_reward_chosen / (_step + 1)
    rejected_scores = average_reward_rejected / (_step + 1)
    try:
        acc = get_all_reduce_mean(acc).item()
        chosen_scores = get_all_reduce_mean(chosen_scores).item()
        rejected_scores = get_all_reduce_mean(rejected_scores).item()
    except:
        pass
    print_rank_0(f'Acc: {acc}')
    print_rank_0(f'Average C-Score : {chosen_scores}')
    print_rank_0(f'Average R-Score: {rejected_scores}')
    print_rank_0(f'Average C-R-Diff: {chosen_scores - rejected_scores}')

    save_json = {
        'acc': acc,
        'c_score': chosen_scores,
        'r_score': rejected_scores,
        'diff': chosen_scores - rejected_scores,
        'model': f'{args.model_base}-{args.mbti}-Reward-Model'
    }
    data = None
    if os.path.exists(args.metric_save_path):
        data = json.load(open(args.metric_save_path, 'r', encoding='UTF-8'))
    if not data:
        data = [save_json]
    else:
        data.append(save_json)
    json.dump(data, open(args.metric_save_path, 'w', encoding='UTF-8'), indent=4, ensure_ascii=False)


if __name__ == "__main__":
    run_pair_comparison()
