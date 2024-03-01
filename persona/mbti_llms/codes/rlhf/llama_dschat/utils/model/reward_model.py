# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import torch
from torch import nn
from ..utils import print_rank_0
import pdb

## Note that the following code is modified from
## https://github.com/CarperAI/trlx/blob/main/examples/summarize_rlhf/reward_model/reward_model.py
class RewardModel(nn.Module):

    def __init__(self,
                 base_model,
                 tokenizer,
                 num_padding_at_beginning=0,
                 compute_fp32_loss=False):
        super().__init__()
        self.config = base_model.config
        self.num_padding_at_beginning = num_padding_at_beginning
        if hasattr(self.config, "word_embed_proj_dim"):
            # `OPT` models use word_embed_proj_dim as final output
            # https://github.com/huggingface/transformers/blob/main/src/transformers/models/opt/modeling_opt.py#L497
            self.v_head = nn.Linear(self.config.word_embed_proj_dim,
                                    1,
                                    bias=False)
        else:
            # `gpt-neo(x)` models use `hidden_size` attribute names instead of `n_embd``
            if base_model.__class__.__name__ == 'MPTForCausalLM':
                setattr(self.config, 'n_embd', self.config.d_model)
            else:
                self.config.n_embd = self.config.hidden_size if hasattr(
                    self.config, "hidden_size") else self.config.n_embd
            self.v_head = nn.Linear(self.config.n_embd, 1, bias=False)
        self.rwtransformer = base_model
        self.PAD_ID = tokenizer.pad_token_id
        self.compute_fp32_loss = compute_fp32_loss
        self.tokenizer = tokenizer

    def gradient_checkpointing_enable(self):
        self.rwtransformer.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.rwtransformer.gradient_checkpointing_disable()

    def forward(self,
                input_ids=None,
                past_key_values=None,
                attention_mask=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                use_cache=False):
        loss = None

        if self.config.model_type == "llama":
            kwargs = dict()
        elif self.config.model_type == 'chatglm':
            kwargs = dict(output_hidden_states=True)
        elif self.config.model_type == "qwen":
            kwargs = dict(output_hidden_states=True)
        else:
            kwargs = dict(head_mask=head_mask)

        transformer_outputs = self.rwtransformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs)

        # pdb.set_trace()

        if self.config.model_type == "llama":
            hidden_states = transformer_outputs[0]
            rewards = self.v_head(hidden_states).squeeze(-1)

        if self.config.model_type == 'chatglm':
            hidden_states = transformer_outputs.hidden_states[-1]
            rewards = self.v_head(hidden_states).squeeze(-1)
            rewards = rewards.permute(1, 0)

        if self.config.model_type == 'qwen':
            hidden_states = transformer_outputs.hidden_states[-1]
            rewards = self.v_head(hidden_states).squeeze(-1)

        chosen_mean_scores = []
        rejected_mean_scores = []

        # Split the inputs and rewards into two parts, chosen and rejected
        assert len(input_ids.shape) == 2
        bs = input_ids.shape[0] // 2
        seq_len = input_ids.shape[1]

        chosen_ids = input_ids[:bs]  # bs x seq x 1
        rejected_ids = input_ids[bs:]
        chosen_rewards = rewards[:bs]
        rejected_rewards = rewards[bs:]

        # Compute pairwise loss. Only backprop on the different tokens before padding
        loss = 0.

        def move_minus(chosen_id, rejected_id):
            start_chosen = (chosen_id != self.PAD_ID).nonzero(as_tuple=False).flatten()[0].item()
            start_rejected = (rejected_id != self.PAD_ID).nonzero(as_tuple=False).flatten()[0].item()
            minus = abs(start_chosen - start_rejected)
            if minus != 0:
                if start_chosen > start_rejected:
                    # pad chosen_id 'minus' padding
                    chosen_id = torch.nn.functional.pad(chosen_id[minus:], (0, minus), mode='constant', value=self.PAD_ID)
                else:
                    rejected_id = torch.nn.functional.pad(rejected_id[minus:], (0, minus), mode='constant', value=self.PAD_ID)
            return chosen_id, rejected_id, minus

        def get_ind(token_id):
            c_inds = (token_id == self.PAD_ID).nonzero()
            c_ind = c_inds[self.num_padding_at_beginning].item() if len(
                c_inds
            ) > self.num_padding_at_beginning else len(token_id)
            return c_ind

        for i in range(bs):
            chosen_id = chosen_ids[i]
            rejected_id = rejected_ids[i]
            chosen_reward = chosen_rewards[i]
            rejected_reward = rejected_rewards[i]

            if self.config.model_type == 'chatglm':
                # 这里需要找到真正token的差别，除了pad差异
                chosen_id, rejected_id, minus = move_minus(chosen_id, rejected_id)
                diff_indices = (chosen_id != rejected_id).nonzero(as_tuple=True)[0]
                # pdb.set_trace()
                if diff_indices.size(0) > 0:  # If there are differences
                    divergence_ind = diff_indices[0]
                else:
                    divergence_ind = seq_len - 1
                #import pdb
                #pdb.set_trace()
                c_truncated_reward = chosen_reward[divergence_ind:]
                r_truncated_reward = rejected_reward[divergence_ind:]
                # chosen and rejected should be difference. c_ind and r_ind
                c_ind = get_ind(chosen_id[divergence_ind:]) + divergence_ind
                r_ind = get_ind(rejected_id[divergence_ind:]) + divergence_ind
                chosen_mean_scores.append(chosen_reward[c_ind - 1])  #use the end score for reference
                rejected_mean_scores.append(rejected_reward[r_ind - 1])
            else:
                c_inds = (chosen_id == self.PAD_ID).nonzero()
                c_ind = c_inds[self.num_padding_at_beginning].item() if len(
                    c_inds
                ) > self.num_padding_at_beginning else seq_len  # OPT model pads the first token, so we need to use the second padding token as the end of the sequence
                check_divergence = (chosen_id != rejected_id).nonzero()

                if len(check_divergence) == 0:
                    end_ind = rejected_reward.size(-1)
                    divergence_ind = end_ind - 1
                    r_ind = c_ind
                else:
                    # Check if there is any padding otherwise take length of sequence
                    r_inds = (rejected_id == self.PAD_ID).nonzero()
                    r_ind = r_inds[self.num_padding_at_beginning].item(
                    ) if len(r_inds) > self.num_padding_at_beginning else seq_len
                    end_ind = max(c_ind, r_ind)
                    divergence_ind = check_divergence[0]
                assert divergence_ind > 0
                c_truncated_reward = chosen_reward[divergence_ind:end_ind]
                r_truncated_reward = rejected_reward[divergence_ind:end_ind]

                # pdb.set_trace()

                chosen_mean_scores.append(chosen_reward[c_ind - 1])  #use the end score for reference
                rejected_mean_scores.append(rejected_reward[r_ind - 1])

            if self.compute_fp32_loss:
                c_truncated_reward = c_truncated_reward.float()
                r_truncated_reward = r_truncated_reward.float()
            loss += -torch.nn.functional.logsigmoid(c_truncated_reward - r_truncated_reward).mean()

        loss = loss / bs
        chosen_mean_scores = torch.stack(chosen_mean_scores)
        rejected_mean_scores = torch.stack(rejected_mean_scores)
        return {
            "loss": loss,
            "chosen_mean_scores": chosen_mean_scores,
            "rejected_mean_scores": rejected_mean_scores,
        }

    def forward_value(self,
                      input_ids=None,
                      attention_mask=None,
                      past_key_values=None,
                      position_ids=None,
                      head_mask=None,
                      inputs_embeds=None,
                      return_value_only=False,
                      prompt_length=0,
                      use_cache=False):

        if self.config.model_type == "llama":
            kwargs = dict()
        elif self.config.model_type == 'chatglm':
            kwargs = dict(output_hidden_states=True)
        elif self.config.model_type == "qwen":
            kwargs = dict(output_hidden_states=True)
        else:
            kwargs = dict(head_mask=head_mask)

        # import pdb
        # pdb.set_trace()

        transformer_outputs = self.rwtransformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs)

        if self.config.model_type == "llama":
            hidden_states = transformer_outputs[0]
            values = self.v_head(hidden_states).squeeze(-1)

        if self.config.model_type == 'chatglm':
            hidden_states = transformer_outputs.hidden_states[-1]
            values = self.v_head(hidden_states).squeeze(-1)
            values = values.permute(1, 0)

        if self.config.model_type == 'qwen':
            hidden_states = transformer_outputs.hidden_states[-1]
            values = self.v_head(hidden_states).squeeze(-1)

        if return_value_only:
            return values
        else:
            # [0 0 0 0 prompt, answer, 0 0 0 0 ] for step 3, we have padding at the beginning
            # [prompt, answer, 0, 0, 0, 0] this is normal
            assert prompt_length > 1, "prompt_length must be greater than 1 to help select the end score"
            bs = values.size(0)
            seq_len = input_ids.shape[1]
            chosen_end_scores = [
            ]  # we use this name for consistency with the original forward function
            debug_value_input_id_list = []
            for i in range(bs):
                input_id = input_ids[i]
                value = values[i]

                c_inds = (input_id[prompt_length:] == self.PAD_ID).nonzero() # padding of answer
                # here we only use the answer part of the sequence so we do not need to care about the padding at the beginning
                c_ind = c_inds[0].item() + prompt_length if len(
                    c_inds) > 0 else seq_len

                # pdb.set_trace()

                chosen_end_scores.append(value[c_ind - 1])
                debug_value_input_id_list.append((input_id[c_ind - 1], value[c_ind - 1]))

            return {
                "values": values,
                "chosen_end_scores": torch.stack(chosen_end_scores),
            }