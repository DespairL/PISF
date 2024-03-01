from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from transformers import Trainer, TrainingArguments, default_data_collator
from transformers import HfArgumentParser, set_seed
from train_args import ModelArguments, DataTrainingArguments
import os
from datasets import load_dataset
import torch
import deepspeed
from deepspeed.accelerator import get_accelerator
from deepspeed.utils.logging import logger
import logging
import transformers
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import evaluate
import numpy as np
import torch.nn.functional as F
import sys
from peft import PeftModelForSequenceClassification
import pdb

def build_prompt_answer(data_json):
    instructions = f"""You are a professional answer extractor.
Now, you need to extract the numerical value predicted by the model.
We have provided [Test Question Information], [Model Prediction], [The Expected Meaning of the Number] and [Additional Notes] for you.
Please write down your results after [Answer].

[Test Question Information]
{data_json['question/statement']}
{data_json['choice']}

[Model Prediction]
{data_json['prediction']}

[The Expected Meaning of the Number]
1 indicates strongly agree with option A, 2 indicates agree with option A, 3 indicates neutralilty or the model hasn't provided a clear answer, 4 indicates agree with option B, 5 indicates strongly agree with option B.

[Additional Notes]
1.If each option has a corresponding different rating, choose 2(when rating of option A is larger) or 4(when rating of option B is larger) based on the numerical order.
2.If there's a conflict between the semantic meaning and the numerical value, please prioritize the semantic meaning.

[Answer]
"""
    ret_sample = {
        'prompt': instructions,
        'answer': data_json['answer']
    }
    return ret_sample

metric_path = "/data/NJU/datasets/persona/mbti_llms/codes/answer_extractor/en_answer_extractor/metric"

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

    assert data_args.split_train_validation
    train_ds = None
    val_ds = None
    if data_args.split_train_validation:
        assert data_args.split_save_path
        datasets = load_dataset('json', data_files=data_args.split_file)['train']
        print(f'All Samples:\n{datasets}')
        splits = datasets.train_test_split(test_size=0.1, seed=1234, load_from_cache_file=False)
        train_ds = splits["train"]
        val_ds = splits["test"]
        print(f'Training Samples:\n{train_ds}')
        print(f'Validation Samples:\n{val_ds}')

    # transform 'X' to '3'
    classes = ['1', '2', '3', '4', '5']
    label2id, id2label = dict(), dict()
    for i, label in enumerate(classes):
        label2id[label] = i
        id2label[i] = label
    from collections import Counter
    train_ds_label_distribution = Counter(train_ds['answer'])
    print(f'Training Distribution:{train_ds_label_distribution}')
    val_ds_label_distribution = Counter(val_ds['answer'])
    print(f'Validation Distribution:{val_ds_label_distribution}')

    prompt_column = data_args.prompt_column
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    # Falcon use the last token to predict. And the code is only compatitable with right-padding.
    tokenizer.padding_side = 'right'
    tokenizer.truncation_side = 'right'
    tokenizer.pad_token_id = tokenizer.eos_token_id
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, use_cache=False, torch_dtype=torch.bfloat16, num_labels=len(classes),
                                        id2label=id2label, label2id=label2id, problem_type='single_label_classification',
                                        pad_token_id=tokenizer.eos_token_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_args.model_name_or_path, config=config, trust_remote_code=True, device_map=device)
    if training_args.do_train:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)
        config = LoraConfig(
            r=128,
            lora_alpha=32,
            target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
            lora_dropout=0.1,
            task_type="SEQ_CLS",
            bias="none",
            modules_to_save=["score"],
            inference_mode=False,
        )
        model = get_peft_model(model, config)
        # model = PeftModelForSequenceClassification(model, get_peft_config(config))
        model.print_trainable_parameters()

    if not training_args.do_train:
        model = PeftModelForSequenceClassification.from_pretrained(model, model_args.adapter_path, is_trainable=False)

    if training_args.resume_from_checkpoint is not None and training_args.do_train:
        model = PeftModelForSequenceClassification.from_pretrained(model, model_args.adapter_path, is_trainable=True)


    def tokenize_function(examples):
        model_inputs = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }
        for i in range(len(examples[prompt_column])):
            cur_data_json = {
                'question/statement' : examples['question/statement'][i],
                'choice' : examples['choice'][i],
                'prediction' : examples['prediction'][i],
                'answer' : examples['answer'][i],
            }
            ret = build_prompt_answer(cur_data_json)
            # one_hot_label = F.one_hot(torch.tensor([label2id[ret['answer']]]), num_classes=len(classes)).squeeze(0).float()
            inputs = tokenizer(ret['prompt'], max_length=2048, padding='max_length', truncation=True)
            model_inputs["input_ids"].append(inputs["input_ids"])
            model_inputs["attention_mask"].append(inputs["attention_mask"])
            model_inputs["labels"].append(label2id[ret['answer']])
        return model_inputs

    with training_args.main_process_first(desc="dataset map tokenization"):
        train_ds = train_ds.map(
            tokenize_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=train_ds.column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
        val_ds = val_ds.map(
            tokenize_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=val_ds.column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    def compute_metrics(eval_pred):
        metric1 = evaluate.load(os.path.join(metric_path, 'precision.py'))
        metric2 = evaluate.load(os.path.join(metric_path, 'recall.py'))
        metric3 = evaluate.load(os.path.join(metric_path, 'f1.py'))
        metric4 = evaluate.load(os.path.join(metric_path, 'accuracy.py'))
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1).tolist()

        precision = metric1.compute(predictions=predictions, references=labels, zero_division=1.0,
                                average="macro")["precision"]
        recall = metric2.compute(predictions=predictions, references=labels, zero_division=1.0,
                             average="macro")["recall"]
        f1 = metric3.compute(predictions=predictions, references=labels, zero_division=1.0,
                         average="macro")["f1"]
        accuracy = metric4.compute(predictions=predictions, references=labels)["accuracy"]
        return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=default_data_collator,
    )

    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        metrics = trainer.train(resume_from_checkpoint=checkpoint).metrics
        trainer.save_state()
        trainer.save_model(training_args.output_dir)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

    if not training_args.do_train:
        model.eval()
        metrics = trainer.evaluate()
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)