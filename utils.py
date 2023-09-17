import os
import json
import yaml
from functools import partial
import torch
from transformers import AutoTokenizer
import bitsandbytes as bnb
from transformers import AutoTokenizer
import deepspeed.comm as dist

from typing import Union

import hashlib


def create_prompt_formats(sample):
    """
    Format various fields of the sample ('instruction', 'context', 'response')
    Then concatenate them using two newline characters
    :param sample: Sample dictionnary
    """
    START_TEXT = "<START_TEXT>"
    END_TEXT = "<END_TEXT>"
    START_REPR = "<START_REPR>"
    END_REPR = "<END_REPR>"

    QUESTION_KEY = "### Instruction:\nYou are a math assistant. Solve the following problem.\n### Problem:"

    question = f"{START_TEXT}\n{QUESTION_KEY}\n{sample['question']}\n{END_TEXT}"
    answer = f"{START_REPR}\nLet's think step by step,\n{sample['answer'].replace('####', '### Answer:')}\n{END_REPR}"

    parts = [part for part in [question, answer] if part]

    formatted_prompt = "\n".join(parts)

    sample["text"] = formatted_prompt

    return sample


def preprocess_batch(batch, tokenizer, max_length):
    """
    Tokenizing a batch
    """
    return tokenizer(
        batch["text"],
        max_length=max_length,
        truncation=True,
    )


# SOURCE https://github.com/databrickslabs/dolly/blob/master/training/trainer.py
def preprocess_dataset(tokenizer: AutoTokenizer, dataset, max_seq_length=2048, seed=42):
    """Format & tokenize it so it is ready for training
    :param tokenizer (AutoTokenizer): Model Tokenizer
    :param max_length (int): Maximum number of tokens to emit from tokenizer
    """

    # Add prompt to each sample
    print("Preprocessing dataset...")
    dataset = dataset.map(create_prompt_formats)  # , batched=True)

    # Apply preprocessing to each batch of the dataset & and remove 'instruction', 'context', 'response', 'category' fields
    _preprocessing_function = partial(preprocess_batch, max_length=max_seq_length, tokenizer=tokenizer)

    dataset = dataset.map(
        _preprocessing_function,
        batched=True,
        remove_columns=["question", "answer"],
    )

    # Filter out samples that have input_ids exceeding max_length
    dataset = dataset.filter(lambda sample: len(sample["input_ids"]) < max_seq_length)

    report = {
        "max_seq_length": max([len(sample["input_ids"]) for sample in dataset]),
    }

    # Shuffle dataset
    dataset = dataset.shuffle(seed=seed)

    return dataset, report


# SOURCE https://github.com/artidoro/qlora/blob/main/qlora.py
def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit  # if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def get_world_size():
    return int(os.environ.get("WORLD_SIZE", 1))


def get_local_rank():
    return int(os.environ.get("LOCAL_RANK", 0))


# SOURCE https://github.com/microsoft/DeepSpeedExamples/blob/bae2afb8417697407ffe7cf6a21388a840679059/applications/DeepSpeed-Chat/training/utils/ds_utils.py
def get_train_ds_config(
    stage=2,
    offload=False,
    enable_hybrid_engine=False,
    inference_tp_size=1,
    release_inference_cache=False,
    pin_parameters=True,
    pin_memory=True,
    tp_gather_partition_size=8,
    max_out_tokens=512,
    enable_mixed_precision_lora=False,
):
    device = "cpu" if offload else "none"
    zero_opt_dict = {
        "stage": stage,
        "offload_param": {"device": device, "pin_memory": pin_memory},
        "offload_optimizer": {"device": device, "pin_memory": pin_memory},
        "stage3_param_persistence_threshold": 1e4,
        "stage3_max_live_parameters": 3e7,
        "stage3_prefetch_bucket_size": 3e7,
        "memory_efficient_linear": False,
    }
    if enable_mixed_precision_lora:
        zero_opt_dict["zero_quantized_nontrainable_weights"] = True
        if dist.get_world_size() != torch.cuda.device_count():
            zero_opt_dict["zero_hpz_partition_size"] = torch.cuda.device_count()
    return {
        "train_batch_size": "auto",
        "gradient_accumulation_steps": "auto",
        "zero_optimization": zero_opt_dict,
        "fp16": {"enabled": "auto", "loss_scale_window": 100},
        "bf16": {"enabled": "auto"},
        "gradient_clipping": "auto",
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
        "hybrid_engine": {
            "enabled": enable_hybrid_engine,
            "max_out_tokens": max_out_tokens,
            "inference_tp_size": inference_tp_size,
            "release_inference_cache": release_inference_cache,
            "pin_parameters": pin_parameters,
            "tp_gather_partition_size": tp_gather_partition_size,
        },
    }


def get_eval_ds_config(offload, pin_memory=True, stage=0):
    device = "cpu" if offload else "none"
    zero_opt_dict = {
        "stage": stage,
        "stage3_param_persistence_threshold": 1e4,
        "offload_param": {"device": device, "pin_memory": pin_memory},
        "memory_efficient_linear": False,
    }
    return {
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "zero_optimization": zero_opt_dict,
        "fp16": {"enabled": "auto", "loss_scale_window": 100},
        "gradient_clipping": "auto",
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
    }


def get_config_md5(dict):
    return hashlib.md5(json.dumps(dict).encode()).hexdigest()


def save_temp_config(config, config_type, tmp_dir=".tmp", format="json"):
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    # get md5 hash of ds_config
    config_hash = get_config_md5(config)

    # get first 8 characters of hash
    config_hash = config_hash[:8]
    config_path = os.path.join(tmp_dir, f"{config_type}.{config_hash}.{format}")

    print(f"Saving {config_type} config to {config_path}")

    with open(config_path, "w") as fp:
        if format == "json":
            json.dump(config, fp)
        elif format == "yaml":
            yaml.dump(config, fp)

    return config_path


def get_accelerate_config(
    task: str = "train",
    distributed: bool = True,
    deepspeed: bool = False,
    stage: int = 2,
    offload: bool = False,
    pin_memory: bool = True,
    mixed_precision: str = "no",
    num_machines: int = 1,
    num_processes: int = 1,
    tmp_dir: Union[str, None] = ".tmp",
):
    distributed_type = "DEEPSPEED" if deepspeed else "MULTI_GPU" if distributed else "NO"

    deepspeed_config = {}

    if deepspeed:
        if task == "eval":
            ds_config = get_eval_ds_config(offload=offload, pin_memory=pin_memory, stage=stage)
        else:
            ds_config = get_train_ds_config(offload=offload, pin_memory=pin_memory, stage=stage)

        ds_config_path = save_temp_config(ds_config, config_type="deepspeed", tmp_dir=tmp_dir, format="json")

        deepspeed_config = {
            "deepspeed_config_file": ds_config_path,
            "zero_init_flag": True,
        }

    return {
        "compute_environment": "LOCAL_MACHINE",
        "debug": False,
        "distributed_type": distributed_type,
        "downcast_bf16": "no",
        "gpu_ids": "all",
        "machine_rank": 0,
        "main_training_function": "main",
        "mixed_precision": mixed_precision,
        "num_machines": num_machines,
        "num_processes": num_processes,
        "rdzv_backend": "static",
        "same_network": True,
        "tpu_env": [],
        "tpu_use_cluster": False,
        "tpu_use_sudo": False,
        "use_cpu": offload,
        "deepspeed_config": deepspeed_config,
    }
