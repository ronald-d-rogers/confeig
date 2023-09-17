import sys
import yaml
from transformers import (
    HfArgumentParser,
    TrainingArguments,
)

from typing import List, Union, NamedTuple

from dataclasses import dataclass, field


@dataclass
class TaskArguments:
    # should only be train or eval
    task: str = field(default="train", metadata={"help": "Task name"})
    clear_data_cache: bool = field(default=False, metadata={"help": "Clear data cache"})
    tmp_dir: str = field(default=".tmp", metadata={"help": "Temporary directory"})


@dataclass
class HuggingFaceHubArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={"help": "Whether to trust remote code when loading a pretrained model"},
    )


@dataclass
class DistributedArguments:
    enable_distributed: bool = field(default=False, metadata={"help": "Whether to use distributed training"})
    num_machines: int = field(default=1, metadata={"help": "Number of machines"})
    num_processes: int = field(default=1, metadata={"help": "Number of processes"})


@dataclass
class DeepSpeedArguments:
    enable_deepspeed: bool = field(default=False, metadata={"help": "Whether to use DeepSpeed"})
    stage: int = field(default=2, metadata={"help": "DeepSpeed ZeRO stage"})
    offload: bool = field(default=False, metadata={"help": "Whether to offload to CPU"})
    pin_memory: bool = field(
        default=True,
        metadata={
            "help": "Offload to page-locked CPU memory. This could boost throughput at the cost of extra memory overhead."
        },
    )


@dataclass
class LoraArguments:
    enable_lora: bool = field(default=False, metadata={"help": "Whether to use LoRA"})
    lora_r: int = field(default=8, metadata={"help": "Lora attention dimension"})
    lora_alpha: int = field(default=8, metadata={"help": "Lora alpha"})
    lora_dropout: float = field(default=0.0, metadata={"help": "Lora dropout"})
    lora_bias: str = field(default="none", metadata={"help": "Bias type for Lora. Can be 'none', 'all' or 'lora_only'"})


@dataclass
class BitsAndBytesArguments:
    # load_in_8bit=False,
    load_in_8bit: bool = field(default=False, metadata={"help": "Whether to load in 8-bit"})

    # load_in_4bit=False,
    load_in_4bit: bool = field(default=False, metadata={"help": "Whether to load in 4-bit"})

    # llm_int8_threshold=6.0,
    llm_int8_threshold: float = field(
        default=6.0,
        metadata={
            "help": "Any hidden states value that is above this threshold will be considered an outlier and the operation on those values will be done in fp16."
        },
    )

    llm_int8_skip_modules: Union[List[str], None] = field(
        default=None,
        metadata={"help": "An explicit list of the modules that we do not want to convert in 8-bit."},
    )

    # llm_int8_enable_fp32_cpu_offload=False,
    llm_int8_enable_fp32_cpu_offload: bool = field(
        default=False,
        metadata={
            "help": "If you want to split your model in different parts and run some parts in int8 on GPU and some parts in fp32 on CPU, you can use this flag. Note that the int8 operations will not be run on CPU."
        },
    )

    # llm_int8_has_fp16_weight=False,
    llm_int8_has_fp16_weight: bool = field(
        default=False,
        metadata={
            "help": "This is useful for fine-tuning as the weights do not have to be converted back and forth for the backward pass."
        },
    )

    # bnb_4bit_compute_dtype=None,
    bnb_4bit_compute_dtype: Union[str, None] = field(
        default=None,
        metadata={
            "help": "This sets the computational type which might be different than the input type. For example, inputs might be fp32, but computation can be set to bf16 for speedups."
        },
    )

    # bnb_4bit_quant_type="fp4",
    bnb_4bit_quant_type: str = field(
        default="fp4",
        metadata={
            "help": "This sets the quantization data type in the bnb.nn.Linear4Bit layers. Options are FP4 and NF4 data types which are specified by `fp4` or `nf4`."
        },
    )

    # bnb_4bit_use_double_quant=False,
    bnb_4bit_use_double_quant: bool = field(
        default=False,
        metadata={
            "help": "This flag is used for nested quantization where the quantization constants from the first quantization are quantized again."
        },
    )


@dataclass
class SFTArguments:
    max_seq_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length to use for training"},
    )
    packing: bool = field(
        default=False,
        metadata={"help": "Whether to pack the sequence length to the batch dimension"},
    )


class MachineLearningArguments(NamedTuple):
    task: TaskArguments
    hf: HuggingFaceHubArguments
    dist: DistributedArguments
    trainer: TrainingArguments
    sft: SFTArguments
    ds: DeepSpeedArguments
    lora: LoraArguments
    bnb: BitsAndBytesArguments
    nargs: list


def parse_args() -> MachineLearningArguments:
    config = yaml.load(open("ml.yaml", "r"), Loader=yaml.FullLoader)

    default_args = [
        *["--quant_method", "bitsandbytes"],
        *["--output_dir", "./results"],
    ]

    for k, v in config.items():
        if v is None:
            continue
        default_args += [f"--{k}", f"{v}"]

    parser = HfArgumentParser(
        (
            TaskArguments,
            HuggingFaceHubArguments,
            DistributedArguments,
            TrainingArguments,
            SFTArguments,
            DeepSpeedArguments,
            LoraArguments,
            BitsAndBytesArguments,
        )
    )

    task, hf, trainer, dist, trl, ds, lora, bnb, nargs = parser.parse_args_into_dataclasses(
        args=[*default_args, *sys.argv[1:]],
        return_remaining_strings=True,
    )

    return MachineLearningArguments(task, hf, trainer, dist, trl, ds, lora, bnb, nargs)


def print_args(args: MachineLearningArguments, print=print):
    for arg_group in args:
        if not isinstance(arg_group, list):
            print(arg_group.__class__.__name__, arg_group.__dict__)
        else:
            print("UnknownArguments", arg_group)
