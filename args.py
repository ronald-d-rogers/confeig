import sys
import yaml
from transformers import (
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
)

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
class TrlArguments:
    max_seq_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length to use for training"},
    )
    packing: bool = field(
        default=False,
        metadata={"help": "Whether to pack the sequence length to the batch dimension"},
    )


@dataclass
class MachineLearningArguments:
    task: TaskArguments = field(default_factory=TaskArguments)
    hf: HuggingFaceHubArguments = field(default_factory=HuggingFaceHubArguments)
    dist: DistributedArguments = field(default_factory=DistributedArguments)
    trainer: TrainingArguments = field(default_factory=TrainingArguments)
    trl: TrlArguments = field(default_factory=TrlArguments)
    ds: DeepSpeedArguments = field(default_factory=DeepSpeedArguments)
    lora: LoraArguments = field(default_factory=LoraArguments)
    bnb: BitsAndBytesConfig = field(default_factory=BitsAndBytesConfig)
    nargs: list = field(default_factory=list)


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
            TrlArguments,
            DeepSpeedArguments,
            LoraArguments,
            BitsAndBytesConfig,
        )
    )

    task, hf, trainer, dist, trl, ds, lora, bnb, nargs = parser.parse_args_into_dataclasses(
        args=[*default_args, *sys.argv[1:]],
        return_remaining_strings=True,
    )

    return MachineLearningArguments(task, hf, trainer, dist, trl, ds, lora, bnb, nargs)
