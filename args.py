import sys
import yaml
import builtins
import torch
from urllib.parse import urlparse
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    BitsAndBytesConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
)
from peft import LoraConfig
from dataclasses import dataclass, field
from typing import Union
from utils import find_all_linear_names, get_local_rank
from trl import SFTTrainer
from sagemaker import TrainingInput


def print(*args, **kwargs):
    if get_local_rank() == 0 or kwargs.get("all_ranks", False):
        kwargs.pop("all_ranks", None)
        builtins.print(*args, **kwargs)


@dataclass
class Arguments:
    def config(self):
        return self.__dict__


@dataclass
class TaskArguments(Arguments):
    # should only be train or eval
    project: str = field(default="huggingface", metadata={"help": "Project name"})
    config: list[str] = field(default=None, metadata={"help": "Config section in the config file to use for this task"})
    tmp_dir: str = field(default=".tmp", metadata={"help": "Temporary directory"})
    save_dir: str = field(default=".", metadata={"help": "Save directory"})
    clear_data_cache: bool = field(default=False, metadata={"help": "Clear data cache"})


@dataclass
class HuggingFaceHubArguments(Arguments):
    pretrained_model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={"help": "Whether to trust remote code when loading a pretrained model"},
    )
    revision: Union[str, None] = field(
        default=None,
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )

    def tokenizer(self, **kwargs):
        return AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.pretrained_model_name_or_path,
            trust_remote_code=self.trust_remote_code,
            revision=self.revision,
            **kwargs,
        )

    def model(self, quantization_config: BitsAndBytesConfig = None, device_map=None, **kwargs):
        return AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self.pretrained_model_name_or_path,
            trust_remote_code=self.trust_remote_code,
            revision=self.revision,
            quantization_config=quantization_config,
            device_map=device_map,
            **kwargs,
        )


@dataclass
class InstanceArguments(Arguments):
    instance_type: str = field(default="ml.p4d.24xlarge", metadata={"help": "Instance type used for the task"})
    volume_size: int = field(default=256, metadata={"help": "The size of the EBS volume in GB"})
    instance_count: int = field(default=1, metadata={"help": "The number of instances used for task"})
    max_run: Union[int, None] = field(
        default=None,
        metadata={"help": "Maximum runtime in seconds"},
    )
    max_wait: Union[int, None] = field(
        default=None,
        metadata={"help": "Maximum wait time in seconds"},
    )
    use_spot_instances: bool = field(
        default=False, metadata={"help": "Whether to use spot instances for cheaper training"}
    )
    input: Union[list[str], None] = field(
        default=None,
        metadata={"help": "A list of input data channels"},
    )
    default_channel: str = field(
        default="data",
        metadata={"help": "The default channel name to use if a channel is not specified in an input uri"},
    )

    def inputs(self) -> dict[str, TrainingInput]:
        val = {}
        for uri in self.input:
            # split the url into url params and get the input_mode and channel params
            parsed_url = urlparse(uri)
            params = dict(kv.split("=") for kv in parsed_url.query.split("&"))
            s3_data = parsed_url.geturl().split("?")[0]
            channel = params.get("channel", self.default_channel)
            input_mode = params.get("input_mode", "File")
            val[channel] = TrainingInput(
                s3_data=s3_data,
                input_mode=input_mode,
            )
        return val


@dataclass
class DistributedArguments(Arguments):
    enable_distributed: bool = field(default=False, metadata={"help": "Whether to use distributed training"})
    num_machines: int = field(default=1, metadata={"help": "Number of machines"})
    num_processes: int = field(default=1, metadata={"help": "Number of processes"})


@dataclass
class DeepSpeedArguments(Arguments):
    enable_deepspeed: bool = field(default=False, metadata={"help": "Whether to use DeepSpeed"})
    zero_stage: int = field(default=2, metadata={"help": "DeepSpeed ZeRO stage"})
    offload: bool = field(default=False, metadata={"help": "Whether to offload to CPU"})
    pin_memory: bool = field(
        default=True,
        metadata={
            "help": "Offload to page-locked CPU memory. This could boost throughput at the cost of extra memory overhead."
        },
    )


@dataclass
class LoraArguments(Arguments):
    enable_lora: bool = field(default=False, metadata={"help": "Whether to use LoRA"})
    lora_r: int = field(default=8, metadata={"help": "Lora attention dimension"})
    lora_alpha: int = field(default=8, metadata={"help": "Lora alpha"})
    lora_dropout: float = field(default=0.0, metadata={"help": "Lora dropout"})
    lora_bias: str = field(default="none", metadata={"help": "Bias type for Lora. Can be 'none', 'all' or 'lora_only'"})
    target_modules: list[str] = field(
        default_factory=lambda: [],
        metadata={"help": "Target modules for Lora. If empty, all modules will be targeted."},
    )
    modules_to_save: list[str] = field(
        default_factory=lambda: [],
        metadata={
            "help": "List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint. These typically include model’s custom head that is randomly initialized for the fine-tuning task."
        },
    )
    lora_task_type: str = field(
        default="CAUSAL_LM",
        metadata={
            "help": "Task type for Lora. Can be 'SEQ_CLS', 'SEQ_2_SEQ_LM', 'CAUSAL_LM', 'TOKEN_CLS', 'QUESTION_ANS', 'FEATURE_EXTRACTION'"
        },
    )

    def config(self, model, **kwargs) -> LoraConfig:
        return LoraConfig(
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=self.target_modules or find_all_linear_names(model),
            r=self.lora_r,
            bias=self.lora_bias,
            modules_to_save=self.modules_to_save,
            task_type=self.lora_task_type,
            **kwargs,
        )


@dataclass
class BitsAndBytesArguments(Arguments):
    load_in_8bit: bool = field(default=False, metadata={"help": "Whether to load in 8-bit"})
    load_in_4bit: bool = field(default=False, metadata={"help": "Whether to load in 4-bit"})
    llm_int8_threshold: float = field(
        default=6.0,
        metadata={
            "help": "Any hidden states value that is above this threshold will be considered an outlier and the operation on those values will be done in fp16."
        },
    )
    llm_int8_skip_modules: Union[list[str], None] = field(
        default=None,
        metadata={"help": "An explicit list of the modules that we do not want to convert in 8-bit."},
    )
    llm_int8_enable_fp32_cpu_offload: bool = field(
        default=False,
        metadata={
            "help": "If you want to split your model in different parts and run some parts in int8 on GPU and some parts in fp32 on CPU, you can use this flag. Note that the int8 operations will not be run on CPU."
        },
    )
    llm_int8_has_fp16_weight: bool = field(
        default=False,
        metadata={
            "help": "This is useful for fine-tuning as the weights do not have to be converted back and forth for the backward pass."
        },
    )
    bnb_4bit_compute_dtype: Union[str, None] = field(
        default=None,
        metadata={
            "help": "This sets the computational type which might be different than the input type. For example, inputs might be fp32, but computation can be set to bf16 for speedups."
        },
    )
    bnb_4bit_quant_type: str = field(
        default="fp4",
        metadata={
            "help": "This sets the quantization data type in the bnb.nn.Linear4Bit layers. Options are FP4 and NF4 data types which are specified by `fp4` or `nf4`."
        },
    )
    bnb_4bit_use_double_quant: bool = field(
        default=False,
        metadata={
            "help": "This flag is used for nested quantization where the quantization constants from the first quantization are quantized again."
        },
    )

    def config(self, **kwargs) -> BitsAndBytesConfig:
        return BitsAndBytesConfig(**self.__dict__, **kwargs)

    def __post_init__(self):
        # Check GPU compatibility with bfloat16
        if self.load_in_4bit and self.bnb_4bit_compute_dtype == torch.float16:
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                # yellow
                print("\033[93m", end="")
                print("Your GPU supports bfloat16: accelerate training with `bnb_4bit_compute_dtype=bfloat16`")
                print("\033[0m", end="")


@dataclass
class SFTArguments(Arguments):
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
    task: TaskArguments
    hf: HuggingFaceHubArguments
    instance: InstanceArguments
    dist: DistributedArguments
    trainer: TrainingArguments
    sft: SFTArguments
    ds: DeepSpeedArguments
    lora: LoraArguments
    bnb: BitsAndBytesArguments
    nargs: list

    def __iter__(self):
        return iter(
            [
                self.task,
                self.hf,
                self.instance,
                self.dist,
                self.trainer,
                self.sft,
                self.ds,
                self.lora,
                self.bnb,
                self.nargs,
            ]
        )

    def lora_config(self, model) -> LoraConfig:
        return self.lora.config(model)

    def bnb_config(self) -> BitsAndBytesConfig:
        return self.bnb.config()

    def tokenizer(self, **kwargs) -> AutoTokenizer:
        return self.hf.tokenizer(**kwargs)

    def model(self, **kwargs) -> AutoModelForCausalLM:
        device_map = None

        if self.lora.enable_lora and not self.ds.enable_deepspeed:
            device_map = {"": get_local_rank()}
            print(f"Device map: {device_map}", all_ranks="true")

        return self.hf.model(
            quantization_config=self.bnb.config() if self.bnb.load_in_4bit or self.bnb.load_in_8bit else None,
            device_map=device_map,
            **kwargs,
        )

    def sft_trainer(self, model, tokenizer, train_dataset, **kwargs) -> SFTTrainer:
        if self.lora.enable_lora and self.dist.enable_distributed:
            print("\033[93m", end="")
            print(
                "Automatically setting `ddp_find_unused_parameters=False` when using LoRA with DistributedDataParallel."
            )
            print("\033[0m", end="")
            self.trainer.ddp_find_unused_parameters = False

        return SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            peft_config=self.lora.config(model) if self.lora.enable_lora else None,
            max_seq_length=self.sft.max_seq_length,
            args=self.trainer,
            packing=self.sft.packing,
            **kwargs,
        )

    def __post_init__(self):
        # Check GPU compatibility with bfloat16
        if self.trainer and self.trainer.fp16:
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                print("\033[93m", end="")
                print("Your GPU supports bfloat16: accelerate training with `bf16=True`")
                print("\033[0m", end="")


def object_to_args(obj):
    args = []
    for k, v in obj.items():
        if v is None:
            continue
        # if it is a list add each arg sepatly
        if isinstance(v, list):
            args += [f"--{k}"]
            for val in v:
                args += [f"{val}"]
        elif isinstance(v, bool):
            if v:
                args += [f"--{k}", "True"]
        else:
            args += [f"--{k}", f"{v}"]
    return args


arg_classes = [
    TaskArguments,
    HuggingFaceHubArguments,
    InstanceArguments,
    DistributedArguments,
    TrainingArguments,
    SFTArguments,
    DeepSpeedArguments,
    LoraArguments,
    BitsAndBytesArguments,
]

arg_keywords = {
    "task": TaskArguments,
    "hf": HuggingFaceHubArguments,
    "instance": InstanceArguments,
    "dist": DistributedArguments,
    "trainer": TrainingArguments,
    "sft": SFTArguments,
    "ds": DeepSpeedArguments,
    "lora": LoraArguments,
    "bnb": BitsAndBytesArguments,
    "nargs": list,
}


def parse_args(*classes) -> MachineLearningArguments:
    full_config = yaml.load(open("ml.yaml", "r"), Loader=yaml.FullLoader)

    current_config = {
        # all defaults and any param that isn't an object
        k: v
        for k, v in full_config.items()
        if not isinstance(v, dict)
    }

    if "defaults" in full_config:
        # add defaults to config
        current_config.update(full_config["defaults"])

    if "config" in full_config:
        config_keys = full_config["config"]

        if isinstance(config_keys, list):
            # if `config` is set, add its settings to run config
            for key in config_keys:
                if key in full_config:
                    current_config.update(full_config[key])
        elif isinstance(config_keys, str):
            key = config_keys
            if key in full_config:
                current_config.update(full_config[key])
        else:
            raise ValueError(f"Invalid config key type: {type(config_keys)}: Must be str or list[str]")

    config_args = object_to_args(current_config)

    parser = HfArgumentParser(classes)

    arg_class_objs = parser.parse_args_into_dataclasses(
        args=[*config_args, *sys.argv[1:]], return_remaining_strings=True
    )

    kwargs = {}
    for k in arg_keywords.keys():
        kwargs[k] = None
    for arg_class_obj in arg_class_objs:
        for k, v in arg_keywords.items():
            if isinstance(arg_class_obj, v):
                kwargs[k] = arg_class_obj

    return MachineLearningArguments(**kwargs)


def print_args(args: MachineLearningArguments, unknown=False):
    for arg_class in args:
        if not arg_class:
            continue
        if not isinstance(arg_class, list):
            print(arg_class.__class__.__name__, arg_class.__dict__)
        if isinstance(arg_class, list) and unknown:
            print("UnknownArguments", arg_class)
