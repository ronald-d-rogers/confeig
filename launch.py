import os

os.system("pip install flash-attn --no-build-isolation --upgrade")
import sys

from transformers import TrainingArguments
from args import (
    parse_args,
    print_args,
    TaskArguments,
    InstanceArguments,
    DistributedArguments,
    DeepSpeedArguments,
)
from config import get_accelerate_config, save_temp_config


def main():
    # Parse args
    args = parse_args(TaskArguments, InstanceArguments, DistributedArguments, DeepSpeedArguments, TrainingArguments)

    print_args(args)

    accelerate_config = get_accelerate_config(
        task=args.task.task,
        distributed=args.dist.enable_distributed,
        deepspeed=args.ds.enable_deepspeed,
        stage=args.ds.zero_stage,
        offload=args.ds.offload,
        pin_memory=args.ds.pin_memory,
        mixed_precision="bf16" if args.trainer.bf16 else "fp16" if args.trainer.fp16 else "no",
        num_machines=args.dist.num_machines,
        num_processes=args.dist.num_processes,
        tmp_dir=args.task.tmp_dir,
    )

    accelerate_config_path = save_temp_config(accelerate_config, "accelerate", args.task.tmp_dir, format="yaml")

    if not os.path.exists("./prepared") or args.task.clear_data_cache:
        os.system("python prepare.py")

    command = f"accelerate launch --config_file {accelerate_config_path} {args.task.task}.py {' '.join(sys.argv[1:])}"

    print("Running accelerate...")
    print(f"command: {command}")

    os.system(command)


if __name__ == "__main__":
    main()
