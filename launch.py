import os
import sys

from args import parse_args
from config import get_accelerate_config, save_temp_config


def main():
    # Parse args
    args = parse_args()

    accelerate_config = get_accelerate_config(
        args.task.task,
        args.dist.enable_distributed,
        args.ds.enable_deepspeed,
        args.ds.stage,
        args.ds.offload,
        args.ds.pin_memory,
        "bf16" if args.trainer.bf16 else "fp16" if args.trainer.fp16 else "no",
        args.dist.num_machines,
        args.dist.num_processes,
        args.task.tmp_dir,
    )

    accelerate_config_path = save_temp_config(accelerate_config, "accelerate", args.task.tmp_dir, format="yaml")

    if not os.path.exists("./prepared") or args.task.clear_data_cache:
        os.system("python prepare.py")

    command = f"accelerate launch --config_file {accelerate_config_path} {args.task.task}.py {' '.join(sys.argv[1:])}"

    print(f"Running command: {command}")

    os.system(command)


if __name__ == "__main__":
    main()
