import builtins
import os
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig
from trl import SFTTrainer

from utils import find_all_linear_names, get_local_rank, prepare_tokenizer, get_local_rank
from args import parse_args, print_args


def print(*args, **kwargs):
    if get_local_rank() == 0 and not kwargs.get("all_ranks", False):
        builtins.print(*args, **kwargs)


def main():
    # Parse args
    args = parse_args()

    print_args(args, print=print)

    if not os.path.exists("./prepared"):
        raise ValueError("Dataset not prepared. Did you run `python prepare.py` first?")

    # Load dataset
    dataset = load_from_disk("./prepared")

    # Load LLaMA tokenizer
    tokenizer = prepare_tokenizer(args.hf.model_name_or_path)

    # Check GPU compatibility with bfloat16
    if args.bnb.bnb_4bit_compute_dtype == torch.float16 and args.bnb.load_in_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print("Your GPU supports bfloat16: accelerate training with bf16=True")
            print("=" * 80)

    device_map = None
    if args.lora.enable_lora:
        device_map = {get_local_rank(): ""}
        print(f"Device map: {device_map}", all_ranks=True)

    quantization_config = BitsAndBytesConfig(**args.bnb.__dict__)

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        args.hf.model_name_or_path,
        quantization_config=quantization_config,
        device_map=device_map,
    )

    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Load LoRA configuration
    peft_config = LoraConfig(
        lora_alpha=args.lora.lora_alpha,
        lora_dropout=args.lora.lora_dropout,
        target_modules=find_all_linear_names(model),
        r=args.lora.lora_r,
        bias=args.lora.lora_bias,
        task_type="CAUSAL_LM",
    )

    # Set supervised fine-tuning parameters
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config if args.lora.enable_lora else None,
        dataset_text_field="text",
        max_seq_length=args.sft.max_seq_length,
        tokenizer=tokenizer,
        args=args.trainer,
        packing=args.sft.packing,
    )

    # Train model
    trainer.train()

    # Save trained model
    trainer.model.save_pretrained("./")

    # Save tokenizer
    tokenizer.save_pretrained("./")


if __name__ == "__main__":
    main()
