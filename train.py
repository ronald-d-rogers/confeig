import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import LoraConfig
from trl import SFTTrainer

from utils import preprocess_dataset, find_all_linear_names, get_local_rank
from args import parse_args


def main():
    # Parse args
    args = parse_args()

    # Load LLaMA tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.hf.model_name_or_path)

    # Add our prompt tokens
    tokenizer.add_tokens(["<START_TEXT>", "<END_TEXT>", "<START_REPR>", "<END_REPR>"])
    tokenizer.add_special_tokens(
        {
            "pad_token": tokenizer.eos_token,
            "cls_token": tokenizer.eos_token,
            "sep_token": tokenizer.eos_token,
            "mask_token": tokenizer.eos_token,
            "additional_special_tokens": ["<START_TEXT>", "<END_TEXT>", "<START_REPR>", "<END_REPR>"],
        }
    )

    tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

    ## Preprocess dataset
    dataset = load_dataset("gsm8k", "main", split="train")

    if args.task.clear_data_cache:
        dataset.cleanup_cache_files()

    dataset, report = preprocess_dataset(tokenizer, dataset, args.trl.max_seq_length)

    print("Dataset report:", report)

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

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        args.hf.model_name_or_path,
        quantization_config=args.bnb,
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
        max_seq_length=args.trl.max_seq_length,
        tokenizer=tokenizer,
        args=args.trainer,
        packing=args.trl.packing,
    )

    # Train model
    trainer.train()

    # Save trained model
    trainer.model.save_pretrained("./")

    # Save tokenizer
    tokenizer.save_pretrained("./")


if __name__ == "__main__":
    main()
