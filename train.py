import builtins
import os

from datasets import load_from_disk

from utils import get_local_rank, get_local_rank
from args import parse_args, print_args
from prepare import prepare_tokenizer

from args import (
    TaskArguments,
    HuggingFaceHubArguments,
    DistributedArguments,
    TrainingArguments,
    SFTArguments,
    DeepSpeedArguments,
    LoraArguments,
    BitsAndBytesArguments,
)


def print(*args, **kwargs):
    if get_local_rank() == 0 and not kwargs.get("all_ranks", False):
        builtins.print(*args, **kwargs)


def main():
    args = parse_args(
        TaskArguments,
        HuggingFaceHubArguments,
        DistributedArguments,
        TrainingArguments,
        SFTArguments,
        DeepSpeedArguments,
        LoraArguments,
        BitsAndBytesArguments,
    )

    print_args(args)

    if not os.path.exists("./prepared"):
        raise ValueError("Dataset not prepared. Did you run `python prepare.py` first?")

    # Load dataset
    dataset = load_from_disk("./prepared")

    # Load tokenizer
    tokenizer = args.tokenizer()
    tokenizer = prepare_tokenizer(tokenizer)

    # Load base model
    model = args.model()

    # Make new learnable parameters for specialized tokens (added by `prepare_tokenizer`)
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)

    # Set Llama 2 specific parameters
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Set supervised fine-tuning parameters
    trainer = args.sft_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"] if "test" in dataset else None,
        dataset_text_field="text",
    )

    # Train model
    trainer.train()

    # Save model & tokenizer
    save_dir = args.task.save_dir

    trainer.model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)


if __name__ == "__main__":
    main()
