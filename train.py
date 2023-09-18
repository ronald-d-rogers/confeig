import builtins
import os
from datasets import load_from_disk

from utils import get_local_rank, get_local_rank
from args import parse_args, print_args
from prepare import prepare_tokenizer


def print(*args, **kwargs):
    if get_local_rank() == 0 and not kwargs.get("all_ranks", False):
        builtins.print(*args, **kwargs)


def main():
    # Parse args
    args = parse_args()

    print_args(args)

    if not os.path.exists("./prepared"):
        raise ValueError("Dataset not prepared. Did you run `python prepare.py` first?")

    # Load dataset
    dataset = load_from_disk("./prepared")

    # Load LLaMA tokenizer
    tokenizer = args.tokenizer()
    tokenizer = prepare_tokenizer(tokenizer)

    # Load base model
    model = args.model()
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Set supervised fine-tuning parameters
    trainer = args.sft_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
    )

    # Train model
    trainer.train()

    # Save trained model
    trainer.model.save_pretrained("./")

    # Save tokenizer
    tokenizer.save_pretrained("./")


if __name__ == "__main__":
    main()
