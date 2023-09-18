import json
from functools import partial
from transformers import AutoTokenizer
from datasets import load_dataset
from args import parse_args


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


def prepare_tokenizer(tokenizer: AutoTokenizer):
    # Add our prompt tokens
    # tokenizer.add_tokens(["<START_TEXT>", "<END_TEXT>", "<START_REPR>", "<END_REPR>"])
    # tokenizer.add_special_tokens(
    #     {
    #         "pad_token": tokenizer.eos_token,
    #         "cls_token": tokenizer.eos_token,
    #         "sep_token": tokenizer.eos_token,
    #         "mask_token": tokenizer.eos_token,
    #         "additional_special_tokens": ["<START_TEXT>", "<END_TEXT>", "<START_REPR>", "<END_REPR>"],
    #     }
    # )

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Fix weird overflow issue with fp15 training

    return tokenizer


def main():
    # Parse args
    args = parse_args()

    tokenizer = args.tokenizer()
    tokenizer = prepare_tokenizer(tokenizer)

    ## Preprocess dataset
    dataset = load_dataset("gsm8k", "main", split="train")

    if args.task.clear_data_cache:
        dataset.cleanup_cache_files()

    dataset, report = preprocess_dataset(tokenizer, dataset, args.sft.max_seq_length)

    print("Dataset report:", report)

    dataset.save_to_disk("./prepared")

    with open("./prepared/report.json", "w") as f:
        json.dump(report, f)


if __name__ == "__main__":
    main()
