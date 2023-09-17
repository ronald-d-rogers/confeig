import os
from functools import partial
from transformers import AutoTokenizer
import bitsandbytes as bnb
from transformers import AutoTokenizer


def prepare_tokenizer(model_name_or_path: str):
    # Load LLaMA tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

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

    tokenizer.padding_side = "right"  # Fix weird overflow issue with fp15 training

    return tokenizer


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


# SOURCE https://github.com/artidoro/qlora/blob/main/qlora.py
def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit  # if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def get_world_size():
    return int(os.environ.get("WORLD_SIZE", 1))


def get_local_rank():
    return int(os.environ.get("LOCAL_RANK", 0))
