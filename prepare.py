import json
from datasets import load_dataset
from utils import preprocess_dataset, prepare_tokenizer
from args import parse_args


def main():
    # Parse args
    args = parse_args()

    tokenizer = prepare_tokenizer(args.hf.model_name_or_path)

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
