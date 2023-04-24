import argparse
import json
from tqdm import tqdm


def format_example(example: dict) -> dict:
    context = f"Instruction: {example['instruction']}\n"
    if example.get("input"):
        context += f"Input: {example['input']}\n"
    context += "Answer: "
    
    target = example["output"]
    return {"context": context, "target": target}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/alpaca_data_zh_51k.json")
    parser.add_argument("--save_path", type=str, default="data/alpaca_data_zh_51k.jsonl")

    args = parser.parse_args()
    with open(args.data_path) as f:
        examples = json.load(f)

    with open(args.save_path, 'w') as f:
        for example in tqdm(examples, desc="formatting.."):
            try:
                example = format_example(example)
            except:
                print(example)
                continue
            f.write(json.dumps(example, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    main()
