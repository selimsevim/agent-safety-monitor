import os
import json
import argparse
from pathlib import Path
import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, Trainer, DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

os.environ["USE_TF"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def load_jsonl(path: str) -> Dataset:
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return Dataset.from_list(records)


def format_prompt(example: dict, tokenizer) -> dict:
    prompt = (
        f"<|user|>\n{example['input']}<|end|>\n"
        f"<|assistant|>\n{example['output']}<|end|>"
    )
    tokenized = tokenizer(
        prompt,
        truncation=True,
        max_length=512,
        padding='max_length'
    )
    tokenized['labels'] = tokenized['input_ids'].copy()
    return tokenized


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='microsoft/Phi-3-mini-4k-instruct')
    parser.add_argument('--train', default='data/processed/train_augmented.jsonl')
    parser.add_argument('--val', default='data/processed/val.jsonl')
    parser.add_argument('--output', default='models/adapters/phi3-safety-v1')
    parser.add_argument('--lora-rank', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=2e-4)
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        load_in_4bit=True,
        torch_dtype=torch.float16,
        device_map='auto',
        trust_remote_code=True
    )

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        target_modules=['q_proj', 'v_proj', 'k_proj', 'o_proj'],
        lora_dropout=0.05,
        bias='none',
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = load_jsonl(args.train)
    val_dataset = load_jsonl(args.val)

    train_dataset = train_dataset.map(
        lambda x: format_prompt(x, tokenizer),
        remove_columns=train_dataset.column_names
    )
    val_dataset = val_dataset.map(
        lambda x: format_prompt(x, tokenizer),
        remove_columns=val_dataset.column_names
    )

    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        fp16=True,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        logging_steps=10,
        report_to='none'
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8)
    )

    print("Training...")
    trainer.train()
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"Saved adapter to {args.output}")


if __name__ == '__main__':
    main()
