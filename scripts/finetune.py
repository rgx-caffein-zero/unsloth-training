"""
ファインチューニングスクリプト
"""
import os
import argparse
import torch
import gc
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel
import warnings

warnings.filterwarnings("ignore")

ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""


def cleanup_memory():
    """メモリのクリーンアップ"""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def get_training_config(gpu_memory_gb: float) -> dict:
    """GPU VRAMに基づいて訓練設定を取得"""
    configs = [
        (24, {"batch_size": 4, "gradient_accumulation": 4,  "max_seq": 2048}),
        (16, {"batch_size": 2, "gradient_accumulation": 8,  "max_seq": 2048}),
        (12, {"batch_size": 1, "gradient_accumulation": 16, "max_seq": 2048}),
        (8,  {"batch_size": 1, "gradient_accumulation": 16, "max_seq": 1024}),
        (0,  {"batch_size": 1, "gradient_accumulation": 32, "max_seq": 512}),
    ]
    
    for threshold, config in configs:
        if gpu_memory_gb >= threshold:
            return config
    return configs[-1][1]


def load_model(model_name: str = "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
               max_seq_length: int = 2048):
    """モデル読み込み"""
    cleanup_memory()
    
    print(f"Loading model: {model_name}")
    print(f"Max sequence length: {max_seq_length}")
    
    is_preloaded_4bit = "bnb-4bit" in model_name
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=torch.float16,
        load_in_4bit=not is_preloaded_4bit,
        device_map="auto",
        trust_remote_code=True,
    )
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        max_seq_length=max_seq_length,
    )
    
    return model, tokenizer


def prepare_dataset(data_path: str, tokenizer, max_seq_length: int = 2048):
    """データセット準備"""
    if data_path.endswith('.jsonl'):
        dataset = load_dataset('json', data_files=data_path, split='train')
    else:
        dataset = load_dataset(data_path, split='train')
    
    def format_prompts(examples):
        instructions = examples.get("instruction", [])
        inputs = examples.get("input", [""] * len(instructions))
        outputs = examples.get("output", [])
        texts = []
        
        for instruction, input_text, output in zip(instructions, inputs, outputs):
            if input_text:
                text = ALPACA_PROMPT.format(instruction, input_text, output)
            else:
                text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
            
            tokens = tokenizer.encode(text)
            if len(tokens) > max_seq_length:
                tokens = tokens[:max_seq_length]
                text = tokenizer.decode(tokens, skip_special_tokens=True)
            texts.append(text)
        
        return {"text": texts}
    
    dataset = dataset.map(format_prompts, batched=True, num_proc=2)
    print(f"Dataset size: {len(dataset)} samples")
    return dataset


def train(model, tokenizer, dataset,
          output_dir: str = "/workspace/models/finetuned",
          batch_size_override: int = None,
          epochs: int = 3):
    """訓練実行"""
    os.makedirs(output_dir, exist_ok=True)
    
    gpu_memory = 0
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    config = get_training_config(gpu_memory)
    batch_size = batch_size_override or config['batch_size']
    
    print(f"\n=== Training Configuration ({gpu_memory:.1f}GB VRAM) ===")
    print(f"  Batch size: {batch_size}")
    print(f"  Gradient accumulation: {config['gradient_accumulation']}")
    print(f"  Max sequence length: {config['max_seq']}")
    print(f"  Epochs: {epochs}")
    print(f"  Dataset size: {len(dataset)} samples")
    print("=" * 50)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=config['gradient_accumulation'],
        warmup_steps=20,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        optim="paged_adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        save_strategy="epoch",
        save_total_limit=2,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=0.3,
        report_to="tensorboard",
        logging_dir=f"{output_dir}/logs",
        dataloader_num_workers=2,
    )
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=config['max_seq'],
        dataset_num_proc=2,
        packing=True,
        args=training_args,
    )
    
    try:
        trainer.train()
        
        print("\nSaving models...")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        merged_dir = f"{output_dir}_merged"
        model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
        
        print(f"✅ LoRA adapter saved to: {output_dir}")
        print(f"✅ Merged model saved to: {merged_dir}")
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"\n❌ OOM Error: {e}")
        print("Try: --batch-size 1, reduce --max-seq-length, or use a smaller model.")
        raise
    finally:
        cleanup_memory()


def main():
    parser = argparse.ArgumentParser(description="Finetune model")
    parser.add_argument("--model", type=str,
                        default="unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
                        help="Model name")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to training data")
    parser.add_argument("--output", type=str,
                        default="/workspace/models/finetuned",
                        help="Output directory")
    parser.add_argument("--max-seq-length", type=int, default=2048,
                        help="Maximum sequence length")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override auto batch size")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    args = parser.parse_args()
    
    cleanup_memory()
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB\n")
    
    model, tokenizer = load_model(args.model, args.max_seq_length)
    dataset = prepare_dataset(args.data, tokenizer, args.max_seq_length)
    train(model, tokenizer, dataset, args.output, args.batch_size, args.epochs)


if __name__ == "__main__":
    main()
