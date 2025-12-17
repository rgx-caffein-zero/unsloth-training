"""
継続事前学習スクリプト
"""
import os
import argparse
import torch
import gc
from datasets import load_dataset, Dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel


def cleanup_memory():
    """メモリのクリーンアップ"""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def get_training_config(gpu_memory_gb: float) -> dict:
    """GPU VRAMに基づいて訓練設定を取得"""
    configs = [
        (24, {"batch_size": 2, "gradient_accumulation": 4, "max_seq": 2048}),
        (16, {"batch_size": 2, "gradient_accumulation": 4, "max_seq": 2048}),
        (12, {"batch_size": 1, "gradient_accumulation": 8, "max_seq": 2048}),
        (8,  {"batch_size": 1, "gradient_accumulation": 16, "max_seq": 1024}),
        (0,  {"batch_size": 1, "gradient_accumulation": 32, "max_seq": 512}),
    ]
    
    for threshold, config in configs:
        if gpu_memory_gb >= threshold:
            return config
    return configs[-1][1]


def load_model_for_pretraining(model_name: str = "unsloth/llama-2-7b",
                                max_seq_length: int = 2048):
    """継続事前学習用のモデル読み込み"""
    cleanup_memory()
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=torch.float16,
        load_in_4bit=True,
        device_map="auto",
    )
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        max_seq_length=max_seq_length,
    )
    
    return model, tokenizer


def prepare_pretraining_dataset(data_path: str, tokenizer, max_seq_length: int = 2048):
    """継続事前学習用データセットの準備"""
    if data_path.endswith('.txt'):
        with open(data_path, 'r', encoding='utf-8') as f:
            texts = [t.strip() for t in f.read().split('\n\n') if t.strip()]
        dataset = Dataset.from_dict({"text": texts})
    elif data_path.endswith('.jsonl'):
        dataset = load_dataset('json', data_files=data_path, split='train')
    else:
        dataset = load_dataset(data_path, split='train')
    
    def truncate_text(examples):
        processed = []
        for text in examples["text"]:
            tokens = tokenizer.encode(text)
            if len(tokens) > max_seq_length:
                tokens = tokens[:max_seq_length]
                text = tokenizer.decode(tokens, skip_special_tokens=True)
            processed.append(text)
        return {"text": processed}
    
    return dataset.map(truncate_text, batched=True, num_proc=2)


def continued_pretrain(model, tokenizer, dataset,
                       output_dir: str = "/workspace/work/models/continued_pretrained"):
    """継続事前学習の実行"""
    os.makedirs(output_dir, exist_ok=True)
    
    gpu_memory = 0
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    config = get_training_config(gpu_memory)
    
    print(f"\n=== Training Configuration ({gpu_memory:.1f}GB VRAM) ===")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Gradient accumulation: {config['gradient_accumulation']}")
    print(f"  Max sequence length: {config['max_seq']}")
    print(f"  Dataset size: {len(dataset)} samples")
    print("=" * 50)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=config['batch_size'],
        gradient_accumulation_steps=config['gradient_accumulation'],
        warmup_steps=100,
        max_steps=1000,
        learning_rate=5e-5,
        fp16=True,
        logging_steps=20,
        optim="paged_adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        report_to="tensorboard",
        logging_dir=f"{output_dir}/logs",
        gradient_checkpointing=True,
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
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"\n✅ Model saved to {output_dir}")
        
        merged_dir = f"{output_dir}_merged"
        model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
        print(f"✅ Merged model saved to {merged_dir}")
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"\n❌ OOM Error: {e}")
        print("Try reducing max_seq_length or using a smaller model.")
        raise
    finally:
        cleanup_memory()


def main():
    parser = argparse.ArgumentParser(description="Continued pre-training script")
    parser.add_argument("--model", type=str, default="unsloth/llama-2-7b",
                        help="Base model name")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to pre-training data")
    parser.add_argument("--output", type=str,
                        default="/workspace/work/models/continued_pretrained",
                        help="Output directory")
    parser.add_argument("--max-seq-length", type=int, default=2048,
                        help="Maximum sequence length")
    args = parser.parse_args()
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB\n")
    
    print(f"Loading model: {args.model}")
    model, tokenizer = load_model_for_pretraining(args.model, args.max_seq_length)
    
    print(f"Preparing dataset: {args.data}")
    dataset = prepare_pretraining_dataset(args.data, tokenizer, args.max_seq_length)
    
    continued_pretrain(model, tokenizer, dataset, args.output)


if __name__ == "__main__":
    main()
