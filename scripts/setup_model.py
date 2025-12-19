"""
モデルセットアップスクリプト
"""
import os

# Unslothキャッシュディレクトリを設定（権限問題を回避）
os.environ["UNSLOTH_COMPILE_CACHE"] = "/tmp/unsloth_compiled_cache"

# Unslothを最初にインポート（最適化のため必須）
from unsloth import FastLanguageModel

import subprocess
import argparse
import torch
import gc

MODEL_CONFIGS = {
    "mistral-7b": {
        "name": "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
        "max_seq_length": 2048,
        "load_in_4bit": False,
        "lora_r": 32,
    },
    "llama2-7b": {
        "name": "unsloth/llama-2-7b-chat-bnb-4bit",
        "max_seq_length": 2048,
        "load_in_4bit": False,
        "lora_r": 32,
    },
    "llama3-8b": {
        "name": "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        "max_seq_length": 2048,
        "load_in_4bit": False,
        "lora_r": 32,
    },
    "gemma-7b": {
        "name": "unsloth/gemma-7b-bnb-4bit",
        "max_seq_length": 2048,
        "load_in_4bit": False,
        "lora_r": 32,
    },
    "gemma-2b": {
        "name": "unsloth/gemma-2b",
        "max_seq_length": 2048,
        "load_in_4bit": True,
        "lora_r": 16,
    },
    "qwen2-7b": {
        "name": "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
        "max_seq_length": 2048,
        "load_in_4bit": False,
        "lora_r": 32,
    },
}

OLLAMA_MODELS = {
    "mistral": "mistral:7b-instruct-q4_0",
    "llama2": "llama2:7b-chat-q4_0",
    "llama3": "llama3.1:8b-instruct-q4_0",
    "gemma": "gemma:7b",
    "qwen2": "qwen2.5:7b-instruct-q4_0",
}


def cleanup_memory():
    """メモリのクリーンアップ"""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def check_gpu() -> float:
    """GPU情報を表示し、VRAM容量を返す"""
    if not torch.cuda.is_available():
        print("❌ No GPU detected!")
        return 0
    
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {gpu_memory:.2f} GB")
    return gpu_memory


def download_ollama_model(model_key: str) -> bool:
    """Ollamaでモデルをダウンロード"""
    model_name = OLLAMA_MODELS.get(model_key, model_key)
    print(f"Downloading {model_name} using Ollama...")
    
    try:
        subprocess.run(["ollama", "pull", model_name], check=True)
        print(f"✅ Successfully downloaded {model_name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error downloading model: {e}")
        return False


def prepare_model(model_type: str = "mistral-7b"):
    """モデルを準備"""
    cleanup_memory()
    
    config = MODEL_CONFIGS.get(model_type, MODEL_CONFIGS["mistral-7b"])
    
    print(f"\n=== Model Configuration ===")
    print(f"  Model: {config['name']}")
    print(f"  Max sequence length: {config['max_seq_length']}")
    print(f"  4bit quantization: {config['load_in_4bit']}")
    print(f"  LoRA rank: {config['lora_r']}")
    print("=" * 30)
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config['name'],
        max_seq_length=config['max_seq_length'],
        dtype=torch.float16,
        load_in_4bit=config['load_in_4bit'],
        device_map="auto",
        trust_remote_code=True,
    )
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=config['lora_r'],
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        max_seq_length=config['max_seq_length'],
    )
    
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    
    print(f"\n✅ Model loaded successfully!")
    print(f"  GPU Memory allocated: {allocated:.2f} GB")
    print(f"  GPU Memory reserved: {reserved:.2f} GB")
    
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Setup model for training")
    parser.add_argument("--model-type", type=str, default="mistral-7b",
                        choices=list(MODEL_CONFIGS.keys()),
                        help="Model type to use")
    parser.add_argument("--download-ollama", action="store_true",
                        help="Download model using Ollama first")
    args = parser.parse_args()
    
    gpu_memory = check_gpu()
    if gpu_memory == 0:
        return
    
    if args.download_ollama:
        model_key = args.model_type.split('-')[0]
        download_ollama_model(model_key)
    
    try:
        print(f"\nInitializing model: {args.model_type}")
        model, tokenizer = prepare_model(args.model_type)
        
        print("\n=== Next Steps (New Method) ===")
        print("1. Copy and edit the configuration file:")
        print("   cp configs/finetune_example.yaml configs/my_config.yaml")
        print("")
        print("2. Run training with the config file:")
        print("   python3 scripts/train.py --config configs/my_config.yaml")
        print("")
        print("3. Or use auto GPU optimization:")
        print("   python3 scripts/train.py --config configs/my_config.yaml --auto-optimize")
        
    except torch.cuda.OutOfMemoryError:
        print("\n❌ Out of Memory Error!")
        print("Try using a smaller model (e.g., gemma-2b)")
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()
