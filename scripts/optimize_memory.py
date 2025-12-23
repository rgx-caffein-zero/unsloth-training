"""
メモリ最適化設定の自動調整ツール
"""
import subprocess
import json
import torch


def get_gpu_info() -> dict | None:
    """GPU情報を取得"""
    if not torch.cuda.is_available():
        return None
    
    gpu_info = {
        "name": torch.cuda.get_device_name(0),
        "total_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3,
        "capability": torch.cuda.get_device_capability(0),
    }
    
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            total, free, util = result.stdout.strip().split(', ')
            gpu_info["memory_free_mb"] = int(free)
            gpu_info["utilization"] = int(util)
    except Exception:
        pass
    
    return gpu_info


def get_recommended_settings(vram_gb: float) -> dict:
    """VRAM容量に基づく推奨設定"""
    settings_map = {
        24: {
            "model": "unsloth/llama-2-13b-chat",
            "max_seq_length": 4096,
            "batch_size": 4,
            "gradient_accumulation": 4,
            "lora_rank": 64,
            "load_in_4bit": True,
        },
        16: {
            "model": "unsloth/llama-2-7b-chat",
            "max_seq_length": 4096,
            "batch_size": 2,
            "gradient_accumulation": 8,
            "lora_rank": 32,
            "load_in_4bit": True,
        },
        12: {
            "model": "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
            "max_seq_length": 2048,
            "batch_size": 1,
            "gradient_accumulation": 16,
            "lora_rank": 32,
            "load_in_4bit": False,
        },
        8: {
            "model": "unsloth/gemma-2b",
            "max_seq_length": 2048,
            "batch_size": 1,
            "gradient_accumulation": 16,
            "lora_rank": 16,
            "load_in_4bit": True,
        },
        0: {
            "model": "unsloth/tinyllama-chat",
            "max_seq_length": 1024,
            "batch_size": 1,
            "gradient_accumulation": 32,
            "lora_rank": 8,
            "load_in_4bit": True,
        },
    }
    
    for threshold in sorted(settings_map.keys(), reverse=True):
        if vram_gb >= threshold:
            return settings_map[threshold]
    return settings_map[0]


def save_config(config: dict, filepath: str = "training_config.json"):
    """設定をファイルに保存"""
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to {filepath}")


def main():
    print("=== GPU Memory Optimization Tool ===\n")
    
    gpu_info = get_gpu_info()
    if not gpu_info:
        print("❌ No GPU detected!")
        return
    
    vram_gb = gpu_info['total_memory_gb']
    
    print(f"GPU: {gpu_info['name']}")
    print(f"Total VRAM: {vram_gb:.2f} GB")
    print(f"Compute Capability: {gpu_info['capability']}")
    
    if 'memory_free_mb' in gpu_info:
        print(f"Free VRAM: {gpu_info['memory_free_mb']/1024:.2f} GB")
        print(f"GPU Utilization: {gpu_info['utilization']}%")
    
    print("\n=== Recommended Settings ===")
    settings = get_recommended_settings(vram_gb)
    
    for key, value in settings.items():
        print(f"  {key}: {value}")
    
    save_config(settings, "/workspace/work/training_config.json")
    
    print("\n=== Quick Start Commands ===")
    print("\n1. Using configuration file (recommended):")
    print("   # Edit the config file as needed")
    print("   cp configs/finetune_example.yaml configs/my_config.yaml")
    print("   python3 scripts/train.py --config configs/my_config.yaml")
    
    print("\n2. With auto GPU optimization:")
    print("   python3 scripts/train.py --config configs/finetune_example.yaml --auto-optimize")
    
    print("\n3. Run inference after training:")
    print("   python3 scripts/inference.py --config configs/inference_example.yaml")
    
    print("\n=== Tips ===")
    print("  • Monitor GPU usage: watch -n 1 nvidia-smi")
    print("  • If OOM occurs, reduce batch_size or max_seq_length in config")
    print("  • Close other GPU applications before training")
    print("  • Check MLflow UI at http://localhost:5000")
    print("  • Check Ollama models: ollama list")


if __name__ == "__main__":
    main()
