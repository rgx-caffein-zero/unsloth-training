"""
統一学習エントリーポイント
ファインチューニングと継続事前学習を1つのスクリプトで実行
学習後のGGUF変換とOllama登録もサポート

使用方法:
    python3 scripts/train.py --config configs/finetune_example.yaml
    python3 scripts/train.py --config configs/pretrain_example.yaml
"""
import os
import sys

# Unslothキャッシュディレクトリを設定（権限問題を回避）
os.environ["UNSLOTH_COMPILE_CACHE"] = "/tmp/unsloth_compiled_cache"

# Unslothを最初にインポート（最適化のため必須）
from unsloth import FastLanguageModel

import argparse
import torch
import gc
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

from datasets import load_dataset, Dataset
from trl import SFTTrainer, SFTConfig

# ローカルモジュール
from config import load_config, validate_config, save_config, TrainJobConfig, get_gpu_optimized_config
from logger import setup_logger, TrainingLogger
from mlflow_tracker import setup_mlflow_tracker, create_mlflow_callback, MLflowTracker


# プロンプトテンプレート
PROMPT_TEMPLATES = {
    "alpaca": """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}""",
    
    "alpaca_no_input": """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{output}""",
    
    "chatml": """<|im_start|>user
{instruction}
{input}<|im_end|>
<|im_start|>assistant
{output}<|im_end|>""",
    
    "llama3": """<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{instruction}
{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{output}<|eot_id|>""",
}


def cleanup_memory():
    """メモリのクリーンアップ"""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def get_dtype(dtype_str: str):
    """文字列からdtypeを取得"""
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return dtype_map.get(dtype_str, torch.float16)


def load_model(config: TrainJobConfig, logger: TrainingLogger):
    """モデルとトークナイザーを読み込み"""
    cleanup_memory()
    
    logger.info(f"Loading model: {config.model.name}")
    logger.info(f"Max sequence length: {config.model.max_seq_length}")
    logger.info(f"4bit quantization: {config.model.load_in_4bit}")
    
    # 事前量子化モデルかどうかを判定
    is_preloaded_4bit = "bnb-4bit" in config.model.name
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model.name,
        max_seq_length=config.model.max_seq_length,
        dtype=get_dtype(config.model.dtype),
        load_in_4bit=config.model.load_in_4bit if not is_preloaded_4bit else False,
        device_map=config.model.device_map,
        trust_remote_code=config.model.trust_remote_code,
    )
    
    # LoRAの適用
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora.r,
        target_modules=config.lora.target_modules,
        lora_alpha=config.lora.lora_alpha,
        lora_dropout=config.lora.lora_dropout,
        bias=config.lora.bias,
        use_gradient_checkpointing=config.lora.use_gradient_checkpointing,
        random_state=config.lora.random_state,
        max_seq_length=config.model.max_seq_length,
    )
    
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info(f"GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
    
    return model, tokenizer


def prepare_finetune_dataset(config: TrainJobConfig, tokenizer, logger: TrainingLogger):
    """ファインチューニング用データセットの準備"""
    data_path = config.data.train_data_path
    max_seq_length = config.model.max_seq_length
    
    logger.info(f"Loading dataset: {data_path}")
    
    if data_path.endswith('.jsonl'):
        dataset = load_dataset('json', data_files=data_path, split='train')
    else:
        dataset = load_dataset(data_path, split='train')
    
    # プロンプトテンプレートの選択
    template = PROMPT_TEMPLATES.get(config.data.prompt_template, PROMPT_TEMPLATES["alpaca"])
    
    def format_prompts(examples):
        instructions = examples.get(config.data.instruction_field, [])
        inputs = examples.get(config.data.input_field, [""] * len(instructions))
        outputs = examples.get(config.data.output_field, [])
        texts = []
        
        for instruction, input_text, output in zip(instructions, inputs, outputs):
            if input_text and "{input}" in template:
                text = template.format(instruction=instruction, input=input_text, output=output)
            else:
                # 入力がない場合はalpaca_no_inputテンプレートを使用
                no_input_template = PROMPT_TEMPLATES.get("alpaca_no_input", template)
                text = no_input_template.format(instruction=instruction, output=output)
            
            # トークン長の制限
            tokens = tokenizer.encode(text)
            if len(tokens) > max_seq_length:
                tokens = tokens[:max_seq_length]
                text = tokenizer.decode(tokens, skip_special_tokens=True)
            texts.append(text)
        
        return {"text": texts}
    
    dataset = dataset.map(format_prompts, batched=True, num_proc=config.training.dataset_num_proc)
    logger.info(f"Dataset size: {len(dataset)} samples")
    
    return dataset


def prepare_pretrain_dataset(config: TrainJobConfig, tokenizer, logger: TrainingLogger):
    """継続事前学習用データセットの準備"""
    data_path = config.data.train_data_path
    max_seq_length = config.model.max_seq_length
    
    logger.info(f"Loading dataset: {data_path}")
    
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
    
    dataset = dataset.map(truncate_text, batched=True, num_proc=config.training.dataset_num_proc)
    logger.info(f"Dataset size: {len(dataset)} samples")
    
    return dataset


def create_trainer_config(config: TrainJobConfig, output_dir: str) -> SFTConfig:
    """SFTConfigを作成"""
    return SFTConfig(
        output_dir=output_dir,
        num_train_epochs=config.training.num_train_epochs,
        max_steps=config.training.max_steps if config.training.max_steps > 0 else -1,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        warmup_steps=config.training.warmup_steps,
        learning_rate=config.training.learning_rate,
        fp16=config.training.fp16,
        bf16=config.training.bf16,
        logging_steps=config.training.logging_steps,
        optim=config.training.optim,
        weight_decay=config.training.weight_decay,
        lr_scheduler_type=config.training.lr_scheduler_type,
        seed=config.seed,
        save_strategy=config.training.save_strategy,
        save_steps=config.training.save_steps,
        save_total_limit=config.training.save_total_limit,
        gradient_checkpointing=config.training.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=config.training.max_grad_norm,
        report_to=config.training.report_to,
        logging_dir=f"{output_dir}/logs",
        dataloader_num_workers=config.training.dataloader_num_workers,
        max_seq_length=config.model.max_seq_length,
        dataset_text_field="text",
        packing=config.training.packing,
        dataset_num_proc=config.training.dataset_num_proc,
    )


def convert_to_gguf(
    config: TrainJobConfig,
    merged_dir: str,
    run_name: str,
    logger: TrainingLogger,
    tracker: MLflowTracker
):
    """学習済みモデルをGGUF形式に変換"""
    from convert_to_gguf import convert_and_register
    
    logger.section("GGUF Conversion")
    
    # GGUF出力ディレクトリの決定
    if config.gguf.output_dir:
        gguf_output_dir = config.gguf.output_dir
    else:
        gguf_output_dir = os.path.join(
            config.output.output_dir,
            config.mlflow.experiment_name,
            run_name,
            "gguf"
        )
    
    # Ollamaモデル名の決定
    ollama_model_name = config.gguf.ollama_model_name
    if not ollama_model_name:
        # experiment_name と run_name から生成
        ollama_model_name = f"{config.mlflow.experiment_name}-{run_name}".lower()
        # Ollamaモデル名に使えない文字を置換
        ollama_model_name = ollama_model_name.replace("_", "-").replace(" ", "-")
    
    logger.info(f"Converting to GGUF format...")
    logger.info(f"  Source: {merged_dir}")
    logger.info(f"  Output: {gguf_output_dir}")
    logger.info(f"  Quantization: {config.gguf.quantization}")
    logger.info(f"  Ollama model name: {ollama_model_name}")
    
    try:
        result = convert_and_register(
            model_path=merged_dir,
            output_dir=gguf_output_dir,
            model_name=ollama_model_name,
            quantization=config.gguf.quantization,
            system_prompt=config.gguf.system_prompt,
            template=config.gguf.template,
            register_ollama=config.gguf.register_ollama,
            logger=logger
        )
        
        if result["success"]:
            logger.info("✅ GGUF conversion completed successfully!")
            
            # MLflowにGGUF情報をログ
            tracker.set_tag("gguf.enabled", "true")
            tracker.set_tag("gguf.quantization", config.gguf.quantization)
            tracker.set_tag("gguf.path", result.get("gguf_path", ""))
            if result.get("registered"):
                tracker.set_tag("ollama.model_name", ollama_model_name)
            
            return result
        else:
            logger.error("❌ GGUF conversion failed")
            tracker.set_tag("gguf.error", "conversion_failed")
            return None
            
    except Exception as e:
        logger.error(f"❌ GGUF conversion error: {e}")
        tracker.set_tag("gguf.error", str(e)[:500])
        return None


def train(
    config: TrainJobConfig,
    logger: TrainingLogger,
    tracker: MLflowTracker,
    run_name: str
):
    """学習を実行"""
    # 出力ディレクトリをベースディレクトリ + experiment_name + run_nameで構築
    base_output_dir = config.output.output_dir
    experiment_name = config.mlflow.experiment_name
    output_dir = os.path.join(base_output_dir, experiment_name, run_name)
    lora_dir = os.path.join(output_dir, "lora")
    merged_dir = os.path.join(output_dir, "merged")
    
    # ディレクトリの作成
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(lora_dir, exist_ok=True)
    if config.output.save_merged_model:
        os.makedirs(merged_dir, exist_ok=True)
    
    # 設定のコピーを保存
    config_save_path = os.path.join(output_dir, "train_config.yaml")
    save_config(config, config_save_path)
    logger.info(f"Configuration saved to: {config_save_path}")
    logger.info(f"Experiment name: {experiment_name}")
    logger.info(f"Run name: {run_name}")
    logger.info(f"Output directory: {output_dir}")
    
    # モデルの読み込み
    logger.section("Loading Model")
    model, tokenizer = load_model(config, logger)
    
    # データセットの準備
    logger.section("Preparing Dataset")
    if config.training_type == "finetune":
        dataset = prepare_finetune_dataset(config, tokenizer, logger)
    else:  # pretrain
        dataset = prepare_pretrain_dataset(config, tokenizer, logger)
    
    # MLflowにデータセット情報をログ
    tracker.log_dataset_info(len(dataset), config.data.train_data_path)
    
    # Trainerの設定
    logger.section("Training Configuration")
    sft_config = create_trainer_config(config, lora_dir)
    
    logger.info(f"Batch size: {config.training.per_device_train_batch_size}")
    logger.info(f"Gradient accumulation: {config.training.gradient_accumulation_steps}")
    logger.info(f"Effective batch size: {config.training.per_device_train_batch_size * config.training.gradient_accumulation_steps}")
    logger.info(f"Learning rate: {config.training.learning_rate}")
    logger.info(f"Epochs: {config.training.num_train_epochs}")
    logger.info(f"Max steps: {config.training.max_steps}")
    
    # コールバックの設定
    callbacks = []
    mlflow_callback = create_mlflow_callback(tracker)
    if mlflow_callback:
        callbacks.append(mlflow_callback)
    
    # Trainerの作成
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=sft_config,
        callbacks=callbacks if callbacks else None,
    )
    
    # 学習実行
    logger.section("Training Started")
    try:
        train_result = trainer.train()
        
        # 学習結果のログ
        if train_result.metrics:
            logger.metrics(train_result.metrics)
            tracker.log_metrics(train_result.metrics)
        
        # モデルの保存
        logger.section("Saving Model")
        model.save_pretrained(lora_dir)
        tokenizer.save_pretrained(lora_dir)
        logger.info(f"LoRA adapter saved to: {lora_dir}")
        
        # マージされたモデルの保存
        if config.output.save_merged_model:
            model.save_pretrained_merged(
                merged_dir, 
                tokenizer, 
                save_method=config.output.save_method
            )
            logger.info(f"Merged model saved to: {merged_dir}")
            tracker.log_model_info(config.model.name, merged_dir)
        
        # 設定ファイルをMLflowにログ
        tracker.log_artifact(config_save_path)
        
        # GGUF変換（有効な場合）
        if config.gguf.enabled and config.output.save_merged_model:
            # メモリ解放
            del model
            del tokenizer
            cleanup_memory()
            
            convert_to_gguf(config, merged_dir, run_name, logger, tracker)
        
        return True
        
    except torch.cuda.OutOfMemoryError as e:
        logger.error(f"CUDA Out of Memory Error: {e}")
        logger.error("Try reducing batch_size, max_seq_length, or using a smaller model.")
        tracker.set_tag("error", "OOM")
        raise
    except Exception as e:
        logger.error(f"Training failed: {e}")
        tracker.set_tag("error", str(e)[:500])
        raise
    finally:
        cleanup_memory()


def main():
    parser = argparse.ArgumentParser(
        description="Unified training script for finetuning and continued pre-training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Finetuning
  python3 scripts/train.py --config configs/finetune_example.yaml
  
  # Continued pre-training  
  python3 scripts/train.py --config configs/pretrain_example.yaml
  
  # With auto GPU optimization
  python3 scripts/train.py --config configs/finetune_example.yaml --auto-optimize
  
  # With GGUF conversion (enable in config or use --convert-gguf)
  python3 scripts/train.py --config configs/finetune_example.yaml --convert-gguf
        """
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--auto-optimize",
        action="store_true",
        help="Automatically optimize settings based on GPU VRAM"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without running training"
    )
    parser.add_argument(
        "--convert-gguf",
        action="store_true",
        help="Convert trained model to GGUF format for Ollama"
    )
    parser.add_argument(
        "--gguf-quantization",
        type=str,
        default=None,
        help="Override GGUF quantization type (e.g., q4_k_m, q8_0)"
    )
    args = parser.parse_args()
    
    # 設定ファイルの読み込み
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    # コマンドライン引数で設定を上書き
    if args.convert_gguf:
        config.gguf.enabled = True
    if args.gguf_quantization:
        config.gguf.quantization = args.gguf_quantization
    
    # 設定の検証
    errors = validate_config(config)
    if errors:
        print("Configuration errors:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)
    
    # GPU最適化
    if args.auto_optimize and torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"Auto-optimizing for {vram_gb:.1f} GB VRAM")
        config = get_gpu_optimized_config(vram_gb, config)
    
    # ドライランの場合は設定を表示して終了
    if args.dry_run:
        # run_nameの決定（nullの場合は自動生成）
        run_name = config.mlflow.run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        print("\n=== Configuration (Dry Run) ===")
        print(f"Training Type: {config.training_type}")
        print(f"Model: {config.model.name}")
        print(f"Data: {config.data.train_data_path}")
        print(f"Experiment Name: {config.mlflow.experiment_name}")
        print(f"Run Name: {run_name}")
        print(f"Output Base Dir: {config.output.output_dir}")
        print(f"Output Dir: {config.output.output_dir}/{config.mlflow.experiment_name}/{run_name}")
        print(f"Batch Size: {config.training.per_device_train_batch_size}")
        print(f"Gradient Accumulation: {config.training.gradient_accumulation_steps}")
        print(f"Max Seq Length: {config.model.max_seq_length}")
        print(f"Epochs: {config.training.num_train_epochs}")
        print(f"Learning Rate: {config.training.learning_rate}")
        print(f"MLflow Enabled: {config.mlflow.enabled}")
        print(f"GGUF Conversion: {config.gguf.enabled}")
        if config.gguf.enabled:
            print(f"  Quantization: {config.gguf.quantization}")
            print(f"  Register Ollama: {config.gguf.register_ollama}")
        print("\nConfiguration is valid. Remove --dry-run to start training.")
        sys.exit(0)
    
    # run_nameの決定（nullの場合は自動生成）
    run_name = config.mlflow.run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # 出力ディレクトリをベースディレクトリ + experiment_name + run_nameで構築
    base_output_dir = config.output.output_dir
    experiment_name = config.mlflow.experiment_name
    output_dir = os.path.join(base_output_dir, experiment_name, run_name)
    
    # 出力ディレクトリの作成
    os.makedirs(output_dir, exist_ok=True)
    
    # ロガーのセットアップ
    logger = setup_logger(
        output_dir=output_dir,
        log_file=config.output.log_file,
        capture_stdout=True
    )
    
    # MLflowトラッカーのセットアップ（run_nameを統一）
    tracker = setup_mlflow_tracker(
        enabled=config.mlflow.enabled,
        tracking_uri=config.mlflow.tracking_uri,
        experiment_name=config.mlflow.experiment_name,
        run_name=run_name,  # main関数で決定したrun_nameを使用
        tags=config.mlflow.tags
    )
    
    success = False
    error_msg = None
    
    try:
        # MLflowセッションの開始
        tracker.setup()
        tracker.start_run()
        
        # 設定をMLflowにログ
        tracker.log_config(config)
        tracker.log_gpu_info()
        
        # 学習開始ログ
        logger.training_start(config)
        
        # GPU情報の表示
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # 学習実行
        success = train(config, logger, tracker, run_name)
        
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        error_msg = "Interrupted by user"
        tracker.set_tag("status", "interrupted")
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Training failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        # 学習終了ログ
        logger.training_end(success=success, error_msg=error_msg)
        
        # MLflowセッションの終了
        tracker.end_run(status="FINISHED" if success else "FAILED")
        
        # ロガーのクローズ
        logger.close()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
