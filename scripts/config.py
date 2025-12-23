"""
YAML設定ファイルの読み込み・検証モジュール
"""
import os
from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path
import yaml


@dataclass
class ModelConfig:
    """モデル設定"""
    name: str = "unsloth/mistral-7b-instruct-v0.2-bnb-4bit"
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    dtype: str = "float16"
    device_map: str = "auto"
    trust_remote_code: bool = True


@dataclass
class LoRAConfig:
    """LoRA設定"""
    r: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    bias: str = "none"
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    use_gradient_checkpointing: str = "unsloth"
    random_state: int = 3407


@dataclass
class TrainingConfig:
    """訓練設定"""
    # 基本設定
    num_train_epochs: int = 3
    max_steps: int = -1
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    
    # 学習率設定
    learning_rate: float = 2e-4
    warmup_steps: int = 20
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.01
    max_grad_norm: float = 0.3
    
    # 精度設定
    fp16: bool = True
    bf16: bool = False
    
    # 最適化設定
    optim: str = "paged_adamw_8bit"
    gradient_checkpointing: bool = True
    
    # データ設定
    packing: bool = True
    dataset_num_proc: int = 2
    dataloader_num_workers: int = 2
    
    # 保存設定
    save_strategy: str = "epoch"
    save_steps: int = 200
    save_total_limit: int = 2
    
    # ログ設定
    logging_steps: int = 10
    report_to: str = "none"


@dataclass
class DataConfig:
    """データ設定"""
    train_data_path: str = ""
    validation_data_path: Optional[str] = None
    text_field: str = "text"
    
    # ファインチューニング用
    instruction_field: str = "instruction"
    input_field: str = "input"
    output_field: str = "output"
    
    # プロンプトテンプレート
    prompt_template: str = "alpaca"


@dataclass
class OutputConfig:
    """出力設定"""
    output_dir: str = "/workspace/work/models/outputs"  # ベースディレクトリ
    save_merged_model: bool = True
    save_method: str = "merged_16bit"
    log_file: str = "training.log"


@dataclass
class GGUFConfig:
    """GGUF変換設定"""
    # GGUF変換を有効にするか
    enabled: bool = False
    
    # 量子化タイプ
    # 利用可能: q4_0, q4_1, q4_k_m, q4_k_s, q5_0, q5_1, q5_k_m, q5_k_s, q6_k, q8_0, f16, f32
    quantization: str = "q4_k_m"
    
    # GGUF出力ディレクトリ（nullの場合はoutput_dir/ggufを使用）
    output_dir: Optional[str] = None
    
    # Ollamaに登録するか
    register_ollama: bool = True
    
    # Ollamaでのモデル名（nullの場合は自動生成）
    ollama_model_name: Optional[str] = None
    
    # Modelfile用のシステムプロンプト
    system_prompt: str = "You are a helpful AI assistant."
    
    # Modelfile用のプロンプトテンプレート
    template: str = "alpaca"


@dataclass
class MLflowConfig:
    """MLflow設定"""
    enabled: bool = True
    tracking_uri: str = "file:///workspace/work/mlruns"
    experiment_name: str = "unsloth-training"
    run_name: Optional[str] = None
    tags: dict = field(default_factory=dict)
    log_model: bool = False


@dataclass
class TrainJobConfig:
    """学習ジョブ全体の設定"""
    # 学習タイプ: "finetune" or "pretrain"
    training_type: str = "finetune"
    
    # 各種設定
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    gguf: GGUFConfig = field(default_factory=GGUFConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)
    
    # メタデータ
    description: str = ""
    seed: int = 3407


def _merge_dict(base: dict, override: dict) -> dict:
    """辞書を再帰的にマージ"""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_dict(result[key], value)
        else:
            result[key] = value
    return result


def _dataclass_from_dict(cls, data: dict):
    """辞書からデータクラスを生成"""
    if data is None:
        return cls()
    
    fieldtypes = {f.name: f.type for f in cls.__dataclass_fields__.values()}
    filtered_data = {}
    
    for key, value in data.items():
        if key in fieldtypes:
            filtered_data[key] = value
    
    return cls(**filtered_data)


def load_config(config_path: str) -> TrainJobConfig:
    """YAML設定ファイルを読み込み、TrainJobConfigを返す"""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        raw_config = yaml.safe_load(f)
    
    if raw_config is None:
        raw_config = {}
    
    # 各セクションをデータクラスに変換
    model_config = _dataclass_from_dict(ModelConfig, raw_config.get('model'))
    lora_config = _dataclass_from_dict(LoRAConfig, raw_config.get('lora'))
    training_config = _dataclass_from_dict(TrainingConfig, raw_config.get('training'))
    data_config = _dataclass_from_dict(DataConfig, raw_config.get('data'))
    output_config = _dataclass_from_dict(OutputConfig, raw_config.get('output'))
    gguf_config = _dataclass_from_dict(GGUFConfig, raw_config.get('gguf'))
    mlflow_config = _dataclass_from_dict(MLflowConfig, raw_config.get('mlflow'))
    
    # トップレベル設定
    job_config = TrainJobConfig(
        training_type=raw_config.get('training_type', 'finetune'),
        model=model_config,
        lora=lora_config,
        training=training_config,
        data=data_config,
        output=output_config,
        gguf=gguf_config,
        mlflow=mlflow_config,
        description=raw_config.get('description', ''),
        seed=raw_config.get('seed', 3407),
    )
    
    return job_config


def save_config(config: TrainJobConfig, config_path: str):
    """設定をYAMLファイルに保存"""
    config_dict = {
        'training_type': config.training_type,
        'description': config.description,
        'seed': config.seed,
        'model': {
            'name': config.model.name,
            'max_seq_length': config.model.max_seq_length,
            'load_in_4bit': config.model.load_in_4bit,
            'dtype': config.model.dtype,
            'device_map': config.model.device_map,
            'trust_remote_code': config.model.trust_remote_code,
        },
        'lora': {
            'r': config.lora.r,
            'lora_alpha': config.lora.lora_alpha,
            'lora_dropout': config.lora.lora_dropout,
            'bias': config.lora.bias,
            'target_modules': config.lora.target_modules,
            'use_gradient_checkpointing': config.lora.use_gradient_checkpointing,
            'random_state': config.lora.random_state,
        },
        'training': {
            'num_train_epochs': config.training.num_train_epochs,
            'max_steps': config.training.max_steps,
            'per_device_train_batch_size': config.training.per_device_train_batch_size,
            'gradient_accumulation_steps': config.training.gradient_accumulation_steps,
            'learning_rate': config.training.learning_rate,
            'warmup_steps': config.training.warmup_steps,
            'lr_scheduler_type': config.training.lr_scheduler_type,
            'weight_decay': config.training.weight_decay,
            'max_grad_norm': config.training.max_grad_norm,
            'fp16': config.training.fp16,
            'bf16': config.training.bf16,
            'optim': config.training.optim,
            'gradient_checkpointing': config.training.gradient_checkpointing,
            'packing': config.training.packing,
            'dataset_num_proc': config.training.dataset_num_proc,
            'dataloader_num_workers': config.training.dataloader_num_workers,
            'save_strategy': config.training.save_strategy,
            'save_steps': config.training.save_steps,
            'save_total_limit': config.training.save_total_limit,
            'logging_steps': config.training.logging_steps,
            'report_to': config.training.report_to,
        },
        'data': {
            'train_data_path': config.data.train_data_path,
            'validation_data_path': config.data.validation_data_path,
            'text_field': config.data.text_field,
            'instruction_field': config.data.instruction_field,
            'input_field': config.data.input_field,
            'output_field': config.data.output_field,
            'prompt_template': config.data.prompt_template,
        },
        'output': {
            'output_dir': config.output.output_dir,
            'save_merged_model': config.output.save_merged_model,
            'save_method': config.output.save_method,
            'log_file': config.output.log_file,
        },
        'gguf': {
            'enabled': config.gguf.enabled,
            'quantization': config.gguf.quantization,
            'output_dir': config.gguf.output_dir,
            'register_ollama': config.gguf.register_ollama,
            'ollama_model_name': config.gguf.ollama_model_name,
            'system_prompt': config.gguf.system_prompt,
            'template': config.gguf.template,
        },
        'mlflow': {
            'enabled': config.mlflow.enabled,
            'tracking_uri': config.mlflow.tracking_uri,
            'experiment_name': config.mlflow.experiment_name,
            'run_name': config.mlflow.run_name,
            'tags': config.mlflow.tags,
            'log_model': config.mlflow.log_model,
        },
    }
    
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def validate_config(config: TrainJobConfig) -> List[str]:
    """設定を検証し、エラーメッセージのリストを返す"""
    errors = []
    
    # 必須項目のチェック
    if not config.data.train_data_path:
        errors.append("data.train_data_path is required")
    elif not Path(config.data.train_data_path).exists():
        errors.append(f"Training data file not found: {config.data.train_data_path}")
    
    # training_typeのチェック
    if config.training_type not in ['finetune', 'pretrain']:
        errors.append(f"Invalid training_type: {config.training_type}. Must be 'finetune' or 'pretrain'")
    
    # 数値範囲のチェック
    if config.training.per_device_train_batch_size < 1:
        errors.append("training.per_device_train_batch_size must be >= 1")
    
    if config.training.learning_rate <= 0:
        errors.append("training.learning_rate must be > 0")
    
    if config.model.max_seq_length < 64:
        errors.append("model.max_seq_length must be >= 64")
    
    if config.lora.r < 1:
        errors.append("lora.r must be >= 1")
    
    # GGUF設定のチェック
    valid_quantizations = [
        "q4_0", "q4_1", "q4_k_m", "q4_k_s",
        "q5_0", "q5_1", "q5_k_m", "q5_k_s",
        "q6_k", "q8_0", "f16", "f32"
    ]
    if config.gguf.enabled and config.gguf.quantization not in valid_quantizations:
        errors.append(f"Invalid gguf.quantization: {config.gguf.quantization}. Must be one of: {valid_quantizations}")
    
    return errors


def get_gpu_optimized_config(vram_gb: float, base_config: TrainJobConfig) -> TrainJobConfig:
    """GPU VRAMに基づいて設定を最適化"""
    config = base_config
    
    if vram_gb >= 24:
        config.training.per_device_train_batch_size = 4
        config.training.gradient_accumulation_steps = 4
        config.model.max_seq_length = min(4096, config.model.max_seq_length)
    elif vram_gb >= 16:
        config.training.per_device_train_batch_size = 2
        config.training.gradient_accumulation_steps = 8
        config.model.max_seq_length = min(2048, config.model.max_seq_length)
    elif vram_gb >= 12:
        config.training.per_device_train_batch_size = 1
        config.training.gradient_accumulation_steps = 16
        config.model.max_seq_length = min(2048, config.model.max_seq_length)
    elif vram_gb >= 8:
        config.training.per_device_train_batch_size = 1
        config.training.gradient_accumulation_steps = 16
        config.model.max_seq_length = min(1024, config.model.max_seq_length)
    else:
        config.training.per_device_train_batch_size = 1
        config.training.gradient_accumulation_steps = 32
        config.model.max_seq_length = min(512, config.model.max_seq_length)
    
    return config
