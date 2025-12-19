"""
MLflow統合モジュール
学習の追跡・管理機能を提供
"""
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Union
from contextlib import contextmanager

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None


class MLflowTracker:
    """MLflowを使った学習追跡クラス"""
    
    def __init__(
        self,
        tracking_uri: str = "file:///workspace/work/mlruns",
        experiment_name: str = "unsloth-training",
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        enabled: bool = True
    ):
        self.enabled = enabled and MLFLOW_AVAILABLE
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.run_name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.tags = tags or {}
        
        self.run = None
        self.run_id = None
        self.experiment_id = None
        
        if not MLFLOW_AVAILABLE and enabled:
            print("⚠️ MLflow is not installed. Tracking will be disabled.")
            print("   Install with: pip install mlflow")
    
    def setup(self):
        """MLflowのセットアップ"""
        if not self.enabled:
            return
        
        # トラッキングURIの設定
        mlflow.set_tracking_uri(self.tracking_uri)
        
        # 実験の取得または作成
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            self.experiment_id = mlflow.create_experiment(self.experiment_name)
        else:
            self.experiment_id = experiment.experiment_id
        
        mlflow.set_experiment(self.experiment_name)
    
    def start_run(self) -> Optional[str]:
        """MLflow runを開始"""
        if not self.enabled:
            return None
        
        self.run = mlflow.start_run(run_name=self.run_name)
        self.run_id = self.run.info.run_id
        
        # タグの設定
        for key, value in self.tags.items():
            mlflow.set_tag(key, value)
        
        return self.run_id
    
    def end_run(self, status: str = "FINISHED"):
        """MLflow runを終了"""
        if not self.enabled or self.run is None:
            return
        
        mlflow.end_run(status=status)
        self.run = None
    
    def log_params(self, params: Dict[str, Any]):
        """パラメータをログ"""
        if not self.enabled:
            return
        
        # MLflowはネストした辞書を直接扱えないのでフラット化
        flat_params = self._flatten_dict(params)
        
        # パラメータ値を文字列に変換（MLflowの制限対応）
        for key, value in flat_params.items():
            try:
                mlflow.log_param(key, str(value)[:500])  # 500文字制限
            except Exception as e:
                print(f"Warning: Could not log param {key}: {e}")
    
    def log_config(self, config):
        """設定オブジェクトをログ"""
        if not self.enabled:
            return
        
        # 設定を辞書に変換してログ
        config_dict = {
            "training_type": config.training_type,
            "model.name": config.model.name,
            "model.max_seq_length": config.model.max_seq_length,
            "model.load_in_4bit": config.model.load_in_4bit,
            "lora.r": config.lora.r,
            "lora.lora_alpha": config.lora.lora_alpha,
            "lora.lora_dropout": config.lora.lora_dropout,
            "training.num_train_epochs": config.training.num_train_epochs,
            "training.per_device_train_batch_size": config.training.per_device_train_batch_size,
            "training.gradient_accumulation_steps": config.training.gradient_accumulation_steps,
            "training.learning_rate": config.training.learning_rate,
            "training.warmup_steps": config.training.warmup_steps,
            "training.lr_scheduler_type": config.training.lr_scheduler_type,
            "training.weight_decay": config.training.weight_decay,
            "training.max_grad_norm": config.training.max_grad_norm,
            "training.fp16": config.training.fp16,
            "training.bf16": config.training.bf16,
            "training.optim": config.training.optim,
            "training.packing": config.training.packing,
            "data.train_data_path": config.data.train_data_path,
            "data.prompt_template": config.data.prompt_template,
            "output.output_dir": config.output.output_dir,
            "seed": config.seed,
        }
        
        self.log_params(config_dict)
    
    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """単一メトリクスをログ"""
        if not self.enabled:
            return
        
        mlflow.log_metric(key, value, step=step)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """複数メトリクスをログ"""
        if not self.enabled:
            return
        
        mlflow.log_metrics(metrics, step=step)
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """アーティファクト（ファイル）をログ"""
        if not self.enabled:
            return
        
        if Path(local_path).exists():
            mlflow.log_artifact(local_path, artifact_path)
    
    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None):
        """アーティファクト（ディレクトリ）をログ"""
        if not self.enabled:
            return
        
        if Path(local_dir).exists():
            mlflow.log_artifacts(local_dir, artifact_path)
    
    def log_model_info(self, model_name: str, model_path: str):
        """モデル情報をログ"""
        if not self.enabled:
            return
        
        mlflow.set_tag("model.name", model_name)
        mlflow.set_tag("model.path", model_path)
    
    def log_gpu_info(self):
        """GPU情報をログ"""
        if not self.enabled:
            return
        
        try:
            import torch
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                mlflow.set_tag("gpu.name", torch.cuda.get_device_name(device))
                mlflow.set_tag("gpu.count", torch.cuda.device_count())
                mlflow.log_param("gpu.total_memory_gb", 
                               f"{torch.cuda.get_device_properties(device).total_memory / 1024**3:.2f}")
        except Exception:
            pass
    
    def log_dataset_info(self, dataset_size: int, data_path: str):
        """データセット情報をログ"""
        if not self.enabled:
            return
        
        mlflow.log_param("dataset.size", dataset_size)
        mlflow.log_param("dataset.path", data_path)
    
    def set_tag(self, key: str, value: str):
        """タグを設定"""
        if not self.enabled:
            return
        
        mlflow.set_tag(key, value)
    
    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
        """ネストした辞書をフラット化"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    @contextmanager
    def run_context(self):
        """コンテキストマネージャとしてrunを管理"""
        try:
            self.setup()
            self.start_run()
            yield self
        except Exception as e:
            self.set_tag("error", str(e)[:500])
            self.end_run(status="FAILED")
            raise
        else:
            self.end_run(status="FINISHED")


class TrainerCallback:
    """Trainerのコールバッククラス（MLflow連携用）"""
    
    def __init__(self, tracker: MLflowTracker):
        self.tracker = tracker
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """ログ時に呼ばれるコールバック"""
        if logs is None:
            return
        
        # metricsをMLflowにログ
        metrics = {}
        for key, value in logs.items():
            if isinstance(value, (int, float)):
                metrics[key] = value
        
        if metrics:
            self.tracker.log_metrics(metrics, step=state.global_step)
    
    def on_train_end(self, args, state, control, **kwargs):
        """学習終了時に呼ばれるコールバック"""
        # 最終メトリクスをログ
        if hasattr(state, 'log_history') and state.log_history:
            final_metrics = {}
            for entry in state.log_history:
                for key, value in entry.items():
                    if isinstance(value, (int, float)) and key not in ['step', 'epoch']:
                        final_metrics[f"final_{key}"] = value
            
            if final_metrics:
                self.tracker.log_metrics(final_metrics)


def create_mlflow_callback(tracker: MLflowTracker):
    """Transformers TrainerCallback を作成"""
    try:
        from transformers import TrainerCallback as HFTrainerCallback
        
        class MLflowTrainerCallback(HFTrainerCallback):
            def __init__(self, mlflow_tracker: MLflowTracker):
                self.tracker = mlflow_tracker
            
            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs is None or not self.tracker.enabled:
                    return
                
                metrics = {k: v for k, v in logs.items() 
                          if isinstance(v, (int, float))}
                
                if metrics:
                    self.tracker.log_metrics(metrics, step=state.global_step)
            
            def on_train_end(self, args, state, control, **kwargs):
                if not self.tracker.enabled:
                    return
                
                if hasattr(state, 'log_history') and state.log_history:
                    final_metrics = {}
                    for entry in state.log_history:
                        for key, value in entry.items():
                            if isinstance(value, (int, float)) and key not in ['step', 'epoch']:
                                final_metrics[f"final_{key}"] = value
                    
                    if final_metrics:
                        self.tracker.log_metrics(final_metrics)
        
        return MLflowTrainerCallback(tracker)
    
    except ImportError:
        return None


def setup_mlflow_tracker(
    enabled: bool = True,
    tracking_uri: str = "file:///workspace/work/mlruns",
    experiment_name: str = "unsloth-training",
    run_name: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None
) -> MLflowTracker:
    """MLflowTrackerをセットアップして返す"""
    return MLflowTracker(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        run_name=run_name,
        tags=tags,
        enabled=enabled
    )
