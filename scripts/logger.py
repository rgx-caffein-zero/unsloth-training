"""
ログ管理モジュール
ファイルとコンソールへの同時出力をサポート
"""
import os
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
import io


class TeeStream:
    """標準出力をファイルにも同時出力するストリーム"""
    
    def __init__(self, original_stream, log_file: io.TextIOWrapper):
        self.original_stream = original_stream
        self.log_file = log_file
    
    def write(self, message):
        self.original_stream.write(message)
        self.log_file.write(message)
        self.flush()
    
    def flush(self):
        self.original_stream.flush()
        self.log_file.flush()
    
    def fileno(self):
        return self.original_stream.fileno()


class TrainingLogger:
    """学習用ロガークラス"""
    
    def __init__(
        self,
        log_dir: str,
        log_file: str = "training.log",
        level: int = logging.INFO,
        capture_stdout: bool = True
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_path = self.log_dir / log_file
        self.level = level
        self.capture_stdout = capture_stdout
        
        # ロガーの設定
        self.logger = logging.getLogger("training")
        self.logger.setLevel(level)
        self.logger.handlers = []  # 既存のハンドラをクリア
        
        # フォーマッタの作成
        formatter = logging.Formatter(
            fmt='%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # コンソールハンドラ
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # ファイルハンドラ
        file_handler = logging.FileHandler(self.log_path, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        self._original_stdout = None
        self._original_stderr = None
        self._log_file_handle = None
        
        # 標準出力のキャプチャを開始
        if capture_stdout:
            self._start_capture()
    
    def _start_capture(self):
        """標準出力・標準エラー出力のキャプチャを開始"""
        self._log_file_handle = open(self.log_path, 'a', encoding='utf-8')
        
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        
        sys.stdout = TeeStream(self._original_stdout, self._log_file_handle)
        sys.stderr = TeeStream(self._original_stderr, self._log_file_handle)
    
    def _stop_capture(self):
        """標準出力・標準エラー出力のキャプチャを停止"""
        if self._original_stdout is not None:
            sys.stdout = self._original_stdout
        if self._original_stderr is not None:
            sys.stderr = self._original_stderr
        if self._log_file_handle is not None:
            self._log_file_handle.close()
    
    def info(self, message: str):
        """INFOレベルのログを出力"""
        self.logger.info(message)
    
    def warning(self, message: str):
        """WARNINGレベルのログを出力"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """ERRORレベルのログを出力"""
        self.logger.error(message)
    
    def debug(self, message: str):
        """DEBUGレベルのログを出力"""
        self.logger.debug(message)
    
    def section(self, title: str, char: str = "=", width: int = 60):
        """セクション区切りを出力"""
        border = char * width
        self.info(border)
        self.info(f" {title}")
        self.info(border)
    
    def config(self, config_dict: dict, title: str = "Configuration"):
        """設定内容をログに出力"""
        self.section(title)
        self._log_dict(config_dict, indent=0)
        self.info("=" * 60)
    
    def _log_dict(self, d: dict, indent: int = 0):
        """辞書を整形してログに出力"""
        prefix = "  " * indent
        for key, value in d.items():
            if isinstance(value, dict):
                self.info(f"{prefix}{key}:")
                self._log_dict(value, indent + 1)
            else:
                self.info(f"{prefix}{key}: {value}")
    
    def metrics(self, metrics_dict: dict, step: Optional[int] = None):
        """メトリクスをログに出力"""
        step_str = f" (step {step})" if step is not None else ""
        metrics_str = ", ".join([f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" 
                                  for k, v in metrics_dict.items()])
        self.info(f"Metrics{step_str}: {metrics_str}")
    
    def gpu_status(self):
        """GPU状態をログに出力"""
        try:
            import torch
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                name = torch.cuda.get_device_name(device)
                total = torch.cuda.get_device_properties(device).total_memory / 1024**3
                allocated = torch.cuda.memory_allocated(device) / 1024**3
                reserved = torch.cuda.memory_reserved(device) / 1024**3
                
                self.info(f"GPU: {name}")
                self.info(f"  Total VRAM: {total:.2f} GB")
                self.info(f"  Allocated: {allocated:.2f} GB")
                self.info(f"  Reserved: {reserved:.2f} GB")
            else:
                self.warning("No GPU available")
        except Exception as e:
            self.error(f"Failed to get GPU status: {e}")
    
    def training_start(self, config):
        """学習開始時のログ"""
        self.section("Training Started", "=", 60)
        self.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.info(f"Training Type: {config.training_type}")
        self.info(f"Model: {config.model.name}")
        self.info(f"Output: {config.output.output_dir}")
        self.info(f"Log File: {self.log_path}")
        self.gpu_status()
    
    def training_end(self, success: bool = True, error_msg: Optional[str] = None):
        """学習終了時のログ"""
        self.section("Training Finished", "=", 60)
        self.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if success:
            self.info("Status: SUCCESS ✅")
        else:
            self.error(f"Status: FAILED ❌")
            if error_msg:
                self.error(f"Error: {error_msg}")
        
        self.gpu_status()
    
    def close(self):
        """ロガーをクローズ"""
        self._stop_capture()
        
        # ハンドラをクローズ
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)


def setup_logger(
    output_dir: str,
    log_file: str = "training.log",
    level: int = logging.INFO,
    capture_stdout: bool = True
) -> TrainingLogger:
    """ロガーをセットアップして返す"""
    return TrainingLogger(
        log_dir=output_dir,
        log_file=log_file,
        level=level,
        capture_stdout=capture_stdout
    )
