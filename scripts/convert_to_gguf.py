"""
GGUF変換モジュール
学習済みモデルをOllamaで使用可能なGGUF形式に変換

使用方法:
    # 単体での使用
    python3 scripts/convert_to_gguf.py --model-path /path/to/merged/model --output-dir /path/to/output
    
    # Ollamaへの登録も行う場合
    python3 scripts/convert_to_gguf.py --model-path /path/to/merged/model --output-dir /path/to/output --register-ollama --model-name my-model
"""
import os
import sys
import argparse
import subprocess
import shutil
from pathlib import Path
from typing import Optional
import tempfile


# 量子化タイプの定義
QUANTIZATION_TYPES = {
    "q4_0": "4-bit quantization (fastest, lowest quality)",
    "q4_1": "4-bit quantization (faster, low quality)",
    "q4_k_m": "4-bit quantization (balanced speed/quality)",
    "q4_k_s": "4-bit quantization (smaller, balanced)",
    "q5_0": "5-bit quantization (medium speed/quality)",
    "q5_1": "5-bit quantization (medium speed/quality)",
    "q5_k_m": "5-bit quantization (balanced)",
    "q5_k_s": "5-bit quantization (smaller)",
    "q6_k": "6-bit quantization (slower, higher quality)",
    "q8_0": "8-bit quantization (slowest, highest quality)",
    "f16": "16-bit float (no quantization)",
    "f32": "32-bit float (no quantization)",
}


def check_llama_cpp():
    """llama.cppがインストールされているか確認"""
    try:
        import llama_cpp
        return True
    except ImportError:
        return False


def convert_to_gguf_unsloth(
    model_path: str,
    output_path: str,
    quantization: str = "q4_k_m",
    logger=None
) -> Optional[str]:
    """
    Unslothの機能を使用してGGUF形式に変換
    
    Args:
        model_path: マージ済みモデルのパス
        output_path: 出力先パス
        quantization: 量子化タイプ
        logger: ロガー（オプション）
    
    Returns:
        生成されたGGUFファイルのパス
    """
    def log(msg):
        if logger:
            logger.info(msg)
        else:
            print(msg)
    
    log(f"Converting model to GGUF format...")
    log(f"Model path: {model_path}")
    log(f"Output path: {output_path}")
    log(f"Quantization: {quantization}")
    
    try:
        # Unslothのsave_pretrained_ggufを使用
        from unsloth import FastLanguageModel
        import torch
        
        # モデルを読み込み
        log("Loading model...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=2048,
            dtype=torch.float16,
            load_in_4bit=False,
        )
        
        # 出力ディレクトリの作成
        os.makedirs(output_path, exist_ok=True)
        
        # GGUF形式で保存
        log(f"Saving as GGUF with {quantization} quantization...")
        model.save_pretrained_gguf(
            output_path,
            tokenizer,
            quantization_method=quantization,
        )
        
        # 生成されたGGUFファイルを探す
        gguf_files = list(Path(output_path).glob("*.gguf"))
        if gguf_files:
            gguf_path = str(gguf_files[0])
            log(f"✅ GGUF file created: {gguf_path}")
            return gguf_path
        else:
            log("❌ GGUF file not found after conversion")
            return None
            
    except Exception as e:
        log(f"❌ Error during conversion: {e}")
        raise


def create_modelfile(
    gguf_path: str,
    output_path: str,
    system_prompt: str = "You are a helpful AI assistant.",
    template: str = "alpaca",
    parameters: Optional[dict] = None
) -> str:
    """
    Ollama用のModelfileを作成
    
    Args:
        gguf_path: GGUFファイルのパス
        output_path: Modelfileの出力先ディレクトリ
        system_prompt: システムプロンプト
        template: プロンプトテンプレート
        parameters: 追加パラメータ
    
    Returns:
        Modelfileのパス
    """
    # テンプレートの定義
    templates = {
        "alpaca": '''TEMPLATE """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{{ .Prompt }}

### Response:
"""''',
        
        "chatml": '''TEMPLATE """<|im_start|>system
{{ .System }}<|im_end|>
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
"""''',
        
        "llama3": '''TEMPLATE """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|><|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""''',
        
        "raw": '''TEMPLATE "{{ .Prompt }}"''',
    }
    
    # デフォルトパラメータ
    default_params = {
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "repeat_penalty": 1.1,
    }
    
    if parameters:
        default_params.update(parameters)
    
    # Modelfileの内容を構築
    modelfile_content = f'FROM {gguf_path}\n\n'
    modelfile_content += f'SYSTEM "{system_prompt}"\n\n'
    modelfile_content += templates.get(template, templates["alpaca"]) + '\n\n'
    
    for key, value in default_params.items():
        modelfile_content += f'PARAMETER {key} {value}\n'
    
    # Modelfileを保存
    modelfile_path = os.path.join(output_path, "Modelfile")
    with open(modelfile_path, 'w') as f:
        f.write(modelfile_content)
    
    return modelfile_path


def register_with_ollama(
    modelfile_path: str,
    model_name: str,
    logger=None
) -> bool:
    """
    OllamaにモデルをModelfileから登録
    
    Args:
        modelfile_path: Modelfileのパス
        model_name: Ollamaでのモデル名
        logger: ロガー（オプション）
    
    Returns:
        成功したかどうか
    """
    def log(msg):
        if logger:
            logger.info(msg)
        else:
            print(msg)
    
    log(f"Registering model '{model_name}' with Ollama...")
    
    try:
        # ollama createコマンドを実行
        result = subprocess.run(
            ["ollama", "create", model_name, "-f", modelfile_path],
            capture_output=True,
            text=True,
            timeout=600  # 10分タイムアウト
        )
        
        if result.returncode == 0:
            log(f"✅ Model '{model_name}' registered successfully!")
            log(f"   Run with: ollama run {model_name}")
            return True
        else:
            log(f"❌ Failed to register model: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        log("❌ Registration timed out")
        return False
    except FileNotFoundError:
        log("❌ Ollama not found. Make sure Ollama is installed and running.")
        return False
    except Exception as e:
        log(f"❌ Error during registration: {e}")
        return False


def convert_and_register(
    model_path: str,
    output_dir: str,
    model_name: str,
    quantization: str = "q4_k_m",
    system_prompt: str = "You are a helpful AI assistant.",
    template: str = "alpaca",
    register_ollama: bool = True,
    logger=None
) -> dict:
    """
    モデルの変換とOllamaへの登録を一括で実行
    
    Args:
        model_path: マージ済みモデルのパス
        output_dir: 出力先ディレクトリ
        model_name: Ollamaでのモデル名
        quantization: 量子化タイプ
        system_prompt: システムプロンプト
        template: プロンプトテンプレート
        register_ollama: Ollamaに登録するか
        logger: ロガー（オプション）
    
    Returns:
        結果の辞書
    """
    def log(msg):
        if logger:
            logger.info(msg)
        else:
            print(msg)
    
    result = {
        "success": False,
        "gguf_path": None,
        "modelfile_path": None,
        "model_name": model_name,
        "registered": False,
    }
    
    try:
        # 出力ディレクトリの作成
        os.makedirs(output_dir, exist_ok=True)
        
        # GGUF形式に変換
        log("=" * 60)
        log("Step 1: Converting to GGUF format")
        log("=" * 60)
        
        gguf_path = convert_to_gguf_unsloth(
            model_path=model_path,
            output_path=output_dir,
            quantization=quantization,
            logger=logger
        )
        
        if not gguf_path:
            return result
        
        result["gguf_path"] = gguf_path
        
        # Modelfileの作成
        log("=" * 60)
        log("Step 2: Creating Modelfile")
        log("=" * 60)
        
        modelfile_path = create_modelfile(
            gguf_path=gguf_path,
            output_path=output_dir,
            system_prompt=system_prompt,
            template=template,
        )
        
        result["modelfile_path"] = modelfile_path
        log(f"✅ Modelfile created: {modelfile_path}")
        
        # Ollamaへの登録
        if register_ollama:
            log("=" * 60)
            log("Step 3: Registering with Ollama")
            log("=" * 60)
            
            registered = register_with_ollama(
                modelfile_path=modelfile_path,
                model_name=model_name,
                logger=logger
            )
            
            result["registered"] = registered
        
        result["success"] = True
        
        log("=" * 60)
        log("Conversion Complete!")
        log("=" * 60)
        log(f"GGUF file: {gguf_path}")
        log(f"Modelfile: {modelfile_path}")
        if register_ollama:
            log(f"Ollama model: {model_name}")
            log(f"Run with: ollama run {model_name}")
        
        return result
        
    except Exception as e:
        log(f"❌ Conversion failed: {e}")
        import traceback
        log(traceback.format_exc())
        return result


def main():
    parser = argparse.ArgumentParser(
        description="Convert trained models to GGUF format for Ollama",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic conversion
  python3 scripts/convert_to_gguf.py --model-path ./models/outputs/my-model/merged --output-dir ./models/gguf/my-model
  
  # With Ollama registration
  python3 scripts/convert_to_gguf.py --model-path ./models/outputs/my-model/merged --output-dir ./models/gguf/my-model --register-ollama --model-name my-finetuned-model
  
  # With custom quantization
  python3 scripts/convert_to_gguf.py --model-path ./models/outputs/my-model/merged --output-dir ./models/gguf/my-model --quantization q8_0

Quantization types:
  q4_0     - 4-bit (fastest, lowest quality)
  q4_k_m   - 4-bit K-quant (balanced, recommended)
  q5_k_m   - 5-bit K-quant (better quality)
  q6_k     - 6-bit K-quant (high quality)
  q8_0     - 8-bit (highest quality)
  f16      - 16-bit float (no quantization)
        """
    )
    parser.add_argument(
        "--model-path", "-m",
        type=str,
        required=True,
        help="Path to the merged model directory"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        required=True,
        help="Output directory for GGUF files"
    )
    parser.add_argument(
        "--quantization", "-q",
        type=str,
        default="q4_k_m",
        choices=list(QUANTIZATION_TYPES.keys()),
        help="Quantization type (default: q4_k_m)"
    )
    parser.add_argument(
        "--register-ollama",
        action="store_true",
        help="Register the model with Ollama"
    )
    parser.add_argument(
        "--model-name", "-n",
        type=str,
        default=None,
        help="Model name for Ollama registration"
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="You are a helpful AI assistant.",
        help="System prompt for the model"
    )
    parser.add_argument(
        "--template",
        type=str,
        default="alpaca",
        choices=["alpaca", "chatml", "llama3", "raw"],
        help="Prompt template for the model"
    )
    
    args = parser.parse_args()
    
    # モデル名の自動生成
    model_name = args.model_name
    if model_name is None:
        model_name = Path(args.model_path).parent.name
        if model_name in ["merged", "lora"]:
            model_name = Path(args.model_path).parent.parent.name
    
    # 変換と登録を実行
    result = convert_and_register(
        model_path=args.model_path,
        output_dir=args.output_dir,
        model_name=model_name,
        quantization=args.quantization,
        system_prompt=args.system_prompt,
        template=args.template,
        register_ollama=args.register_ollama,
    )
    
    sys.exit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
