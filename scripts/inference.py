"""
Ollamaを使った推論スクリプト
学習済みモデルまたはHugging Faceモデルを使用して推論を実行

使用方法:
    # インタラクティブモード（対話形式）
    python3 scripts/inference.py --config configs/inference_example.yaml
    
    # 単一プロンプト
    python3 scripts/inference.py --config configs/inference_example.yaml --prompt "こんにちは"
    
    # バッチ推論
    python3 scripts/inference.py --config configs/inference_example.yaml --input prompts.txt --output results.jsonl
    
    # モデル一覧の表示
    python3 scripts/inference.py --list-models
"""
import os
import sys
import argparse
import json
import requests
from pathlib import Path
from typing import Optional, List, Dict, Any, Generator
from dataclasses import dataclass, field

import yaml


# Ollama API エンドポイント
OLLAMA_BASE_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
if not OLLAMA_BASE_URL.startswith("http"):
    OLLAMA_BASE_URL = f"http://{OLLAMA_BASE_URL}"


# プロンプトテンプレート
PROMPT_TEMPLATES = {
    "alpaca": """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
""",
    
    "alpaca_with_input": """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
""",
    
    "chatml": """<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
""",
    
    "llama3": """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system}<|eot_id|><|start_header_id|>user<|end_header_id|>

{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

""",
    
    "raw": "{instruction}",
    
    "none": "{instruction}",
}


@dataclass
class ModelConfig:
    """モデル設定"""
    # Ollamaモデル名（例: "llama2", "mistral", "my-finetuned-model"）
    name: str = "llama2"
    # 学習済みモデルのパス（GGUF変換前のパス、オプション）
    local_path: Optional[str] = None


@dataclass
class SamplingConfig:
    """サンプリング設定"""
    # 最大生成トークン数
    num_predict: int = 512
    # 温度（0.0で決定論的）
    temperature: float = 0.7
    # Top-p サンプリング
    top_p: float = 0.9
    # Top-k サンプリング
    top_k: int = 40
    # 繰り返しペナルティ
    repeat_penalty: float = 1.1
    # 停止トークン
    stop: List[str] = field(default_factory=list)
    # シード（再現性のため）
    seed: Optional[int] = None


@dataclass
class ServerConfig:
    """サーバー設定"""
    # Ollama API ホスト
    host: str = "localhost"
    # Ollama API ポート
    port: int = 11434
    # タイムアウト（秒）
    timeout: int = 300


@dataclass
class PromptConfig:
    """プロンプト設定"""
    # プロンプトテンプレート
    template: str = "none"
    # システムプロンプト
    system_prompt: str = "You are a helpful AI assistant."


@dataclass
class InferenceConfig:
    """推論設定全体"""
    description: str = ""
    model: ModelConfig = field(default_factory=ModelConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)


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


def load_inference_config(config_path: str) -> InferenceConfig:
    """YAML設定ファイルを読み込み"""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        raw_config = yaml.safe_load(f)
    
    if raw_config is None:
        raw_config = {}
    
    model_config = _dataclass_from_dict(ModelConfig, raw_config.get('model'))
    sampling_config = _dataclass_from_dict(SamplingConfig, raw_config.get('sampling'))
    server_config = _dataclass_from_dict(ServerConfig, raw_config.get('server'))
    prompt_config = _dataclass_from_dict(PromptConfig, raw_config.get('prompt'))
    
    return InferenceConfig(
        description=raw_config.get('description', ''),
        model=model_config,
        sampling=sampling_config,
        server=server_config,
        prompt=prompt_config,
    )


class OllamaClient:
    """Ollama APIクライアント"""
    
    def __init__(self, host: str = "localhost", port: int = 11434, timeout: int = 300):
        self.base_url = f"http://{host}:{port}"
        self.timeout = timeout
    
    def _make_request(
        self,
        endpoint: str,
        method: str = "GET",
        data: Optional[dict] = None,
        stream: bool = False
    ) -> requests.Response:
        """APIリクエストを送信"""
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method == "GET":
                response = requests.get(url, timeout=self.timeout)
            elif method == "POST":
                response = requests.post(
                    url,
                    json=data,
                    timeout=self.timeout,
                    stream=stream
                )
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            return response
            
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Could not connect to Ollama at {self.base_url}. "
                "Make sure Ollama is running."
            )
        except requests.exceptions.Timeout:
            raise TimeoutError(f"Request to Ollama timed out after {self.timeout} seconds")
    
    def list_models(self) -> List[dict]:
        """利用可能なモデル一覧を取得"""
        response = self._make_request("/api/tags")
        
        if response.status_code == 200:
            return response.json().get("models", [])
        else:
            raise RuntimeError(f"Failed to list models: {response.text}")
    
    def pull_model(self, model_name: str) -> bool:
        """モデルをダウンロード"""
        print(f"Pulling model: {model_name}")
        
        response = self._make_request(
            "/api/pull",
            method="POST",
            data={"name": model_name},
            stream=True
        )
        
        if response.status_code == 200:
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    status = data.get("status", "")
                    if "pulling" in status:
                        print(f"  {status}")
                    elif "success" in status:
                        print(f"✅ {status}")
            return True
        else:
            print(f"❌ Failed to pull model: {response.text}")
            return False
    
    def model_exists(self, model_name: str) -> bool:
        """モデルが存在するか確認"""
        models = self.list_models()
        return any(m.get("name", "").startswith(model_name) for m in models)
    
    def generate(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        options: Optional[dict] = None,
        stream: bool = False
    ) -> Generator[str, None, None] | str:
        """テキスト生成"""
        data = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
        }
        
        if system:
            data["system"] = system
        
        if options:
            data["options"] = options
        
        response = self._make_request(
            "/api/generate",
            method="POST",
            data=data,
            stream=stream
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"Generation failed: {response.text}")
        
        if stream:
            def stream_response():
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line)
                        if "response" in data:
                            yield data["response"]
                        if data.get("done", False):
                            break
            return stream_response()
        else:
            result = response.json()
            return result.get("response", "")
    
    def chat(
        self,
        model: str,
        messages: List[dict],
        options: Optional[dict] = None,
        stream: bool = False
    ) -> Generator[str, None, None] | str:
        """チャット形式での生成"""
        data = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }
        
        if options:
            data["options"] = options
        
        response = self._make_request(
            "/api/chat",
            method="POST",
            data=data,
            stream=stream
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"Chat failed: {response.text}")
        
        if stream:
            def stream_response():
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line)
                        message = data.get("message", {})
                        if "content" in message:
                            yield message["content"]
                        if data.get("done", False):
                            break
            return stream_response()
        else:
            result = response.json()
            return result.get("message", {}).get("content", "")


class OllamaInference:
    """Ollamaを使った推論クラス"""
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.client = OllamaClient(
            host=config.server.host,
            port=config.server.port,
            timeout=config.server.timeout
        )
        self.model_name = config.model.name
    
    def check_model(self) -> bool:
        """モデルの存在確認"""
        try:
            if self.client.model_exists(self.model_name):
                print(f"✅ Model '{self.model_name}' is available")
                return True
            else:
                print(f"⚠️ Model '{self.model_name}' not found locally")
                return False
        except Exception as e:
            print(f"❌ Error checking model: {e}")
            return False
    
    def pull_model_if_needed(self) -> bool:
        """必要に応じてモデルをダウンロード"""
        if not self.check_model():
            print(f"Attempting to pull model '{self.model_name}'...")
            return self.client.pull_model(self.model_name)
        return True
    
    def _get_options(self) -> dict:
        """サンプリングオプションを取得"""
        options = {
            "num_predict": self.config.sampling.num_predict,
            "temperature": self.config.sampling.temperature,
            "top_p": self.config.sampling.top_p,
            "top_k": self.config.sampling.top_k,
            "repeat_penalty": self.config.sampling.repeat_penalty,
        }
        
        if self.config.sampling.stop:
            options["stop"] = self.config.sampling.stop
        
        if self.config.sampling.seed is not None:
            options["seed"] = self.config.sampling.seed
        
        return options
    
    def format_prompt(self, instruction: str, input_text: str = "", system: str = "") -> str:
        """プロンプトをフォーマット"""
        template_name = self.config.prompt.template
        template = PROMPT_TEMPLATES.get(template_name, PROMPT_TEMPLATES["none"])
        
        system = system or self.config.prompt.system_prompt
        
        if template_name == "alpaca" and input_text:
            template = PROMPT_TEMPLATES["alpaca_with_input"]
            return template.format(instruction=instruction, input=input_text)
        elif template_name in ["chatml", "llama3"]:
            return template.format(system=system, instruction=instruction)
        else:
            return template.format(instruction=instruction)
    
    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        stream: bool = False
    ) -> Generator[str, None, None] | str:
        """テキスト生成"""
        system_prompt = system or self.config.prompt.system_prompt
        options = self._get_options()
        
        return self.client.generate(
            model=self.model_name,
            prompt=prompt,
            system=system_prompt,
            options=options,
            stream=stream
        )
    
    def chat(
        self,
        messages: List[dict],
        stream: bool = False
    ) -> Generator[str, None, None] | str:
        """チャット形式での生成"""
        options = self._get_options()
        
        return self.client.chat(
            model=self.model_name,
            messages=messages,
            options=options,
            stream=stream
        )
    
    def generate_single(
        self,
        instruction: str,
        input_text: str = "",
        system: str = ""
    ) -> str:
        """単一プロンプトの生成"""
        prompt = self.format_prompt(instruction, input_text, system)
        return self.generate(prompt, system)
    
    def interactive_mode(self, use_chat: bool = True):
        """インタラクティブモード（対話形式）"""
        print("\n" + "=" * 60)
        print("Interactive Mode (Ollama)")
        print(f"Model: {self.model_name}")
        print("Type 'quit' or 'exit' to end the session")
        print("Type 'clear' to clear the screen")
        print("Type '/chat' to toggle chat mode")
        print("=" * 60 + "\n")
        
        chat_history = []
        system_prompt = self.config.prompt.system_prompt
        
        if use_chat and system_prompt:
            chat_history.append({"role": "system", "content": system_prompt})
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit']:
                    print("Goodbye!")
                    break
                
                if user_input.lower() == 'clear':
                    os.system('clear' if os.name == 'posix' else 'cls')
                    chat_history = []
                    if use_chat and system_prompt:
                        chat_history.append({"role": "system", "content": system_prompt})
                    continue
                
                if user_input.lower() == '/chat':
                    use_chat = not use_chat
                    print(f"Chat mode: {'ON' if use_chat else 'OFF'}")
                    chat_history = []
                    if use_chat and system_prompt:
                        chat_history.append({"role": "system", "content": system_prompt})
                    continue
                
                print("\nAssistant: ", end="", flush=True)
                
                if use_chat:
                    chat_history.append({"role": "user", "content": user_input})
                    
                    # ストリーミング応答
                    full_response = ""
                    for chunk in self.chat(chat_history, stream=True):
                        print(chunk, end="", flush=True)
                        full_response += chunk
                    
                    chat_history.append({"role": "assistant", "content": full_response})
                else:
                    prompt = self.format_prompt(user_input)
                    
                    # ストリーミング応答
                    for chunk in self.generate(prompt, stream=True):
                        print(chunk, end="", flush=True)
                
                print("\n")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")
    
    def batch_inference(
        self,
        input_file: str,
        output_file: str
    ):
        """バッチ推論"""
        input_path = Path(input_file)
        output_path = Path(output_file)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        print(f"Reading prompts from: {input_file}")
        
        prompts = []
        if input_path.suffix == ".txt":
            with open(input_path, 'r', encoding='utf-8') as f:
                prompts = [{"prompt": line.strip()} for line in f if line.strip()]
        elif input_path.suffix == ".jsonl":
            with open(input_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        if isinstance(data, str):
                            prompts.append({"prompt": data})
                        elif isinstance(data, dict):
                            prompts.append(data)
        else:
            raise ValueError(f"Unsupported file format: {input_path.suffix}")
        
        print(f"Processing {len(prompts)} prompts...")
        
        # 結果の保存
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, item in enumerate(prompts):
                print(f"  Processing {i+1}/{len(prompts)}...", end="\r")
                
                instruction = item.get("instruction", item.get("prompt", ""))
                input_text = item.get("input", "")
                
                prompt = self.format_prompt(instruction, input_text)
                response = self.generate(prompt)
                
                output_data = {
                    "id": i,
                    "prompt": instruction,
                    "input": input_text,
                    "response": response
                }
                f.write(json.dumps(output_data, ensure_ascii=False) + "\n")
        
        print(f"\n✅ Batch inference completed. Results saved to: {output_file}")


def list_models(host: str = "localhost", port: int = 11434):
    """利用可能なモデル一覧を表示"""
    try:
        client = OllamaClient(host=host, port=port)
        models = client.list_models()
        
        print("\n" + "=" * 60)
        print("Available Ollama Models")
        print("=" * 60)
        
        if not models:
            print("No models found. Pull a model with: ollama pull <model-name>")
        else:
            for model in models:
                name = model.get("name", "unknown")
                size = model.get("size", 0) / (1024**3)  # Convert to GB
                modified = model.get("modified_at", "")
                print(f"  {name}")
                print(f"    Size: {size:.2f} GB")
                print(f"    Modified: {modified}")
                print()
        
        print("=" * 60)
        
    except ConnectionError as e:
        print(f"❌ {e}")
        print("Make sure Ollama is running with: ollama serve")


def main():
    parser = argparse.ArgumentParser(
        description="Ollama inference script for trained models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python3 scripts/inference.py --config configs/inference_example.yaml
  
  # Single prompt
  python3 scripts/inference.py --config configs/inference_example.yaml --prompt "Hello, how are you?"
  
  # Batch inference
  python3 scripts/inference.py --config configs/inference_example.yaml --input prompts.txt --output results.jsonl
  
  # List available models
  python3 scripts/inference.py --list-models
  
  # Pull a model
  python3 scripts/inference.py --pull llama2
        """
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        default=None,
        help="Single prompt for inference"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=None,
        help="Input file for batch inference (txt or jsonl)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file for batch inference results (jsonl)"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Override model name from config"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available Ollama models"
    )
    parser.add_argument(
        "--pull",
        type=str,
        default=None,
        help="Pull a model from Ollama registry"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Ollama API host"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=11434,
        help="Ollama API port"
    )
    parser.add_argument(
        "--no-chat",
        action="store_true",
        help="Disable chat mode in interactive mode"
    )
    
    args = parser.parse_args()
    
    # モデル一覧の表示
    if args.list_models:
        list_models(args.host, args.port)
        return
    
    # モデルのダウンロード
    if args.pull:
        client = OllamaClient(host=args.host, port=args.port)
        success = client.pull_model(args.pull)
        sys.exit(0 if success else 1)
    
    # 設定ファイルが必要な操作
    if args.config is None:
        # デフォルト設定で実行
        config = InferenceConfig()
        if args.model:
            config.model.name = args.model
        else:
            print("Error: Either --config or --model is required")
            parser.print_help()
            sys.exit(1)
    else:
        # 設定ファイルの読み込み
        print(f"Loading configuration from: {args.config}")
        config = load_inference_config(args.config)
    
    # コマンドライン引数で設定を上書き
    if args.model:
        config.model.name = args.model
    if args.host:
        config.server.host = args.host
    if args.port:
        config.server.port = args.port
    
    # 推論インスタンスの作成
    inference = OllamaInference(config)
    
    # モデルの確認
    if not inference.pull_model_if_needed():
        print(f"❌ Model '{config.model.name}' is not available")
        sys.exit(1)
    
    # バッチ推論モード
    if args.input:
        output_file = args.output or "inference_results.jsonl"
        inference.batch_inference(args.input, output_file)
        return
    
    # 単一プロンプトモード
    if args.prompt:
        response = inference.generate_single(args.prompt)
        print(f"\nPrompt: {args.prompt}")
        print(f"\nResponse: {response}")
        return
    
    # インタラクティブモード（デフォルト）
    inference.interactive_mode(use_chat=not args.no_chat)


if __name__ == "__main__":
    main()
