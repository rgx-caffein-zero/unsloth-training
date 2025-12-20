"""
vLLMを使った推論スクリプト
学習済みモデルまたはHugging Faceモデルを使用して推論を実行

使用方法:
    # インタラクティブモード（対話形式）
    python3 scripts/inference.py --config configs/inference_example.yaml
    
    # 単一プロンプト
    python3 scripts/inference.py --config configs/inference_example.yaml --prompt "こんにちは"
    
    # APIサーバーモード
    python3 scripts/inference.py --config configs/inference_example.yaml --server
    
    # バッチ推論
    python3 scripts/inference.py --config configs/inference_example.yaml --input prompts.txt --output results.jsonl
"""
import os
import sys
import argparse
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

import yaml

try:
    from vllm import LLM, SamplingParams
    from vllm.entrypoints.openai.api_server import run_server
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("Warning: vLLM is not installed. Install with: pip install vllm")


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

{system}<|eot_id|><|start_header_id|>user<|end_header_id|}

{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

""",
    
    "raw": "{instruction}",
}


@dataclass
class ModelConfig:
    """モデル設定"""
    path: str = ""
    tokenizer_path: Optional[str] = None
    dtype: str = "auto"
    quantization: Optional[str] = None
    trust_remote_code: bool = True


@dataclass
class EngineConfig:
    """vLLMエンジン設定"""
    gpu_memory_utilization: float = 0.9
    max_model_len: Optional[int] = None
    tensor_parallel_size: int = 1
    swap_space: int = 4
    kv_cache_dtype: str = "auto"
    max_num_seqs: int = 256
    seed: Optional[int] = None


@dataclass
class SamplingConfig:
    """サンプリング設定"""
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 0
    repetition_penalty: float = 1.1
    stop_tokens: List[str] = field(default_factory=list)
    stop_token_ids: List[int] = field(default_factory=list)


@dataclass
class ServerConfig:
    """サーバー設定"""
    host: str = "0.0.0.0"
    port: int = 8000
    api_key: Optional[str] = None
    openai_api: bool = True
    chat_template: Optional[str] = None


@dataclass
class PromptConfig:
    """プロンプト設定"""
    template: str = "alpaca"
    system_prompt: str = "You are a helpful AI assistant."


@dataclass
class InferenceConfig:
    """推論設定全体"""
    description: str = ""
    model: ModelConfig = field(default_factory=ModelConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)
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
    engine_config = _dataclass_from_dict(EngineConfig, raw_config.get('engine'))
    sampling_config = _dataclass_from_dict(SamplingConfig, raw_config.get('sampling'))
    server_config = _dataclass_from_dict(ServerConfig, raw_config.get('server'))
    prompt_config = _dataclass_from_dict(PromptConfig, raw_config.get('prompt'))
    
    return InferenceConfig(
        description=raw_config.get('description', ''),
        model=model_config,
        engine=engine_config,
        sampling=sampling_config,
        server=server_config,
        prompt=prompt_config,
    )


class VLLMInference:
    """vLLMを使った推論クラス"""
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.llm = None
        self.sampling_params = None
    
    def load_model(self):
        """モデルを読み込み"""
        if not VLLM_AVAILABLE:
            raise RuntimeError("vLLM is not installed")
        
        model_path = self.config.model.path
        tokenizer_path = self.config.model.tokenizer_path or model_path
        
        print(f"Loading model: {model_path}")
        print(f"Tokenizer: {tokenizer_path}")
        print(f"GPU Memory Utilization: {self.config.engine.gpu_memory_utilization}")
        
        # vLLM LLMインスタンスの作成
        llm_kwargs = {
            "model": model_path,
            "tokenizer": tokenizer_path,
            "dtype": self.config.model.dtype,
            "trust_remote_code": self.config.model.trust_remote_code,
            "gpu_memory_utilization": self.config.engine.gpu_memory_utilization,
            "tensor_parallel_size": self.config.engine.tensor_parallel_size,
            "swap_space": self.config.engine.swap_space,
            "max_num_seqs": self.config.engine.max_num_seqs,
        }
        
        if self.config.model.quantization:
            llm_kwargs["quantization"] = self.config.model.quantization
        
        if self.config.engine.max_model_len:
            llm_kwargs["max_model_len"] = self.config.engine.max_model_len
        
        if self.config.engine.seed is not None:
            llm_kwargs["seed"] = self.config.engine.seed
        
        self.llm = LLM(**llm_kwargs)
        
        # サンプリングパラメータの作成
        self.sampling_params = SamplingParams(
            max_tokens=self.config.sampling.max_tokens,
            temperature=self.config.sampling.temperature,
            top_p=self.config.sampling.top_p,
            top_k=self.config.sampling.top_k if self.config.sampling.top_k > 0 else -1,
            repetition_penalty=self.config.sampling.repetition_penalty,
            stop=self.config.sampling.stop_tokens if self.config.sampling.stop_tokens else None,
            stop_token_ids=self.config.sampling.stop_token_ids if self.config.sampling.stop_token_ids else None,
        )
        
        print("✅ Model loaded successfully!")
    
    def format_prompt(self, instruction: str, input_text: str = "", system: str = "") -> str:
        """プロンプトをフォーマット"""
        template_name = self.config.prompt.template
        template = PROMPT_TEMPLATES.get(template_name, PROMPT_TEMPLATES["raw"])
        
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
        prompts: List[str],
        sampling_params: Optional[SamplingParams] = None
    ) -> List[str]:
        """テキスト生成"""
        if self.llm is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        params = sampling_params or self.sampling_params
        outputs = self.llm.generate(prompts, params)
        
        results = []
        for output in outputs:
            generated_text = output.outputs[0].text
            results.append(generated_text)
        
        return results
    
    def generate_single(
        self,
        instruction: str,
        input_text: str = "",
        system: str = "",
        sampling_params: Optional[SamplingParams] = None
    ) -> str:
        """単一プロンプトの生成"""
        prompt = self.format_prompt(instruction, input_text, system)
        results = self.generate([prompt], sampling_params)
        return results[0]
    
    def interactive_mode(self):
        """インタラクティブモード（対話形式）"""
        print("\n" + "=" * 60)
        print("Interactive Mode")
        print("Type 'quit' or 'exit' to end the session")
        print("Type 'clear' to clear the screen")
        print("=" * 60 + "\n")
        
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
                    continue
                
                response = self.generate_single(user_input)
                print(f"\nAssistant: {response}\n")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def batch_inference(
        self,
        input_file: str,
        output_file: str,
        input_format: str = "text"
    ):
        """バッチ推論"""
        input_path = Path(input_file)
        output_path = Path(output_file)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        print(f"Reading prompts from: {input_file}")
        
        prompts = []
        if input_format == "text" or input_path.suffix == ".txt":
            with open(input_path, 'r', encoding='utf-8') as f:
                prompts = [line.strip() for line in f if line.strip()]
        elif input_format == "jsonl" or input_path.suffix == ".jsonl":
            with open(input_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        if isinstance(data, str):
                            prompts.append(data)
                        elif isinstance(data, dict):
                            instruction = data.get('instruction', data.get('prompt', ''))
                            input_text = data.get('input', '')
                            prompts.append(self.format_prompt(instruction, input_text))
        
        print(f"Processing {len(prompts)} prompts...")
        
        # フォーマット済みプロンプトの生成
        formatted_prompts = []
        for p in prompts:
            if not p.startswith(("Below is", "<|", "###")):
                formatted_prompts.append(self.format_prompt(p))
            else:
                formatted_prompts.append(p)
        
        # バッチ生成
        results = self.generate(formatted_prompts)
        
        # 結果の保存
        print(f"Saving results to: {output_file}")
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, (prompt, result) in enumerate(zip(prompts, results)):
                output_data = {
                    "id": i,
                    "prompt": prompt,
                    "response": result
                }
                f.write(json.dumps(output_data, ensure_ascii=False) + "\n")
        
        print(f"✅ Batch inference completed. Results saved to: {output_file}")


def start_api_server(config: InferenceConfig):
    """vLLM APIサーバーを起動"""
    import subprocess
    
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", config.model.path,
        "--host", config.server.host,
        "--port", str(config.server.port),
        "--gpu-memory-utilization", str(config.engine.gpu_memory_utilization),
        "--tensor-parallel-size", str(config.engine.tensor_parallel_size),
    ]
    
    if config.model.tokenizer_path:
        cmd.extend(["--tokenizer", config.model.tokenizer_path])
    
    if config.model.quantization:
        cmd.extend(["--quantization", config.model.quantization])
    
    if config.engine.max_model_len:
        cmd.extend(["--max-model-len", str(config.engine.max_model_len)])
    
    if config.model.trust_remote_code:
        cmd.append("--trust-remote-code")
    
    if config.server.api_key:
        cmd.extend(["--api-key", config.server.api_key])
    
    if config.server.chat_template:
        cmd.extend(["--chat-template", config.server.chat_template])
    
    print("=" * 60)
    print("Starting vLLM API Server")
    print("=" * 60)
    print(f"Model: {config.model.path}")
    print(f"Host: {config.server.host}")
    print(f"Port: {config.server.port}")
    print(f"OpenAI Compatible API: {config.server.openai_api}")
    print("=" * 60)
    print("\nAPI Endpoints:")
    print(f"  - http://{config.server.host}:{config.server.port}/v1/models")
    print(f"  - http://{config.server.host}:{config.server.port}/v1/completions")
    print(f"  - http://{config.server.host}:{config.server.port}/v1/chat/completions")
    print("=" * 60 + "\n")
    
    subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser(
        description="vLLM inference script for trained models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python3 scripts/inference.py --config configs/inference_example.yaml
  
  # Single prompt
  python3 scripts/inference.py --config configs/inference_example.yaml --prompt "Hello, how are you?"
  
  # API server mode
  python3 scripts/inference.py --config configs/inference_example.yaml --server
  
  # Batch inference
  python3 scripts/inference.py --config configs/inference_example.yaml --input prompts.txt --output results.jsonl
        """
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        default=None,
        help="Single prompt for inference"
    )
    parser.add_argument(
        "--server", "-s",
        action="store_true",
        help="Start vLLM API server"
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
        "--model-path",
        type=str,
        default=None,
        help="Override model path from config"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Override max tokens from config"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Override temperature from config"
    )
    
    args = parser.parse_args()
    
    if not VLLM_AVAILABLE:
        print("Error: vLLM is not installed. Install with: pip install vllm")
        sys.exit(1)
    
    # 設定ファイルの読み込み
    print(f"Loading configuration from: {args.config}")
    config = load_inference_config(args.config)
    
    # コマンドライン引数で設定を上書き
    if args.model_path:
        config.model.path = args.model_path
    if args.max_tokens:
        config.sampling.max_tokens = args.max_tokens
    if args.temperature is not None:
        config.sampling.temperature = args.temperature
    
    # モデルパスの検証
    if not config.model.path:
        print("Error: Model path is required. Set it in config or use --model-path")
        sys.exit(1)
    
    # APIサーバーモード
    if args.server:
        start_api_server(config)
        return
    
    # 推論インスタンスの作成とモデル読み込み
    inference = VLLMInference(config)
    inference.load_model()
    
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
    inference.interactive_mode()


if __name__ == "__main__":
    main()
