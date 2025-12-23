# Unsloth Training Environment with Ollama Inference

Unsloth公式Dockerイメージをベースにした、ローカルLLMのファインチューニング・継続事前学習環境です。
Ollamaによるメモリ効率の良い推論機能を搭載しています。

## 特徴

- **Unsloth公式Dockerイメージ**をベース（依存関係の問題なし）
- **Ollama**によるメモリ効率の良い推論（GGUF形式対応）
- **統一された学習エントリーポイント**（`train.py`）
- **YAML設定ファイル**による柔軟なパラメータ管理
- **MLflow**による学習管理・追跡
- **学習後の自動GGUF変換・Ollama登録**
- Jupyter Lab環境がプリインストール済み
- GPU VRAMに応じた自動設定

## クイックスタート

### 1. 環境の構築

```bash
# リポジトリのクローン
git clone <repository-url>
cd unsloth-training

# Dockerイメージのビルドと起動
docker-compose up -d

# Jupyter Labにアクセス
# http://localhost:8888 (パスワード: unsloth)

# MLflow UIにアクセス
# http://localhost:5000
```

### 2. コンテナでの作業

```bash
# コンテナに入る
docker exec -it unsloth-training bash

# または、Jupyter Lab上のターミナルを使用
```

### 3. 学習の実行

#### ファインチューニング

```bash
# 設定ファイルを使用して学習を実行
python3 scripts/train.py --config configs/finetune_example.yaml

# GPU VRAMに応じた自動最適化を有効にする場合
python3 scripts/train.py --config configs/finetune_example.yaml --auto-optimize

# 学習後にGGUF変換も行う場合
python3 scripts/train.py --config configs/finetune_example.yaml --convert-gguf

# 設定を確認するだけ（ドライラン）
python3 scripts/train.py --config configs/finetune_example.yaml --dry-run
```

#### 継続事前学習

```bash
python3 scripts/train.py --config configs/pretrain_example.yaml
```

### 4. 推論の実行（Ollama）

#### ベースモデルでの推論

```bash
# モデルのダウンロード
python3 scripts/inference.py --pull llama2

# インタラクティブモード
python3 scripts/inference.py --model llama2

# 単一プロンプト
python3 scripts/inference.py --model llama2 --prompt "こんにちは"
```

#### 学習済みモデルでの推論

学習時に `--convert-gguf` オプションを使用した場合、モデルは自動的にOllamaに登録されます。

```bash
# 学習済みモデルで推論（モデル名は学習時のexperiment_name-run_nameで自動生成）
python3 scripts/inference.py --model unsloth-finetuning-run-20241220-143052

# 設定ファイルを使用
python3 scripts/inference.py --config configs/inference_example.yaml
```

#### 手動でのGGUF変換

```bash
# 学習済みモデルをGGUF形式に変換
python3 scripts/convert_to_gguf.py \
  --model-path ./models/outputs/unsloth-finetuning/run_20241220_143052/merged \
  --output-dir ./models/gguf/my-model \
  --register-ollama \
  --model-name my-finetuned-model

# 変換したモデルで推論
python3 scripts/inference.py --model my-finetuned-model
```

#### バッチ推論

```bash
# プロンプトファイルから一括推論
python3 scripts/inference.py --config configs/inference_example.yaml \
  --input prompts.txt --output results.jsonl
```

### 5. 設定ファイルのカスタマイズ

`configs/` ディレクトリ内の設定ファイルをコピーして編集：

```bash
# 学習用
cp configs/finetune_example.yaml configs/my_finetune.yaml

# 推論用
cp configs/inference_example.yaml configs/my_inference.yaml
```

## 設定ファイル構成

### ファインチューニング設定例 (`configs/finetune_example.yaml`)

```yaml
# 学習タイプ
training_type: finetune

# モデル設定
model:
  name: unsloth/mistral-7b-instruct-v0.2-bnb-4bit
  max_seq_length: 2048

# LoRA設定
lora:
  r: 32
  lora_alpha: 32

# 訓練設定
training:
  num_train_epochs: 3
  per_device_train_batch_size: 2
  learning_rate: 2.0e-4

# データ設定
data:
  train_data_path: /workspace/work/data/sample_finetune.jsonl
  prompt_template: alpaca

# 出力設定
output:
  output_dir: /workspace/work/models/outputs
  save_merged_model: true

# GGUF変換設定（Ollama用）
gguf:
  enabled: false  # --convert-gguf オプションで有効化可能
  quantization: q4_k_m
  register_ollama: true

# MLflow設定
mlflow:
  enabled: true
  experiment_name: unsloth-finetuning
```

### 推論設定例 (`configs/inference_example.yaml`)

```yaml
# モデル設定
model:
  # Ollamaモデル名（ベースモデルまたは学習済みモデル）
  name: llama2

# サンプリング設定
sampling:
  num_predict: 512
  temperature: 0.7
  top_p: 0.9

# サーバー設定
server:
  host: localhost
  port: 11434

# プロンプト設定
prompt:
  template: none  # Ollamaのデフォルトを使用
  system_prompt: "You are a helpful AI assistant."
```

## GGUF変換と量子化

### 量子化タイプ

| タイプ | 説明 | 推奨用途 |
|--------|------|----------|
| `q4_k_m` | 4-bit K-quant（バランス良し） | **推奨デフォルト** |
| `q4_0` | 4-bit（最速、品質低） | 速度重視 |
| `q5_k_m` | 5-bit K-quant | バランス |
| `q6_k` | 6-bit K-quant | 品質重視 |
| `q8_0` | 8-bit（高品質） | 品質最重視 |
| `f16` | 16-bit float | 量子化なし |

### 変換コマンド例

```bash
# デフォルト量子化（q4_k_m）
python3 scripts/convert_to_gguf.py \
  --model-path ./models/outputs/my-model/merged \
  --output-dir ./models/gguf/my-model \
  --register-ollama

# 高品質量子化（q8_0）
python3 scripts/convert_to_gguf.py \
  --model-path ./models/outputs/my-model/merged \
  --output-dir ./models/gguf/my-model \
  --quantization q8_0 \
  --register-ollama \
  --model-name my-model-q8

# カスタムシステムプロンプトとテンプレート
python3 scripts/convert_to_gguf.py \
  --model-path ./models/outputs/my-model/merged \
  --output-dir ./models/gguf/my-model \
  --register-ollama \
  --system-prompt "あなたは親切なAIアシスタントです。" \
  --template chatml
```

## Ollama 推論機能

### 推論モード一覧

| モード | コマンド | 説明 |
|-------|---------|------|
| インタラクティブ | `--model` のみ | 対話形式で推論 |
| 単一プロンプト | `--prompt` | 1つのプロンプトを処理 |
| バッチ処理 | `--input` `--output` | ファイルから一括処理 |
| モデル一覧 | `--list-models` | 利用可能なモデルを表示 |
| モデルダウンロード | `--pull` | モデルをダウンロード |

### コマンド例

```bash
# モデル一覧の表示
python3 scripts/inference.py --list-models

# モデルのダウンロード
python3 scripts/inference.py --pull llama2

# インタラクティブモード（チャット履歴あり）
python3 scripts/inference.py --model llama2

# インタラクティブモード（チャット履歴なし）
python3 scripts/inference.py --model llama2 --no-chat

# 単一プロンプト
python3 scripts/inference.py --model llama2 --prompt "Pythonでフィボナッチ数列を生成する関数を書いてください"

# バッチ推論
python3 scripts/inference.py --model llama2 --input prompts.txt --output results.jsonl
```

### インタラクティブモードのコマンド

- `quit` / `exit`: 終了
- `clear`: 画面とチャット履歴をクリア
- `/chat`: チャットモードの切り替え

## MLflow による学習管理

### MLflow UI

学習の進捗や結果はMLflow UIで確認できます：

```
http://localhost:5000
```

### 記録される情報

- **パラメータ**: モデル設定、訓練設定、LoRA設定など
- **メトリクス**: loss、learning_rate、epoch など
- **アーティファクト**: 設定ファイル、ログファイル
- **タグ**: GPU情報、学習タイプ、GGUF変換情報

## ログファイル

学習中のログは自動的にファイルに保存されます：

- **保存場所**: `<output_dir>/training.log`
- **内容**: 設定情報、GPU状態、学習進捗、エラー情報

## 出力ファイル構成

学習完了後、以下のファイルが生成されます：

```
models/outputs/{experiment_name}/{run_name}/
├── lora/                    # LoRAアダプター
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── tokenizer files...
├── merged/                  # マージされたモデル
│   ├── config.json
│   ├── model.safetensors
│   └── tokenizer files...
├── gguf/                    # GGUF変換後（--convert-gguf使用時）
│   ├── model.gguf
│   └── Modelfile
├── train_config.yaml        # 使用した設定ファイル
└── training.log             # 学習ログ
```

## ディレクトリ構成

```
/workspace/work/
├── configs/              # YAML設定ファイル
│   ├── finetune_example.yaml
│   ├── pretrain_example.yaml
│   └── inference_example.yaml
├── data/                 # 学習データ
│   ├── sample_finetune.jsonl
│   └── pretrain_data.txt
├── models/
│   ├── cache/            # モデルキャッシュ
│   │   └── huggingface/
│   ├── outputs/          # 学習出力
│   │   └── {experiment_name}/
│   │       └── {run_name}/
│   │           ├── lora/
│   │           ├── merged/
│   │           └── gguf/
│   └── gguf/             # 手動変換したGGUFモデル
├── mlruns/               # MLflowのデータ
├── scripts/              # Pythonスクリプト
│   ├── train.py          # 学習エントリーポイント
│   ├── inference.py      # Ollama推論スクリプト
│   ├── convert_to_gguf.py # GGUF変換スクリプト
│   ├── config.py         # 設定管理
│   ├── logger.py         # ログ管理
│   ├── mlflow_tracker.py # MLflow統合
│   ├── setup_model.py    # モデルセットアップ
│   └── optimize_memory.py # メモリ最適化ツール
└── notebooks/            # Jupyterノートブック
```

## 利用可能なモデル

### Unslothモデル（学習用）

| モデルタイプ | モデル名 | 推奨VRAM |
|-------------|---------|----------|
| mistral-7b | unsloth/mistral-7b-instruct-v0.2-bnb-4bit | 12GB+ |
| llama2-7b | unsloth/llama-2-7b-chat-bnb-4bit | 12GB+ |
| llama3-8b | unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit | 16GB+ |
| gemma-7b | unsloth/gemma-7b-bnb-4bit | 12GB+ |
| gemma-2b | unsloth/gemma-2b | 8GB+ |
| qwen2-7b | unsloth/Qwen2.5-7B-Instruct-bnb-4bit | 12GB+ |

### Ollamaモデル（推論用）

| モデル名 | 説明 |
|---------|------|
| llama2 | Llama 2 |
| llama3 | Llama 3 |
| llama3.2 | Llama 3.2 |
| mistral | Mistral 7B |
| gemma | Gemma |
| qwen2 | Qwen 2 |

## データフォーマット

### ファインチューニング用（JSONL）

```json
{"instruction": "次の文章を要約してください", "input": "...", "output": "..."}
{"instruction": "質問に答えてください", "input": "...", "output": "..."}
```

### 継続事前学習用（テキスト）

```text
最初の段落のテキスト...

2番目の段落のテキスト...
```

### バッチ推論用（テキスト or JSONL）

```text
# prompts.txt
最初のプロンプト
2番目のプロンプト
```

```json
// prompts.jsonl
{"instruction": "質問に答えてください", "input": "日本の首都は？"}
{"prompt": "翻訳してください: Hello"}
```

## 環境変数

| 変数名 | デフォルト値 | 説明 |
|--------|-------------|------|
| JUPYTER_PASSWORD | unsloth | Jupyter Labのパスワード |
| JUPYTER_PORT | 8888 | Jupyter Labのポート |
| USER_PASSWORD | unsloth2024 | sudo用パスワード |
| MLFLOW_TRACKING_URI | file:///workspace/work/mlruns | MLflowのトラッキングURI |
| OLLAMA_HOST | 0.0.0.0:11434 | Ollama APIホスト |
| OLLAMA_MODELS | /home/unsloth/.ollama/models | Ollamaモデルの保存先 |

## トラブルシューティング

### CUDA out of memoryエラー

設定ファイルで以下を調整：

```yaml
training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16

model:
  max_seq_length: 1024
```

または、自動最適化を使用：

```bash
python3 scripts/train.py --config configs/finetune_example.yaml --auto-optimize
```

### Ollamaに接続できない場合

```bash
# Ollamaサーバーの状態を確認
curl http://localhost:11434/api/tags

# 手動でOllamaを起動
ollama serve &
```

### GGUF変換が失敗する場合

- マージされたモデルが存在することを確認（`save_merged_model: true`）
- GPU メモリを解放してから再試行
- より軽い量子化タイプ（q4_0）を試す

### MLflowが起動しない場合

```bash
# 手動で起動
mlflow ui --host 0.0.0.0 --port 5000 --backend-store-uri file:///workspace/work/mlruns &
```

### GPUが認識されない場合

```bash
# nvidia-smiでGPUを確認
nvidia-smi

# PyTorchでGPUを確認
python3 -c "import torch; print(torch.cuda.is_available())"
```

## 参考リンク

- [Unsloth公式ドキュメント](https://docs.unsloth.ai/)
- [Unsloth Dockerガイド](https://docs.unsloth.ai/get-started/install-and-update/docker)
- [Ollama公式ドキュメント](https://ollama.com/)
- [MLflow公式ドキュメント](https://mlflow.org/docs/latest/index.html)
