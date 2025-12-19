# Ollama + Unsloth Training Environment

Unsloth公式Dockerイメージをベースにした、ローカルLLMのファインチューニング・継続事前学習環境です。
モデルのダウンロードにはOllamaを利用できます。

## 特徴

- **Unsloth公式Dockerイメージ**をベース（依存関係の問題なし）
- **Ollama**によるモデルダウンロード対応
- **統一された学習エントリーポイント**（`train.py`）
- **YAML設定ファイル**による柔軟なパラメータ管理
- **MLflow**による学習管理・追跡
- **ログファイル出力**対応
- Jupyter Lab環境がプリインストール済み
- GPU VRAMに応じた自動設定

## クイックスタート

### 1. 環境の構築

```bash
# リポジトリのクローン
git clone <repository-url>
cd ollama-unsloth-training

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
docker exec -it ollama-unsloth-training bash

# または、Jupyter Lab上のターミナルを使用
```

### 3. 学習の実行

#### ファインチューニング

```bash
# 設定ファイルを使用して学習を実行
python3 scripts/train.py --config configs/finetune_example.yaml

# GPU VRAMに応じた自動最適化を有効にする場合
python3 scripts/train.py --config configs/finetune_example.yaml --auto-optimize

# 設定を確認するだけ（ドライラン）
python3 scripts/train.py --config configs/finetune_example.yaml --dry-run
```

#### 継続事前学習

```bash
python3 scripts/train.py --config configs/pretrain_example.yaml
```

### 4. 設定ファイルのカスタマイズ

`configs/` ディレクトリ内の設定ファイルをコピーして編集：

```bash
cp configs/finetune_example.yaml configs/my_finetune.yaml
# 設定ファイルを編集
python3 scripts/train.py --config configs/my_finetune.yaml
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
  load_in_4bit: false

# LoRA設定
lora:
  r: 32
  lora_alpha: 32
  lora_dropout: 0.05

# 訓練設定
training:
  num_train_epochs: 3
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 4
  learning_rate: 2.0e-4

# データ設定
data:
  train_data_path: /workspace/work/data/sample_finetune.jsonl
  prompt_template: alpaca

# 出力設定
output:
  output_dir: /workspace/work/models/outputs  # ベースディレクトリ
  save_merged_model: true
  log_file: training.log

# MLflow設定
mlflow:
  enabled: true
  experiment_name: unsloth-finetuning  # 実験名（出力サブディレクトリ）
  run_name: null  # 実行名（nullで自動生成: run_YYYYMMDD_HHMMSS）
```

**出力先**: `{output_dir}/{experiment_name}/{run_name}/`

例: `/workspace/work/models/outputs/unsloth-finetuning/run_20241220_143052/`

### 主要な設定項目

| セクション | 項目 | 説明 |
|-----------|------|------|
| `training_type` | - | `finetune` または `pretrain` |
| `model.name` | モデル名 | Hugging Face Hub のモデル名 |
| `model.max_seq_length` | 最大シーケンス長 | トークン数の上限 |
| `training.per_device_train_batch_size` | バッチサイズ | GPUあたりのサンプル数 |
| `training.learning_rate` | 学習率 | 最適化の学習率 |
| `data.train_data_path` | データパス | 学習データのパス |
| `output.output_dir` | 出力ベースディレクトリ | 学習出力のベースパス |
| `mlflow.enabled` | MLflow有効化 | 学習追跡の有効/無効 |
| `mlflow.experiment_name` | 実験名 | 出力サブディレクトリ名（第1階層） |
| `mlflow.run_name` | 実行名 | 出力サブディレクトリ名（第2階層、nullで自動生成） |

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
- **タグ**: GPU情報、学習タイプ、エラー情報

### コマンドラインからの確認

```bash
# 実験一覧の表示
mlflow experiments search

# 実行一覧の表示
mlflow runs list --experiment-id <experiment_id>
```

## ログファイル

学習中のログは自動的にファイルに保存されます：

- **保存場所**: `<output_dir>/training.log`
- **内容**: 
  - 設定情報
  - GPU状態
  - 学習進捗
  - エラー情報

## 出力ファイル構成

学習完了後、以下のファイルが生成されます：

```
models/outputs/{experiment_name}/{run_name}/
├── lora/                    # LoRAアダプター
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── tokenizer files...
├── merged/                  # マージされたモデル（save_merged_model: true の場合）
│   ├── config.json
│   ├── model.safetensors
│   └── tokenizer files...
├── train_config.yaml        # 使用した設定ファイル
└── training.log             # 学習ログ
```

- `experiment_name`: `mlflow.experiment_name` の値
- `run_name`: `mlflow.run_name` の値（nullの場合は `run_YYYYMMDD_HHMMSS` 形式で自動生成）

## ディレクトリ構成

```
/workspace/work/
├── configs/              # YAML設定ファイル
│   ├── finetune_example.yaml
│   └── pretrain_example.yaml
├── data/                 # 学習データ
│   ├── sample_finetune.jsonl
│   └── pretrain_data.txt
├── models/
│   ├── cache/            # モデルキャッシュ
│   │   ├── ollama/       # Ollamaモデル
│   │   └── huggingface/  # HuggingFaceモデル
│   └── outputs/          # 学習出力
│       └── {experiment_name}/
│           └── {run_name}/
│               ├── lora/     # LoRAアダプター
│               ├── merged/   # マージされたモデル
│               ├── train_config.yaml
│               └── training.log
├── mlruns/              # MLflowのデータ
├── scripts/             # Pythonスクリプト
│   ├── train.py         # 統一エントリーポイント
│   ├── config.py        # 設定管理
│   ├── logger.py        # ログ管理
│   ├── mlflow_tracker.py # MLflow統合
│   ├── setup_model.py   # モデルセットアップ
│   └── optimize_memory.py # メモリ最適化ツール
└── notebooks/           # Jupyterノートブック
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

```bash
ollama pull mistral:7b-instruct-q4_0
ollama pull llama2:7b-chat-q4_0
ollama pull llama3.1:8b-instruct-q4_0
ollama pull gemma:7b
ollama pull qwen2.5:7b-instruct-q4_0
```

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

3番目の段落のテキスト...
```

## 環境変数

| 変数名 | デフォルト値 | 説明 |
|--------|-------------|------|
| JUPYTER_PASSWORD | unsloth | Jupyter Labのパスワード |
| JUPYTER_PORT | 8888 | Jupyter Labのポート |
| USER_PASSWORD | unsloth2024 | sudo用パスワード |
| OLLAMA_HOST | 0.0.0.0 | Ollama APIのホスト |
| MLFLOW_TRACKING_URI | file:///workspace/work/mlruns | MLflowのトラッキングURI |

## トラブルシューティング

### CUDA out of memoryエラー

設定ファイルで以下を調整：

```yaml
training:
  per_device_train_batch_size: 1  # バッチサイズを小さく
  gradient_accumulation_steps: 16  # 勾配累積を増やす

model:
  max_seq_length: 1024  # シーケンス長を短く
```

または、自動最適化を使用：

```bash
python3 scripts/train.py --config configs/finetune_example.yaml --auto-optimize
```

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
- [Ollama公式サイト](https://ollama.com/)
- [MLflow公式ドキュメント](https://mlflow.org/docs/latest/index.html)
