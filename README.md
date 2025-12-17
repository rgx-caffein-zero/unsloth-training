# Ollama + Unsloth Training Environment

Unsloth公式Dockerイメージをベースにした、ローカルLLMのファインチューニング・継続事前学習環境です。
モデルのダウンロードにはOllamaを利用できます。

## 特徴

- **Unsloth公式Dockerイメージ**をベース（依存関係の問題なし）
- **Ollama**によるモデルダウンロード対応
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
```

### 2. コンテナでの作業

```bash
# コンテナに入る
docker exec -it ollama-unsloth-training bash

# または、Jupyter Lab上のターミナルを使用
```

### 3. Ollamaでモデルをダウンロード

```bash
# Ollamaでモデルをダウンロード（推論用）
ollama pull mistral:7b-instruct-q4_0
ollama pull llama3.1:8b-instruct-q4_0

# ダウンロード済みモデルの確認
ollama list
```

### 4. ファインチューニングの実行

```bash
python3 scripts/finetune.py \
    --model unsloth/mistral-7b-instruct-v0.2-bnb-4bit \
    --data /workspace/work/data/sample_finetune.jsonl \
    --output /workspace/work/models/finetuned \
    --epochs 3
```

### 5. 継続事前学習の実行

```bash
python3 scripts/continued_pretrain.py \
    --model unsloth/llama-2-7b \
    --data /workspace/work/data/pretrain_data.txt \
    --output /workspace/work/models/continued_pretrained
```

### 6. 推奨設定の確認

```bash
python3 scripts/optimize_memory.py
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

## ディレクトリ構成

```
/workspace/
├── work/                    # マウントされた作業ディレクトリ
│   ├── data/               # 学習データ
│   ├── models/             # 保存されたモデル
│   ├── scripts/            # Python スクリプト
│   └── notebooks/          # カスタムノートブック
└── unsloth-notebooks/      # Unsloth公式サンプルノートブック
```

## 環境変数

| 変数名 | デフォルト値 | 説明 |
|--------|-------------|------|
| JUPYTER_PASSWORD | unsloth | Jupyter Labのパスワード |
| JUPYTER_PORT | 8888 | Jupyter Labのポート |
| USER_PASSWORD | unsloth2024 | sudo用パスワード |
| OLLAMA_HOST | 0.0.0.0 | Ollama APIのホスト |

## 注意事項

### GPU要件
- NVIDIA GPUが必要
- CUDAドライバーとnvidia-container-toolkitがインストールされていること

### メモリ要件
| モデルサイズ | 推奨VRAM |
|-------------|----------|
| 2Bモデル | 8GB以上 |
| 7Bモデル | 12-16GB以上 |
| 13Bモデル | 24GB以上 |

### データフォーマット
- **ファインチューニング**: JSONL形式（instruction, input, outputフィールド）
- **継続事前学習**: プレーンテキストまたはJSONL形式

## トラブルシューティング

### Ollamaが起動しない場合

```bash
# Ollamaサービスの手動起動
ollama serve &

# モデルリストの確認
ollama list
```

### CUDA out of memoryエラー

```bash
# バッチサイズと系列長を小さくする
python3 scripts/finetune.py \
    --batch-size 1 \
    --max-seq-length 1024 \
    ...
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
