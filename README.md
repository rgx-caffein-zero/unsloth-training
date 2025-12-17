# Ollama + Unsloth Training Environment

UnslothによるローカルLLMのファインチューニング・継続事前学習環境です。
モデルのダウンロードにはOllamaを利用します。

## 使い方

### 1. 環境の構築

```bash
# リポジトリのクローン/作成
mkdir ollama-unsloth-training
cd ollama-unsloth-training

# 必要なファイルを配置

# Dockerイメージのビルドと起動
docker-compose up -d

# コンテナに入る
docker exec -it ollama-unsloth-training bash
```

### 2. モデルのセットアップ

```bash
# Ollamaでモデルをダウンロード
python3 scripts/setup_model.py --model-type mistral-7b
```

利用可能なモデルタイプ:
- `mistral-7b` (デフォルト)
- `llama2-7b`
- `gemma-7b`
- `gemma-2b`
- `phi-2`

### 3. ファインチューニングの実行

```bash
python3 scripts/finetune.py \
    --model unsloth/mistral-7b-instruct-v0.2-bnb-4bit \
    --data /workspace/data/sample_finetune.jsonl \
    --output /workspace/models/finetuned \
    --epochs 3
```

### 4. 継続事前学習の実行

```bash
python3 scripts/continued_pretrain.py \
    --model unsloth/llama-2-7b \
    --data /workspace/data/pretrain_data.txt \
    --output /workspace/models/continued_pretrained
```

### 5. メモリ最適化設定の確認

```bash
python3 scripts/optimize_memory.py
```

### 6. Jupyter Notebookの起動

```bash
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

## 注意事項

### GPU要件
- NVIDIA GPUが必要です
- CUDAドライバーとnvidia-dockerがインストールされている必要があります

### メモリ要件
| モデル | 推奨VRAM |
|--------|----------|
| 7Bモデル | 16GB以上 |
| 13Bモデル | 24GB以上 |
| 2Bモデル | 8GB以上 |

### ストレージ
モデルファイルのために十分なディスク容量（50GB以上推奨）が必要です。

### データフォーマット
- **ファインチューニング**: JSONL形式（instruction, input, outputフィールド）
- **継続事前学習**: プレーンテキストまたはJSONL形式

## トラブルシューティング

### Ollamaが起動しない場合

```bash
# Ollamaサービスの再起動
ollama serve &

# モデルリストの確認
ollama list
```

### CUDA out of memoryエラー

```bash
# バッチサイズを小さくする
python3 scripts/finetune.py \
    --batch-size 1 \
    --max-seq-length 1024 \
    ...
```

### Unslothのインストールエラー

```bash
# 最新版を再インストール
pip3 uninstall unsloth -y
pip3 install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```
