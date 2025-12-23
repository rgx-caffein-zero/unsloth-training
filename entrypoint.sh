#!/bin/bash

# ディレクトリの存在確認と作成（sudoなし）
mkdir -p /workspace/work/models/cache/huggingface 2>/dev/null
mkdir -p /workspace/work/models/outputs 2>/dev/null
mkdir -p /workspace/work/models/gguf 2>/dev/null
mkdir -p /workspace/work/mlruns 2>/dev/null
mkdir -p /workspace/work/data 2>/dev/null
mkdir -p /workspace/work/notebooks 2>/dev/null
mkdir -p /home/unsloth/.ollama/models 2>/dev/null
mkdir -p /tmp/unsloth_compiled_cache 2>/dev/null

# Ollamaサーバーをバックグラウンドで起動
echo "Starting Ollama server..."
ollama serve &

# Ollamaが起動するまで待機
sleep 3

# MLflow UIをバックグラウンドで起動
mlflow ui --host 0.0.0.0 --port 5000 --backend-store-uri file:///workspace/work/mlruns &

# 引数がある場合はそれを実行
if [ $# -gt 0 ]; then
    exec "$@"
else
    # Jupyter Labを起動（Unsloth公式イメージのデフォルト動作）
    jupyter lab \
        --ip=0.0.0.0 \
        --port=${JUPYTER_PORT:-8888} \
        --no-browser \
        --NotebookApp.token='' \
        --NotebookApp.password="$(python3 -c "from jupyter_server.auth import passwd; print(passwd('${JUPYTER_PASSWORD:-unsloth}'))")" \
        --notebook-dir=/workspace
fi
