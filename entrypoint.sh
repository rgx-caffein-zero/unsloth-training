#!/bin/bash

# Ollamaモデルディレクトリの作成と権限設定
if [ -n "$OLLAMA_MODELS" ]; then
    echo "$USER_PASSWORD" | sudo -S mkdir -p "$OLLAMA_MODELS" 2>/dev/null
    echo "$USER_PASSWORD" | sudo -S chown -R unsloth:unsloth "$(dirname "$OLLAMA_MODELS")" 2>/dev/null
fi

# 作業ディレクトリの権限設定（マウントされたディレクトリ用）
echo "$USER_PASSWORD" | sudo -S chown -R unsloth:unsloth /workspace/work/models 2>/dev/null
echo "$USER_PASSWORD" | sudo -S chown -R unsloth:unsloth /workspace/work/data 2>/dev/null
echo "$USER_PASSWORD" | sudo -S chown -R unsloth:unsloth /workspace/work/notebooks 2>/dev/null
echo "$USER_PASSWORD" | sudo -S chown -R unsloth:unsloth /workspace/work/configs 2>/dev/null
echo "$USER_PASSWORD" | sudo -S chown -R unsloth:unsloth /workspace/work/mlruns 2>/dev/null

# Unslothコンパイルキャッシュ用ディレクトリの作成
mkdir -p /workspace/work/unsloth_compiled_cache 2>/dev/null
echo "$USER_PASSWORD" | sudo -S chown -R unsloth:unsloth /workspace/work/unsloth_compiled_cache 2>/dev/null

# Ollamaサービスをバックグラウンドで起動
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
