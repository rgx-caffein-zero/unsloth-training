#!/bin/bash

# Ollamaサービスをバックグラウンドで起動
ollama serve &

# Ollamaが起動するまで待機
sleep 3

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
