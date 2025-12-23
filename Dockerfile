FROM unsloth/unsloth

# MLflow、YAMLライブラリ、GGUF変換用ライブラリのインストール
USER root
RUN pip install --break-system-packages mlflow pyyaml llama-cpp-python

# Ollamaのインストール
RUN curl -fsSL https://ollama.com/install.sh | sh

# 作業ディレクトリとスクリプトのセットアップ
WORKDIR /workspace/work

# スクリプトとデータのコピー
COPY --chown=unsloth:unsloth scripts/ ./scripts/
COPY --chown=unsloth:unsloth data/ ./data/
COPY --chown=unsloth:unsloth configs/ ./configs/

# モデル保存用ディレクトリの作成
RUN mkdir -p ./models/cache/huggingface ./models/outputs ./models/gguf && \
    chown -R unsloth:unsloth ./models

# MLflow保存用ディレクトリの作成
RUN mkdir -p ./mlruns && chown -R unsloth:unsloth ./mlruns

# Ollama用ディレクトリの作成
RUN mkdir -p /home/unsloth/.ollama && chown -R unsloth:unsloth /home/unsloth/.ollama

# カスタムエントリーポイント
COPY --chown=unsloth:unsloth entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# unslothユーザーに戻す
USER unsloth

# Ollama環境変数
ENV OLLAMA_HOST=0.0.0.0:11434
ENV OLLAMA_MODELS=/home/unsloth/.ollama/models

ENTRYPOINT ["/entrypoint.sh"]
