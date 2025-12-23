FROM unsloth/unsloth

# rootユーザーで必要なセットアップを実行
USER root

# unslothユーザーにパスワードなしでsudoを許可
RUN echo "unsloth ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# MLflow、YAMLライブラリ、GGUF変換用ライブラリのインストール
RUN pip install --break-system-packages mlflow pyyaml llama-cpp-python

# Ollamaのインストール
RUN curl -fsSL https://ollama.com/install.sh | sh

# 作業ディレクトリのセットアップ
WORKDIR /workspace/work

# 必要なディレクトリをすべてビルド時に作成し、権限を設定
RUN mkdir -p ./models/cache/huggingface ./models/outputs ./models/gguf && \
    mkdir -p ./mlruns && \
    mkdir -p ./data ./notebooks ./scripts ./configs && \
    mkdir -p /home/unsloth/.ollama/models && \
    mkdir -p /tmp/unsloth_compiled_cache && \
    chown -R unsloth:unsloth /workspace/work && \
    chown -R unsloth:unsloth /home/unsloth/.ollama && \
    chown -R unsloth:unsloth /tmp/unsloth_compiled_cache

# スクリプトとデータのコピー
COPY --chown=unsloth:unsloth scripts/ ./scripts/
COPY --chown=unsloth:unsloth data/ ./data/
COPY --chown=unsloth:unsloth configs/ ./configs/

# カスタムエントリーポイント
COPY --chown=unsloth:unsloth entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# unslothユーザーに切り替え
USER unsloth

# 環境変数
ENV OLLAMA_HOST=0.0.0.0:11434
ENV OLLAMA_MODELS=/home/unsloth/.ollama/models
ENV UNSLOTH_COMPILE_CACHE=/tmp/unsloth_compiled_cache

ENTRYPOINT ["/entrypoint.sh"]
