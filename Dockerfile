FROM unsloth/unsloth

# vLLMとMLflow、YAMLライブラリのインストール
USER root
RUN pip install --break-system-packages mlflow pyyaml vllm

# 作業ディレクトリとスクリプトのセットアップ
WORKDIR /workspace/work

# スクリプトとデータのコピー
COPY --chown=unsloth:unsloth scripts/ ./scripts/
COPY --chown=unsloth:unsloth data/ ./data/
COPY --chown=unsloth:unsloth configs/ ./configs/

# モデル保存用ディレクトリの作成
RUN mkdir -p ./models/cache/huggingface ./models/outputs && \
    chown -R unsloth:unsloth ./models

# MLflow保存用ディレクトリの作成
RUN mkdir -p ./mlruns && chown -R unsloth:unsloth ./mlruns

# カスタムエントリーポイント
COPY --chown=unsloth:unsloth entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# unslothユーザーに戻す
USER unsloth

ENTRYPOINT ["/entrypoint.sh"]
