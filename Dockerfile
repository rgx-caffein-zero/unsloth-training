FROM unsloth/unsloth

# Ollamaのインストール（rootユーザーで実行）
USER root
RUN curl -fsSL https://ollama.com/install.sh | sh

# 作業ディレクトリとスクリプトのセットアップ
WORKDIR /workspace/work

# スクリプトとデータのコピー
COPY --chown=unsloth:unsloth scripts/ ./scripts/
COPY --chown=unsloth:unsloth data/ ./data/

# モデル保存用ディレクトリの作成
RUN mkdir -p ./models && chown -R unsloth:unsloth ./models

# カスタムエントリーポイント
COPY --chown=unsloth:unsloth entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# unslothユーザーに戻す
USER unsloth

ENTRYPOINT ["/entrypoint.sh"]
