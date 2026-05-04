#!/bin/bash
set -e
echo "[entrypoint] device=$DIART_DEVICE models_dir=$DIART_MODELS_DIR"
ls -la "$DIART_MODELS_DIR" || { echo "models dir missing"; exit 1; }
exec "$@"
