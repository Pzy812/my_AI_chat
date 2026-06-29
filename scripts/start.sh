#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ ! -f .env ]]; then
  if [[ -f .env.example ]]; then
    echo "[提示] 未找到 .env，已从 .env.example 复制一份，请编辑 .env 填入 ZHIPUAI_API_KEY"
    cp .env.example .env
  else
    echo "[错误] 未找到 .env，请创建并配置 ZHIPUAI_API_KEY"
    exit 1
  fi
fi

echo "Starting AI Chat Web on http://localhost:5001"
echo "MCP will auto-start if not running on port 8090"
echo
exec python app.py
