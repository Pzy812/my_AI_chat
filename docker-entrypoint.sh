#!/bin/sh
set -e
cd /app

# 从环境读 Redis（Compose 里应为 REDIS_HOST=redis），本地单容器可默认 127.0.0.1
export REDIS_HOST="${REDIS_HOST:-127.0.0.1}"
export REDIS_PORT="${REDIS_PORT:-6379}"

echo "Waiting for Redis at ${REDIS_HOST}:${REDIS_PORT} ..."
python - <<'PY'
import os
import socket
import time

host = os.environ.get("REDIS_HOST", "127.0.0.1")
port = int(os.environ.get("REDIS_PORT", "6379"))
for i in range(90):
    try:
        s = socket.create_connection((host, port), timeout=3)
        s.close()
        print("Redis is reachable.")
        break
    except OSError as e:
        if i == 0:
            print(f"  (still trying: {e})")
        time.sleep(0.5)
else:
    raise SystemExit(
        f"Could not connect to Redis at {host}:{port}. "
        "In Docker Compose, set REDIS_HOST to the Redis service name (e.g. redis), not 127.0.0.1."
    )
PY

echo "Waiting for Milvus at ${MILVUS_URI:-http://milvus:19530} ..."
python - <<'PY'
import os
import time

uri = os.environ.get("MILVUS_URI", "http://milvus:19530")
for i in range(120):
    try:
        from pymilvus import MilvusClient

        MilvusClient(uri=uri)
        print("Milvus is reachable.")
        break
    except Exception as e:
        if i == 0:
            print(f"  (still trying: {e})")
        time.sleep(1)
else:
    raise SystemExit(f"Could not connect to Milvus at {uri}.")
PY

python mcp_server.py &
MCP_PID=$!

cleanup() {
  echo "Stopping MCP (pid $MCP_PID)..."
  kill "$MCP_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "Waiting for MCP on 127.0.0.1:8081 ..."
python - <<'PY'
import socket
import time

for _ in range(60):
    try:
        s = socket.create_connection(("127.0.0.1", 8081), timeout=2)
        s.close()
        break
    except OSError:
        time.sleep(0.5)
else:
    raise SystemExit("MCP did not open port 8081 in time")
time.sleep(0.8)
PY

exec python app.py
