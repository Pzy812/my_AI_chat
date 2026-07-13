"""应用路径、环境变量与日志配置。"""
import logging
import os
from pathlib import Path

from config.env_config import ensure_zhipuai_api_key_in_environ
from observability.langsmith_config import configure_langsmith, public_status as langsmith_public_status

BASE_DIR = Path(__file__).resolve().parent.parent
EXPORTS_DIR = (BASE_DIR / "exports").resolve()
UPLOADS_DIR = (BASE_DIR / "uploads").resolve()
EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

MCP_TABLE_ATTACH_MAX = 120_000

LOG_LLM_PROMPT = os.getenv("LOG_LLM_PROMPT", "1").strip().lower() not in (
    "0",
    "false",
    "no",
)
LOG_LLM_PROMPT_MAX = int(os.getenv("LOG_LLM_PROMPT_MAX", "12000"))

logger = logging.getLogger("ai_chat")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )

ensure_zhipuai_api_key_in_environ()
LANGSMITH_ENABLED = configure_langsmith()
LANGSMITH_STATUS = langsmith_public_status()
MCP_HOST = os.getenv("MCP_HOST", "localhost")
MCP_PORT = int(os.getenv("MCP_PORT", "8090"))
MCP_URL = f"http://{MCP_HOST}:{MCP_PORT}/mcp"

# APP_* 优先；兼容旧 FLASK_* 环境变量
APP_HOST = os.getenv("APP_HOST") or os.getenv("FLASK_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT") or os.getenv("FLASK_PORT", "5001"))
FLASK_HOST = APP_HOST  # 兼容旧引用
FLASK_PORT = APP_PORT

# PostgreSQL：聊天权威存储 + LangGraph Checkpointer（可共用同一库）
POSTGRES_URI = (
    os.getenv("POSTGRES_URI", "").strip()
    or os.getenv("DATABASE_URL", "").strip()
)
CHAT_HISTORY_MAX_MESSAGES = int(os.getenv("CHAT_HISTORY_MAX_MESSAGES", "500"))
CHAT_AGENT_CONTEXT_MESSAGES = int(
    os.getenv(
        "CHAT_AGENT_CONTEXT_MESSAGES",
        os.getenv("REDIS_CHAT_MAX_MESSAGES", "80"),
    )
)
# 超过 N 轮用户发言时，对更早对话做小模型摘要（保留最近 CHAT_SUMMARY_KEEP_ROUNDS 轮原文）
CHAT_SUMMARY_ENABLED = os.getenv("CHAT_SUMMARY_ENABLED", "1").strip().lower() not in (
    "0",
    "false",
    "no",
)
CHAT_SUMMARY_ROUNDS = int(os.getenv("CHAT_SUMMARY_ROUNDS", "20"))
CHAT_SUMMARY_KEEP_ROUNDS = int(
    os.getenv("CHAT_SUMMARY_KEEP_ROUNDS", os.getenv("CHAT_SUMMARY_ROUNDS", "20"))
)
CHAT_SUMMARY_MAX_INPUT_CHARS = int(os.getenv("CHAT_SUMMARY_MAX_INPUT_CHARS", "48000"))
CHAT_SUMMARY_MAX_MSG_CHARS = int(os.getenv("CHAT_SUMMARY_MAX_MSG_CHARS", "6000"))
AGENT_CHECKPOINT_ENABLED = os.getenv("AGENT_CHECKPOINT_ENABLED", "1").strip().lower() not in (
    "0",
    "false",
    "no",
)
UPLOAD_META_TTL_SEC = int(os.getenv("UPLOAD_META_TTL_SEC", str(7 * 24 * 3600)))
HITL_ENABLED = os.getenv("HITL_ENABLED", "1").strip().lower() not in ("0", "false", "no")

# Task Harness：复杂任务计划 / 阶段 gate / 上下文裁剪
AGENT_TASK_HARNESS = os.getenv("AGENT_TASK_HARNESS", "1").strip().lower() not in (
    "0",
    "false",
    "no",
)
AGENT_TASK_HARNESS_MIN_SIGNALS = int(os.getenv("AGENT_TASK_HARNESS_MIN_SIGNALS", "1"))
AGENT_LLM_CONTEXT_MESSAGES = int(os.getenv("AGENT_LLM_CONTEXT_MESSAGES", "10"))
AGENT_REANCHOR_EVERY_N_TOOLS = int(os.getenv("AGENT_REANCHOR_EVERY_N_TOOLS", "3"))
AGENT_RECURSION_LIMIT = int(os.getenv("AGENT_RECURSION_LIMIT", "100"))
AGENT_PHASE_GATE_MAX_RETRIES = int(os.getenv("AGENT_PHASE_GATE_MAX_RETRIES", "3"))
MAX_TASK_CONTINUATIONS = int(os.getenv("MAX_TASK_CONTINUATIONS", "5"))
