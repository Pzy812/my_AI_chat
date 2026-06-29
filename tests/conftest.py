"""pytest 全局配置：在导入业务模块前注入 CI / 本地测试所需环境变量。"""
from __future__ import annotations

import os

# 须在 import config.app_config 之前设置
os.environ.setdefault("ZHIPUAI_API_KEY", "test_key_for_pytest")
os.environ.setdefault("AGENT_TASK_HARNESS", "1")
os.environ.setdefault("HITL_ENABLED", "0")
os.environ.setdefault("AGENT_CHECKPOINT_ENABLED", "0")
os.environ.setdefault("RAG_ENABLED", "0")
os.environ.setdefault("GRAPHRAG_ENABLED", "0")
