"""模型与 API Key 设置相关 API。"""
from fastapi import APIRouter, Request

from llm.model_config import normalize_llm_config, server_llm_defaults

router = APIRouter(tags=["settings"])


@router.get("/settings/llm")
async def settings_llm_get():
    """返回智谱预设与服务端默认配置（不含完整密钥）。"""
    return {"code": 0, **server_llm_defaults()}


@router.post("/settings/llm/validate")
async def settings_llm_validate(request: Request):
    """校验配置是否可创建 LLM 实例（不发起实际对话）。"""
    data = await request.json() or {}
    cfg = normalize_llm_config(data.get("llm_config") or data)
    try:
        from llm.model_config import make_llm_from_config

        make_llm_from_config(cfg)
        label = cfg["model"]
        if cfg["provider"] == "custom":
            label = f"{cfg['model']}（自定义）"
        return {"code": 0, "msg": f"配置有效：{label}"}
    except Exception as e:
        return {"code": -1, "msg": str(e)}
