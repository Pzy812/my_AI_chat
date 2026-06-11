"""通用小工具。"""


def format_error(e: BaseException) -> str:
    """展开 asyncio TaskGroup / ExceptionGroup，便于前端展示真实原因。"""
    if isinstance(e, BaseExceptionGroup):
        parts = [format_error(x) for x in e.exceptions]
        joined = "; ".join(p for p in parts if p)
        return joined or str(e)
    return str(e).strip() or repr(e)
