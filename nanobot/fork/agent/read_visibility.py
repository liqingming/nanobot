"""运行内保守重读：磁盘状态不等于模型（含原生压缩后）仍能看到正文。"""

from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from functools import wraps
from typing import ParamSpec, TypeVar

_P = ParamSpec("_P")
_R = TypeVar("_R")
_read_evidence_required: ContextVar[bool] = ContextVar("nanobot_read_evidence_required", default=False)


def read_evidence_required() -> bool:
    """仅查询本异步上下文；不推断远端窗口，也不修改文件状态。"""
    return _read_evidence_required.get()


def require_read_evidence(func: Callable[_P, Awaitable[_R]]) -> Callable[_P, Awaitable[_R]]:
    """装饰 runner.run，含其工具子任务；独立工具调用维持原去重行为。"""
    @wraps(func)
    async def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        token = _read_evidence_required.set(True)
        try:
            return await func(*args, **kwargs)
        finally:
            _read_evidence_required.reset(token)

    return wrapped
