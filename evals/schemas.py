from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvalCase:
    id: str
    category: str
    difficulty: str
    input: str
    plan: list[str]
    fixtures: dict[str, Any] = field(default_factory=dict)
    expected: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "EvalCase":
        return cls(
            id=str(raw["id"]),
            category=str(raw.get("category") or "other"),
            difficulty=str(raw.get("difficulty") or "medium"),
            input=str(raw["input"]),
            plan=[str(x) for x in raw.get("plan") or []],
            fixtures=dict(raw.get("fixtures") or {}),
            expected=dict(raw.get("expected") or {}),
        )


@dataclass
class EvalResult:
    task_id: str
    category: str
    variant: str
    repeat: int
    success: bool
    score: float
    tool_calls: int
    latency_ms: int
    failures: list[str]
    trajectory: list[dict[str, Any]]
    final_answer: str
    task_state: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)
