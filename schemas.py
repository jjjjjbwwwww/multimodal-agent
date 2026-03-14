
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional


class ToolCall(BaseModel):
    tool: str
    input: Dict[str, Any]
    output: Optional[Dict[str, Any]] = None
    ok: bool = True
    error: Optional[str] = None


class AgentPlan(BaseModel):
    steps: List[str] = Field(default_factory=list)
    questions: List[str] = Field(default_factory=list)


class AgentRunResponse(BaseModel):
    trace_id: str
    device: str = "unknown"
    goal: str
    plan: AgentPlan
    tool_calls: List[ToolCall]
    final_answer: str
    outputs: Dict[str, Any] = Field(default_factory=dict)