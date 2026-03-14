
from __future__ import annotations
import json
import time
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, List

from schemas import ToolCall, AgentPlan
from tools_client import ToolClient


def _now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def run_agent_once(
    image_path: str,
    goal: str,
    tool_client: ToolClient,
    model_dir: Optional[str] = None,
    offline: bool = True,
    max_new_tokens: int = 20,
    out_dir: str = "runs",
) -> Dict[str, Any]:
    """
    Returns dict: trace_id, final_answer, plan, tool_calls, outputs...
    Also writes:
      runs/<trace_id>_trace.json
      runs/<trace_id>_result.md
    """
    trace_id = f"trace_{_now_ts()}_{uuid.uuid4().hex[:8]}"
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    tool_calls: List[ToolCall] = []
    outputs: Dict[str, Any] = {}

    # 1) caption
    try:
        call = ToolCall(tool="caption", input={"image_path": image_path, "offline": offline, "model_dir": model_dir})
        cap = tool_client.caption(image_path, model_dir=model_dir, offline=offline)
        call.output = cap
        tool_calls.append(call)
        outputs["caption"] = cap
    except Exception as e:
        tool_calls.append(ToolCall(tool="caption", input={"image_path": image_path}, ok=False, error=str(e)))
        outputs["caption_error"] = str(e)

    # 2) batch questions
    from agent_planner import build_plan
    plan: AgentPlan = build_plan(goal)

    try:
        call = ToolCall(
            tool="vqa_batch",
            input={
                "image_path": image_path,
                "num_questions": len(plan.questions),
                "offline": offline,
                "model_dir": model_dir,
                "max_new_tokens": max_new_tokens,
            },
        )
        batch = tool_client.vqa_batch(
            image_path=image_path,
            questions=plan.questions,
            model_dir=model_dir,
            offline=offline,
            max_new_tokens=max_new_tokens,
        )
        call.output = batch
        tool_calls.append(call)
        outputs["vqa_batch"] = batch
    except Exception as e:
        tool_calls.append(ToolCall(tool="vqa_batch", input={"image_path": image_path}, ok=False, error=str(e)))
        outputs["vqa_batch_error"] = str(e)

    # 3) summarize
    caption_text = ""
    if isinstance(outputs.get("caption"), dict):
        caption_text = outputs["caption"].get("caption") or outputs["caption"].get("answer") or ""

    qa_pairs = []
    if isinstance(outputs.get("vqa_batch"), dict) and isinstance(outputs["vqa_batch"].get("results"), list):
        qa_pairs = outputs["vqa_batch"]["results"]

    # Build a readable final answer (deterministic template)
    lines = []
    lines.append(f"Goal: {goal}")
    if caption_text:
        lines.append(f"Caption: {caption_text}")
    if qa_pairs:
        lines.append("\nKey Q&A:")
        for i, r in enumerate(qa_pairs, 1):
            q = r.get("question", "")
            a = r.get("answer", "")
            lines.append(f"  Q{i}. {q}")
            lines.append(f"      A: {a}")
    else:
        lines.append("\nKey Q&A: (no results)")

    lines.append("\nSummary:")
    # Heuristic summary from a few answers
    scene = ""
    main_obj = ""
    extra = ""
    for r in qa_pairs:
        q = (r.get("question") or "").lower()
        a = (r.get("answer") or "")
        if ("场景" in q) or ("room" in q) or ("scene" in q):
            if not scene:
                scene = a
        if ("主要" in q) or ("main object" in q):
            if not main_obj:
                main_obj = a
        if ("细节" in q) or ("显著" in q):
            if not extra:
                extra = a

    if scene:
        lines.append(f"- Scene: {scene}")
    if main_obj:
        lines.append(f"- Main object: {main_obj}")
    if extra:
        lines.append(f"- Details: {extra}")
    if not any([scene, main_obj, extra]) and caption_text:
        lines.append(f"- {caption_text}")
    if not any([scene, main_obj, extra]) and not caption_text:
        lines.append("- (Insufficient signals; check tool outputs and thresholds.)")

    final_answer = "\n".join(lines)

    # 4) save artifacts
    trace_json = {
        "trace_id": trace_id,
        "image_path": image_path,
        "goal": goal,
        "plan": plan.model_dump(),
        "tool_calls": [tc.model_dump() for tc in tool_calls],
        "final_answer": final_answer,
        "outputs": outputs,
    }
    (outp / f"{trace_id}_trace.json").write_text(json.dumps(trace_json, ensure_ascii=False, indent=2), encoding="utf-8")

    md = []
    md.append(f"# Agent Run: {trace_id}\n")
    md.append(f"**Goal**: {goal}\n")
    md.append(f"**Image**: `{image_path}`\n")
    md.append("## Plan\n")
    for s in plan.steps:
        md.append(f"- {s}")
    md.append("\n## Final Answer\n")
    md.append(final_answer)
    md.append("\n## Tool Calls (Trace)\n")
    for i, tc in enumerate(tool_calls, 1):
        md.append(f"### {i}. {tc.tool}  {'✅' if tc.ok else '❌'}")
        md.append("**Input**:")
        md.append("```json")
        md.append(json.dumps(tc.input, ensure_ascii=False, indent=2))
        md.append("```")
        if tc.ok and tc.output is not None:
            md.append("**Output**:")
            md.append("```json")
            md.append(json.dumps(tc.output, ensure_ascii=False, indent=2)[:20000])
            md.append("```")
        if not tc.ok:
            md.append(f"**Error**: {tc.error}")
        md.append("")
    (outp / f"{trace_id}_result.md").write_text("\n".join(md), encoding="utf-8")

    return trace_json
