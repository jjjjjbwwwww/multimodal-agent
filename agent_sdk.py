
# agent_sdk.py
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests


@dataclass
class AgentConfig:
    test6_base: str = "http://127.0.0.1:8006"
    model_dir: str = r"E:\python\test6\hf_tmp_blip_vqa"
    offline: bool = True
    max_new_tokens: int = 40
    timeout_sec: int = 120

   
    steps: Tuple[str, ...] = (
        "1) Generate a short caption for the image",
        "2) Ask key questions to extract scene/objects/details",
        "3) Summarize results aligned with the goal",
        "4) Return trace for debugging and reproducibility",
    )


def _post_multipart(url: str, image_path: Path, data: Dict[str, Any], timeout_sec: int) -> Dict[str, Any]:
    with image_path.open("rb") as f:
        files = {"image": (image_path.name, f, "image/jpeg")}
        resp = requests.post(url, data=data, files=files, timeout=timeout_sec)
    ct = resp.headers.get("content-type", "")
    body = resp.text
    if resp.status_code != 200:
        raise RuntimeError(f"Upstream error: url={url} status={resp.status_code} content_type={ct} body={body}")
    try:
        return resp.json()
    except Exception:
        raise RuntimeError(f"Upstream returned non-JSON: url={url} content_type={ct} body={body[:300]}")


def _build_questions(goal: str) -> List[str]:
    # v1：固定模板（稳定、可解释）
    return [
        "What is shown in the image? Answer in one short sentence.",
        "Is this indoor or outdoor?",
        "What room or place is this?",
        "What is the main object in the image?",
        "List three important objects you can see.",
        "Are there any people? If yes, how many?",
        "Is there any visible text, sign, or logo?",
        "这张图大概是什么场景？",
        "主要物体是什么？",
    ]


def run_agent(
    image_path: str | Path,
    goal: str,
    history: Optional[List[Dict[str, str]]] = None,
    cfg: Optional[AgentConfig] = None,
) -> Dict[str, Any]:
    """
    返回结构：
    {
      trace_id, plan, final_answer, outputs, trace
    }
    """
    cfg = cfg or AgentConfig()
    history = history or []
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"image not found: {image_path}")

    trace_id = f"trace_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    # 1) caption
    caption_question = "What is the image about? Answer in one short sentence."
    cap_out = _post_multipart(
        url=f"{cfg.test6_base}/vqa",
        image_path=image_path,
        data={
            "question": caption_question,
            "offline": str(cfg.offline).lower(),
            "max_new_tokens": str(cfg.max_new_tokens),
            "model_dir": cfg.model_dir,
        },
        timeout_sec=cfg.timeout_sec,
    )

    # 2) batch questions
    questions = _build_questions(goal)
    batch_out = _post_multipart(
        url=f"{cfg.test6_base}/vqa_batch",
        image_path=image_path,
        data={
            # 这里必须是 JSON 字符串（数组）
            "questions_json": json.dumps(questions, ensure_ascii=False),
            "offline": str(cfg.offline).lower(),
            "max_new_tokens": str(cfg.max_new_tokens),
            "model_dir": cfg.model_dir,
        },
        timeout_sec=cfg.timeout_sec,
    )

    # 3) summarize（规则化拼接，稳定，不靠 LLM）
    caption = cap_out.get("answer", "")
    qa_lines = []
    for i, r in enumerate(batch_out.get("results", []), start=1):
        q = r.get("question", "")
        a = r.get("answer", "")
        qa_lines.append(f"  Q{i}. {q}\n      A: {a}")

    final = (
        f"Goal: {goal}\n"
        f"Caption: {caption}\n\n"
        f"Key Q&A:\n" + "\n".join(qa_lines) + "\n\n"
        f"Summary:\n"
        f"- Scene: {batch_out.get('results', [{}])[2].get('answer', '') if batch_out.get('results') else ''}\n"
        f"- Main object: {batch_out.get('results', [{}])[3].get('answer', '') if batch_out.get('results') else ''}\n"
    )

    return {
        "trace_id": trace_id,
        "plan": {"steps": list(cfg.steps), "goal": goal},
        "final_answer": final,
        "outputs": {"caption": cap_out, "vqa_batch": batch_out},
        "trace": {
            "trace_id": trace_id,
            "inputs": {
                "goal": goal,
                "image": str(image_path),
                "offline": cfg.offline,
                "max_new_tokens": cfg.max_new_tokens,
                "model_dir": cfg.model_dir,
                "history_turns": len(history),
            },
            "tool_calls": [
                {"tool": "vqa (caption)", "ok": True, "error": None, "output": cap_out},
                {"tool": "vqa_batch", "ok": True, "error": None, "output": batch_out},
            ],
        },
        "config": asdict(cfg),
    }


def save_trace(out: Dict[str, Any], out_dir: str | Path = "runs") -> Dict[str, str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    trace_id = out.get("trace_id", f"trace_{uuid.uuid4().hex[:8]}")
    json_path = out_dir / f"{trace_id}.json"
    md_path = out_dir / f"{trace_id}.md"

    json_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    md = f"# Trace {trace_id}\n\n" \
         f"## Goal\n{out['plan']['goal']}\n\n" \
         f"## Plan\n" + "\n".join([f"- {s}" for s in out["plan"]["steps"]]) + "\n\n" \
         f"## Final Answer\n```\n{out['final_answer']}\n```\n"
    md_path.write_text(md, encoding="utf-8")

    return {"json": str(json_path), "md": str(md_path)}
