
# -*- coding: utf-8 -*-
"""
test7/api.py

Agent Chat（多轮 + 记忆）
- 上游：test6 (VQA / VQA_BATCH)  http://127.0.0.1:8006
- 本服务：test7                 http://127.0.0.1:8007

Endpoints
- GET  /                 -> UI (ui.html)
- GET  /health           -> health check
- POST /agent/run        -> single-shot plan + Q&A + summary + trace
- POST /agent/chat       -> multi-turn chat with memory (session_id)
uvicorn api:app --host 127.0.0.1 --port 8007
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware


# ==============
# Config
# ==============
TEST6_BASE = "http://127.0.0.1:8006"
DEFAULT_MODEL_DIR = r"E:\python\test6\hf_tmp_blip_vqa"  # 本地离线模型目录
RUNS_DIR = Path("runs")
RUNS_DIR.mkdir(parents=True, exist_ok=True)

UI_FILE = Path("ui.html")  


# ==============
# In-memory sessions
# ==============
@dataclass
class SessionState:
    image_bytes: Optional[bytes] = None
    image_name: Optional[str] = None
    history: List[Dict[str, str]] = None  # [{"q":..., "a":...}, ...]

    def __post_init__(self):
        if self.history is None:
            self.history = []


SESSIONS: Dict[str, SessionState] = {}


# ==============
# Utils
# ==============
def now_trace_id() -> str:
    t = time.strftime("%Y%m%d_%H%M%S")
    return f"trace_{t}_{uuid.uuid4().hex[:8]}"


def ensure_utf8_json_dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def ensure_utf8_text_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(text)


async def post_multipart(
    url: str,
    files: Dict[str, Tuple[str, bytes, str]],
    data: Dict[str, Any],
    timeout_s: float = 120.0,
) -> Dict[str, Any]:
    """
    Robust multipart caller.
    Important: trust_env=False -> ignore system proxy, avoids random 502.
    """
    async with httpx.AsyncClient(timeout=timeout_s, trust_env=False) as client:
        resp = await client.post(url, files=files, data=data)
        ct = resp.headers.get("content-type", "")
        body = resp.text if resp.text else ""
        if resp.status_code >= 400:
            raise RuntimeError(
                f"Upstream error: url={url} status={resp.status_code} content_type={ct} body={body[:500]}"
            )
        if "application/json" not in ct:
            # 也有可能 返回空 content-type
            try:
                return resp.json()
            except Exception:
                raise RuntimeError(
                    f"Upstream non-json: url={url} status={resp.status_code} content_type={ct} body={body[:500]}"
                )
        return resp.json()


def build_plan(goal: str) -> Dict[str, Any]:
    steps = [
        "1) 先生成一句话 Caption（快速概括）",
        "2) 再提关键问题（场景/主体/细节/文字/数量）",
        "3) 汇总为要点报告（贴合 goal）",
        "4) 返回 trace 便于复现与调试",
    ]
    return {"steps": steps, "goal": goal}


def default_questions(goal: str) -> List[str]:
    # 中英各一套，goal 含中文就偏中文
    has_zh = any("\u4e00" <= ch <= "\u9fff" for ch in goal)
    if has_zh:
        return [
            "图里是什么？（用一句话概括）",
            "这是室内还是室外？",
            "这是什么场景/房间/环境？",
            "主要物体是什么？",
            "还有哪些显著物体或细节？（列 3 个）",
            "有没有文字/标志/屏幕内容？如果有，是什么？",
            "有没有人？如果有，大概几个人？",
        ]
    else:
        return [
            "What is shown in the image? Answer in one short sentence.",
            "Is this indoor or outdoor?",
            "What room or place is this?",
            "What is the main object in the image?",
            "List three important objects you can see.",
            "Is there any visible text, sign, or logo?",
            "Are there any people? If yes, how many?",
        ]


def render_markdown(trace: Dict[str, Any]) -> str:
    plan = trace.get("plan", {})
    outputs = trace.get("outputs", {})
    qa = outputs.get("vqa_batch", {}).get("results", [])

    lines = []
    lines.append(f"# Trace: {trace.get('trace_id','')}\n")
    lines.append("## Goal\n")
    lines.append(f"- {plan.get('goal','')}\n")
    lines.append("## Plan\n")
    for s in plan.get("steps", []):
        lines.append(f"- {s}\n")

    lines.append("\n## Caption\n")
    cap = outputs.get("caption", {})
    if cap:
        lines.append(f"- Q: {cap.get('question','')}\n")
        lines.append(f"- A: {cap.get('answer','')}\n")

    lines.append("\n## Key Q&A\n")
    for i, item in enumerate(qa, 1):
        lines.append(f"- Q{i}. {item.get('question','')}\n")
        lines.append(f"  - A: {item.get('answer','')}\n")

    lines.append("\n## Final Answer\n")
    lines.append(trace.get("final_answer", "") + "\n")
    return "".join(lines)


def summarize(goal: str, caption: str, qa: List[Dict[str, str]]) -> str:
   
    lines = []
    lines.append(f"Goal: {goal}\n")
    lines.append(f"Caption: {caption}\n\n")
    lines.append("Key Q&A:\n")
    for i, item in enumerate(qa, 1):
        lines.append(f"  Q{i}. {item['question']}\n")
        lines.append(f"      A: {item['answer']}\n")
   
    scene = ""
    main_obj = ""
    details = ""
    for item in qa:
        q = item["question"].lower()
        a = item["answer"]
        if ("场景" in item["question"]) or ("room" in q) or ("place" in q):
            scene = a
        if ("主要物体" in item["question"]) or ("main object" in q):
            main_obj = a
        if ("细节" in item["question"]) or ("three important" in q):
            details = a

    lines.append("\nSummary:\n")
    if scene:
        lines.append(f"- Scene: {scene}\n")
    if main_obj:
        lines.append(f"- Main object: {main_obj}\n")
    if details:
        lines.append(f"- Details: {details}\n")
    return "".join(lines).strip()


# ==============
# FastAPI app
# ==============
app = FastAPI(title="test7 Agent", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def ui_index():
    if UI_FILE.exists():
        return FileResponse(str(UI_FILE))
    return JSONResponse(
        {"error": f"ui.html not found at: {UI_FILE.resolve()}"}, status_code=404
    )


@app.get("/health")
def health():
    return {
        "ok": True,
        "service": "test7",
        "test6_base": TEST6_BASE,
        "ui_exists": UI_FILE.exists(),
        "sessions": len(SESSIONS),
    }


async def call_test6_vqa(
    image_name: str,
    image_bytes: bytes,
    question: str,
    offline: bool,
    max_new_tokens: int,
    model_dir: str,
) -> Dict[str, Any]:
    url = f"{TEST6_BASE}/vqa"
    files = {"image": (image_name, image_bytes, "image/jpeg")}
    data = {
        "question": question,
        "offline": "true" if offline else "false",
        "max_new_tokens": str(max_new_tokens),
        "model_dir": model_dir,
    }
    return await post_multipart(url, files=files, data=data, timeout_s=180.0)


async def call_test6_vqa_batch(
    image_name: str,
    image_bytes: bytes,
    questions: List[str],
    offline: bool,
    max_new_tokens: int,
    model_dir: str,
) -> Dict[str, Any]:
    url = f"{TEST6_BASE}/vqa_batch"
    files = {"image": (image_name, image_bytes, "image/jpeg")}
    questions_json = json.dumps(questions, ensure_ascii=False)
    data = {
        "questions_json": questions_json,
        "offline": "true" if offline else "false",
        "max_new_tokens": str(max_new_tokens),
        "model_dir": model_dir,
    }
    return await post_multipart(url, files=files, data=data, timeout_s=240.0)


@app.post("/agent/run")
async def agent_run(
    image: UploadFile = File(...),
    goal: str = Form(...),
    offline: bool = Form(True),
    max_new_tokens: int = Form(40),
    model_dir: str = Form(DEFAULT_MODEL_DIR),
):
    trace_id = now_trace_id()
    try:
        img_bytes = await image.read()
        plan = build_plan(goal)
        cap_q = "What is the image about? Answer in one short sentence."
        cap_out = await call_test6_vqa(
            image.filename or "image.jpg",
            img_bytes,
            cap_q,
            offline=offline,
            max_new_tokens=max_new_tokens,
            model_dir=model_dir,
        )

        qs = default_questions(goal)
        batch_out = await call_test6_vqa_batch(
            image.filename or "image.jpg",
            img_bytes,
            qs,
            offline=offline,
            max_new_tokens=max_new_tokens,
            model_dir=model_dir,
        )

        caption = cap_out.get("answer", "")
        qa = batch_out.get("results", [])
        final_answer = summarize(goal, caption, qa)

        trace = {
            "trace_id": trace_id,
            "plan": plan,
            "final_answer": final_answer,
            "outputs": {
                "caption": cap_out,
                "vqa_batch": batch_out,
            },
        }

        # save files
        json_path = RUNS_DIR / f"{trace_id}.json"
        md_path = RUNS_DIR / f"{trace_id}.md"
        ensure_utf8_json_dump(json_path, trace)
        ensure_utf8_text_write(md_path, render_markdown(trace))

        return {
            "trace_id": trace_id,
            "plan": plan,
            "final_answer": final_answer,
            "saved_json": str(json_path),
            "saved_md": str(md_path),
            "outputs": trace["outputs"],
        }
    except Exception as e:
        return JSONResponse({"error": str(e), "trace_id": trace_id}, status_code=500)


@app.post("/agent/chat")
async def agent_chat(
    # image 可选
    image: Optional[UploadFile] = File(None),
    question: str = Form(...),
    session_id: Optional[str] = Form(None),
    offline: bool = Form(True),
    max_new_tokens: int = Form(40),
    model_dir: str = Form(DEFAULT_MODEL_DIR),
):
    trace_id = now_trace_id()
    try:
        q = (question or "").strip()
        if not q:
            return JSONResponse({"error": "问题不能为空", "trace_id": trace_id}, status_code=400)

        # session init
        if not session_id:
            session_id = uuid.uuid4().hex[:12]
        if session_id not in SESSIONS:
            SESSIONS[session_id] = SessionState()

        st = SESSIONS[session_id]

        # update image memory if provided
        if image is not None:
            st.image_bytes = await image.read()
            st.image_name = image.filename or "image.jpg"

        if not st.image_bytes:
            return JSONResponse(
                {"error": "未选择图片（第一轮必须上传图片）", "trace_id": trace_id},
                status_code=400,
            )

        
        history_text = ""
        if st.history:
            lines = []
            for i, it in enumerate(st.history[-6:], 1):  # 最近 6 轮就够
                lines.append(f"Q{i}: {it['q']}\nA{i}: {it['a']}")
            history_text = "\n\n".join(lines)

        plan = {
            "steps": [
                "1) 读取图片（使用已上传的图像记忆）",
                "2) 结合历史对话（若有）理解当前问题",
                "3) 调用上游 VQA 得到答案",
                "4) 返回 trace 便于调试",
            ],
            "goal": "多轮问答",
        }

       
        # 给一点“系统角色提示”作为上下文，但仍保持可控
        system_hint = (
            "You are a multimodal assistant that can understand images and answer questions.\n"
            "Answer the question briefly and accurately."
        )
        if history_text:
            user_q = f"{system_hint}\n\nConversation so far:\n{history_text}\n\nNow answer:\n{q}"
        else:
            user_q = f"{system_hint}\n\nNow answer:\n{q}"

        vqa_out = await call_test6_vqa(
            st.image_name or "image.jpg",
            st.image_bytes,
            user_q,
            offline=offline,
            max_new_tokens=max_new_tokens,
            model_dir=model_dir,
        )
        a = (vqa_out.get("answer") or "").strip()

        # update memory
        st.history.append({"q": q, "a": a})

        trace = {
            "trace_id": trace_id,
            "session_id": session_id,
            "plan": plan,
            "question": q,
            "answer": a,
            "history_len": len(st.history),
            "outputs": {"vqa": vqa_out},
        }

        # 每轮也保存一份 trace
        json_path = RUNS_DIR / f"{trace_id}.json"
        md_path = RUNS_DIR / f"{trace_id}.md"
        ensure_utf8_json_dump(json_path, trace)
        ensure_utf8_text_write(
            md_path,
            "# Agent Chat Trace\n\n"
            f"- trace_id: {trace_id}\n"
            f"- session_id: {session_id}\n"
            f"- question: {q}\n"
            f"- answer: {a}\n",
        )

        return {
            "trace_id": trace_id,
            "session_id": session_id,
            "round": len(st.history),          # 第几轮（从 1 开始）
            "question": q,
            "answer": a,
            "plan": plan,                      # UI 可选择显示/隐藏
            "history": st.history,            
            "saved_json": str(json_path),
            "saved_md": str(md_path),
        }
    except Exception as e:
       
        return JSONResponse(
            {"error": str(e), "trace_id": trace_id, "session_id": session_id},
            status_code=500,
        )
