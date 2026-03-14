
实验 7：多模态 Agent（自动规划 + 多轮记忆）

Experiment 7: Multimodal Agent with Planning, Memory, and Tool Use

一、项目简介（Project Overview）
中文版

本实验实现了一个 多模态智能体（Multimodal Agent），系统能够：

接收 图像 + 自然语言目标

自动生成解决策略（Plan）

主动调用下游模态模型（VQA / Caption）

进行 多轮对话 + 对话记忆

输出 结构化 JSON + 可追踪 Trace

该 Agent 不再是“问一句答一句”的模型，而是一个 具备规划、决策和工具调用能力的智能系统。

English Version

This project implements a multimodal agent that can:

Take images and natural language goals as input

Automatically generate a reasoning plan

Call downstream multimodal tools (VQA, Captioning)

Maintain multi-turn conversation memory

Produce structured JSON outputs with full traces

Integrate seamlessly with a chat-style web UI

Unlike traditional VQA systems, this agent plans before acting, making it closer to real-world AI assistants.

二、系统架构（Architecture – Text Diagram）
┌──────────────┐
│   Web UI     │  (Apple-style chat)
│ (ui.html)    │
└──────┬───────┘
       │ HTTP (multipart / JSON)
       ▼
┌──────────────────────────┐
│     Agent Server         │  test7 (port 8007)
│     FastAPI              │
│                          │
│  - Goal Parsing          │
│  - Strategy Planning     │
│  - Memory Management     │
│  - Tool Orchestration    │
│  - Trace Generation      │
└──────┬───────────────────┘
       │ Internal Tool Calls
       ▼
┌──────────────────────────┐
│   Multimodal Tool Server │  test6 (port 8006)
│   BLIP VQA / Caption     │
│                          │
│  /vqa                    │
│  /vqa_batch              │
└──────────────────────────┘

三、核心亮点：生成策略（Plan）的

1 本实验中的生成策略示例
"plan": {
  "mode": "agent",
  "steps": [
    "Generate a short caption for the image",
    "Ask key questions to extract scene/objects/details",
    "Summarize results aligned with the goal"
  ],
  "goal": "请描述图片内容，并总结关键物体与细节"
}


四、API 使用示例（API Examples）
1️ 多模态 Agent Chat
curl -X POST http://127.0.0.1:8007/agent/chat \
  -F "image=@sample.jpg" \
  -F "question=请描述图片内容，并总结关键物体与细节" \
  -F "offline=true"

2️ 返回示例（简化）
{
  "session_id": "a1b2c3",
  "trace_id": "trace_20260131_xxxx",
  "results": [
    { "role": "assistant", "text": "这是一个卧室场景，主要物体是床..." }
  ],
  "plan": {
    "mode": "agent",
    "steps": ["caption", "key questions", "summary"]
  },
  "saved": {
    "trace_json": "/outputs/trace_xxx_chat.json"
  }
}

五、前端 UI 说明（UI Explanation）
 多轮对话记忆（左侧历史轮次）
 友好错误提示：

 图片仅首轮上传，后续复用