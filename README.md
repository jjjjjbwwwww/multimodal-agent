
# Multimodal Agent with Planning and Tool Use

An AI agent capable of reasoning about images by generating a plan and calling external vision tools.

The agent integrates with a **vision-language model service** to perform image analysis.

---

# Features

- Multimodal reasoning (image + text)
- Automatic planning
- Tool calling
- Multi-step reasoning
- FastAPI API service
- Interactive Web UI

---

# System Architecture

User
 │
 ▼
Web UI
 │
 ▼
FastAPI API Server
 │
 ▼
Agent Runner
 │
 ▼
Agent Planner
 │
 ▼
Tools Client
 │
 ▼
Vision Tool Server (BLIP)

---

# Project Structure

multimodal-agent
│
├── api.py              # FastAPI service
├── agent_runner.py     # agent execution logic
├── agent_planner.py    # reasoning planner
├── tools_client.py     # tool API client
├── agent_sdk.py        # agent utilities
├── schemas.py          # request/response schemas
│
├── cli.py              # CLI testing
├── questions.json      # example prompts
│
├── ui.html             # web interface
│
├── docs
│   └── architecture.md
│
├── requirements.txt
└── README.md
Agent Workflow

1️⃣ Receive user query and image
2️⃣ Generate reasoning plan
3️⃣ Execute steps sequentially
4️⃣ Call external vision tools
5️⃣ Aggregate results
6️⃣ Return final answer

Example plan:

1 Generate caption
2 Ask key visual questions
3 Summarize results
Run Server
python api.py

Server runs at:

http://localhost:8007
Example Request
curl -X POST http://localhost:8007/agent/chat \
  -F "image=@sample.jpg" \
  -F "question=Describe the scene"
Tech Stack

Python

FastAPI

Agent Architecture

REST APIs

BLIP Vision-Language Model
