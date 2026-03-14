# System Architecture

## Overview

This project implements a multimodal AI agent that can analyze images and answer user questions using vision-language models.

The system is composed of two main services:

1. Agent Service (test7)
2. Multimodal Tool Service (test6)

The agent receives user requests, generates a reasoning plan, and calls external tools to complete the task.

---

## Architecture Diagram

User
 │
 ▼
Web UI (ui.html)
 │
 ▼
Agent API Server (FastAPI)
 │
 │  planning
 ▼
Agent Planner
 │
 │ tool call
 ▼
Tools Client
 │
 ▼
Vision Tool API (BLIP VQA / Caption)
 │
 ▼
Result
 │
 ▼
Agent Summary Response

---

## Components

### Agent API (api.py)

FastAPI service exposing endpoints:

- `/agent/chat`
- `/agent/run`

Responsibilities:

- receive user request
- manage session
- call agent runner

---

### Agent Planner (agent_planner.py)

Responsible for generating the reasoning plan.

Example plan:


1 generate caption
2 ask key visual questions
3 summarize result


---

### Agent Runner (agent_runner.py)

Executes the plan step by step.

---

### Tools Client (tools_client.py)

Handles communication with the multimodal tool server.

Example:


POST http://localhost:8006/vqa


---

### Vision Tool Server (test6)

Provides:

- image caption
- visual question answering