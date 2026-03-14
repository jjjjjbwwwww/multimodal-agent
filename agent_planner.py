
from __future__ import annotations
from typing import List
from schemas import AgentPlan


def build_plan(goal: str) -> AgentPlan:
    """
    Simple rule-based planner.
    Goal: generate a stable multi-step plan + good questions.
    """
    goal = (goal or "").strip()
    if not goal:
        goal = "Understand the image and answer questions."

    steps = [
        "1) Generate a short caption for the image",
        "2) Ask key questions to extract scene/object/details",
        "3) Summarize results into a final answer aligned with the goal",
        "4) Return trace (tool calls) for debugging and reproducibility",
    ]

    # Base questions: scene / objects / details / text / count / actions
    questions = [
        # English first (BLIP VQA is much more reliable in English)
        "What is shown in the image? Answer in one short sentence.",
        "Is this indoor or outdoor?",
        "What room or place is this?",
        "What is the main object in the image?",
        "List 3 important objects you can see.",
        "Are there any people? If yes, how many?",
        "Is there any text/sign/logo visible?",

        # Chinese backup (optional)
        "这张图大概是什么场景？",
        "主要物体是什么？",
]

    # Goal-aware add-ons
    g = goal.lower()
    if "color" in g or "颜色" in goal:
        questions.append("主要物体的颜色是什么？")
    if "count" in g or "多少" in goal or "几" in goal:
        questions.append("请数一数最重要对象的数量。")
    if "risk" in g or "安全" in goal or "危险" in goal:
        questions.append("这张图里有没有潜在风险/危险源？")
    if "summary" in g or "报告" in goal:
        questions.append("用三条要点总结画面信息。")

    return AgentPlan(steps=steps, questions=questions)
