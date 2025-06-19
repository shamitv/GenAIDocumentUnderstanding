"""
agentic_pdf_assessor_team_vision.py
-----------------------------------
• Planner / Executor / Reporter / User agents in an AutoGen *Team*
• PDF pages rendered to PNG → data-URL → sent to GPT-4o as vision messages
"""
from __future__ import annotations

import asyncio
import base64
import datetime as dt
import io
import json
import pathlib
import sys
from typing import List, Dict

import fitz  # PyMuPDF

from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient

# ────────────────────────────────────────────────────────────────────
# 0 ▪ Helper: PDF → list[image_url-dict]
# ────────────────────────────────────────────────────────────────────
def pdf_to_image_messages(pdf_path: str, dpi: int = 220) -> List[Dict]:
    """
    Render each page to PNG, base-64 encode, and wrap as an OpenAI
    vision-message image object.
    """
    msgs: List[Dict] = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            pix = page.get_pixmap(dpi=dpi)
            buf = io.BytesIO(pix.tobytes("png"))
            b64 = base64.b64encode(buf.getvalue()).decode()
            msgs.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                }
            )
    return msgs


# ────────────────────────────────────────────────────────────────────
# 1 ▪ Vision model client
# ────────────────────────────────────────────────────────────────────
model_client = OpenAIChatCompletionClient(model="gpt-4o")  # vision + text


# ────────────────────────────────────────────────────────────────────
# 2 ▪ Define the four agents
# ────────────────────────────────────────────────────────────────────
planner = AssistantAgent(
    name="planner",
    model_client=model_client,
    system_message=(
        "You output STRICT JSON only.\n"
        "Each task object MUST have: description, objective, instructions, "
        "success_criteria.\n"
        "If you need clarification, create a task with `agent` = user.\n"
        "When everything is complete, create ONE task for `reporter` then "
        "respond DONE."
    ),
)

executor = AssistantAgent(
    name="executor",
    model_client=model_client,
    system_message=(
        "You receive ONE task object plus PDF page images.\n"
        "Base answers ONLY on the images; no external knowledge.\n"
        "Return JSON {result:str, citations:[{page:int, quote:str}]}.\n"
        "Quotes ≤200 characters."
    ),
)

reporter = AssistantAgent(
    name="reporter",
    model_client=model_client,
    system_message=(
        "Write a Markdown report containing:\n"
        "  # Objective\n  # Findings (inline footnote markers ¹,²,…)\n"
        "  # Conclusion\n"
        "Follow with a 'Footnotes' section listing each citation as "
        "⟦n⟧ Page N: \"truncated quote…\".\n"
        "Return the report *directly* (no JSON)."
    ),
)

user = UserProxyAgent(
    name="user",
    human_input_mode="ALWAYS",      # pauses to ask the real user
    max_consecutive_auto_reply=0,
    system_message="Answer clarification questions briefly and factually.",
)

# convenience list for team creation
PARTICIPANTS = [planner, executor, reporter, user]


# ────────────────────────────────────────────────────────────────────
# 3 ▪ Build a Round-Robin GroupChat as the Team
# ────────────────────────────────────────────────────────────────────
def build_team(img_msgs: List[Dict], objective: str) -> RoundRobinGroupChat:
    """
    Creates a Round-Robin chat team whose *startup task* includes:
      • JSON blob describing the overall job
      • the image messages (one per PDF page)
    """
    root_json = {
        "mode": "INIT",
        "today": dt.date.today().isoformat(),
        "objective": objective,
        "note": "PDF pages follow as images.",
    }
    # first message = JSON instructions, followed by images
    startup_msgs = [{"type": "text", "content": json.dumps(root_json)}, *img_msgs]

    rr_chat = RoundRobinGroupChat(
        participants=PARTICIPANTS,
        allow_parallel=False,   # sequential turns
        max_rounds=None,        # stop when planner says DONE
    )
    rr_chat.startup_task = startup_msgs  # type: ignore
    return rr_chat


# ────────────────────────────────────────────────────────────────────
# 4 ▪ Orchestrator: run everything and return markdown report
# ────────────────────────────────────────────────────────────────────
async def assess_pdf(pdf_path: str, objective: str) -> str:
    images = pdf_to_image_messages(pdf_path)
    team = build_team(images, objective)

    # Run interactively; may pause for user clarification
    await team.run()

    # The final reporter message is the last one sent by 'reporter'.
    # We grab it from the chat history.
    for msg in reversed(team.chat_history):            # type: ignore
        if msg["agent_name"] == "reporter":
            return msg["content"]

    raise RuntimeError("No report produced.")


# ────────────────────────────────────────────────────────────────────
# 5 · CLI / demo
# ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pdf_file= None

    if len(sys.argv) < 3:
        print("Usage: python agentic_pdf_assessor_team_vision.py <PDF> <objective>")
        sys.exit(1)
    else:
        print("Running agentic PDF assessor with default input…")
        pdf_file = pathlib.Path("data/test_pdfs/sample.pdf").expanduser()
        objective = ("WHO’s recommended daily limit for free "
                     "sugar intake and give at least two health "
                     "risks of excessive sugar consumption?")


    pdf_file = pathlib.Path(sys.argv[1]).expanduser()
    if not pdf_file.is_file():
        print(f"PDF not found: {pdf_file}")
        sys.exit(1)

    objective = " ".join(sys.argv[2:])

    print("\n=== Starting interactive assessment… ===\n")
    md_report = asyncio.run(assess_pdf(str(pdf_file), objective))
    print("\n=== FINAL MARKDOWN REPORT ===\n")
    print(md_report)
