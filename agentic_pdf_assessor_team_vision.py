#!/usr/bin/env python
"""
agentic_pdf_assessor_team_vision.py
-----------------------------------
• Accepts a PDF path and an objective
• Renders every page → PNG (PyMuPDF) → base-64 data-URL
• Four-agent Round-Robin team (planner ▸ executor ▸ reporter ▸ user)
• Vision-capable GPT-4o model
Usage:
    python agentic_pdf_assessor_team_vision.py <PDF_PATH> "<objective text>"
"""

from __future__ import annotations
import asyncio, base64, datetime as dt, io, json, pathlib, sys
from typing import List, Dict

import fitz                                          # PyMuPDF

from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv

load_dotenv(verbose=True)

# ────────────────────────────────────────────────────────────────────
# 0 ▪ PDF → OpenAI “image_url” helper
# ────────────────────────────────────────────────────────────────────
def pdf_to_image_msgs(pdf_path: str, dpi: int = 220) -> List[Dict]:
    """Render each page to PNG and wrap as image objects for the vision model."""
    msgs: List[Dict] = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            pix = page.get_pixmap(dpi=dpi)
            buf = io.BytesIO(pix.tobytes("png"))
            b64 = base64.b64encode(buf.getvalue()).decode()
            msgs.append(
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
            )
    return msgs


# ────────────────────────────────────────────────────────────────────
# 1 ▪ Vision model client
# ────────────────────────────────────────────────────────────────────
model_client = OpenAIChatCompletionClient(model="gpt-4o")   # vision + text


# ────────────────────────────────────────────────────────────────────
# 2 ▪ Agents
# ────────────────────────────────────────────────────────────────────
planner = AssistantAgent(
    "planner",
    model_client=model_client,
    system_message=(
        "Output STRICT JSON only.\n"
        "Return an array of task objects with keys: description, objective, "
        "instructions, success_criteria.\n"
        "If you need clarification, create a task for `user`.\n"
        "When finished, create ONE task for `reporter` then reply DONE."
    ),
)

executor = AssistantAgent(
    "executor",
    model_client=model_client,
    system_message=(
        "You receive ONE task object plus PDF page images.\n"
        "Base answers ONLY on those images.\n"
        "Return JSON {result:str, citations:[{page:int, quote:str}]} "
        "(quotes ≤200 chars)."
    ),
)

reporter = AssistantAgent(
    "reporter",
    model_client=model_client,
    system_message=(
        "Write a Markdown report:\n"
        "  # Objective\n  # Findings (inline footnotes ¹,²,…)\n  # Conclusion\n"
        "Then a 'Footnotes' section listing ⟦n⟧ Page N: \"quote…\".\n"
        "Return the markdown directly."
    ),
)

# ---  User proxy: construct with legacy-compatible signature, then set prompt ---
user = UserProxyAgent("user", human_input_mode="ALWAYS")
user.update_system_message(
    "You are the document owner.  Answer clarification questions briefly."
)

PARTICIPANTS = [planner, executor, reporter, user]


# ────────────────────────────────────────────────────────────────────
# 3 ▪ Build Round-Robin team
# ────────────────────────────────────────────────────────────────────
def build_team(img_msgs: List[Dict], objective: str) -> RoundRobinGroupChat:
    root_blob = {
        "mode": "INIT",
        "today": dt.date.today().isoformat(),
        "objective": objective,
        "note": "PDF pages follow as images.",
    }
    startup = [{"type": "text", "content": json.dumps(root_blob)}, *img_msgs]

    team = RoundRobinGroupChat(
        participants=PARTICIPANTS,
        allow_parallel=False,     # sequential turns
        max_rounds=None,          # planner decides when to stop
    )
    team.startup_task = startup          # type: ignore
    return team


# ────────────────────────────────────────────────────────────────────
# 4 ▪ Orchestrator
# ────────────────────────────────────────────────────────────────────
async def assess_pdf(pdf_path: str, objective: str) -> str:
    images = pdf_to_image_msgs(pdf_path)
    team = build_team(images, objective)
    await team.run()                      # interactive (may ask the user)

    # Reporter’s last message = final markdown
    for msg in reversed(team.messages):
        if msg["agent_name"] == "reporter":
            return msg["content"]
    raise RuntimeError("Reporter did not produce a summary.")


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
