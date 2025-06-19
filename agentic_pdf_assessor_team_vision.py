#!/usr/bin/env python
"""
agentic_pdf_assessor_team_vision.py
----------------------------------
• PyMuPDF renders a PDF to PNGs (base-64 data-URLs)
• Four-agent Round-Robin team (planner ▸ executor ▸ reporter ▸ user)
• GPT-4o handles vision + text
Run:
    python agentic_pdf_assessor_team_vision.py <PDF_PATH> "<objective>"
"""

from __future__ import annotations
import asyncio, base64, datetime as dt, io, json, pathlib, sys
from typing import List, Dict

import fitz  # PyMuPDF
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv

load_dotenv(verbose=True)

# ────────────────────────────────────────────────────────────────────
# 0 ▪ PDF → image_url helpers
# ────────────────────────────────────────────────────────────────────
def pdf_to_image_messages(pdf_path: str, dpi: int = 220) -> List[Dict]:
    """Render each PDF page to PNG and wrap as OpenAI image_url objects."""
    out: List[Dict] = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            pix = page.get_pixmap(dpi=dpi)
            buf = io.BytesIO(pix.tobytes("png"))
            b64 = base64.b64encode(buf.getvalue()).decode()
            out.append({"type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"}})
    return out


# ────────────────────────────────────────────────────────────────────
# 1 ▪ Vision-capable model client
# ────────────────────────────────────────────────────────────────────
model_client = OpenAIChatCompletionClient(model="gpt-4o")   # or gpt-4o-mini


# ────────────────────────────────────────────────────────────────────
# 2 ▪ Agents
# ────────────────────────────────────────────────────────────────────
planner = AssistantAgent(
    "planner",
    model_client=model_client,
    system_message=(
        "Output STRICT JSON only.\n"
        "Return an array of task objects with keys: "
        "description, objective, instructions, success_criteria.\n"
        "If clarification is required, emit a task addressed to `user`.\n"
        "When all work is done, create ONE task for `reporter` then reply DONE."
    ),
)

executor = AssistantAgent(
    "executor",
    model_client=model_client,
    system_message=(
        "Execute ONE task using ONLY the provided PDF page images.\n"
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
        "Then append a 'Footnotes' section listing ⟦n⟧ Page N: \"quote…\".\n"
        "Return the markdown directly, no JSON wrapper."
    ),
)

# Newer API: UserProxyAgent no longer takes human_input_mode / max_auto_reply
user = UserProxyAgent(
    "user",
    system_message="You are the document owner.  Answer clarification questions briefly.",
)

# Participants list for convenience
PARTICIPANTS = [planner, executor, reporter, user]


# ────────────────────────────────────────────────────────────────────
# 3 ▪ Round-Robin Team builder
# ────────────────────────────────────────────────────────────────────
def build_team(images: List[Dict], objective: str) -> RoundRobinGroupChat:
    root_json = {
        "mode": "INIT",
        "today": dt.date.today().isoformat(),
        "objective": objective,
        "note": "PDF pages follow as images.",
    }
    startup_msgs = [{"type": "text", "content": json.dumps(root_json)}, *images]

    team = RoundRobinGroupChat(
        participants=PARTICIPANTS,
        allow_parallel=False,   # sequential turns
        max_rounds=None,        # planner stops with DONE
    )
    team.startup_task = startup_msgs             # type: ignore
    return team


# ────────────────────────────────────────────────────────────────────
# 4 ▪ Orchestrator
# ────────────────────────────────────────────────────────────────────
async def assess_pdf(pdf_path: str, objective: str) -> str:
    imgs = pdf_to_image_messages(pdf_path)
    team = build_team(imgs, objective)
    await team.run()                            # interactive: may ask the user

    # The final reporter post is the last message from 'reporter'
    for msg in reversed(team.messages):         # newest → oldest
        if msg["agent_name"] == "reporter":
            return msg["content"]

    raise RuntimeError("Reporter did not return a summary.")




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
