#!/usr/bin/env python
"""
Vision-PDF assessor (AutoGen 0.6.1)

CLI
----
python agentic_pdf_assessor_team_vision.py  <PDF_PATH>  "<objective>"

What it does
------------
1.  Renders each PDF page to PNG with PyMuPDF.
2.  Wraps every PNG in an `autogen_core.Image` → `MultiModalMessage`.
3.  Creates a four-agent Round-Robin team
        planner ▸ executor ▸ reporter ▸ user-proxy
4.  Feeds the JSON “INIT” blob + images to the team via `team.run(task=…)`.
5.  Prints the reporter’s markdown summary (with footnote citations).
"""

from __future__ import annotations
import asyncio, base64, datetime as dt, io, json, pathlib, sys
from typing import List

import fitz                                         # PyMuPDF

from autogen_core import Image as AGImage
from autogen_agentchat.messages import TextMessage, MultiModalMessage
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv

load_dotenv(verbose=True)


# ────────────────────────────────────────────────────────────────
# 0 ▪ helper:  PDF  →  [TextMessage, *MultiModalMessage]
# ────────────────────────────────────────────────────────────────
def pdf_to_init_messages(pdf_path: str, objective: str,
                         dpi: int = 220) -> List:
    """Return the first JSON blob + one image message per PDF page."""
    # ① JSON blob describing the task
    init_blob = TextMessage(
        content=json.dumps(
            {
                "mode": "INIT",
                "today": dt.date.today().isoformat(),
                "objective": objective,
                "note": "PDF pages follow as images.",
            }
        ),
        source="user",
    )

    # ② multi-modal messages (one per page)
    image_msgs: List[MultiModalMessage] = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            png_bytes = page.get_pixmap(dpi=dpi).tobytes("png")
            data_url = f"data:image/png;base64,{base64.b64encode(png_bytes).decode()}"
            image_msgs.append(
                MultiModalMessage(content=[AGImage.from_uri(data_url)],
                                  source="user")
            )

    return [init_blob, *image_msgs]


# ────────────────────────────────────────────────────────────────
# 1 ▪ model client (GPT-4o vision)
# ────────────────────────────────────────────────────────────────
MODEL = OpenAIChatCompletionClient(model="gpt-4o-mini")


# ────────────────────────────────────────────────────────────────
# 2 ▪ agents
# ────────────────────────────────────────────────────────────────
planner = AssistantAgent(
    "planner",
    model_client=MODEL,
    system_message=(
        "Output STRICT JSON only.  Produce an array of task objects "
        "(description, objective, instructions, success_criteria). "
        "Add a task addressed to `user` whenever clarification is required. "
        "When finished, create ONE task for `reporter` then reply DONE."
    ),
)

executor = AssistantAgent(
    "executor",
    model_client=MODEL,
    system_message=(
        "Execute ONE task using ONLY the provided PDF images. "
        "Return JSON {result:str, citations:[{page:int, quote:str}]}. "
        "Quotes ≤200 characters."
    ),
)

reporter = AssistantAgent(
    "reporter",
    model_client=MODEL,
    system_message=(
        "Write a Markdown report:\n"
        "  # Objective\n  # Findings (inline footnotes ¹,²,…)\n  # Conclusion\n"
        "Then list a 'Footnotes' section with ⟦n⟧ Page N: \"quote…\". "
        "Return the markdown directly."
    ),
)

def console_input(prompt: str, *_):
    """
    Custom input function for UserProxyAgent.
    AutoGen passes the OTHER agent’s message in `prompt`.
    We print it so the user sees the question, then wait for stdin.
    """
    print("\n🔵  QUESTION for you ➜", prompt.strip(), "\n")
    return input("📝  Your reply: ")

user = UserProxyAgent(
    "user",
    description="Document owner who answers clarification questions briefly.",
    input_func=console_input,
)

TEAM_PARTICIPANTS = [planner, executor, reporter, user]


# ────────────────────────────────────────────────────────────────
# 3 ▪ run everything and fetch reporter’s summary
# ────────────────────────────────────────────────────────────────
async def assess_pdf(pdf_path: str, objective: str) -> str:
    task_messages = pdf_to_init_messages(pdf_path, objective)

    team = RoundRobinGroupChat(
        participants=TEAM_PARTICIPANTS,
        max_turns=None,                 # planner stops via DONE
    )

    task_result = await team.run(task=task_messages)
    return task_result.messages[-1].content   # reporter's markdown

# ────────────────────────────────────────────────────────────────────
# 5 · CLI / demo
# ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pdf_file= None

    if len(sys.argv) < 3:
        print("Running agentic PDF assessor with default input…")
        pdf_file = pathlib.Path("./data/test_pdfs/sugars-factsheet.pdf").expanduser()
        objective = ("WHO’s recommended daily limit for free "
                     "sugar intake and give at least two health "
                     "risks of excessive sugar consumption?")
    else:
        pdf_file = pathlib.Path(sys.argv[1]).expanduser()
        objective = " ".join(sys.argv[2:])


    if not pdf_file.is_file():
        print(f"PDF not found: {pdf_file}")
        sys.exit(1)

    print("\n=== Starting interactive assessment… ===\n")
    md_report = asyncio.run(assess_pdf(str(pdf_file), objective))
    print("\n=== FINAL MARKDOWN REPORT ===\n")
    print(md_report)
