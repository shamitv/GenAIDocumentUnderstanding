#!/usr/bin/env python
"""
Vision‑enabled PDF assessor — AutoGen 0.6.1‑compatible
=====================================================
Run from a shell:

    python agentic_pdf_assessor_team_vision.py <PDF_PATH> "<objective text>"

*   Renders each page of the PDF to a PNG image (via **PyMuPDF**).
*   Embeds every PNG as a base‑64 data‑URL → `autogen_core.Image` → `MultiModalMessage`.
*   Creates a four‑agent **Round‑Robin** team:
        • **planner**   – breaks work into task objects, may ask the human user.
        • **executor**  – inspects the page images and returns findings + citations.
        • **reporter**  – compiles a Markdown report with numbered footnotes.
        • **user**      – real human; only asked when the planner explicitly requests.
*   Feeds the JSON blob + images to `team.run(task=…)` (required by 0.6 API).
*   Prints the reporter’s Markdown summary when finished.

Notes & API‑compliance (AutoGen 0.6.1)
--------------------------------------
* **No deprecated kwargs** — `allow_parallel`, `startup_task`, `human_input_mode`,
  etc. were removed after 0.4; this script uses only parameters present in 0.6.1.
* **Message objects** — initial prompt and every page image are concrete
  `TextMessage` / `MultiModalMessage` instances (all subclass `BaseChatMessage`).
* **User input** — we supply a custom `input_func` so the *actual* clarification
  question is printed before the console waits for an answer.
"""

from __future__ import annotations
import asyncio, base64, datetime as dt, io, json, logging, pathlib, sys
from typing import List

import fitz                                         # PyMuPDF

from autogen_core import Image as AGImage
from autogen_agentchat.messages import TextMessage, MultiModalMessage, BaseChatMessage
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.models.cache import ChatCompletionCache, CHAT_CACHE_VALUE_TYPE
from autogen_ext.cache_store.diskcache import DiskCacheStore
from diskcache import Cache
from dotenv import load_dotenv

load_dotenv(verbose=True)


# ────────────────────────────────────────────────────────────────
# 0 ▪ logger
# ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S.%f",
)

def log(msg: str):
    """Log a message with a timestamp."""
    logging.info(msg)


# ────────────────────────────────────────────────────────────────
# 0 ▪ helper:  PDF  →  [TextMessage, *MultiModalMessage]
# ────────────────────────────────────────────────────────────────
def pdf_to_init_messages(pdf_path: str, objective: str,
                         dpi: int = 220) -> List:
    """Return the first JSON blob + one image message per PDF page."""
    log(f"Reading PDF: {pdf_path}")
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
        log(f"Processing {len(doc)} pages")
        for page in doc:
            png_bytes = page.get_pixmap(dpi=dpi).tobytes("png")
            data_url = f"data:image/png;base64,{base64.b64encode(png_bytes).decode()}"
            image_msgs.append(
                MultiModalMessage(content=[AGImage.from_uri(data_url)],
                                  source="user")
            )

    log("Finished processing PDF")
    return [init_blob, *image_msgs]


# ────────────────────────────────────────────────────────────────
# 1 ▪ model client (GPT-4o vision)
# ────────────────────────────────────────────────────────────────
MODEL_upstream = OpenAIChatCompletionClient(model="gpt-4o-mini")
cache_store = DiskCacheStore[CHAT_CACHE_VALUE_TYPE](Cache(".autogen_cache"))
openai_client = ChatCompletionCache(MODEL_upstream, cache_store)


# ────────────────────────────────────────────────────────────────
# 2 ▪ agents
# ────────────────────────────────────────────────────────────────
planner = AssistantAgent(
    "planner",
    model_client=openai_client,
    system_message=(
        "Output STRICT JSON only.  Produce an array of task objects "
        "(description, objective, instructions, success_criteria). "
        "If clarification is required, emit a task addressed to `user` with "
        "the question in *instructions*.  When every task is complete, "
        "create ONE task for `reporter` and reply DONE."
    ),
)

executor = AssistantAgent(
    "executor",
    model_client=openai_client,
    system_message=(
        "Execute ONE task using ONLY the provided PDF page images. "
        "Return JSON {result:str, citations:[{page:int, quote:str}]} "
        "(quotes ≤200 chars)."
    ),
)

reporter = AssistantAgent(
    "reporter",
    model_client=openai_client,
    system_message=(
        "Write a Markdown report:\n"
        "  # Objective\n  # Findings (inline footnotes ¹,²,…)\n  # Conclusion\n"
        "Then list a 'Footnotes' section with ⟦n⟧ Page N: \"quote…\". "
        "Return the markdown directly."
    ),
)

# custom console prompt so the human sees the actual question

def console_input(prompt: str):
    """Custom stdin helper that prints the question *only if it’s meaningful*.

    AutoGen’s runtime passes a default prompt string ("Enter your response:")
    when it merely needs *any* user message to advance the loop.  That
    situation doesn’t require real user input, so we auto‑return an empty
    string and let the workflow continue.
    """
    prompt_clean = (prompt or "").strip()

    # Ignore the stock sentinel prompt that contains no real question
    if not prompt_clean or prompt_clean.lower().startswith("enter your response"):
        return ""  # no‑op

    print(f"  QUESTION for you ➜ {prompt_clean}")
    return input("📝  Your reply: ")

user = UserProxyAgent(
    "user",
    description="Document owner who answers clarification questions briefly.",
    input_func=console_input,
)


TEAM = [planner, executor, reporter, user]

# ────────────────────────────────────────────────────────────────
# 3 ▪ pretty‑print helper for plans
# ────────────────────────────────────────────────────────────────

def _print_plan(raw: str):
    """Try to parse & pretty‑print a JSON array of task objects."""
    try:
        tasks = json.loads(raw)
        if isinstance(tasks, list) and tasks and isinstance(tasks[0], dict):
            print("\n📑  Current plan (", len(tasks), " tasks)\n" + "=" * 40)
            for idx, t in enumerate(tasks, 1):
                desc = t.get("description") or t.get("objective") or ""
                print(f"{idx:>2}. {desc}")
            print("=" * 40)
    except Exception:
        pass  # not a plan update


# ────────────────────────────────────────────────────────────────
# 4 ▪ orchestrator (streaming, with plan logging)
# ────────────────────────────────────────────────────────────────
async def assess_pdf(pdf_path: str, objective: str) -> str:
    init_msgs = pdf_to_init_messages(pdf_path, objective)

    team = RoundRobinGroupChat(participants=TEAM, max_turns=None)

    reporter_markdown: str | None = None

    from autogen_agentchat.teams import events as _events  # local import for clarity

    # ------------------------------------------------------------------
    # Stream through all events; pretty‑print plans and capture final report
    # ------------------------------------------------------------------
    from autogen_agentchat.teams.events import GroupChatAgentResponse  # v0.6.1

    async for evt in team.run_stream(task=init_msgs):
        # AutoGen ≥0.6 emits high‑level *event* objects, not raw messages.
        if isinstance(evt, GroupChatAgentResponse):
            msg = evt.chat_message  # TextMessage / MultiModalMessage

            if msg.source == "planner":
                raw_json = (
                    msg.content.strip()
                    .removeprefix("```json").removeprefix("```")
                    .removesuffix("```")
                )
                _print_plan(raw_json)

            elif msg.source == "reporter":
                reporter_markdown = msg.content

    if reporter_markdown is None:
        raise RuntimeError("Reporter did not produce a summary.")("Reporter did not produce a summary.")
    return reporter_markdown


# ────────────────────────────────────────────────────────────────
# 4 ▪ orchestrator
# ────────────────────────────────────────────────────────────────
async def assess_pdf(pdf_path: str, objective: str) -> str:
    init_msgs = pdf_to_init_messages(pdf_path, objective)

    team = RoundRobinGroupChat(participants=TEAM, max_turns=None)

    # Provide the initial messages list via the `task=` param (0.6 API)
    task_result = await team.run(task=init_msgs)

    # Reporter’s markdown is the last message content
    return task_result.messages[-1].content

# ────────────────────────────────────────────────────────────────────
# 5 · CLI / demo
# ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pdf_file= None

    if len(sys.argv) < 3:
        log("Running agentic PDF assessor with default input…")
        pdf_file = pathlib.Path("./data/test_pdfs/sugars-factsheet.pdf").expanduser()
        objective = ("WHO’s recommended daily limit for free "
                     "sugar intake and give at least two health "
                     "risks of excessive sugar consumption?")
    else:
        pdf_file = pathlib.Path(sys.argv[1]).expanduser()
        objective = " ".join(sys.argv[2:])

    log(f"Objective: {objective}")

    if not pdf_file.is_file():
        log(f"PDF not found: {pdf_file}")
        sys.exit(1)

    log("Starting interactive assessment…")
    md_report = asyncio.run(assess_pdf(str(pdf_file), objective))
    log("FINAL MARKDOWN REPORT")
    print(md_report)
