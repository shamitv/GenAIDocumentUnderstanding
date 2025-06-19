#!/usr/bin/env python
"""
Vision‑enabled PDF assessor — AutoGen 0.6.1‑compatible
=====================================================
Run from a shell:

    python agentic_pdf_assessor_team_vision.py <PDF_PATH> "<objective text>"

*   Renders each page of the PDF to a PNG image (via **PyMuPDF**).
*   Embeds every PNG as a base‑64 data‑URL → `autogen_core.Image` → `MultiModalMessage`.
*   Creates a four‑agent **Round‑Robin** team:
        • **planner**   – breaks work into task objects, may ask the human user.
        • **executor**  – inspects the page images and returns findings + citations.
        • **reporter**  – compiles a Markdown report with numbered footnotes.
        • **user**      – real human; only asked when the planner explicitly requests.
*   Feeds the JSON blob + images to `team.run(task=…)` (required by 0.6 API).
*   Prints the reporter’s Markdown summary when finished.

Notes & API‑compliance (AutoGen 0.6.1)
--------------------------------------
* **No deprecated kwargs** — `allow_parallel`, `startup_task`, `human_input_mode`,
  etc. were removed after 0.4; this script uses only parameters present in 0.6.1.
* **Message objects** — initial prompt and every page image are concrete
  `TextMessage` / `MultiModalMessage` instances (all subclass `BaseChatMessage`).
* **User input** — we supply a custom `input_func` so the *actual* clarification
  question is printed before the console waits for an answer.
"""

from __future__ import annotations
import asyncio, base64, datetime as dt, io, json, logging, pathlib, sys
from typing import List

import fitz                                         # PyMuPDF
from autogen_agentchat.teams._group_chat._events import GroupChatAgentResponse, GroupChatMessage

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

# ──────────────────────────────────────────────────────────────
# Debugging helper – subclass of AssistantAgent
# ──────────────────────────────────────────────────────────────

class DebugAssistantAgent(AssistantAgent):
    """AssistantAgent that lets you break *right before* the model is called.

    AutoGen ≥0.6 funnels *all* sync & async completions through the protected
    `_chat()` method.  By overriding it we intercept **every** request — even
    function‑calling branches — without touching the public API.
    """

    # single sync/async entry‑point used by AssistantAgent
    async def _chat(self, messages, **kwargs):  # type: ignore[override]
        # Pretty preview (first 180 chars of the user/system delta)
        last_user = next((m for m in reversed(messages) if m["role"] != "tool"), {})
        prompt_snip = (last_user.get("content", "")[:180] + "…")
        print(f" 🛠️  {self.name} → about to call LLM. Prompt preview: {prompt_snip} {'-'*60}")

        # Uncomment for interactive debugging
        # breakpoint()

        return await super()._chat(messages, **kwargs)


class DebugOpenAIChatCompletionClient(OpenAIChatCompletionClient):
    """Wraps the real OpenAI client so you can set a breakpoint on *every* call.

    This is the most reliable interception point because *all* AutoGen
    assistants eventually call `create_chat_completion`.  We log/preview the
    first user‑visible part of the messages list (trimmed) before delegating to
    the superclass.  Place a `breakpoint()` for interactive debugging.
    """

    async def create_chat_completion(self, messages, *args, **kwargs):  # type: ignore[override]
        # Extract a short preview from the last *non‑tool* message
        last_msg = next((m for m in reversed(messages) if m.get("role") != "tool"), {})
        snippet = (last_msg.get("content", "")[:200] + "…")
        print(f"⚙️  LLM call — preview:{snippet}{'-'*60}")
        # Uncomment for step‑through debugging
        # breakpoint()
        return await super().create_chat_completion(messages, *args, **kwargs)


# ────────────────────────────────────────────────────────────────
# 1 ▪ model client (GPT-4o vision)
# ────────────────────────────────────────────────────────────────
MODEL_upstream = OpenAIChatCompletionClient(model="gpt-4o-mini")
cache_store = DiskCacheStore[CHAT_CACHE_VALUE_TYPE](Cache(".autogen_cache"))
openai_client = ChatCompletionCache(MODEL_upstream, cache_store)


# ────────────────────────────────────────────────────────────────
# 2 ▪ agents
# ────────────────────────────────────────────────────────────────
planner = DebugAssistantAgent(
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

executor = DebugAssistantAgent(
    "executor",
    model_client=openai_client,
    system_message=(
        "Execute ONE task using ONLY the provided PDF page images. "
        "Return JSON {result:str, citations:[{page:int, quote:str}]} "
        "(quotes ≤200 chars)."
    ),
)

reporter = DebugAssistantAgent(
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

def _pretty_plan(raw: str):
    try:
        tasks = json.loads(raw)
        if isinstance(tasks, list):
            print("\n📑 Plan (" + str(len(tasks)) + " tasks)\n" + "=" * 32)
            for i, t in enumerate(tasks, 1):
                print(f"{i:>2}. {t.get('description', t.get('objective', ''))}")
            print("=" * 32)
    except Exception:
        pass


# ────────────────────────────────────────────────────────────────
# 4 ▪ orchestrator (streaming, with plan logging)
# ────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────
# 3 ▪ orchestrator – stream & log plans
# ──────────────────────────────────────────────────────────────
async def assess_pdf(pdf: str, objective: str) -> str:
    team = RoundRobinGroupChat(participants=TEAM)
    reporter_md: str | None = None
    init_msgs = pdf_to_init_messages(pdf, objective)
    aiter = team.run_stream(task=init_msgs)

    first_event = await anext(aiter)
    print("First event:", first_event)
    async for ev in aiter:

        if isinstance(ev, GroupChatAgentResponse):
            msg = ev.chat_message
        elif isinstance(ev, GroupChatMessage):
            msg = ev.message  # covers some reply paths
        else:
            continue

        if msg.source == "planner":
            cleaned = (
                msg.content.strip()
                .removeprefix("```json").removeprefix("```")
                .removesuffix("```")
            )
            _pretty_plan(cleaned)
        elif msg.source == "reporter":
            reporter_md = msg.content

    if reporter_md is None:
        raise RuntimeError("Reporter did not produce a summary.")
    return reporter_md

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
        pdf_file = pathlib.Path("./data/test_pdfs/Intro_AutoGen.pdf").expanduser()
        objective = ("What is this document about? "
                     "Which patterns are discussed? "
                     )
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
